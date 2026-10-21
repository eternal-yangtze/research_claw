"""Reuse a local formula model only on disputed regions with a tighter crop.

This is the same recognizer with a different image, not an independent vote.
"""
from __future__ import annotations

import json
import os
import time
import uuid
from importlib.metadata import version
from pathlib import Path

from researchops.core.artifacts import file_hash, write_json


def refine_disputed_regions(reports: list[Path], output_root: Path, *, max_regions: int = 12,
                            max_new_tokens: int = 512) -> list[dict]:
    os.environ.setdefault("HF_HUB_OFFLINE", "1")
    import torch
    import pymupdf
    from PIL import Image
    from docling.datamodel.pipeline_options import CodeFormulaVlmOptions
    from docling.datamodel.accelerator_options import AcceleratorOptions, AcceleratorDevice
    from docling.datamodel.vlm_engine_options import TransformersVlmEngineOptions
    from docling.models.stages.code_formula.code_formula_vlm_model import CodeFormulaVlmModel
    from docling.models.inference_engines.vlm import VlmEngineInput
    from docling.models.utils.generation_utils import TailRepetitionStopper

    if max_regions < 1 or not 1 <= max_new_tokens <= 2048:
        raise ValueError("positive region cap and formula token cap <= 2048 required")
    torch.set_num_threads(4)
    options = CodeFormulaVlmOptions.from_preset("codeformulav2")
    options.engine_options = TransformersVlmEngineOptions(compile_model=False, load_in_8bit=False)
    options.model_spec.max_new_tokens = max_new_tokens
    stage = CodeFormulaVlmModel(enabled=True, enable_remote_services=False, artifacts_path=None,
                                options=options, accelerator_options=AcceleratorOptions(num_threads=4, device=AcceleratorDevice.CPU))
    from huggingface_hub import try_to_load_from_cache
    config_path = try_to_load_from_cache("docling-project/CodeFormulaV2", "config.json")
    checkpoint = {"repo": "docling-project/CodeFormulaV2",
                  "cached_revision": Path(config_path).parent.name if isinstance(config_path, str) else "unknown",
                  "config_sha256": file_hash(config_path) if isinstance(config_path, str) else None,
                  "versions": {name: version(name) for name in ["docling", "transformers", "torch"]}}
    results = []
    remaining = max_regions
    for report_path in reports:
        report = json.loads(report_path.read_text(encoding="utf-8"))
        pdf_path = Path(report["pdf_path"])
        parser_path = Path(report["parser_artifact"])
        if file_hash(pdf_path) != report["pdf_sha256"] or file_hash(parser_path) != report["parser_sha256"]:
            raise ValueError("formula refinement input hash changed")
        parser = json.loads(parser_path.read_text(encoding="utf-8"))
        root = output_root.resolve() / uuid.uuid4().hex
        root.mkdir(parents=True, exist_ok=False)
        by_ref = {item["self_ref"]: item for item in parser["document"].get("texts", [])}
        candidates = []
        with pymupdf.open(pdf_path) as pdf:
            for region in report["regions"]:
                if remaining == 0 or region["decision"]["status"] == "accepted":
                    continue
                locations = region["locations"]
                if len(locations) != 1 or "bbox_top_left" not in locations[0]:
                    continue
                location = locations[0]
                page = pdf[location["page"] - 1]
                box = pymupdf.Rect(location["bbox_top_left"])
                box = pymupdf.Rect(box.x0 - 1, box.y0 - 1, box.x1 + 1, box.y1 + 1) & page.rect
                pixmap = page.get_pixmap(matrix=pymupdf.Matrix(1.67, 1.67), clip=box)
                image = Image.frombytes("RGB", [pixmap.width, pixmap.height], pixmap.samples)
                crop_path = root / f"crop_{len(candidates):04d}.png"
                image.save(crop_path)
                started = time.perf_counter()
                stopper = TailRepetitionStopper()
                prediction = stage.engine.predict_batch([VlmEngineInput(
                    image=image, prompt="<formula>", temperature=0.0, max_new_tokens=max_new_tokens,
                    extra_generation_config={"skip_special_tokens": False, "custom_stopping_criteria": [stopper]})])[0]
                # Match Docling's output cleanup while preserving the complete raw prediction.
                text = prediction.text.split("<end_of_utterance>")[0]
                for token in ["</code>", "</formula>", "<loc_0><loc_0><loc_500><loc_500>"]:
                    text = text.replace(token, "")
                text = stopper.strip(text).strip()
                candidates.append({"region_id": region["region_id"], "original_latex": region["latex"],
                                   "latex": text, "raw_prediction": prediction.text,
                                   "crop_path": str(crop_path), "crop_sha256": file_hash(crop_path),
                                   "bbox_top_left": list(box), "elapsed_seconds": time.perf_counter() - started,
                                   "source_relation": "same_model_different_crop_not_independent_vote"})
                by_ref[region["region_id"]]["text"] = text
                remaining -= 1
        parser["refinement"] = {"method": "tight_crop_local_codeformulav2", "original_parser": str(parser_path),
                                "original_parser_sha256": file_hash(parser_path), "input_report": str(report_path.resolve()),
                                "input_report_sha256": file_hash(report_path), "max_new_tokens": max_new_tokens,
                                "region_cap_shared_across_reports": max_regions,
                                "checkpoint": checkpoint, "implementation_sha256": file_hash(Path(__file__)),
                                "candidates": candidates, "llm_api_calls": 0, "training": False}
        write_json(root / "refined_docling.json", parser)
        results.append({"parser_path": str(root / "refined_docling.json"), "refined_regions": len(candidates),
                        "input_report": str(report_path.resolve())})
        write_json(root / "result.json", results[-1])
    return results

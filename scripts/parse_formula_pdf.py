"""Explicit, bounded Docling formula pass; preserves existing ingestion outputs."""
from __future__ import annotations

import argparse
import json
import os
import sys
import time
from importlib.metadata import version
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))


def main() -> None:
    sys.stdout.reconfigure(encoding="utf-8")
    parser = argparse.ArgumentParser()
    parser.add_argument("--pdf", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--first-page", type=int, default=1)
    parser.add_argument("--last-page", type=int, required=True)
    parser.add_argument("--formula-margin", type=float, default=0.18)
    args = parser.parse_args()
    # Reuse cached weights without repeated HTTP metadata probes on restricted hosts.
    os.environ.setdefault("HF_HUB_OFFLINE", "1")
    import torch
    torch.set_num_threads(4)
    from docling.datamodel.base_models import InputFormat
    from docling.datamodel.pipeline_options import PdfPipelineOptions
    from docling.datamodel.accelerator_options import AcceleratorOptions, AcceleratorDevice
    from docling.document_converter import DocumentConverter, PdfFormatOption
    from researchops.core.artifacts import file_hash

    output = Path(args.output)
    output.mkdir(parents=True, exist_ok=False)
    options = PdfPipelineOptions(do_ocr=False, do_table_structure=False,
                                 do_formula_enrichment=True)
    options.accelerator_options = AcceleratorOptions(num_threads=4, device=AcceleratorDevice.CPU)
    options.code_formula_options.expansion_factor = args.formula_margin
    from docling.datamodel.vlm_engine_options import TransformersVlmEngineOptions
    options.code_formula_options.engine_options = TransformersVlmEngineOptions(compile_model=False, load_in_8bit=False)
    started = time.perf_counter()
    print("Starting local Docling layout + formula recognition", flush=True)
    converter = DocumentConverter(format_options={InputFormat.PDF: PdfFormatOption(pipeline_options=options)})
    result = converter.convert(args.pdf, page_range=(args.first_page, args.last_page))
    document = result.document.export_to_dict()
    payload = {"schema_version": "formula-parser-v1", "pdf_path": str(Path(args.pdf).resolve()),
               "pdf_sha256": file_hash(Path(args.pdf)), "parser": "docling",
               "parser_version": version("docling"), "page_range": [args.first_page, args.last_page],
               "formula_generation_limit": "upstream Docling stage sets input max_new_tokens=2048",
               "device": "cpu", "torch_threads": 4,
               "formula_margin": args.formula_margin,
               "elapsed_seconds": time.perf_counter() - started, "status": str(result.status),
               "errors": [str(e) for e in result.errors], "document": document}
    (output / "docling.json").write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")
    (output / "document.md").write_text(result.document.export_to_markdown(), encoding="utf-8")
    print(json.dumps({k: v for k, v in payload.items() if k != "document"}, ensure_ascii=False))


if __name__ == "__main__":
    main()

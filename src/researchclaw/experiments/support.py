"""Trusted manifest, RNG and checkpoint utilities exposed to generated experiments.

No model implementation is supplied here: the Coding Agent writes it.
"""
from __future__ import annotations

import hashlib
import json
import random
from pathlib import Path

import numpy as np
import torch
from PIL import Image
from torch.utils.data import Dataset
from torchvision import transforms as T


def sha256(path):
    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def spec_hash(spec):
    return hashlib.sha256(json.dumps(spec, sort_keys=True, ensure_ascii=False).encode()).hexdigest()


def seed_everything(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


class ManifestDataset(Dataset):
    def __init__(self, dataset_spec, split, *, fraction=1.0, smoke=False, allow_test=False,
                 mean=(0.485, 0.456, 0.406), std=(0.229, 0.224, 0.225)):
        if split not in {"train", "val", "test"}:
            raise ValueError("Unknown split")
        if split == "test" and (smoke or not allow_test):
            raise ValueError("Test access prohibited for this stage")
        self.root = Path(dataset_spec["root"]).resolve()
        manifest_path = Path(dataset_spec["manifest_path"])
        if sha256(manifest_path) != dataset_spec["manifest_sha256"]:
            raise ValueError("Dataset manifest hash mismatch")
        self.rows = [json.loads(line) for line in manifest_path.read_text(encoding="utf-8").splitlines() if line.strip()]
        self.rows = [row for row in self.rows if row["included"] and row["split"] == split]
        if split == "train":
            subset = dataset_spec["train_fractions"][str(float(fraction))]
            if sha256(subset["path"]) != subset["sha256"]:
                raise ValueError("Training subset hash mismatch")
            selected = set(json.loads(Path(subset["path"]).read_text(encoding="utf-8")))
            self.rows = [row for row in self.rows if row["sample_id"] in selected]
            if len(self.rows) != len(selected):
                raise ValueError("Subset contains missing or excluded samples")
        if not self.rows:
            raise ValueError("Empty split")
        self.smoke = smoke
        self.arrays = {}
        size = 32 if smoke else 224
        operations = ([T.RandomResizedCrop(size, scale=(0.8, 1.0), ratio=(0.75, 4/3)), T.RandomHorizontalFlip()]
                      if split == "train" else [T.Resize(36 if smoke else 256), T.CenterCrop(size)])
        self.transform = T.Compose([*operations, T.ToTensor(), T.Normalize(mean, std)])

    def __len__(self):
        return len(self.rows)

    def __getitem__(self, index):
        row = self.rows[index]
        if row.get("loader") == "numpy_rgb":
            path = (self.root / row["array_path"]).resolve()
            if self.root not in path.parents:
                raise ValueError("Array path escapes dataset")
            if str(path) not in self.arrays:
                self.arrays[str(path)] = np.load(path, mmap_mode="r", allow_pickle=False)
            array = self.arrays[str(path)][row["index"]]
            digest = hashlib.sha256(str((32, 32)).encode() + array.tobytes()).hexdigest()
            if digest != row["content_sha256"]:
                raise ValueError("Image pixels changed")
            image = Image.fromarray(array)
        else:
            path = (self.root / row["path"]).resolve()
            if self.root not in path.parents:
                raise ValueError("Image path escapes dataset")
            if row.get("encoded_sha256") and sha256(path) != row["encoded_sha256"]:
                raise ValueError("Image encoded bytes changed")
            with Image.open(path) as source:
                image = source.convert("RGB")
            digest = hashlib.sha256(str(image.size).encode() + image.tobytes()).hexdigest()
            if digest != row["content_sha256"]:
                raise ValueError("Image pixels changed")
        return self.transform(image), int(row["label"])


def save_checkpoint(path, model, optimizer, scheduler, *, spec_hash, job_id, global_step, extra=None):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    value = {"model": model.state_dict(), "optimizer": optimizer.state_dict(),
             "scheduler": scheduler.state_dict() if scheduler is not None else None,
             "spec_hash": spec_hash, "job_id": job_id, "global_step": global_step,
             "rng": {"python": random.getstate(), "numpy": np.random.get_state(), "torch": torch.get_rng_state(),
                     "cuda": torch.cuda.get_rng_state_all() if torch.cuda.is_available() else []},
             "extra": extra or {}}
    temporary = path.with_suffix(path.suffix + ".tmp")
    torch.save(value, temporary)
    temporary.replace(path)
    return value


def load_checkpoint(path, model, optimizer, scheduler, *, spec_hash, job_id):
    # Only checkpoints created by this isolated experiment are accepted here.
    value = torch.load(path, map_location="cpu", weights_only=False)
    if value["spec_hash"] != spec_hash or value["job_id"] != job_id:
        raise ValueError("Checkpoint belongs to another configuration/job")
    model.load_state_dict(value["model"])
    optimizer.load_state_dict(value["optimizer"])
    if scheduler is not None and value["scheduler"] is not None:
        scheduler.load_state_dict(value["scheduler"])
    random.setstate(value["rng"]["python"])
    np.random.set_state(value["rng"]["numpy"])
    torch.set_rng_state(value["rng"]["torch"])
    if torch.cuda.is_available() and value["rng"]["cuda"]:
        torch.cuda.set_rng_state_all(value["rng"]["cuda"])
    return value

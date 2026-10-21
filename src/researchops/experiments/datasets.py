"""Dataset acquisition and reproducible manifests, without running training code.

Official URLs/checksums and native layout adapters follow torchvision's BSD-licensed
dataset implementations. No third-party dataset download scripts are executed.
"""
from __future__ import annotations

import hashlib
import io
import json
import math
import random
import re
import shutil
import ssl
import tarfile
import threading
import time
import urllib.error
import urllib.parse
import urllib.request
import zipfile
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path, PurePosixPath
from typing import Any

from researchops.core.artifacts import canonical_hash, file_hash, write_json, workspace_lock as dataset_lock
from researchops.core.layout import ensure_workspace_assets

DATASET_VERSION = "researchops-datasets-v2"
MAX_FILE_BYTES = 4 * 1024**3
MAX_EXTRACT_BYTES = 12 * 1024**3
HTTP_SLOTS = threading.BoundedSemaphore(12)
HF_REPOSITORIES = {"cifar10": "uoft-cs/cifar10", "cifar100": "uoft-cs/cifar100",
                   "flowers102": "xingslong/flowers-102", "pets": "timm/oxford-iiit-pet",
                   "dtd": "mcimpoi/dtd_split_1"}


def _file(name: str, md5: str, url: str) -> dict:
    # Canonical university download hosts, verified independently of the old pages.
    replacements = {
        "https://www.cs.toronto.edu/~kriz/": "https://cave.cs.toronto.edu/kriz/",
        "https://www.robots.ox.ac.uk/~vgg/data/flowers/102/": "https://thor.robots.ox.ac.uk/flowers/102/",
        "https://www.robots.ox.ac.uk/~vgg/data/pets/data/": "https://thor.robots.ox.ac.uk/pets/",
        "https://www.robots.ox.ac.uk/~vgg/data/dtd/download/": "https://thor.robots.ox.ac.uk/dtd/",
    }
    urls = [url]
    for legacy, current in replacements.items():
        if url.startswith(legacy):
            urls.insert(0, current + url[len(legacy):])
            break
    return {"filename": name, "md5": md5, "urls": urls}


REGISTRY = {
    "cifar10": {
        "name": "CIFAR-10", "classes": 10, "samples": 60000,
        "source_splits": {"train": 50000, "test": 10000},
        "landing_url": "https://www.cs.toronto.edu/~kriz/cifar.html",
        "aliases": ["cifar-10", "cifar10"], "layout": "", "task": "image_classification",
        "usage": "Research use; consult the source page and original image rights.",
        "files": [_file("cifar-10-python.tar.gz", "c58f30108f718f92721af3b95e74349a",
                        "https://www.cs.toronto.edu/~kriz/cifar-10-python.tar.gz")],
    },
    "cifar100": {
        "name": "CIFAR-100", "classes": 100, "samples": 60000,
        "source_splits": {"train": 50000, "test": 10000},
        "landing_url": "https://www.cs.toronto.edu/~kriz/cifar.html",
        "aliases": ["cifar-100", "cifar100"], "layout": "", "task": "image_classification",
        "usage": "Research use; consult the source page and original image rights.",
        "files": [_file("cifar-100-python.tar.gz", "eb9058c3a382ffc7106e4002c42a8d85",
                        "https://www.cs.toronto.edu/~kriz/cifar-100-python.tar.gz")],
    },
    "flowers102": {
        "name": "Oxford Flowers-102", "classes": 102, "samples": 8189,
        "source_splits": {"train": 1020, "val": 1020, "test": 6149},
        "landing_url": "https://www.robots.ox.ac.uk/~vgg/data/flowers/102/",
        "aliases": ["flowers102", "flowers-102", "flowers 102", "oxford flowers", "oxford-flowers", "flower102"],
        "layout": "flowers-102", "task": "image_classification",
        "usage": "Academic dataset; retain source citations and follow the source terms.",
        "files": [
            _file("102flowers.tgz", "52808999861908f626f3c1f4e79d11fa", "https://www.robots.ox.ac.uk/~vgg/data/flowers/102/102flowers.tgz"),
            _file("imagelabels.mat", "e0620be6f572b9609742df49c70aed4d", "https://www.robots.ox.ac.uk/~vgg/data/flowers/102/imagelabels.mat"),
            _file("setid.mat", "a5357ecc9cb78c4bef273ce3793fc85c", "https://www.robots.ox.ac.uk/~vgg/data/flowers/102/setid.mat"),
        ],
    },
    "pets": {
        "name": "Oxford-IIIT Pet", "classes": 37, "samples": 7349,
        "source_splits": {"train": 3680, "test": 3669},
        "landing_url": "https://www.robots.ox.ac.uk/~vgg/data/pets/",
        "aliases": ["oxford-iiit", "oxford iiit", "oxford pets", "oxford-pets", "oxford pet", "oxfordpet"],
        "layout": "oxford-iiit-pet", "task": "image_classification",
        "usage": "Research dataset; consult source terms and cite the original authors.",
        "files": [
            _file("images.tar.gz", "5c4f3ee8e5d25df40f4fd59a7f44e54c", "https://www.robots.ox.ac.uk/~vgg/data/pets/data/images.tar.gz"),
            _file("annotations.tar.gz", "95a8c909bbe2e81eed6a22bccdf3f68f", "https://www.robots.ox.ac.uk/~vgg/data/pets/data/annotations.tar.gz"),
        ],
    },
    "dtd": {
        "name": "Describable Textures Dataset (DTD)", "classes": 47, "samples": 5640,
        "source_splits": {"train": 1880, "val": 1880, "test": 1880},
        "landing_url": "https://www.robots.ox.ac.uk/~vgg/data/dtd/",
        "aliases": ["describable textures", "dtd"], "layout": "dtd", "task": "image_classification",
        "usage": "Academic dataset; consult source terms and retain dataset citation.",
        "files": [_file("dtd-r1.0.1.tar.gz", "fff73e5086ae6bdbea199a49dfb8a4c1",
                        "https://www.robots.ox.ac.uk/~vgg/data/dtd/download/dtd-r1.0.1.tar.gz")],
    },
}


class _HTTPSRedirect(urllib.request.HTTPRedirectHandler):
    def redirect_request(self, req, fp, code, msg, headers, newurl):
        # Some academic hosts still publish HTTP redirects for HTTPS archives.
        if newurl.startswith("http://"):
            newurl = "https://" + newurl[7:]
        if not newurl.startswith("https://"):
            raise ValueError("Dataset redirects must use HTTPS")
        return super().redirect_request(req, fp, code, msg, headers, newurl)


def download_verified(item: dict, destination: Path, *, timeout: int = 60) -> dict:
    """Bounded streaming download, byte-range resume, trusted MD5 and local SHA256."""
    destination.parent.mkdir(parents=True, exist_ok=True)
    algorithm = "sha256" if "sha256" in item else "md5"
    expected_hash = item[algorithm]
    if destination.is_file() and file_hash(destination, algorithm) == expected_hash:
        return {"path": str(destination), "bytes": destination.stat().st_size,
                "sha256": file_hash(destination), algorithm: expected_hash, "cached": True}
    import certifi

    opener = urllib.request.build_opener(
        urllib.request.HTTPSHandler(context=ssl.create_default_context(cafile=certifi.where())),
        _HTTPSRedirect(),
    )
    part = destination.with_suffix(destination.suffix + ".part")
    errors = []
    started = time.monotonic()
    for url in item["urls"]:
        if not url.startswith("https://"):
            raise ValueError("Dataset URLs must use HTTPS")
        if item["filename"].endswith((".tar.gz", ".tgz", ".zip")):
            try:
                ranged = _parallel_ranges(opener, url, destination, item, timeout=timeout)
                if ranged is not None:
                    return ranged
            except (OSError, ValueError, RuntimeError, urllib.error.URLError) as error:
                errors.append(f"Ranged {url}: {type(error).__name__}: {error}")
        for attempt in range(12):
            try:
                while True:
                    offset = part.stat().st_size if part.exists() else 0
                    end = offset + 4 * 1024 * 1024 - 1
                    # Range-specific query prevents intermediary caches returning a
                    # different previously cached range for the same static URL.
                    separator = "&" if "?" in url else "?"
                    request_url = f"{url}{separator}researchops_range={offset}-{end}"
                    headers = {"User-Agent": "ResearchOps/1.0 (academic dataset preparation)",
                               "Accept-Encoding": "identity", "Range": f"bytes={offset}-{end}"}
                    with opener.open(urllib.request.Request(request_url, headers=headers), timeout=timeout) as response:
                        partial = response.status == 206
                        content_range = response.headers.get("Content-Range", "")
                        match = re.fullmatch(r"bytes (\d+)-(\d+)/(\d+)", content_range)
                        total = None
                        if partial:
                            if not match or int(match[1]) != offset or int(match[2]) > end:
                                raise ValueError("Invalid download resume range")
                            total = int(match[3])
                            if total > MAX_FILE_BYTES:
                                raise ValueError("Archive exceeds download budget")
                        else:
                            offset = 0
                        expected = int(response.headers.get("Content-Length") or 0)
                        if offset + expected > MAX_FILE_BYTES:
                            raise ValueError("Archive exceeds per-file download budget")
                        size = offset
                        with part.open("ab" if partial and offset else "wb") as target:
                            while True:
                                block = response.read(64 * 1024)
                                if not block:
                                    break
                                size += len(block)
                                if size > MAX_FILE_BYTES or time.monotonic() - started > 1800:
                                    raise ValueError("Download byte/time budget exceeded")
                                target.write(block)
                        if expected and size - offset != expected:
                            raise ValueError("Incomplete HTTP response")
                        if partial and size - offset != int(match[2]) - int(match[1]) + 1:
                            raise ValueError("Incomplete HTTP range")
                        final_url = response.geturl()
                    if total is None or size == total:
                        break
                    if size > total or size <= offset:
                        raise ValueError("Download did not advance")
                actual_checksum = file_hash(part, algorithm)
                if actual_checksum != expected_hash:
                    # Keep the bad payload for inspection, but never extract it.
                    part.replace(part.with_suffix(part.suffix + ".rejected"))
                    raise ValueError(f"Dataset {algorithm} mismatch: {actual_checksum}")
                part.replace(destination)
                return {"path": str(destination), "bytes": size, "sha256": file_hash(destination),
                        algorithm: actual_checksum, "cached": False, "url": url, "final_url": final_url}
            except (OSError, ValueError, urllib.error.URLError) as error:
                errors.append(f"{url}: {type(error).__name__}: {error}")
                if isinstance(error, urllib.error.HTTPError) and error.code in {401, 403, 404, 416}:
                    break
    raise RuntimeError("; ".join(errors[-4:]))


def _parallel_ranges(opener, url: str, destination: Path, item: dict, *, timeout: int) -> dict | None:
    """Four resumable ranges per large archive, with twelve HTTP requests globally.

    Each complete range is hash-checked on resume; the assembled archive must match
    the official checksum before any parsing or extraction. Servers without byte
    ranges fall back to the sequential streaming path.
    """
    separator = "&" if "?" in url else "?"
    with HTTP_SLOTS, opener.open(urllib.request.Request(
        f"{url}{separator}researchops_range=probe",
        headers={"Range": "bytes=0-0", "User-Agent": "ResearchOps/1.0", "Accept-Encoding": "identity"},
    ), timeout=timeout) as response:
        if response.status != 206:
            return None
        match = re.fullmatch(r"bytes 0-0/(\d+)", response.headers.get("Content-Range", ""))
        if not match:
            return None
        total = int(match[1])
        if total > MAX_FILE_BYTES:
            raise ValueError("Archive exceeds download budget")
    if total < 16 * 1024 * 1024:
        return None
    chunk_bytes = 8 * 1024 * 1024
    directory = destination.parent / (destination.name + ".ranges")
    directory.mkdir(parents=True, exist_ok=True)
    metadata_path = directory / "source.json"
    algorithm = "sha256" if "sha256" in item else "md5"
    source_key = {"total": total, algorithm: item[algorithm], "chunk_bytes": chunk_bytes}
    if metadata_path.exists() and json.loads(metadata_path.read_text()) != source_key:
        raise ValueError("Ranged source metadata changed; preserve old chunks for inspection")
    write_json(metadata_path, source_key)
    # Reuse complete ranges from an interrupted sequential transfer.
    old_part = destination.with_suffix(destination.suffix + ".part")
    if old_part.exists():
        with old_part.open("rb") as stream:
            for start in range(0, min(old_part.stat().st_size, total), chunk_bytes):
                size = min(chunk_bytes, total - start)
                data = stream.read(size)
                if len(data) != size:
                    break
                target = directory / f"{start:012d}.bin"
                if not target.exists():
                    target.write_bytes(data)
                    write_json(target.with_suffix(".json"), {"sha256": file_hash(target), "bytes": size})
    started = time.monotonic()
    cancelled = threading.Event()

    def fetch(start: int) -> Path:
        end = min(total, start + chunk_bytes) - 1
        expected = end - start + 1
        target = directory / f"{start:012d}.bin"
        marker = target.with_suffix(".json")
        if target.exists() and marker.exists() and target.stat().st_size == expected:
            if file_hash(target) == json.loads(marker.read_text())["sha256"]:
                return target
        last_error = None
        for attempt in range(3):
            if cancelled.is_set() or time.monotonic() - started > 1800:
                raise ValueError("Ranged download cancelled or time budget exceeded")
            try:
                request = urllib.request.Request(
                    f"{url}{separator}researchops_range={start}-{end}",
                    headers={"Range": f"bytes={start}-{end}", "Accept-Encoding": "identity", "User-Agent": "ResearchOps/1.0"},
                )
                temporary = target.with_suffix(".part")
                with HTTP_SLOTS, opener.open(request, timeout=timeout) as response:
                    if response.status != 206 or response.headers.get("Content-Range") != f"bytes {start}-{end}/{total}":
                        raise ValueError("Server did not honor requested byte range")
                    size = 0
                    with temporary.open("wb") as stream:
                        while True:
                            block = response.read(64 * 1024)
                            if not block:
                                break
                            size += len(block)
                            if size > expected or cancelled.is_set() or time.monotonic() - started > 1800:
                                raise ValueError("Range exceeded byte/time budget")
                            stream.write(block)
                    if size != expected:
                        raise ValueError("Truncated range download")
                temporary.replace(target)
                write_json(marker, {"sha256": file_hash(target), "bytes": expected})
                return target
            except (OSError, ValueError, urllib.error.URLError) as error:
                last_error = error
        raise RuntimeError(f"Range {start}-{end} failed: {last_error}")

    with ThreadPoolExecutor(max_workers=4) as pool:
        futures = {pool.submit(fetch, start): start for start in range(0, total, chunk_bytes)}
        try:
            for future in as_completed(futures):
                future.result()
        except Exception:
            cancelled.set()
            for future in futures:
                future.cancel()
            raise
    assembled = destination.with_suffix(destination.suffix + ".assembled")
    with assembled.open("wb") as stream:
        for start in range(0, total, chunk_bytes):
            with (directory / f"{start:012d}.bin").open("rb") as source:
                shutil.copyfileobj(source, stream)
    if file_hash(assembled, algorithm) != item[algorithm]:
        assembled.replace(assembled.with_suffix(".rejected"))
        raise ValueError("Assembled archive does not match official checksum")
    assembled.replace(destination)
    return {"path": str(destination), "bytes": total, "sha256": file_hash(destination),
            algorithm: item[algorithm], "cached": False, "url": url, "transfer": "four_bounded_parallel_ranges"}


def safe_extract(archive: Path, target: Path) -> None:
    """Reject traversal, links, special files and oversized archives before extracting."""
    target = target.resolve()
    target.mkdir(parents=True, exist_ok=True)

    def check(name: str) -> Path:
        normalized = name.replace("\\", "/")
        member_path = PurePosixPath(normalized)
        if member_path.is_absolute() or ".." in member_path.parts or ":" in normalized:
            raise ValueError(f"Unsafe archive member: {name}")
        result = (target / normalized).resolve()
        if result != target and target not in result.parents:
            raise ValueError(f"Archive member escapes destination: {name}")
        return result

    if zipfile.is_zipfile(archive):
        with zipfile.ZipFile(archive) as source:
            members = source.infolist()
            if sum(item.file_size for item in members) > MAX_EXTRACT_BYTES or len(members) > 250000:
                raise ValueError("Archive extraction budget exceeded")
            for item in members:
                check(item.filename)
                if (item.external_attr >> 16) & 0o170000 == 0o120000:
                    raise ValueError("Archive symlinks are not allowed")
            for item in members:
                destination = check(item.filename)
                if item.is_dir():
                    destination.mkdir(parents=True, exist_ok=True)
                else:
                    destination.parent.mkdir(parents=True, exist_ok=True)
                    with source.open(item) as inp, destination.open("wb") as out:
                        shutil.copyfileobj(inp, out)
    else:
        with tarfile.open(archive, "r:*") as source:
            members = source.getmembers()
            if sum(item.size for item in members) > MAX_EXTRACT_BYTES or len(members) > 250000:
                raise ValueError("Archive extraction budget exceeded")
            for item in members:
                check(item.name)
                if not (item.isfile() or item.isdir()):
                    raise ValueError(f"Archive links/special files are not allowed: {item.name}")
            for item in members:
                destination = check(item.name)
                if item.isdir():
                    destination.mkdir(parents=True, exist_ok=True)
                else:
                    destination.parent.mkdir(parents=True, exist_ok=True)
                    with source.extractfile(item) as inp, destination.open("wb") as out:
                        shutil.copyfileobj(inp, out)


def stratified_validation(rows: list[dict], ratio: float = 0.1, seed: int = 42) -> list[dict]:
    """Split only the official training pool; the official test set is untouched."""
    groups: dict[int, list[dict]] = {}
    for row in rows:
        groups.setdefault(row["label"], []).append(row)
    result = []
    for label, group in sorted(groups.items()):
        ordered = sorted(group, key=lambda row: row["sample_id"])
        random.Random(f"{seed}:{label}").shuffle(ordered)
        count = max(1, int(len(ordered) * ratio))
        if count >= len(ordered):
            raise ValueError("Insufficient class samples for train/validation")
        for index, row in enumerate(ordered):
            result.append({**row, "split": "val" if index < count else "train"})
    return result


def native_samples(dataset_id: str, root: Path) -> tuple[list[dict], list[str]]:
    """Use installed torchvision readers; no network access or download=True here."""
    import numpy as np
    from PIL import Image
    from torchvision import datasets as tv

    root = root.resolve()
    rows = []
    classes = []
    hf_index = root / "hf_index.json"
    if hf_index.exists():
        index = json.loads(hf_index.read_text(encoding="utf-8"))
        classes = index["classes"]
        if index.get("array_storage"):
            arrays = {}
            paths = {}
            for item in index["array_storage"]:
                path = (root / item["path"]).resolve()
                if root not in path.parents or file_hash(path) != item["sha256"]:
                    raise ValueError("Prepared image array differs from pinned conversion")
                paths[item["path"]] = path
            try:
                for key, path in paths.items():
                    arrays[key] = np.load(path, mmap_mode="r", allow_pickle=False)
                for source in index["samples"]:
                    image = arrays[source["array_path"]][source["index"]]
                    if image.shape != (32, 32, 3) or image.dtype != np.uint8:
                        raise ValueError("Invalid normalized CIFAR image")
                    content_hash = hashlib.sha256(str((32, 32)).encode() + image.tobytes()).hexdigest()
                    if content_hash != source["expected_pixel_sha256"]:
                        raise ValueError("Normalized image does not match decoded source pixels")
                    rows.append({**source, "content_sha256": content_hash})
            finally:
                # Windows cannot replace an array file while a mmap still holds it,
                # including through an exception traceback during automatic repair.
                for array in arrays.values():
                    array._mmap.close()
            train = [r for r in rows if r["split"] == "train"]
            return ([r for r in rows if r["split"] != "train"] + stratified_validation(train), classes)
        def verify_image(source):
            path = (root / source["path"]).resolve()
            if root not in path.parents:
                raise ValueError("Image path escapes dataset root")
            if source.get("encoded_sha256") and file_hash(path) != source["encoded_sha256"]:
                raise ValueError(f"Prepared image differs from pinned mirror bytes: {path.name}")
            with Image.open(path) as image:
                image.verify()
            with Image.open(path) as image:
                image = image.convert("RGB")
                content_hash = hashlib.sha256(str(image.size).encode() + image.tobytes()).hexdigest()
            return {**source, "content_sha256": content_hash}
        with ThreadPoolExecutor(max_workers=8) as pool:
            # Keep pending futures bounded even for 60,000 tiny CIFAR images.
            for offset in range(0, len(index["samples"]), 256):
                rows.extend(pool.map(verify_image, index["samples"][offset:offset + 256]))
        if dataset_id in {"cifar10", "cifar100", "pets"}:
            train = [r for r in rows if r["split"] == "train"]
            rows = [r for r in rows if r["split"] != "train"] + stratified_validation(train)
        return rows, classes
    if dataset_id in {"cifar10", "cifar100"}:
        cls = tv.CIFAR10 if dataset_id == "cifar10" else tv.CIFAR100
        for train in (True, False):
            data = cls(str(root), train=train, download=False)
            classes = list(data.classes)
            source_split = "train" if train else "test"
            current = []
            for index, (image, label) in enumerate(zip(data.data, data.targets)):
                if image.shape != (32, 32, 3) or image.dtype != np.uint8:
                    raise ValueError("Invalid CIFAR image tensor")
                current.append({"sample_id": f"{source_split}:{index}", "label": int(label),
                                "source_split": source_split, "split": source_split,
                                "loader": dataset_id, "index": index,
                                "content_sha256": hashlib.sha256(image.tobytes()).hexdigest()})
            rows.extend(stratified_validation(current) if train else current)
    else:
        definitions = {
            "flowers102": (tv.Flowers102, ["train", "val", "test"]),
            "pets": (tv.OxfordIIITPet, ["trainval", "test"]),
            "dtd": (tv.DTD, ["train", "val", "test"]),
        }
        cls, splits = definitions[dataset_id]
        for split in splits:
            data = cls(str(root), split=split, download=False)
            classes = list(data.classes)
            paths = data._images if dataset_id == "pets" else data._image_files
            current = []
            for path, label in zip(paths, data._labels):
                path = Path(path).resolve()
                if root not in path.parents:
                    raise ValueError("Image path escapes dataset root")
                with Image.open(path) as image:
                    image.verify()
                # Reopen and decode for a pixel-level hash (catches format-equivalent duplicates).
                with Image.open(path) as image:
                    image = image.convert("RGB")
                    if min(image.size) < 1:
                        raise ValueError(f"Empty image: {path}")
                    content_hash = hashlib.sha256(
                        str(image.size).encode() + image.tobytes()
                    ).hexdigest()
                relative = path.relative_to(root).as_posix()
                current.append({"sample_id": relative, "label": int(label),
                                "source_split": split, "split": split, "path": relative,
                                "content_sha256": content_hash})
            rows.extend(stratified_validation(current) if split == "trainval" else current)
    return rows, classes


def build_manifest(dataset_id: str, root: Path, output: Path) -> dict:
    rows, classes = native_samples(dataset_id, root)
    definition = REGISTRY[dataset_id]
    if len(rows) != definition["samples"] or len(classes) != definition["classes"]:
        raise ValueError(f"Unexpected sample/class count for {dataset_id}")
    source_counts = {}
    for row in rows:
        source_split = "train" if row["source_split"] == "trainval" else row["source_split"]
        source_counts[source_split] = source_counts.get(source_split, 0) + 1
    if definition.get("source_splits") and source_counts != definition["source_splits"]:
        raise ValueError(f"Mirror does not preserve official source split counts: {source_counts}")
    if len({row["sample_id"] for row in rows}) != len(rows):
        raise ValueError("Overlapping sample IDs between official splits")
    if any(row["label"] < 0 or row["label"] >= len(classes) for row in rows):
        raise ValueError("Invalid class label")
    # Keep official evaluation pools intact. Exclude train/val pixel duplicates of
    # test, then train duplicates of validation, then duplicates within training.
    test_hashes = {r["content_sha256"] for r in rows if r["split"] == "test"}
    val_hashes = {r["content_sha256"] for r in rows if r["split"] == "val"}
    seen_train = set()
    exclusions: dict[str, int] = {}
    for row in sorted(rows, key=lambda r: r["sample_id"]):
        reason = None
        if row["split"] != "test" and row["content_sha256"] in test_hashes:
            reason = "duplicate_of_test"
        elif row["split"] == "train" and row["content_sha256"] in val_hashes:
            reason = "duplicate_of_validation"
        elif row["split"] == "train" and row["content_sha256"] in seen_train:
            reason = "duplicate_within_train"
        if row["split"] == "train":
            seen_train.add(row["content_sha256"])
        row["included"] = reason is None
        if reason:
            row["exclusion_reason"] = reason
            exclusions[reason] = exclusions.get(reason, 0) + 1
    for split in ("train", "val", "test"):
        labels = {r["label"] for r in rows if r["split"] == split and r["included"]}
        if labels != set(range(len(classes))):
            raise ValueError(f"Missing classes in {split} after leakage filtering")
    output.mkdir(parents=True, exist_ok=True)
    manifest_path = output / "samples.jsonl"
    temporary = manifest_path.with_suffix(".jsonl.tmp")
    with temporary.open("w", encoding="utf-8") as stream:
        for row in sorted(rows, key=lambda r: (r["split"], r["sample_id"])):
            stream.write(json.dumps(row, sort_keys=True) + "\n")
    temporary.replace(manifest_path)
    fractions = {}
    groups: dict[int, list[str]] = {}
    for row in rows:
        if row["split"] == "train" and row["included"]:
            groups.setdefault(row["label"], []).append(row["sample_id"])
    for label, group in groups.items():
        group.sort()
        random.Random(f"42:{dataset_id}:{label}").shuffle(group)
    for fraction in (0.1, 0.25, 1.0):
        ids = sorted(sample_id for group in groups.values()
                     for sample_id in group[:max(1, math.ceil(len(group) * fraction))])
        fraction_path = output / f"train_fraction_{fraction:g}.json"
        write_json(fraction_path, ids)
        fractions[str(fraction)] = {"count": len(ids), "path": str(fraction_path),
                                    "sha256": file_hash(fraction_path)}
    result = {
        "schema_version": DATASET_VERSION, "dataset_id": dataset_id, "name": definition["name"],
        "status": "verified", "root": str(root), "classes": classes, "class_count": len(classes),
        "sample_count": len(rows), "split_counts": {
            split: sum(r["split"] == split and r["included"] for r in rows)
            for split in ("train", "val", "test")
        },
        "original_split_counts": {s: sum(r["split"] == s for r in rows) for s in ("train", "val", "test")},
        "source_split_counts": source_counts,
        "exclusions": exclusions, "manifest_path": str(manifest_path), "manifest_sha256": file_hash(manifest_path),
        "train_fractions": fractions, "split_seed": 42, "official_test_preserved": True,
        "test_used_for": "identity leakage audit and final evaluation only; never tuning",
        "checks": {"every_sample_readable": True, "labels_valid": True, "sample_ids_disjoint": True,
                   "pixel_duplicates_filtered": True, "all_classes_present_per_split": True},
        "provenance": definition,
    }
    write_json(output / "verification.json", result)
    return result


class DatasetManager:
    def __init__(self, workspace: Path, source: str = "auto"):
        self.workspace = workspace.resolve()
        ensure_workspace_assets(self.workspace)
        self.root = self.workspace / "datasets"
        self.root.mkdir(parents=True, exist_ok=True)
        if source not in {"auto", "official", "huggingface"}:
            raise ValueError("Invalid dataset source")
        self.source = source

    def acquire_many(self, dataset_ids: list[str], workers: int = 3) -> list[dict]:
        if not 1 <= workers <= 5:
            raise ValueError("Dataset workers must be between 1 and 5")
        results = {}
        with ThreadPoolExecutor(max_workers=workers) as pool:
            futures = {pool.submit(self.acquire, dataset_id): dataset_id for dataset_id in dataset_ids}
            for future in as_completed(futures):
                dataset_id = futures[future]
                results[dataset_id] = future.result()
        return [results[dataset_id] for dataset_id in dataset_ids]

    def acquire(self, dataset_id: str, *, offline: bool = False) -> dict:
        with dataset_lock(self.root / dataset_id):
            return self._acquire_locked(dataset_id, offline=offline)

    def _acquire_locked(self, dataset_id: str, *, offline: bool = False) -> dict:
        hf_error = None
        if self.source in {"auto", "huggingface"} and dataset_id in HF_REPOSITORIES:
            try:
                return self._acquire_huggingface(dataset_id, offline=offline)
            except Exception as error:
                hf_error = f"{type(error).__name__}: {error}"
                if self.source == "huggingface":
                    definition = REGISTRY[dataset_id]
                    result = {"dataset_id": dataset_id, "name": definition["name"], "status": "manual_required",
                              "files": [], "error": hf_error, "landing_url": definition["landing_url"],
                              "mirror_url": f"https://huggingface.co/datasets/{HF_REPOSITORIES[dataset_id]}",
                              "import_command": f'.\\.venv\\Scripts\\python.exe researchops.py --workspace "{self.workspace}" dataset-import {dataset_id} --path "YOUR_DOWNLOAD_DIRECTORY"'}
                    write_json(self.root / dataset_id / "state.json", result)
                    return result
        definition = REGISTRY[dataset_id]
        directory = self.root / dataset_id
        raw = directory / "raw"
        native = directory / "native"
        raw.mkdir(parents=True, exist_ok=True)
        native.mkdir(parents=True, exist_ok=True)
        state = {"dataset_id": dataset_id, "name": definition["name"], "status": "acquiring", "files": []}
        if hf_error:
            state["mirror_error"] = hf_error
        state_path = directory / "state.json"
        write_json(state_path, state)
        try:
            for item in definition["files"]:
                path = raw / item["filename"]
                if offline:
                    if not path.is_file() or file_hash(path, "md5") != item["md5"]:
                        raise ValueError(f"Provide original official file: {item['filename']}")
                    downloaded = {"path": str(path), "bytes": path.stat().st_size,
                                  "sha256": file_hash(path), "md5": item["md5"], "cached": True}
                else:
                    downloaded = download_verified(item, path)
                state["files"].append(downloaded)
                write_json(state_path, state)
                target = native / definition["layout"]
                extraction_hash = canonical_hash({"archive": downloaded["sha256"], "version": DATASET_VERSION})
                marker = directory / f"extracted_{item['filename']}.json"
                if item["filename"].endswith((".tar.gz", ".tgz", ".zip", ".tar")):
                    if not marker.is_file() or json.loads(marker.read_text())["hash"] != extraction_hash:
                        safe_extract(path, target)
                        write_json(marker, {"hash": extraction_hash})
                else:
                    target.mkdir(parents=True, exist_ok=True)
                    destination = target / item["filename"]
                    if not destination.exists() or file_hash(destination) != downloaded["sha256"]:
                        shutil.copy2(path, destination)
            # Recheck/read all native samples on resume too; existence is not verification.
            try:
                verification = build_manifest(dataset_id, native, directory / "prepared")
            except (OSError, ValueError, RuntimeError):
                # Verified raw archives repair interrupted or damaged extraction;
                # ask for manual action only if the repair also cannot validate.
                for item in definition["files"]:
                    path = raw / item["filename"]
                    target = native / definition["layout"]
                    if item["filename"].endswith((".tar.gz", ".tgz", ".zip", ".tar")):
                        safe_extract(path, target)
                    else:
                        target.mkdir(parents=True, exist_ok=True)
                        shutil.copy2(path, target / item["filename"])
                verification = build_manifest(dataset_id, native, directory / "prepared")
            state.update(status="verified", verification=verification)
            write_json(state_path, state)
        except Exception as error:
            state.update(status="manual_required", error=f"{type(error).__name__}: {error}",
                         landing_url=definition["landing_url"],
                         required_files=[item["filename"] for item in definition["files"]],
                         download_urls=[url for item in definition["files"] for url in item["urls"]],
                         import_command=f'.\\.venv\\Scripts\\python.exe researchops.py --workspace "{self.workspace}" dataset-import {dataset_id} --path "YOUR_DOWNLOAD_DIRECTORY"')
            write_json(state_path, state)
        return state

    def _acquire_huggingface(self, dataset_id: str, *, offline: bool, force_rebuild: bool = False) -> dict:
        import pyarrow.parquet as pq
        import certifi

        directory = self.root / dataset_id
        directory.mkdir(parents=True, exist_ok=True)
        metadata_path = directory / "hf_source.json"
        repo = HF_REPOSITORIES[dataset_id]

        def get_json(url):
            with urllib.request.urlopen(url, timeout=30, context=ssl.create_default_context(cafile=certifi.where())) as response:
                return json.load(response)

        if metadata_path.exists():
            metadata = json.loads(metadata_path.read_text(encoding="utf-8"))
            if metadata.get("repo") != repo:
                write_json(directory / "source_history" / f"{metadata['revision']}.json", metadata)
                metadata = None
        else:
            metadata = None
        if metadata is None:
            if offline:
                raise FileNotFoundError("No pinned public-mirror metadata available offline")
            info = get_json(f"https://huggingface.co/api/datasets/{repo}")
            if info.get("gated"):
                raise ValueError("Mirror requires authentication; no token requested")
            revision = info["sha"]
            if not re.fullmatch("[0-9a-f]{40}", revision):
                raise ValueError("Invalid mirror revision")
            tree = get_json(f"https://huggingface.co/api/datasets/{repo}/tree/{revision}?recursive=true&limit=1000")
            files = []
            for entry in tree:
                path = entry.get("path", "")
                is_archive = dataset_id == "flowers102" and path.endswith((".tar.gz", ".tgz"))
                if not path.endswith(".parquet") and not is_archive:
                    continue
                if ".." in PurePosixPath(path).parts or PurePosixPath(path).is_absolute():
                    raise ValueError("Unsafe mirror path")
                split_match = re.search(r"(?:^|/)(train|validation|val|test)[-/.]", path)
                checksum = entry.get("lfs", {}).get("oid", "")
                if (not split_match and not is_archive) or not re.fullmatch("[0-9a-f]{64}", checksum):
                    raise ValueError(f"Mirror shard lacks split or SHA256: {path}")
                files.append({"filename": path.replace("/", "__"), "remote_path": path,
                              "split": {"validation": "val"}.get(split_match[1], split_match[1]) if split_match else None,
                              "sha256": checksum, "bytes": entry["size"],
                              "urls": [f"https://huggingface.co/datasets/{repo}/resolve/{revision}/{path}"]})
            if not files or sum(f["bytes"] for f in files) > 8 * 1024**3:
                raise ValueError("No mirror shards or dataset exceeds size budget")
            metadata = {"repo": repo, "revision": revision, "files": sorted(files, key=lambda item: item["filename"]),
                        "storage_namespace": revision,
                        "note": "Public repackaging; pinned revision and LFS SHA256, not official-archive byte equivalence."}
            write_json(metadata_path, metadata)
        state = {"dataset_id": dataset_id, "name": REGISTRY[dataset_id]["name"], "status": "acquiring",
                 "source_format": "huggingface_parquet", "source": metadata, "files": []}
        write_json(directory / "state.json", state)
        raw = directory / "mirror_raw"
        if metadata.get("storage_namespace"):
            raw = raw / metadata["storage_namespace"]
        raw.mkdir(parents=True, exist_ok=True)

        def acquire_shard(item):
            path = raw / item["filename"]
            if offline:
                if not path.is_file() or file_hash(path) != item["sha256"]:
                    raise ValueError(f"Mirror shard missing or hash mismatch: {path.name}")
                return {"path": str(path), "bytes": path.stat().st_size, "sha256": item["sha256"], "cached": True}
            return download_verified(item, path)

        with ThreadPoolExecutor(max_workers=2) as pool:
            for result in pool.map(acquire_shard, metadata["files"]):
                state["files"].append(result)
                write_json(directory / "state.json", state)
        if any(item["filename"].endswith((".tar.gz", ".tgz")) for item in metadata["files"]):
            return self._prepare_flowers_archive(directory, raw, metadata, state, offline=offline, force_rebuild=force_rebuild)
        if dataset_id in {"cifar10", "cifar100"}:
            return self._prepare_cifar_arrays(dataset_id, directory, raw, metadata, state, force_rebuild=force_rebuild)
        flower_ids = None
        flower_labels = None
        if dataset_id == "flowers102" and metadata["repo"] == "huggan/flowers-102-categories":
            # This compact mirror has one aggregate split. Recover the actual
            # official protocol from checksum-verified annotations and image IDs,
            # never from row order or a new random split.
            from scipy.io import loadmat
            annotations = directory / "official_annotations"
            annotation_files = []
            for item in REGISTRY[dataset_id]["files"]:
                if not item["filename"].endswith(".mat"):
                    continue
                path = annotations / item["filename"]
                if offline:
                    if not path.is_file() or file_hash(path, "md5") != item["md5"]:
                        raise ValueError("Official Flowers split/label annotations are required")
                    result = {"path": str(path), "bytes": path.stat().st_size,
                              "sha256": file_hash(path), "md5": item["md5"], "cached": True}
                else:
                    result = download_verified(item, path)
                state["files"].append(result)
                annotation_files.append({key: result[key] for key in ("path", "bytes", "sha256", "md5")})
            sets = loadmat(annotations / "setid.mat", squeeze_me=True)
            flower_ids = {int(image_id): split for split, key in (("train", "trnid"), ("val", "valid"), ("test", "tstid"))
                          for image_id in sets[key].tolist()}
            flower_labels = loadmat(annotations / "imagelabels.mat", squeeze_me=True)["labels"].tolist()
            metadata["official_annotation_files"] = annotation_files
            metadata["split_reconstruction"] = "Official .mat files matched to original image_XXXXX.jpg IDs, not mirror row order"
            write_json(metadata_path, metadata)
        native = directory / "mirror_native"
        native.mkdir(exist_ok=True)
        index_path = native / "hf_index.json"
        input_hash = canonical_hash({"version": DATASET_VERSION, "source": metadata})
        cached_index = json.loads(index_path.read_text()) if index_path.exists() else {}
        if force_rebuild or cached_index.get("input_hash") != input_hash:
            samples, classes, split_indices = [], None, {}
            for item in metadata["files"]:
                parquet = pq.ParquetFile(raw / item["filename"])
                features = json.loads(parquet.schema_arrow.metadata[b"huggingface"])["info"]["features"]
                label_column = "fine_label" if dataset_id == "cifar100" else "label"
                names = features[label_column]["names"]
                if flower_ids is not None:
                    from torchvision.datasets import Flowers102
                    names = list(Flowers102.classes)
                if classes is not None and names != classes:
                    raise ValueError("Class names differ across mirror shards")
                classes = names
                image_column = "img" if "img" in features else "image"
                source_shard_split = item["split"]
                for batch in parquet.iter_batches(batch_size=64, columns=[image_column, label_column]):
                    for row in batch.to_pylist():
                        encoded = row[image_column].get("bytes")
                        if not encoded:
                            raise ValueError("Mirror image has no embedded bytes")
                        split = source_shard_split
                        label = int(row[label_column])
                        original_id = None
                        if flower_ids is not None:
                            original_path = row[image_column].get("path") or ""
                            match = re.search(r"image_(\d{5})\.", original_path)
                            if not match:
                                raise ValueError("Compact Flowers mirror lacks original image IDs; cannot recover official splits")
                            original_id = int(match[1])
                            split = flower_ids[original_id]
                            label = int(flower_labels[original_id - 1]) - 1
                        index = split_indices.get(split, 0)
                        split_indices[split] = index + 1
                        # Encoded bytes are retained exactly; Pillow detects the format.
                        extension = (".jpg" if encoded.startswith(b"\xff\xd8") else
                                     ".png" if encoded.startswith(b"\x89PNG") else
                                     ".webp" if encoded.startswith(b"RIFF") else ".image")
                        relative = f"images/{split}/{index:07d}{extension}"
                        path = native / relative
                        path.parent.mkdir(parents=True, exist_ok=True)
                        path.write_bytes(encoded)
                        sample = {"sample_id": relative, "path": relative, "label": label,
                                  "source_split": split, "split": split,
                                  "encoded_sha256": hashlib.sha256(encoded).hexdigest()}
                        if original_id is not None:
                            sample["original_image_id"] = original_id
                            sample["mirror_label"] = int(row[label_column])
                        samples.append(sample)
            if flower_ids is not None and {r["original_image_id"] for r in samples} != set(flower_ids):
                raise ValueError("Compact Flowers mirror has missing/duplicate original image identities")
            write_json(index_path, {"input_hash": input_hash, "classes": classes, "samples": samples})
        state["status"] = "validating"
        write_json(directory / "state.json", state)
        try:
            verification = build_manifest(dataset_id, native, directory / "prepared")
        except (OSError, ValueError, RuntimeError):
            if force_rebuild:
                raise
            return self._acquire_huggingface(dataset_id, offline=True, force_rebuild=True)
        verification["provenance"] = {**verification["provenance"], "retrieval_source": metadata}
        write_json(directory / "prepared" / "verification.json", verification)
        state.update(status="verified", verification=verification)
        write_json(directory / "state.json", state)
        return state

    def _prepare_cifar_arrays(self, dataset_id, directory, raw, metadata, state, *, force_rebuild=False):
        import numpy as np
        import pyarrow.parquet as pq
        from PIL import Image
        state["prepared_format"] = "numpy_rgb"
        native = directory / "array_native"
        native.mkdir(exist_ok=True)
        index_path = native / "hf_index.json"
        input_hash = canonical_hash({"version": DATASET_VERSION, "storage": "numpy-rgb-v1", "source": metadata})
        cached = json.loads(index_path.read_text()) if index_path.exists() else {}
        if force_rebuild or cached.get("input_hash") != input_hash:
            samples, array_storage, classes = [], [], None
            for item in metadata["files"]:
                parquet = pq.ParquetFile(raw / item["filename"])
                features = json.loads(parquet.schema_arrow.metadata[b"huggingface"])["info"]["features"]
                label_column = "fine_label" if dataset_id == "cifar100" else "label"
                names = features[label_column]["names"]
                if classes is not None and names != classes:
                    raise ValueError("Mirror class names differ across shards")
                classes = names
                image_column = "img" if "img" in features else "image"
                count = parquet.metadata.num_rows
                images = np.empty((count, 32, 32, 3), dtype=np.uint8)
                relative = item["filename"] + ".images.npy"
                row_index = 0
                for batch in parquet.iter_batches(batch_size=256, columns=[image_column, label_column]):
                    for row in batch.to_pylist():
                        with Image.open(io.BytesIO(row[image_column]["bytes"])) as image:
                            images[row_index] = np.asarray(image.convert("RGB"))
                        pixel_hash = hashlib.sha256(str((32, 32)).encode() + images[row_index].tobytes()).hexdigest()
                        samples.append({"sample_id": f"{item['split']}:{row_index}", "label": int(row[label_column]),
                                        "split": item["split"], "source_split": item["split"], "loader": "numpy_rgb",
                                        "array_path": relative, "index": row_index, "expected_pixel_sha256": pixel_hash})
                        row_index += 1
                path = native / relative
                temporary = path.with_suffix(".npy.tmp")
                with temporary.open("wb") as stream:
                    np.save(stream, images, allow_pickle=False)
                temporary.replace(path)
                array_storage.append({"path": relative, "sha256": file_hash(path), "shape": list(images.shape)})
            write_json(index_path, {"input_hash": input_hash, "classes": classes, "samples": samples, "array_storage": array_storage})
        state["status"] = "validating"
        write_json(directory / "state.json", state)
        try:
            verification = build_manifest(dataset_id, native, directory / "prepared")
        except (OSError, ValueError, RuntimeError):
            if force_rebuild:
                raise
            return self._prepare_cifar_arrays(dataset_id, directory, raw, metadata, state, force_rebuild=True)
        verification["provenance"] = {**verification["provenance"], "retrieval_source": metadata,
                                      "normalization": "Lossless uint8 RGB numpy arrays; no image resizing; allow_pickle=False"}
        write_json(directory / "prepared" / "verification.json", verification)
        state.update(status="verified", verification=verification)
        write_json(directory / "state.json", state)
        return state

    def _prepare_flowers_archive(self, directory, raw, metadata, state, *, offline, force_rebuild=False):
        from scipy.io import loadmat
        from torchvision.datasets import Flowers102
        state["source_format"] = "huggingface_archive"
        native = directory / "archive_native"
        native.mkdir(exist_ok=True)
        source_hash = canonical_hash({"version": DATASET_VERSION, "source": metadata})
        marker = native / "extraction.json"
        if force_rebuild or not marker.exists() or json.loads(marker.read_text()).get("input_hash") != source_hash:
            for item in metadata["files"]:
                safe_extract(raw / item["filename"], native)
            write_json(marker, {"input_hash": source_hash})
        annotations = directory / "official_annotations"
        annotations.mkdir(exist_ok=True)
        for item in REGISTRY["flowers102"]["files"]:
            if not item["filename"].endswith(".mat"):
                continue
            target = annotations / item["filename"]
            if not target.exists() or file_hash(target, "md5") != item["md5"]:
                candidate = next((p for p in native.rglob(item["filename"]) if file_hash(p, "md5") == item["md5"]), None)
                if candidate:
                    shutil.copy2(candidate, target)
                elif not offline:
                    state["files"].append(download_verified(item, target))
                else:
                    raise ValueError("Official Flowers annotations missing offline")
            if file_hash(target, "md5") != item["md5"]:
                raise ValueError("Official Flowers annotation checksum mismatch")
        # Some mirrors retain the original official archive inside a wrapper.
        nested = next((p for p in native.rglob("102flowers.tgz")
                       if file_hash(p, "md5") == REGISTRY["flowers102"]["files"][0]["md5"]), None)
        if nested and not list(native.rglob("image_00001.jpg")):
            safe_extract(nested, native / "official_images")
        sets = loadmat(annotations / "setid.mat", squeeze_me=True)
        splits = {int(image_id): split for split, key in (("train", "trnid"), ("val", "valid"), ("test", "tstid"))
                  for image_id in sets[key].tolist()}
        labels = loadmat(annotations / "imagelabels.mat", squeeze_me=True)["labels"].tolist()
        images = {}
        for path in native.rglob("image_*.jpg"):
            match = re.fullmatch(r"image_(\d{5})\.jpg", path.name)
            if not match:
                continue
            image_id = int(match[1])
            if image_id in images and file_hash(path) != file_hash(images[image_id]):
                raise ValueError("Mirror contains conflicting copies of an original image")
            images[image_id] = path
        if set(images) != set(splits):
            raise ValueError("Archive mirror lacks complete original Flowers image identities")
        samples = [{"sample_id": f"image_{image_id:05d}", "path": path.relative_to(native).as_posix(),
                    "original_image_id": image_id, "label": int(labels[image_id - 1]) - 1,
                    "split": splits[image_id], "source_split": splits[image_id], "encoded_sha256": file_hash(path)}
                   for image_id, path in sorted(images.items())]
        write_json(native / "hf_index.json", {"input_hash": source_hash, "classes": list(Flowers102.classes), "samples": samples})
        state["status"] = "validating"
        write_json(directory / "state.json", state)
        verification = build_manifest("flowers102", native, directory / "prepared")
        verification["provenance"] = {**verification["provenance"], "retrieval_source": metadata,
                                      "split_reconstruction": "Original image IDs joined to official MD5-verified setid.mat and imagelabels.mat"}
        write_json(directory / "prepared" / "verification.json", verification)
        state.update(status="verified", verification=verification)
        write_json(directory / "state.json", state)
        return state

    def import_local(self, dataset_id: str, path: Path) -> dict:
        """Accept a downloaded file/directory; copy only checksum-verified originals.

        Never unpickle an untrusted CIFAR import. Other native formats are also
        reconstructed from official checked archives, leaving user originals intact.
        """
        with dataset_lock(self.root / dataset_id):
            return self._import_locked(dataset_id, path)

    def _import_locked(self, dataset_id: str, path: Path) -> dict:
        definition = REGISTRY[dataset_id]
        path = path.resolve()
        if not path.exists():
            raise FileNotFoundError(path)
        directory = self.root / dataset_id / "raw"
        directory.mkdir(parents=True, exist_ok=True)
        imported = []
        missing = []
        for item in definition["files"]:
            destination = directory / item["filename"]
            if destination.is_file() and file_hash(destination, "md5") == item["md5"]:
                continue
            candidates = [path] if path.is_file() else list(path.rglob(item["filename"]))
            candidate = next((p for p in candidates if p.is_file() and file_hash(p, "md5") == item["md5"]), None)
            if candidate is None:
                missing.append(item["filename"])
            else:
                shutil.copy2(candidate, destination)
                imported.append(item["filename"])
        # An explicit official-file import takes priority over a mirror cache.
        previous_source = self.source
        self.source = "official"
        try:
            result = self._acquire_locked(dataset_id, offline=True)
        finally:
            self.source = previous_source
        result["imported_files"] = imported
        result["missing_files"] = missing
        return result

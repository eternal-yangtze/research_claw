import io
import json
import tarfile
import hashlib
from pathlib import Path

import pytest

from researchops.datasets import (
    DatasetManager, build_manifest, file_hash, safe_extract, stratified_validation,
    download_verified,
)
from researchops.experiment_preparation import build_vision_contract


def test_safe_extraction_rejects_traversal_before_writing_any_member(tmp_path):
    archive = tmp_path / "payload.tar"
    with tarfile.open(archive, "w") as tar:
        for name in ("good.txt", "../escaped.txt"):
            member = tarfile.TarInfo(name)
            member.size = 1
            tar.addfile(member, io.BytesIO(b"x"))
    with pytest.raises(ValueError, match="Unsafe archive"):
        safe_extract(archive, tmp_path / "out")
    assert not (tmp_path / "escaped.txt").exists()
    assert not (tmp_path / "out" / "good.txt").exists()


def test_safe_extraction_rejects_links_and_windows_paths(tmp_path):
    for name, link in (("C:/escape.txt", False), ("link", True)):
        archive = tmp_path / "payload.tar"
        with tarfile.open(archive, "w") as tar:
            member = tarfile.TarInfo(name)
            if link:
                member.type = tarfile.SYMTYPE
                member.linkname = "../secret"
            tar.addfile(member)
        with pytest.raises(ValueError):
            safe_extract(archive, tmp_path / "out")


def test_stratified_split_is_fixed_balanced_and_training_only():
    rows = [{"sample_id": f"{label}:{i}", "label": label, "source_split": "train"}
            for label in (0, 1) for i in range(20)]
    first = stratified_validation(rows)
    second = stratified_validation(list(reversed(rows)))
    assert first == second
    assert sum(row["split"] == "val" for row in first) == 4
    assert {r["source_split"] for r in first} == {"train"}
    assert len({r["sample_id"] for r in first}) == 40


def test_manifest_filters_leakage_and_freezes_nested_subsets(tmp_path, monkeypatch):
    from researchops import datasets
    rows = []
    for split in ("train", "val", "test"):
        for label in (0, 1):
            for index in range(10):
                rows.append({"sample_id": f"{split}:{label}:{index}", "label": label,
                             "split": split, "source_split": split,
                             "content_sha256": f"{split}:{label}:{index}"})
    # Two train images leak across splits, one duplicates another train image.
    rows[0]["content_sha256"] = "test:0:0"
    rows[1]["content_sha256"] = "val:0:0"
    rows[2]["content_sha256"] = "train:0:3"
    monkeypatch.setitem(datasets.REGISTRY, "toy", {"name": "Toy", "samples": 60, "classes": 2})
    monkeypatch.setattr(datasets, "native_samples", lambda *_: (rows, ["a", "b"]))
    report = build_manifest("toy", tmp_path, tmp_path / "prepared")
    assert report["split_counts"] == {"train": 17, "val": 20, "test": 20}
    assert report["exclusions"] == {"duplicate_of_test": 1, "duplicate_of_validation": 1, "duplicate_within_train": 1}
    subsets = [set(json.loads(Path(v["path"]).read_text())) for v in report["train_fractions"].values()]
    assert subsets[0] <= subsets[1] <= subsets[2]
    assert len(subsets[2]) == 17
    manifest = [json.loads(line) for line in Path(report["manifest_path"]).read_text().splitlines()]
    assert all(r["source_split"] == "train" for r in manifest if r["sample_id"] in subsets[2])
    datasets = {"one": {"status": "verified", "verification": {**report, "dataset_id": "one"}},
                "two": {"status": "verified", "verification": {**report, "dataset_id": "two"}}}
    recommendations = {"baseline_citations": ["c1", "c2"], "recommendations": []}
    contract = build_vision_contract({"idea_id": "toy"}, ["one", "two"], datasets, recommendations, {})
    assert contract["run_budget"]["main"] == 90
    assert contract["run_budget"]["total"] == len(contract["jobs"]) == 128
    assert len({job["job_id"] for job in contract["jobs"]}) == 128
    assert all(not job["evaluate_test"] for job in contract["jobs"] if job["stage"] == "validation_tuning")
    Path(report["manifest_path"]).write_text("tampered")
    with pytest.raises(ValueError, match="manifest changed"):
        build_vision_contract({}, ["one", "two"], datasets, recommendations, {})


def test_import_rejects_untrusted_pickle_before_extraction(tmp_path):
    source = tmp_path / "cifar-10-python.tar.gz"
    source.write_bytes(b"not the official archive")
    manager = DatasetManager(tmp_path / "workspace")
    report = manager.import_local("cifar10", source)
    assert report["status"] == "manual_required"
    assert "cifar-10-python.tar.gz" in report["missing_files"]
    assert source.read_bytes() == b"not the official archive"
    assert not (manager.root / "cifar10" / "raw" / source.name).exists()


def test_verified_import_can_continue_without_network(tmp_path, monkeypatch):
    from researchops import datasets
    source_dir = tmp_path / "user_downloads"
    source_dir.mkdir()
    archive = source_dir / "toy.tar.gz"
    with tarfile.open(archive, "w:gz") as tar:
        member = tarfile.TarInfo("images/a.txt")
        member.size = 5
        tar.addfile(member, io.BytesIO(b"image"))
    monkeypatch.setitem(datasets.REGISTRY, "toy", {
        "name": "Toy", "layout": "", "landing_url": "https://example.test/toy",
        "files": [{"filename": archive.name, "md5": file_hash(archive, "md5"), "urls": ["https://example.test/toy.tar.gz"]}],
    })
    monkeypatch.setattr(datasets, "build_manifest", lambda *args: {"sample_count": 1, "status": "verified"})
    monkeypatch.setattr(datasets, "download_verified", lambda *args, **kwargs: pytest.fail("Import must be offline"))
    manager = DatasetManager(tmp_path / "workspace")
    report = manager.import_local("toy", source_dir)
    assert report["status"] == "verified"
    assert (manager.root / "toy" / "native" / "images" / "a.txt").read_bytes() == b"image"
    assert archive.is_file()


class HTTPResponse(io.BytesIO):
    def __init__(self, data, start, end, total):
        super().__init__(data)
        self.status = 206
        self.headers = {"Content-Length": str(end - start + 1), "Content-Range": f"bytes {start}-{end}/{total}"}

    def geturl(self):
        return "https://example.test/data"


def test_download_recovers_after_truncated_response_and_verifies_checksum(tmp_path, monkeypatch):
    from researchops import datasets
    payload = b"correct official file content"
    destination = tmp_path / "labels.mat"
    destination.with_suffix(".mat.part").write_bytes(payload[:3])
    starts = []

    class Opener:
        def open(self, request, timeout):
            start = int(request.headers["Range"].split("=")[1].split("-")[0])
            starts.append(start)
            data = payload[start:8] if len(starts) == 1 else payload[start:]
            return HTTPResponse(data, start, len(payload) - 1, len(payload))

    monkeypatch.setattr(datasets.urllib.request, "build_opener", lambda *args: Opener())
    item = {"filename": "labels.mat", "md5": hashlib.md5(payload).hexdigest(), "urls": ["https://example.test/labels.mat"]}
    result = download_verified(item, destination)
    assert starts == [3, 8]
    assert destination.read_bytes() == payload
    assert result["sha256"] == hashlib.sha256(payload).hexdigest()
    assert download_verified(item, destination)["cached"] is True
    assert starts == [3, 8]


def test_parallel_ranges_reuse_completed_transfer_and_assemble_official_hash(tmp_path, monkeypatch):
    from researchops import datasets
    payload = b"0" * (8 * 1024 * 1024) + b"1" * (8 * 1024 * 1024) + b"tail" * (512 * 1024)
    destination = tmp_path / "toy.tar.gz"
    destination.with_suffix(".gz.part").write_bytes(payload[:8 * 1024 * 1024])
    starts = []

    class Opener:
        def open(self, request, timeout):
            start, end = map(int, request.headers["Range"].split("=")[1].split("-"))
            end = min(len(payload) - 1, end)
            starts.append(start)
            return HTTPResponse(payload[start:end + 1], start, end, len(payload))

    monkeypatch.setattr(datasets.urllib.request, "build_opener", lambda *args: Opener())
    item = {"filename": "toy.tar.gz", "md5": hashlib.md5(payload).hexdigest(), "urls": ["https://example.test/data"]}
    report = download_verified(item, destination)
    assert report["transfer"] == "four_bounded_parallel_ranges"
    assert destination.read_bytes() == payload
    assert sorted(starts) == [0, 8 * 1024 * 1024, 16 * 1024 * 1024]


def test_pinned_parquet_mirror_is_decoded_validated_and_resumable_offline(tmp_path, monkeypatch):
    pa = pytest.importorskip("pyarrow")
    import pyarrow.parquet as pq
    from PIL import Image
    from researchops import datasets

    dataset_id = "toyhf"
    monkeypatch.setitem(datasets.HF_REPOSITORIES, dataset_id, "test/public-toy")
    monkeypatch.setitem(datasets.REGISTRY, dataset_id, {"name": "Toy HF", "samples": 20, "classes": 2, "landing_url": "https://example.test"})
    manager = DatasetManager(tmp_path / "workspace", source="huggingface")
    directory = manager.root / dataset_id
    raw = directory / "mirror_raw"
    raw.mkdir(parents=True)
    metadata = {"repo": "test/public-toy", "revision": "a" * 40, "files": []}
    features = {"image": {"_type": "Image"}, "label": {"_type": "ClassLabel", "names": ["a", "b"]}}
    schema = pa.schema([("image", pa.struct([("bytes", pa.binary()), ("path", pa.string())])), ("label", pa.int64())],
                       metadata={b"huggingface": json.dumps({"info": {"features": features}}).encode()})
    serial = 0
    for split, count in (("train", 12), ("val", 4), ("test", 4)):
        rows = []
        for index in range(count):
            serial += 1
            encoded = io.BytesIO()
            Image.new("RGB", (4, 4), color=(serial, 0, 0)).save(encoded, format="PNG")
            rows.append({"image": {"bytes": encoded.getvalue(), "path": None}, "label": index % 2})
        path = raw / f"{split}.parquet"
        pq.write_table(pa.Table.from_pylist(rows, schema=schema), path)
        metadata["files"].append({"filename": path.name, "split": split, "sha256": file_hash(path)})
    datasets.write_json(directory / "hf_source.json", metadata)
    first = manager.acquire(dataset_id, offline=True)
    assert first["status"] == "verified", first
    assert first["verification"]["split_counts"] == {"train": 12, "val": 4, "test": 4}
    assert first["verification"]["class_count"] == 2
    assert first["verification"]["provenance"]["retrieval_source"]["revision"] == "a" * 40
    second = manager.acquire(dataset_id, offline=True)
    assert second["verification"]["manifest_sha256"] == first["verification"]["manifest_sha256"]
    image = next((directory / "mirror_native" / "images").rglob("*.png"))
    image.write_bytes(b"corrupted prepared image")
    repaired = manager.acquire(dataset_id, offline=True)
    assert repaired["status"] == "verified", repaired
    assert repaired["verification"]["manifest_sha256"] == first["verification"]["manifest_sha256"]


def test_custom_mirror_splits_cannot_be_reported_as_official(tmp_path, monkeypatch):
    from researchops import datasets
    rows = [{"sample_id": str(i), "label": i % 2, "split": "train" if i < 4 else "test",
             "source_split": "train" if i < 4 else "test", "content_sha256": str(i)} for i in range(6)]
    monkeypatch.setitem(datasets.REGISTRY, "toy", {"name": "Toy", "samples": 6, "classes": 2,
                                                "source_splits": {"train": 2, "val": 2, "test": 2}})
    monkeypatch.setattr(datasets, "native_samples", lambda *args: (rows, ["a", "b"]))
    with pytest.raises(ValueError, match="official source split counts"):
        build_manifest("toy", tmp_path, tmp_path / "prepared")


def test_flowers_aggregate_mirror_recovers_official_splits_and_labels(tmp_path, monkeypatch):
    pa = pytest.importorskip("pyarrow")
    import pyarrow.parquet as pq
    from scipy.io import savemat
    from PIL import Image
    from torchvision.datasets import Flowers102
    from researchops import datasets

    monkeypatch.setitem(datasets.HF_REPOSITORIES, "flowers102", "huggan/flowers-102-categories")

    directory = tmp_path / "datasets" / "flowers102"
    raw = directory / "mirror_raw"
    annotations = directory / "official_annotations"
    raw.mkdir(parents=True)
    annotations.mkdir()
    savemat(annotations / "setid.mat", {"trnid": [1, 2, 3, 4, 5, 6], "valid": [7, 8], "tstid": [9, 10]})
    savemat(annotations / "imagelabels.mat", {"labels": [1, 2] * 5})
    files = [{"filename": name, "md5": file_hash(annotations / name, "md5"), "urls": ["https://example.test/data"]}
             for name in ("imagelabels.mat", "setid.mat")]
    monkeypatch.setitem(datasets.REGISTRY, "flowers102", {
        "name": "Toy Flowers", "samples": 10, "classes": 2, "files": files,
        "source_splits": {"train": 6, "val": 2, "test": 2}, "landing_url": "https://example.test"})
    monkeypatch.setattr(Flowers102, "classes", ["a", "b"])
    rows = []
    # Deliberately reorder IDs and reverse mirror labels; official annotations win.
    for image_id in reversed(range(1, 11)):
        image = io.BytesIO()
        Image.new("RGB", (4, 4), color=(image_id, 1, 0)).save(image, format="PNG")
        rows.append({"image": {"bytes": image.getvalue(), "path": f"image_{image_id:05d}.jpg"}, "label": image_id % 2})
    features = {"image": {"_type": "Image"}, "label": {"_type": "ClassLabel", "names": ["a", "b"]}}
    table = pa.Table.from_pylist(rows).replace_schema_metadata({b"huggingface": json.dumps({"info": {"features": features}}).encode()})
    shard = raw / "train.parquet"
    pq.write_table(table, shard)
    datasets.write_json(directory / "hf_source.json", {
        "repo": "huggan/flowers-102-categories", "revision": "a" * 40,
        "files": [{"filename": "train.parquet", "split": "train", "sha256": file_hash(shard)}],
    })
    manager = DatasetManager(tmp_path, source="huggingface")
    result = manager.acquire("flowers102", offline=True)
    assert result["status"] == "verified", result
    assert result["verification"]["source_split_counts"] == {"test": 2, "val": 2, "train": 6}
    samples = [json.loads(line) for line in Path(result["verification"]["manifest_path"]).read_text().splitlines()]
    assert all(row["label"] == (row["original_image_id"] - 1) % 2 for row in samples)
    assert {row["original_image_id"] for row in samples if row["split"] == "val"} == {7, 8}


def test_cifar_parquet_normalizes_to_safe_arrays_and_repairs_tampering(tmp_path, monkeypatch):
    pa = pytest.importorskip("pyarrow")
    import pyarrow.parquet as pq
    import numpy as np
    from PIL import Image
    from researchops import datasets
    directory = tmp_path / "datasets" / "cifar10"
    raw = directory / "mirror_raw"
    raw.mkdir(parents=True)
    monkeypatch.setitem(datasets.REGISTRY, "cifar10", {"name": "Toy CIFAR", "samples": 6, "classes": 2,
                                                     "source_splits": {"train": 4, "test": 2}, "landing_url": "https://example.test"})
    features = {"img": {"_type": "Image"}, "label": {"_type": "ClassLabel", "names": ["a", "b"]}}
    metadata = {"repo": "uoft-cs/cifar10", "revision": "a" * 40, "files": []}
    serial = 0
    for split, count in (("train", 4), ("test", 2)):
        rows = []
        for index in range(count):
            serial += 1
            encoded = io.BytesIO()
            Image.new("RGB", (32, 32), color=(serial * 20, 0, 0)).save(encoded, format="PNG")
            rows.append({"img": {"bytes": encoded.getvalue(), "path": None}, "label": index % 2})
        table = pa.Table.from_pylist(rows).replace_schema_metadata({b"huggingface": json.dumps({"info": {"features": features}}).encode()})
        path = raw / f"{split}.parquet"
        pq.write_table(table, path)
        metadata["files"].append({"filename": path.name, "split": split, "sha256": file_hash(path)})
    datasets.write_json(directory / "hf_source.json", metadata)
    manager = DatasetManager(tmp_path, source="huggingface")
    first = manager.acquire("cifar10", offline=True)
    assert first["status"] == "verified", first
    assert first["verification"]["split_counts"] == {"train": 2, "val": 2, "test": 2}
    assert not list((directory / "array_native").rglob("*.png"))
    array_path = next((directory / "array_native").glob("*.npy"))
    array = np.load(array_path, mmap_mode="r+", allow_pickle=False)
    array[0] = 0
    array.flush()
    del array
    repaired = manager.acquire("cifar10", offline=True)
    assert repaired["status"] == "verified"
    assert repaired["verification"]["manifest_sha256"] == first["verification"]["manifest_sha256"]

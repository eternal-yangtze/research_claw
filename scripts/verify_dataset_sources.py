"""Read-only source diagnostics; fetch just a small range, never execute data."""
import json
import io
import ssl
import sys
import time
from pathlib import Path
import urllib.request
from concurrent.futures import ThreadPoolExecutor

import certifi

URLS = [
    "https://thor.robots.ox.ac.uk/datasets/flowers-102/102flowers.tgz",
    "https://thor.robots.ox.ac.uk/datasets/flowers-102/imagelabels.mat",
    "https://thor.robots.ox.ac.uk/datasets/flowers-102/setid.mat",
    "https://thor.robots.ox.ac.uk/datasets/pets/images.tar.gz",
    "https://thor.robots.ox.ac.uk/datasets/pets/annotations.tar.gz",
    "https://thor.robots.ox.ac.uk/datasets/dtd/dtd-r1.0.1.tar.gz",
    "https://cave.cs.toronto.edu/kriz/cifar-10-python.tar.gz",
    "https://cave.cs.toronto.edu/kriz/cifar-100-python.tar.gz",
    "https://storage.googleapis.com/tensorflow/tf-keras-datasets/cifar-10-batches-py.tar.gz",
    "https://storage.googleapis.com/tensorflow/tf-keras-datasets/cifar-100-python.tar.gz",
    "https://ossci-datasets.s3.amazonaws.com/cifar-10-python.tar.gz",
    "https://huggingface.co/api/datasets?search=cifar100&limit=3",
    "https://huggingface.co/api/datasets?search=flowers102&limit=3",
]


def probe(url):
    try:
        started = time.monotonic()
        transfer = "--transfer" in sys.argv
        size = 1024 * 1024 if transfer else 512
        request = urllib.request.Request(url, headers={"Range": f"bytes=0-{size-1}", "User-Agent": "ResearchOps/1.0"})
        with urllib.request.urlopen(request, timeout=15, context=ssl.create_default_context(cafile=certifi.where())) as response:
            prefix = response.read(size)
            return {"url": url, "status": response.status, "final": response.url,
                    "length": response.headers.get("Content-Length"),
                    "bytes_read": len(prefix), "seconds": round(time.monotonic() - started, 2),
                    "range": response.headers.get("Content-Range"),
                    "prefix": repr(prefix[:140])}
    except Exception as error:
        return {"url": url, "error": f"{type(error).__name__}: {error}"}


if __name__ == "__main__":
    if "--parquet-info" in sys.argv:
        import pyarrow.parquet as pq
        source = json.loads(Path("workspace_batch_vit_v2/datasets/flowers102/hf_source.json").read_text())
        entry = source["files"][0]
        size = entry["bytes"]
        url = entry["urls"][0]

        class RemoteFooter(io.RawIOBase):
            position = 0

            def readable(self):
                return True

            def seekable(self):
                return True

            def tell(self):
                return self.position

            def seek(self, offset, whence=0):
                self.position = offset if whence == 0 else self.position + offset if whence == 1 else size + offset
                return self.position

            def read(self, count=-1):
                count = min(count if count >= 0 else size - self.position, size - self.position)
                if count > 1024 * 1024:
                    raise ValueError("Read-only footer diagnostic byte limit")
                request = urllib.request.Request(url, headers={"Range": f"bytes={self.position}-{self.position+count-1}"})
                with urllib.request.urlopen(request, timeout=30, context=ssl.create_default_context(cafile=certifi.where())) as response:
                    if response.status != 206:
                        raise ValueError("Footer source ignored range")
                    data = response.read(count)
                self.position += len(data)
                return data

        parquet = pq.ParquetFile(RemoteFooter())
        print(json.dumps({"rows": parquet.metadata.num_rows, "schema": str(parquet.schema_arrow),
                          "columns": [{"path": parquet.metadata.row_group(0).column(index).path_in_schema,
                                       "stats": str(parquet.metadata.row_group(0).column(index).statistics)}
                                      for index in range(parquet.metadata.num_columns)]}))
        sys.exit(0)
    if "--candidate-cards" in sys.argv:
        URLS = ["https://huggingface.co/datasets/huggan/flowers-102-categories/raw/main/README.md",
                "https://huggingface.co/datasets/Multimodal-Fatima/OxfordFlowers_train/raw/main/README.md",
                "https://huggingface.co/datasets/Multimodal-Fatima/OxfordFlowers_validation/raw/main/README.md"]

        def card(url):
            try:
                with urllib.request.urlopen(url, timeout=20, context=ssl.create_default_context(cafile=certifi.where())) as response:
                    return {"url": url, "card": response.read(10000).decode("utf-8")}
            except Exception as error:
                return {"url": url, "error": str(error)}

        probe = card
    if "--mirror-search" in sys.argv:
        def get_json(url):
            with urllib.request.urlopen(url, timeout=20, context=ssl.create_default_context(cafile=certifi.where())) as response:
                return json.load(response)

        repos = []
        for keyword in ("flowers102", "oxford-flowers", "dtd"):
            for entry in get_json(f"https://huggingface.co/api/datasets?search={keyword}&limit=12&sort=downloads&direction=-1"):
                if not entry.get("gated") and not entry.get("private"):
                    repos.append(entry["id"])
        URLS = sorted(set(repos))

        def inspect_repo(repo):
            try:
                info = get_json(f"https://huggingface.co/api/datasets/{repo}")
                tree = get_json(f"https://huggingface.co/api/datasets/{repo}/tree/{info['sha']}?recursive=true&limit=1000")
                files = [item for item in tree if item.get("path", "").endswith((".parquet", ".tar.gz", ".tgz", ".zip"))]
                return {"repo": repo, "revision": info["sha"], "total_mb": round(sum(item.get("size", 0) for item in files) / 1024**2, 2),
                        "files": [item["path"] for item in files][:12]}
            except Exception as error:
                return {"repo": repo, "error": str(error)}

        probe = inspect_repo
    if "--huggingface" in sys.argv:
        URLS = [f"https://huggingface.co/api/datasets/{repo}" for repo in (
            "pytorch/oxford-flowers", "mcimpoi/dtd_split_1",
        )]

        def hf_metadata(url):
            try:
                with urllib.request.urlopen(url, timeout=20, context=ssl.create_default_context(cafile=certifi.where())) as response:
                    value = json.load(response)
                return {"url": url, "sha": value.get("sha"), "gated": value.get("gated"),
                        "files": [item["rfilename"] for item in value.get("siblings", [])]}
            except Exception as error:
                return {"url": url, "error": str(error)}

        probe = hf_metadata
    with ThreadPoolExecutor(max_workers=6) as pool:
        for result in pool.map(probe, URLS):
            print(json.dumps(result), flush=True)

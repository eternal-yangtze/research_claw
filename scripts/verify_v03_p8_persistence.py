"""Restart the existing P8 Qdrant container and verify actual indexed results."""
from __future__ import annotations

import argparse
import subprocess
import time

from verify_v03_p8 import canonical_hash, check_pins, environment, read, write_json
from pathlib import Path


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output",type=Path,required=True)
    args=parser.parse_args()
    root=args.output.resolve()
    environment()
    check_pins(root)
    from qdrant_client import QdrantClient
    client=QdrantClient(url="http://127.0.0.1:6333",timeout=10)
    prefix="researchops_p8_"+root.name.lower()+"_"
    aliases=sorted(a.alias_name for a in client.get_aliases().aliases if a.alias_name.startswith(prefix))
    if not aliases:
        raise ValueError("No P8 collections to verify")
    saved={}
    for alias in aliases:
        info=client.get_collection(alias)
        points,_=client.scroll(alias,limit=1,with_payload=True,with_vectors=True)
        if not points:
            raise ValueError("P8 collection has no persisted points")
        vector=points[0].vector
        before=client.query_points(alias,query=vector,limit=3,with_payload=True).points
        saved[alias]={"points":info.points_count,"vector":vector,
            "vector_sha256":canonical_hash(vector),"query_ids":[str(p.id) for p in before],
            "payload_hashes":[canonical_hash(p.payload) for p in before]}
    started=time.monotonic()
    p=subprocess.run(["docker","restart","researchops-qdrant-v03"],capture_output=True,text=True,timeout=45)
    if p.returncode:
        raise RuntimeError(p.stderr)
    deadline=time.monotonic()+30
    while True:
        try:
            client.get_collections()
            break
        except Exception:
            if time.monotonic()>=deadline:
                raise
            time.sleep(0.5)
    checks={}
    result={}
    for alias,before in saved.items():
        info=client.get_collection(alias)
        after=client.query_points(alias,query=before["vector"],limit=3,with_payload=True).points
        checks[alias+"_points_survive"]=info.points_count==before["points"]
        checks[alias+"_query_and_payload_survive"]=before["query_ids"]==[str(p.id) for p in after] and before["payload_hashes"]==[canonical_hash(p.payload) for p in after]
        result[alias]={k:v for k,v in before.items() if k!="vector"}
    out={"passed":all(checks.values()),"checks":checks,"container":"researchops-qdrant-v03",
        "seconds":time.monotonic()-started,"collections":result}
    write_json(root/"qdrant_persistence.json",out)
    import json
    print(json.dumps(out,indent=2))
    return 0 if out["passed"] else 1


if __name__=="__main__":
    raise SystemExit(main())

"""Remove only Qdrant collections provably owned by disposable local test DBs."""
import argparse
import hashlib
import os
import sqlite3
from pathlib import Path

from qdrant_client import QdrantClient

ROOT=next(p for p in Path(__file__).resolve().parents if (p / "pyproject.toml").is_file())


def main():
    parser=argparse.ArgumentParser()
    parser.add_argument('--apply',action='store_true')
    args=parser.parse_args()
    roots=[p for p in ROOT.iterdir() if p.is_dir() and p.name.startswith(('.live27','.browser27','.w27','.reuse27'))]
    roots.append(ROOT/'self-revise/backend-revise/internal_experiments/runs')
    client=QdrantClient(url='http://127.0.0.1:6333',timeout=10)
    present={c.name for c in client.get_collections().collections}
    selected={}
    for root in roots:
        for parent,dirs,files in os.walk(root,followlinks=False):
            dirs[:]=[d for d in dirs if not (Path(parent)/d).is_junction() and d not in {'node_modules','datasets','.git'}]
            for name in files:
                if name not in {'researchops.db','study_corpus.db'}:
                    continue
                path=Path(parent)/name
                # A copied DB cannot authorize deletion of its original vector index.
                expected='project_'+hashlib.sha256(str(path.resolve()).encode()).hexdigest()[:16]
                try:
                    with sqlite3.connect(path.resolve().as_uri()+'?mode=ro',uri=True) as db:
                        rows=db.execute('SELECT version_collection,project_id,collection_name FROM vector_sync_state').fetchall()
                except (sqlite3.Error,OSError):
                    continue
                for collection,project,alias in rows:
                    default_alias='researchclaw_'+hashlib.sha256(expected.encode()).hexdigest()[:12]
                    if project!=expected or alias!=default_alias or collection not in present:
                        continue
                    info=client.get_collection(collection)
                    metadata=getattr(info.config,'metadata',None) or {}
                    if metadata.get('project_id')!=expected:
                        continue
                    selected[collection]=path.relative_to(ROOT).as_posix()
    for collection,path in sorted(selected.items()):
        if args.apply:
            client.delete_collection(collection)
        print(f'{"DELETED" if args.apply else "PREVIEW"} {collection} <- {path}')
    print(f'collections={len(selected)} applied={args.apply}; original/user/shared collections untouched')


if __name__=='__main__':
    main()

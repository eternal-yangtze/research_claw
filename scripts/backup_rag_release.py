"""Consistent, non-destructive backup of the published PG/Qdrant library."""
import hashlib
import json
from pathlib import Path
import shutil
import subprocess
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / 'src'))
from researchclaw.core.qdrant_store import QdrantResearchStore
import requests


def main():
    workspace = ROOT / 'knowledge_base/rag_corpora/time_series_domain_release_v3_1'
    target = ROOT / 'self-revise/backend-revise/internal_experiments/assets/rag_backup_194_v20261109'
    target.mkdir(parents=True, exist_ok=True)
    if (target / 'manifest.json').exists():
        print('Verified backup already exists; refusing overwrite')
        for row in json.loads((target / 'manifest.json').read_text())['files']:
            assert hashlib.sha256((target / row['name']).read_bytes()).hexdigest() == row['sha256']
        return
    store = QdrantResearchStore(workspace, recover=False)
    with store.database.connect(read_only=True) as db:
        count = db.execute('SELECT count(*) FROM papers').fetchone()[0]
        assert count == 194, f'Expected original 194 papers, got {count}'
        states = [dict(r) for r in db.execute("SELECT * FROM vector_sync_state WHERE status='ready'")]
    dump = target / 'postgres.dump'
    subprocess.run(['docker', 'exec', 'researchclaw-infra-postgres-1', 'pg_dump', '-U',
                    'researchclaw', '-d', 'researchclaw', '-Fc', '-n', store.database.schema,
                    '-f', '/tmp/researchclaw-rag194.dump'], check=True, capture_output=True)
    subprocess.run(['docker', 'cp', 'researchclaw-infra-postgres-1:/tmp/researchclaw-rag194.dump', str(dump)], check=True, capture_output=True)
    subprocess.run(['docker', 'exec', 'researchclaw-infra-postgres-1', 'pg_restore', '--list', '/tmp/researchclaw-rag194.dump'], check=True, capture_output=True)
    lexical = json.loads((workspace / 'rag-search-index.json').read_text(encoding='utf8'))
    collections = {r['version_collection'] for r in states if r['version_collection']}
    collections.add(lexical['collection'] if 'collection' in lexical else lexical['sparse_collection'])
    snapshots = []
    for collection in sorted(collections):
        snapshot = store.client.create_snapshot(collection_name=collection, wait=True)
        response = requests.get(f'http://127.0.0.1:6333/collections/{collection}/snapshots/{snapshot.name}', timeout=180)
        response.raise_for_status()
        name = collection + '.snapshot'
        (target / name).write_bytes(response.content)
        snapshots.append({'collection': collection, 'file': name, 'points': store.client.count(collection, exact=True).count})
    for name in ['knowledge-storage.json', 'rag-search-index.json']:
        shutil.copy2(workspace / name, target / name)
    files = [{'name': p.name, 'bytes': p.stat().st_size, 'sha256': hashlib.sha256(p.read_bytes()).hexdigest()}
             for p in target.iterdir() if p.is_file()]
    manifest = {'papers': count, 'schema': store.database.schema, 'workspace': str(workspace),
                'states': states, 'snapshots': snapshots, 'files': files,
                'validation': 'pg_restore --list passed; exact Qdrant counts and SHA256 recorded; no isolated restore performed'}
    (target / 'manifest.json').write_text(json.dumps(manifest, indent=2, default=str), encoding='utf8')
    print(json.dumps({'papers': count, 'collections': len(snapshots), 'backup': str(target)}))


if __name__ == '__main__':
    main()

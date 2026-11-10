"""Restore the verified cold backup into the accepted, empty Qdrant volume.

Never deletes the volume or backup. Empty-instance files are retained both in
an archive and a hidden quarantine directory. Refuses any nonempty collection.
"""
import argparse
import hashlib
import json
from pathlib import Path
import subprocess
import time
import urllib.request

ROOT = Path(__file__).resolve().parents[1]
COMPOSE = ['docker', 'compose', '--env-file', 'infra/.env', '-f', 'infra/compose.yaml']
VOLUME = 'researchclaw_qdrant_data'
IMAGE = 'postgres:16.6-bookworm'


def command(argv, timeout=30):
    result = subprocess.run(argv, cwd=ROOT, capture_output=True, text=True,
                            encoding='utf-8', errors='replace', timeout=timeout)
    if result.returncode:
        raise RuntimeError(result.stderr.strip() or 'Docker operation failed')
    return result.stdout.strip()


def get(path):
    opener = urllib.request.build_opener(urllib.request.ProxyHandler({}))
    with opener.open('http://127.0.0.1:6333' + path, timeout=5) as response:
        return json.load(response)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--apply', action='store_true', required=True)
    args = parser.parse_args()
    evidence = ROOT / 'self-revise/backend-revise/stage23-evidence'
    report = json.loads((evidence / 'backup-preflight.json').read_text(encoding='utf-8'))
    backup = ROOT / 'self-revise/backend-revise/internal_experiments/assets/docker_backup_20261030/qdrant-storage.tar.gz'
    expected = json.loads((ROOT / 'self-revise/backend-revise/environment-evidence/qdrant-cold-backup-sha256.json').read_text(encoding='utf-8-sig'))['Hash'].lower()
    with backup.open('rb') as stream:
        actual = hashlib.file_digest(stream, 'sha256').hexdigest()
    if actual != expected or report['sha256'] != actual or report['collection_count'] != 74:
        raise RuntimeError('Backup verification missing or changed')
    if get('/collections')['result']['collections'] or get('/aliases')['result']['aliases']:
        raise RuntimeError('Target Qdrant has new data/aliases; restore refused')
    containers = command([*COMPOSE, 'ps', '-q', 'qdrant']).splitlines()
    if len(containers) != 1:
        raise RuntimeError('Expected one running Qdrant service')
    mounts = json.loads(command(['docker', 'inspect', containers[0], '--format', '{{json .Mounts}}']))
    if not any(item.get('Name') == VOLUME and item['Destination'] == '/qdrant/storage' for item in mounts):
        raise RuntimeError('Unexpected target volume')
    original_image = command(['docker', 'inspect', containers[0], '--format', '{{.Config.Image}}'])
    if original_image != 'qdrant/qdrant@sha256:9bfc89f9914fc14876a88c05270685921a55b8877b254a98bcc1cc2f0a424cd2':
        raise RuntimeError('Qdrant image differs from accepted backup baseline')
    if (evidence / 'target-before-restore.tar.gz').exists() or (evidence / 'restore-result.json').exists():
        raise RuntimeError('Existing recovery evidence; do not repeat restore')
    command([*COMPOSE, 'stop', 'qdrant'])
    print('Qdrant stopped; original backup and empty target state will be retained', flush=True)
    operation = '''set -eu
cd /restore
test ! -e .pre_restore_20261101
test ! -e /evidence/target-before-restore.tar.gz
test ! -d collections || test -z "$(find collections -mindepth 1 -maxdepth 1 -print -quit)"
tar -czf /evidence/target-before-restore.tar.gz .
mkdir .pre_restore_20261101
find . -mindepth 1 -maxdepth 1 ! -name .pre_restore_20261101 -exec mv -t .pre_restore_20261101 -- {} +
tar -xzf /backup/qdrant-storage.tar.gz -C /restore --keep-old-files
'''
    command(['docker', 'run', '--rm', '--network', 'none',
             '--mount', f'type=volume,src={VOLUME},dst=/restore',
             '--mount', f'type=bind,src={backup.parent},dst=/backup,readonly',
             '--mount', f'type=bind,src={evidence},dst=/evidence',
             '--entrypoint', 'sh', IMAGE, '-c', operation], timeout=600)
    print('Archive extracted; starting the same Qdrant image/volume', flush=True)
    command([*COMPOSE, 'start', 'qdrant'])
    collections = None
    for _ in range(120):
        try:
            collections = sorted(item['name'] for item in get('/collections')['result']['collections'])
            if collections == report['collections']:
                break
        except Exception:
            pass
        time.sleep(.5)
    if collections != report['collections']:
        raise RuntimeError('Recovered collection inventory differs from archive')
    result = dict(status='passed', backup_sha256=actual, original_backup_preserved=True,
                  volume=VOLUME, image=original_image, collection_count=len(collections),
                  collections=collections, aliases=get('/aliases')['result']['aliases'],
                  target_before_restore='target-before-restore.tar.gz',
                  original_empty_volume_files='.pre_restore_20261101')
    with (evidence / 'restore-result.json').open('x', encoding='utf-8') as stream:
        json.dump(result, stream, ensure_ascii=False, indent=2)
    print('RESTORE PASS: 74 original collections; original aliases and backup preserved')


if __name__ == '__main__':
    main()

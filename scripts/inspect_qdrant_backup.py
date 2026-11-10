"""Verify historical backup without extracting or modifying it."""
import argparse
from collections import Counter
import hashlib
import json
from pathlib import Path
import tarfile

ROOT = Path(__file__).resolve().parents[1]


def inspect():
    backup = ROOT / 'self-revise/backend-revise/internal_experiments/assets/docker_backup_20261030/qdrant-storage.tar.gz'
    expected = json.loads((ROOT / 'self-revise/backend-revise/environment-evidence/qdrant-cold-backup-sha256.json').read_text(encoding='utf-8-sig'))['Hash'].lower()
    with backup.open('rb') as stream:
        actual = hashlib.file_digest(stream, 'sha256').hexdigest()
    if actual != expected:
        raise RuntimeError('Backup SHA256 mismatch; restore forbidden')
    types, collections, entries, roots = Counter(), set(), 0, set()
    # GzipFile's buffered seek avoids tarfile._Stream's quadratic buffering
    # when skipping the large memory-mapped vector files in this archive.
    with tarfile.open(backup, 'r:gz') as archive:
        for member in archive:
            parts = Path(member.name.replace('\\', '/')).parts
            if member.name.startswith('/') or '..' in parts or member.issym() or member.islnk():
                raise RuntimeError('Unsafe archive member')
            parts = tuple(part for part in parts if part != '.')
            if parts:
                roots.add(parts[0])
            if len(parts) >= 2 and parts[0] == 'collections':
                collections.add(parts[1])
            types['files' if member.isfile() else 'directories' if member.isdir() else 'other'] += 1
            entries += 1
    return dict(backup=str(backup), bytes=backup.stat().st_size, sha256=actual,
                hash_matches=True, entries=entries, types=dict(types), roots=sorted(roots),
                collection_count=len(collections), collections=sorted(collections))


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output', type=Path, required=True)
    args = parser.parse_args()
    result = inspect()
    args.output.parent.mkdir(parents=True, exist_ok=True)
    with args.output.open('x', encoding='utf-8') as stream:
        json.dump(result, stream, ensure_ascii=False, indent=2)
    print(json.dumps({key: value for key, value in result.items() if key != 'collections'}, ensure_ascii=False))

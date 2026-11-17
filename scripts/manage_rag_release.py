"""Inspect, validate, explicitly activate or roll back an isolated RAG channel."""
import argparse
import json
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / 'src'))
from researchclaw.knowledge.rag_release import ReleaseRegistry


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('action', choices=['status', 'verify', 'activate', 'rollback'])
    parser.add_argument('--registry', required=True)
    parser.add_argument('--channel', default='pilot')
    parser.add_argument('--release')
    parser.add_argument('--expected-generation', type=int)
    args = parser.parse_args()
    # Even status must refer to a pre-existing registry, not silently create one.
    root = Path(args.registry).resolve()
    if not (root / 'manifests').is_dir():
        parser.error('Existing isolated registry with registered manifests required')
    registry = ReleaseRegistry(root)
    if args.action in ('activate', 'rollback'):
        if args.release is None or args.expected_generation is None:
            parser.error('--release and --expected-generation are required for writes')
        snapshot = getattr(registry, args.action)(args.channel, args.release,
                                                 expected_generation=args.expected_generation)
    else:
        snapshot = registry.resolve(args.channel)
    manifest = snapshot.manifest
    print(json.dumps(dict(status='verified', channel=args.channel, release=snapshot.release_id,
                          generation=snapshot.generation, manifest_hash=snapshot.manifest_hash,
                          inventory=manifest['inventory'], models=manifest['models']),
                     ensure_ascii=False, indent=2))


if __name__ == '__main__':
    main()

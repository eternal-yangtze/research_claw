"""Explicit allowlist cleanup after the final measured release is complete.

Gold reviews, source PDFs, parser caches needed for baselines, final studies and
database backups are never candidates. Default is a dry run.
"""
import argparse
import hashlib
import json
from pathlib import Path
import shutil

ROOT = next(p for p in Path(__file__).resolve().parents if (p / "pyproject.toml").is_file())
BASE = ROOT / 'self-revise/backend-revise/rag-evidence'
OUT = BASE / 'final-rag-v20261109'


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--apply', action='store_true')
    args = parser.parse_args()
    candidates = []
    # Superseded line-level candidate caches; retain their compact comparison
    # reports and all human-labelled cases. Winning typed pilot remains intact.
    for name in ['coordinate-phase3-1', 'coordinate-phase3-1-rawline',
                 'coordinate-phase3-1-word-union', 'coordinate-phase3-1-word-union-indexed',
                 'hybrid-phase3-sample', 'hybrid-phase3-sample-v2']:
        candidates += list((BASE / name).glob('rag_*.json'))
    for name in ['rag-v1.5-candidate-50-retrieval-eval.json', 'rag-v1.5-candidate-60-retrieval-eval.json',
                 'rag-v1.5-rrf-20-retrieval-eval.json', 'rag-v1.5-rrf-40-retrieval-eval.json',
                 'rag-v1.5-rrf-80-retrieval-eval.json', 'time-series-domain-release-v1-retrieval-eval-c120.json',
                 'time-series-domain-release-v1-retrieval-eval-c80.json']:
        if (BASE / name).exists(): candidates.append(BASE / name)
    duplicate = BASE / 'practical-retrieval-v1/partial-results.json'
    if duplicate.exists(): candidates.append(duplicate)
    rows = []
    for p in candidates:
        p.resolve().relative_to(BASE.resolve())
        assert p.is_file() and p.suffix == '.json' and 'gold' not in p.name.lower()
        rows.append({'path': str(p.relative_to(ROOT)), 'bytes': p.stat().st_size,
                     'sha256': hashlib.sha256(p.read_bytes()).hexdigest()})
    report = {'apply': args.apply, 'files': rows, 'bytes': sum(r['bytes'] for r in rows),
        'retained': ['194-paper PostgreSQL/Qdrant backup', 'all source PDFs', 'all human gold and reviews',
                     'GROBID TEI/schema source cache', 'typed-structure pilot used by A/B baseline',
                     'practical-retrieval-v1 baselines', 'final-rag-v20261109', 'compact historical comparison reports'],
        'recovery': 'Deleted candidate caches can be regenerated with retained experiment scripts; no trash backup created'}
    if args.apply:
        acceptance = json.loads((OUT / 'acceptance.json').read_text(encoding='utf8'))
        assert acceptance['ragas_expected'] == 48 and acceptance['full_pipeline_ragas_scored'] == 24
        assert acceptance['status'].startswith('passed_for_personal_research_rag')
        prior_path = OUT / 'cleanup-manifest.json'
        if prior_path.exists():
            prior = json.loads(prior_path.read_text(encoding='utf8'))
            union = {r['path']: r for r in prior['files'] + rows}
            report['files'] = list(union.values())
            report['bytes'] = sum(r['bytes'] for r in union.values())
        protected = set(BASE.rglob('*gold*.json')) | set(BASE.rglob('REVIEW.md')) | set(BASE.rglob('*followup.md'))
        protected |= set((ROOT / 'self-revise').glob('金标*.md')) | set(BASE.glob('金标*.md'))
        protected_hashes = {str(p.relative_to(ROOT)): hashlib.sha256(p.read_bytes()).hexdigest()
                            for p in protected if p.is_file()}
        report['protected_hashes'] = protected_hashes
        (OUT / 'cleanup-manifest.json').write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding='utf8')
        for p in candidates: p.unlink()
        assert all(hashlib.sha256((ROOT / p).read_bytes()).hexdigest() == digest
                   for p, digest in protected_hashes.items())
        pending = []
        for name in ['pytest-temp', 'pytest-temp2']:
            p = OUT / name
            p.resolve().relative_to(OUT.resolve())
            if p.exists():
                try:
                    shutil.rmtree(p)
                except PermissionError:
                    pending.append({'path': str(p.relative_to(ROOT)), 'reason': 'Windows ACL denies current user; native deletion, ACL reset and takeown also denied'})
        report['pending_temporary_directories'] = pending
        report['protected_hashes_verified_after_cleanup'] = True
        (OUT / 'cleanup-manifest.json').write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding='utf8')
        if pending:
            print(json.dumps({'removed_files_total': len(report['files']), 'pending': pending}))
            raise SystemExit(2)
    print(json.dumps({'apply': args.apply, 'files': len(rows), 'bytes': report['bytes']}, ensure_ascii=False))


if __name__ == '__main__': main()

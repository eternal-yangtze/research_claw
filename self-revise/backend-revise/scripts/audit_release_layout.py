"""Read-only delivery audit; never starts stores, models, or external services."""
import hashlib
import json
from pathlib import Path
import re
import sys

ROOT = next(p for p in Path(__file__).resolve().parents if (p / 'pyproject.toml').is_file())
sys.path[:0] = [str(ROOT / 'src'), str(ROOT / 'backend/src')]
from researchclaw.core.postgres import PostgresDatabase


def main():
    checks = {}
    children = sorted(p.name for p in (ROOT / 'self-revise').iterdir())
    assert children == ['backend-revise', 'frontend-revise'], children
    checks['self_revise_children'] = children
    root_markdown = sorted(p.name for p in ROOT.glob('*.md'))
    assert root_markdown == ['README.md', 'README_myself.md'], root_markdown
    checks['root_markdown'] = root_markdown
    forbidden = ['tests', '.maintenance', '.pytest_cache', 'internal_experiments', '__pycache__']
    assert not [name for name in forbidden if (ROOT / name).exists()]
    checks['root_intermediate_directories_absent'] = forbidden
    links = []
    for label, target in re.findall(r'\[([^\]]+)\]\(([^)]+)\)', (ROOT / 'README.md').read_text(encoding='utf8')):
        if re.match(r'https?://|mailto:|#', target):
            continue
        destination = ROOT / target.split('#')[0]
        assert destination.exists(), (label, target)
        links.append(target)
    checks['readme_local_links_verified'] = links
    evidence = [
        'backend-revise/后端测试与性能总报告_v20261110.md',
        'backend-revise/后端逐题测评附录_v20261110.md',
        'backend-revise/preserved-specs/金标1.md',
        'backend-revise/rag-evidence/金标2.md',
        'frontend-revise/README_frontend.md',
        'frontend-revise/evidence/readme-cases-v20261110/cases.json',
        'frontend-revise/evidence/readme-cases-v20261110/browser/real-browser-answer.png',
        'frontend-revise/evidence/registration-rollback-v20261110/real-chain.json',
        'frontend-revise/evidence/release-layout-67-plus-namespace.xml',
    ]
    assert all((ROOT / 'self-revise' / p).is_file() for p in evidence)
    checks['key_evidence_retained'] = evidence
    marker = json.loads((ROOT / 'web_workspace/web-runtime-binding.json').read_text(encoding='utf8'))
    schema = PostgresDatabase(ROOT / 'web_workspace', kind='web').schema
    assert schema == marker['schema']
    checks['existing_web_namespace_unchanged'] = schema
    manifest = ROOT / 'knowledge_base/rag_corpora/time_series_domain_release_v3_1/rag-search-index.json'
    digest = hashlib.sha256(manifest.read_bytes()).hexdigest()
    assert digest == '1c1689eda16e908b2e9415056d24e49156ece93cc486214705d9e6351143862b'
    checks['published_manifest_sha256'] = digest
    compiled = 0
    for folder in ['src', 'backend/src', 'scripts', 'self-revise/backend-revise/tests',
                   'self-revise/backend-revise/scripts', 'self-revise/frontend-revise/scripts']:
        for path in (ROOT / folder).rglob('*.py'):
            compile(path.read_text(encoding='utf-8-sig'), str(path), 'exec')
            compiled += 1
    checks['python_files_syntax_verified'] = compiled
    print(json.dumps({'status': 'verified', 'checks': checks}, ensure_ascii=False, indent=2))


if __name__ == '__main__':
    main()

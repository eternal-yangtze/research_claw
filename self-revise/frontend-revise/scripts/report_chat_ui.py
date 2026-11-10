"""Summarize preserved UI regression and real-model smoke, without model calls."""
import hashlib
import json
from pathlib import Path
import shutil
import xml.etree.ElementTree as ET

ROOT = next(p for p in Path(__file__).resolve().parents if (p / "pyproject.toml").is_file())
OUT = ROOT / 'self-revise/frontend-revise/evidence/v20261110'


def read(path):
    return json.loads(path.read_text(encoding='utf8'))


def main():
    browser = read(OUT / 'browser-verification.json')
    assert browser['status'] == 'passed' and not browser['errors']
    built = []
    for job in browser['build_jobs']:
        folder = Path(job['report_path']).parent
        destination = OUT / 'incremental-publication' / job['id']
        destination.mkdir(parents=True, exist_ok=True)
        for name in ('build-report.json', 'state.events.jsonl'):
            source = folder / name
            if source.is_file():
                shutil.copyfile(source, destination / name)
        built.append({'summary': job['summary'], 'seconds': job['elapsed_seconds'],
            'evidence': str(destination.relative_to(ROOT))})
    xml = ET.parse(OUT / 'backend-regression.xml').getroot()
    api = ET.parse(OUT / 'chat-surface-regression.xml').getroot()
    assert not list(xml.iter('failure')) + list(xml.iter('error')) + list(api.iter('failure')) + list(api.iter('error'))
    # These two suites overlap. Count unique classname/name pairs, never add both totals.
    cases = {(c.attrib['classname'], c.attrib['name']) for root in (xml, api) for c in root.iter('testcase')}
    live = read(OUT / 'real-llm-smoke.json')
    live_rows = []
    for index, case in enumerate(live['cases'], 1):
        folder = Path(live['workspace']) / 'users/live/projects/default/runs' / case['graph']['run_id']
        journal = read(folder / 'agent.json')
        destination = OUT / 'real-llm-runs' / f'case-{index}'
        destination.mkdir(parents=True, exist_ok=True)
        for name in ('agent.json', 'context.json'):
            if (folder / name).is_file():
                shutil.copyfile(folder / name, destination / name)
        observations = journal.get('observations', [])
        live_rows.append({'question': case['question'], 'status': case['graph']['status'],
            'seconds': round(case['seconds'], 2), 'sources': len(case['sources']),
            'tools': [{'tool': o['tool'], 'status': o['result']['status']} for o in observations],
            'guard_stop': journal.get('guard_stop')})
    result = {'regression_unique_cases': len(cases), 'browser_checks': len(browser['checks']),
        'browser_errors': browser['errors'], 'incremental_builds': built, 'real_llm_cases': live_rows,
        'shared_papers_seen': live['published_papers'],
        'scope': 'Frontend/API integration checks and real-model smoke; not RAG/LongMemEval benchmark reruns'}
    retry_file = OUT / 'real-llm-reference-retry.json'
    if retry_file.is_file():
        retry = read(retry_file)
        result['same_session_reference_retry'] = {k: retry[k] for k in ('question', 'seconds', 'tools', 'guard_stop')}
        result['same_session_reference_retry']['status'] = retry['graph']['status']
        folder = Path(retry['workspace']) / 'users/live/projects/default/runs' / retry['graph']['run_id']
        target = OUT / 'real-llm-runs/reference-retry'
        target.mkdir(parents=True, exist_ok=True)
        for name in ('agent.json', 'context.json'):
            if (folder / name).is_file():
                shutil.copyfile(folder / name, target / name)
    result['evidence_sha256'] = {p.relative_to(OUT).as_posix(): hashlib.sha256(p.read_bytes()).hexdigest()
        for p in OUT.rglob('*') if p.is_file() and p.name not in {'summary.json', 'README.md'}}
    (OUT / 'summary.json').write_text(json.dumps(result, ensure_ascii=False, indent=2), encoding='utf8')
    print(json.dumps({k: v for k, v in result.items() if k != 'evidence_sha256'}, ensure_ascii=False))


if __name__ == '__main__':
    main()

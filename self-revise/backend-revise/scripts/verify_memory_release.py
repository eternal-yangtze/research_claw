"""Persist the actual regression/build results and evidence fingerprints."""
import hashlib
import json
from pathlib import Path
import subprocess
import sys
import time
import uuid

ROOT=next(p for p in Path(__file__).resolve().parents if (p / "pyproject.toml").is_file())
sys.path[:0]=[str(ROOT/'src')]
from researchclaw.core.artifacts import write_json,canonical_hash,file_hash


def main():
    out=ROOT/'self-revise/backend-revise/memory-evidence/v20261109'
    start=time.perf_counter()
    tests=['tests/api/test_conversation_memory.py','tests/api/test_separated_backend.py',
           'tests/api/test_session_knowledge.py','tests/harness/test_skills.py']
    run=subprocess.run([sys.executable,'-m','pytest',*tests,'-q','-p','no:cacheprovider',
        '--basetemp=self-revise/frontend-revise/local-runtime/memory-recorded-'+uuid.uuid4().hex[:8],f'--junitxml={out / "regression.xml"}'],
        cwd=ROOT,capture_output=True,text=True,encoding='utf8',errors='replace')
    build=subprocess.run(['cmd','/c','npm.cmd','run','build'],cwd=ROOT/'frontend',
        capture_output=True,text=True,encoding='utf8',errors='replace')
    suite=json.loads((out/'suite.json').read_text(encoding='utf8'))
    frozen=json.loads((out/'freeze.json').read_text(encoding='utf8'))
    summary=json.loads((out/'summary.json').read_text(encoding='utf8'))
    environment=file_hash(ROOT/'self-revise/环境.md')
    valid=environment=='3535fc8a2a82cf5c6b58136eae6c288ad58acecc525563c691fed9feeabfe237' and canonical_hash(suite)==frozen['sha256']==summary['suite_hash']
    live=json.loads((out/'live-api.json').read_text(encoding='utf8'))
    sources=['backend/src/researchclaw_api/'+x for x in ('conversation.py','conversation_tools.py','store.py','main.py')]
    sources += ['src/researchclaw/harness/context.py','src/researchclaw/harness/skills/loader.py',
                'src/researchclaw/harness/skills/runtime.py',
                'src/researchclaw/core/qdrant_store.py','self-revise/backend-revise/scripts/evaluate_conversation_memory.py','self-revise/backend-revise/scripts/verify_conversation_live.py']
    write_json(out/'baseline-extension.json',dict(suite_changed=False,suite_sha256=frozen['sha256'],
        original_arms=['recent-window','summary-window','state-source-memory'],
        added_arm='llm-summary-window',reason='Stronger standard summary baseline, query-independent',
        failed_attempts_preserved=len(list((out/'failed-attempts').glob('*.json'))),retuned_answers=False))
    report=dict(regression_returncode=run.returncode,regression_stdout=run.stdout,regression_stderr=run.stderr,
        frontend_returncode=build.returncode,frontend_stdout=build.stdout,frontend_stderr=build.stderr,
        frozen_suite_and_environment_verified=valid,environment_sha256=environment,live_api_passed=live['passed'],
        source_sha256={p:file_hash(ROOT/p) for p in sources},seconds=time.perf_counter()-start,
        passed=run.returncode==build.returncode==0 and valid and live['passed'] and summary['failed']==0)
    write_json(out/'verification.json',report)
    evidence=[p for p in out.iterdir() if p.is_file() and p.name not in {'evidence-manifest.json'}]
    for name in ('calls','summaries','failed-attempts'):
        evidence.extend((out/name).glob('*.json'))
    write_json(out/'evidence-manifest.json',{str(p.relative_to(out)):file_hash(p) for p in evidence})
    print(run.stdout,flush=True)
    print('Frontend build return code:',build.returncode,'Release verified:',report['passed'],flush=True)
    if not report['passed']:
        raise SystemExit(1)


if __name__=='__main__':
    main()

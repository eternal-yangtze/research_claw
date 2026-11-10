r"""Run from activated CMD: python scripts\serve_backend.py"""
import argparse
import os
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path[:0] = [str(ROOT / 'src'), str(ROOT / 'backend' / 'src')]
os.chdir(ROOT)

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Start ResearchClaw API; local .env.web supplies credentials')
    parser.add_argument('--host', default='127.0.0.1')
    parser.add_argument('--port', default=8011, type=int)
    parser.add_argument('--offline', action='store_true', help='Use real deterministic hashing retrieval, no embedding downloads')
    parser.add_argument('--workspace', help='Optional Web state root; defaults to .env.web configuration')
    args = parser.parse_args()
    if args.workspace:
        os.environ['RESEARCHCLAW_WEB_WORKSPACE'] = str(Path(args.workspace).resolve())
    from dotenv import load_dotenv
    from researchclaw.core.config import load_project_env
    load_dotenv(ROOT/'.env.web',override=False)
    load_project_env()
    # Interactive uploads use the text-layer parser; explicit environment values
    # still allow Docling/OCR when the operator needs scanned PDF support.
    os.environ.setdefault('RESEARCHCLAW_PDF_PARSER', 'pymupdf')
    os.environ.setdefault('RESEARCHCLAW_KNOWLEDGE_BACKEND', 'qdrant')
    os.environ.setdefault('RESEARCHCLAW_VECTOR_BACKEND', 'qdrant')
    os.environ.setdefault('RESEARCHCLAW_RETRIEVAL_BACKEND', 'neural')
    if args.offline:
        raise SystemExit('统一环境使用 Qdrant；请启动 Qdrant 并移除 --offline。')
    if os.environ['RESEARCHCLAW_KNOWLEDGE_BACKEND'] != 'qdrant':
        raise SystemExit('统一 Web 环境只允许 RESEARCHCLAW_KNOWLEDGE_BACKEND=qdrant，请清除旧 CMD 中的 sqlite 配置。')
    os.environ['RESEARCHCLAW_VECTOR_BACKEND']='qdrant'
    from researchclaw.core.postgres import preflight
    preflight()
    print('Storage: PostgreSQL authority/runtime/checkpoints + Qdrant index; legacy SQLite frozen.',flush=True)
    import uvicorn
    uvicorn.run('researchclaw_api.main:create_app', factory=True, host=args.host, port=args.port)

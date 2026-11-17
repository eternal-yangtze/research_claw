"""Start the existing Web/Harness with explicit CPU retrieval; GPU entry is unchanged."""
import os
import sys
from pathlib import Path
import runpy

if __name__=='__main__':
    root=Path(__file__).resolve().parents[1]
    os.environ['RESEARCHCLAW_NEURAL_BACKEND']='local_cpu'
    os.environ.setdefault('RESEARCHCLAW_CPU_THREADS','4')
    os.environ.setdefault('RESEARCHCLAW_CPU_LOCAL_ONLY','true')
    # A short, isolated artifact root avoids Windows MAX_PATH failures for
    # per-user run/checkpoint filenames. Explicit --workspace still wins.
    if '--workspace' not in sys.argv:
        os.environ.setdefault('RESEARCHCLAW_WEB_WORKSPACE',str(root/'.cpu-runtime'))
    runpy.run_path(str(root/'scripts/serve_backend.py'),run_name='__main__')

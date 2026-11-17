"""CPU entry for the published PG/Qdrant corpus: ask, ingest and structure-build."""
import os
from pathlib import Path
import runpy
import sys

if __name__=='__main__':
    if 'ask-fullcorpus' in sys.argv[1:]:
        raise SystemExit('CPU entry uses ask against the published corpus; ask-fullcorpus remains the frozen GPU job interface.')
    os.environ['RESEARCHCLAW_NEURAL_BACKEND']='local_cpu'
    os.environ.setdefault('RESEARCHCLAW_CPU_THREADS','4')
    os.environ.setdefault('RESEARCHCLAW_CPU_LOCAL_ONLY','true')
    runpy.run_path(str(Path(__file__).with_name('research_rag.py')),run_name='__main__')

"""Private, resident CUDA inference entrypoint."""
import argparse
from pathlib import Path
import sys
sys.path.insert(0,str(Path(__file__).resolve().parents[1]/'src'))
from researchclaw.knowledge.gpu_service import serve

if __name__=='__main__':
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--port',type=int,default=8766)
    serve(parser.parse_args().port)

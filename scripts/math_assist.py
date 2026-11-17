"""Explicit standalone mathematical assistant using the configured answer API."""
import argparse
import json
from pathlib import Path
import sys
ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT/'src'))

def main():
    from researchclaw.core.config import load_project_env
    from researchclaw.harness.runtime import OpenAICompatibleModel
    from researchclaw.knowledge.math_tools import MathReasoner,MathSandbox
    parser=argparse.ArgumentParser()
    parser.add_argument('question')
    parser.add_argument('--no-tools',action='store_true')
    args=parser.parse_args()
    load_project_env()
    engine=MathReasoner(OpenAICompatibleModel(max_attempts=1),sandbox=None if args.no_tools else MathSandbox())
    result=engine.run(args.question)
    print(json.dumps({'answer':result['answer'],'status':result['status'],'claims':result.get('claims',[]),
                      'evidence_directory':str(engine.run_dir)},ensure_ascii=False,indent=2))

if __name__=='__main__':main()

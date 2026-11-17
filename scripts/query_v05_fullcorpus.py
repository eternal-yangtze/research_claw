"""Run an actual natural query against the isolated whole-corpus CUDA index."""
import argparse
import json
from pathlib import Path
import re
import sys
ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT/'src'))
def main():
    p=argparse.ArgumentParser();p.add_argument('--question',required=True);p.add_argument('--training');p.add_argument('--arm',choices=['hard-lora','distilled-lora']);p.add_argument('--fusion',choices=['rank','none']);p.add_argument('--run',required=True);a=p.parse_args()
    assert re.fullmatch(r'[A-Za-z0-9_-]{1,80}',a.run)
    if a.training:assert re.fullmatch(r'[A-Za-z0-9_-]{1,80}',a.training)
    from researchclaw.knowledge.fullcorpus_retrieval import FullCorpusRetriever
    base=ROOT/'self-revisev0.5';adapter=None;expected=None;fusion=a.fusion or 'rank'
    if a.training:
        summary=json.loads((base/'artifacts/p3'/a.training/'summary.json').read_text())
        arm=a.arm or summary['selected_by_validation']
        descriptor=base/'workspace/p3/fullcorpus-r1/selected.json'
        if descriptor.exists() and not a.arm:
            selected=json.loads(descriptor.read_text())
            if selected['training']!=a.training:raise ValueError('Selected retrieval release belongs to another training run')
            arm=selected['arm'];fusion=a.fusion or selected['fusion'];expected=selected['adapter']['identity_sha256'] if selected['adapter'] else None
            if json.loads((base/'workspace/p3/fullcorpus-r1/retrieval/summary.json').read_text())['index_sha256']!=selected['index_sha256']:raise ValueError('Selected release index changed')
        if arm!='base':adapter=base/'artifacts/p3'/a.training/arm
    engine=FullCorpusRetriever(base/'workspace/p3/fullcorpus-r1/retrieval',adapter=adapter,expected_adapter_identity=expected)
    result=engine.search(a.question,fusion=fusion)
    out=base/'artifacts/p3'/(a.run+'.json');assert not out.exists()
    out.write_text(json.dumps(result,ensure_ascii=False,indent=2),encoding='utf8');print(json.dumps(result,ensure_ascii=False),flush=True)
if __name__=='__main__':main()

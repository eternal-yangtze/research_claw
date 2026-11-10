"""Summarize the personal-tool acceptance without promoting development scores."""
import json
from pathlib import Path
import statistics

ROOT=next(p for p in Path(__file__).resolve().parents if (p / "pyproject.toml").is_file())

def main():
    out=ROOT/'self-revise/backend-revise/rag-evidence/practical-retrieval-v1'
    read=lambda name:json.loads((out/name).read_text(encoding='utf8'))
    report=read('published-runtime-bm25-report.json')
    quality=read('evidence-quality.json');gates=read('practical-gates.json')
    hypotheses=read('report.json')
    current=read('published-runtime-bm25-hits.json');ab=read('retrieved-evidence.json')
    same=all([h['child_text'] for h in current[q][:3]]==[h['text'] for h in ab['baseline:'+q][:3]] for q in current)
    checks={'legacy_recall_at_10':report['groups']['legacy']['recall@10']>=gates['minimum_legacy_recall_at_10'],
            'target_supported_at_3_silver':quality['summaries']['baseline']['supported_at_3']>=gates['minimum_target_supported_at_3'],
            'all_retrieved_pages_known':report['missing_page_hits']==0,
            'official_entrypoint_uses_qdrant_bm25':report['lexical_backend']=='qdrant_bm25',
            'official_top3_matches_silver_assessed_evidence':same,
            'no_postgres_vector_copies':report['inventory']['postgres_embedding_copies']==0,
            'no_op_has_no_dense_encoding':read('lexical-index-noop.json')['dense_vectors_encoded']==0}
    warm=[c['seconds'] for c in report['cases'][1:]]
    result={'status':'passed_for_personal_retrieval' if all(checks.values()) else 'needs_attention',
            'checks':checks,'scope':'194-paper personal research retrieval; small topic-labeled suite and LLM silver, not production/RAGAS certification',
            'selected_chunk_policy':'page-parent-word-child-v1','structure_candidate_promoted':False,
            'decision_reason':'Paper rankings tie, baseline direct evidence at3 is 6/8 versus 3/8 for the structure candidate',
            'paper_retrieval_metrics':report['groups'],'silver_evidence_metrics':quality['summaries'],
            'warm_median_seconds':statistics.median(warm),'warm_max_seconds':max(warm),
            'first_query_including_model_load_seconds':report['cases'][0]['seconds'],
            'sparse_initial_build':read('lexical-index-build.json'),'sparse_no_op':read('lexical-index-noop.json'),
            'chunking_ab':{'baseline_children':hypotheses['base_children'],'candidate_children':hypotheses['structured_children'],
                           'changed_papers':hypotheses['changed_papers'],'candidate_dense_vectors_encoded':hypotheses['vectors_encoded']},
            'known_gaps':['Only one of three labeled streaming PLA papers retrieved for one legacy question',
                          'Some top3 passages are background/irrelevant; do not treat every retrieved passage as supporting evidence',
                          'Paper relevance labels are incomplete; labels and LLM silver are not independent human gold',
                          'Generation/RAGAS, memory, context and agent runtime are outside this retrieval acceptance']}
    (out/'acceptance.json').write_text(json.dumps(result,ensure_ascii=False,indent=2),encoding='utf8')
    print(json.dumps({'status':result['status'],'checks':checks},ensure_ascii=True,indent=2))

if __name__=='__main__':main()

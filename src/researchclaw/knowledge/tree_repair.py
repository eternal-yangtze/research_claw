"""Untrained, bounded tree priors and traceable fine evidence views."""
from researchclaw.knowledge.token_safety import source_windows,token_count


def focused_windows(document, tokenizer, maximum=160):
    heading=document['section']
    while heading and token_count(tokenizer,heading+'\n')>=maximum//2:
        heading=heading[:len(heading)//2]
    return [dict(w,embedding_text=heading+'\n'+w['text'])
            for w in source_windows(document['text'],tokenizer,prefix=heading+'\n',maximum=maximum)]


def soft_paper_prior(fusion,documents,ranked_papers,weight=.25):
    """Add a bounded third signal without deleting any global candidate."""
    if not 0<=weight<=1:raise ValueError('Bounded prior weight required')
    ranks={paper:rank for rank,paper in enumerate(ranked_papers,1)}
    return {i:score+(weight/(60+ranks[documents[i]['paper_id']])
                     if documents[i]['paper_id'] in ranks else 0.) for i,score in fusion.items()}

"""Standard binary/graded IR metrics; query IDs remain the original benchmark IDs."""
import math


def retrieval_metrics(ranking, relevant, *, k=10):
    if k <= 0:
        raise ValueError('k must be positive')
    # One retrieved item can only contribute once, even if a caller expands
    # multiple child windows to the same paper/parent identity.
    ranking = list(dict.fromkeys(ranking))
    hits = [max(0, relevant.get(doc, 0)) for doc in ranking[:k]]
    positive = sum(v > 0 for v in relevant.values())
    recall = sum(v > 0 for v in hits) / max(1, positive)
    reciprocal = next((1/i for i,v in enumerate(hits,1) if v > 0), 0.)
    dcg = sum((2**v-1)/math.log2(i+1) for i,v in enumerate(hits,1))
    ideal = sum((2**v-1)/math.log2(i+1) for i,v in enumerate(sorted(relevant.values(), reverse=True)[:k],1))
    return {f'Recall@{k}':recall, f'MRR@{k}':reciprocal, f'nDCG@{k}':dcg/ideal if ideal else 0.,
            f'Precision@{k}':sum(v > 0 for v in hits)/k,
            f'HitRate@{k}':float(any(v > 0 for v in hits))}

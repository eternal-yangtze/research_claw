"""Reject silently truncated, malformed or mismatched dense vectors before storage."""
import math

def validate_vectors(vectors, *, expected, dimension=384):
    if len(vectors)!=expected:raise ValueError('Embedding count mismatch; refusing partial publication')
    for vector in vectors:
        if len(vector)!=dimension or any(not math.isfinite(float(x)) for x in vector):
            raise ValueError('Embedding dimension/non-finite value mismatch')
        norm=math.sqrt(sum(float(x)**2 for x in vector))
        if abs(norm-1)>1e-3:raise ValueError('Embedding must be normalized; refusing invalid vector')

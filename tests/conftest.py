import os


# Unit tests stay network-independent. Dedicated retrieval tests inject neural-model fakes;
# real-model verification is performed by scripts/verify_neural_retrieval.py.
os.environ.setdefault("RESEARCHOPS_RETRIEVAL_BACKEND", "hashing")

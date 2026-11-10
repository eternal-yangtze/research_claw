import os


# Unit tests stay network-independent. Dedicated retrieval tests inject neural-model fakes;
# real-model verification is performed by self-revise/backend-revise/scripts/verify_neural_retrieval.py.
os.environ.setdefault("RESEARCHCLAW_RETRIEVAL_BACKEND", "hashing")
# New tests must exercise Qdrant. Legacy SQLite import tests set their backend
# explicitly and are not allowed to change the application default.
os.environ.setdefault("RESEARCHCLAW_KNOWLEDGE_BACKEND", "qdrant")
os.environ.setdefault('RESEARCHCLAW_SHARED_CORPUS','off')
# Historical API fixtures use a shared password; real Web defaults to invited accounts.
os.environ.setdefault('RESEARCHCLAW_ALLOW_LEGACY_LOGIN', '1')
os.environ.setdefault('RESEARCHCLAW_ALLOW_DIRECT_INGEST', '1')

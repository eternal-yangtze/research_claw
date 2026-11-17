"""Bound future human annotation and retire the full-library identity queue."""
MAX_FUTURE_HUMAN_PAPERS=40
LEGACY_IDENTITY_QUEUE_ACTIVE=False


def require_legacy_identity_queue():
    if not LEGACY_IDENTITY_QUEUE_ACTIVE:
        raise ValueError('Full-library identity review retired by user; do not request more identity submissions.')


def validate_future_annotation_scope(paper_ids, *, previously_requested=()):
    """Count distinct papers across future tasks, not per batch; reuse costs zero."""
    combined=set(previously_requested)|set(paper_ids)
    if len(combined)>MAX_FUTURE_HUMAN_PAPERS:
        raise ValueError('Future human annotation exceeds the cumulative 40-paper limit')
    return sorted(combined)

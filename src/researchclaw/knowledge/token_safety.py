"""Exact-token budgets with lossless source character spans."""
from dataclasses import replace
import hashlib
import re

POLICY = 'page-parent-token-child-v05'


def token_count(tokenizer, text):
    return len(tokenizer.encode(text, add_special_tokens=True, truncation=False))


def source_windows(text, tokenizer, *, prefix='', maximum=512):
    """Partition source without decode/re-encode corruption or silent truncation.

    Every character belongs to exactly one span. Prefix and special tokens count
    against each window; long identifiers and formulae are split if necessary.
    """
    if token_count(tokenizer, prefix) >= maximum:
        raise ValueError('Section/query prefix exhausts token budget')
    spans = []
    start = 0
    while start < len(text):
        remaining = text[start:]
        remaining_tokens = token_count(tokenizer, prefix + remaining)
        if remaining_tokens <= maximum:
            spans.append({'text':remaining,'char_start':start,'char_end':len(text),'tokens':remaining_tokens})
            break
        low, high = start + 1, len(text)
        if token_count(tokenizer, prefix + text[start:low]) > maximum:
            raise ValueError('Cannot fit one source character in token budget')
        best = low
        while low <= high:
            mid = (low + high) // 2
            if token_count(tokenizer, prefix + text[start:mid]) <= maximum:
                best, low = mid, mid + 1
            else:
                high = mid - 1
        # Prefer complete paragraphs/sentences near the budget, retaining every character.
        if best < len(text):
            minimum=start+max(1,int((best-start)*.6))
            paragraph=text.rfind('\n\n',minimum,best)
            sentences=list(re.finditer(r'[.!?](?:\s+|$)',text[minimum:best]))
            if paragraph>=minimum:best=paragraph+2
            elif sentences:best=minimum+sentences[-1].end()
            else:
                boundary=text.rfind(' ',minimum,best)
                if boundary>start:best=boundary+1
        value = text[start:best]
        count = token_count(tokenizer, prefix + value)
        assert count <= maximum
        spans.append({'text': value, 'char_start': start, 'char_end': best, 'tokens': count})
        start = best
    return spans


def split_children(nodes, tokenizer, *, maximum=512):
    result = []
    for node in nodes:
        if node.node_type != 'child':
            result.append(node)
            continue
        spans = source_windows(node.text, tokenizer, prefix=node.section + '\n', maximum=maximum)
        for span in spans:
            text = span['text']
            identity = f'{POLICY}:{node.chunk_id}:{span["char_start"]}:{text}'
            result.append(replace(node, chunk_id=hashlib.sha256(identity.encode()).hexdigest()[:40],
                                  text=text, content_hash=hashlib.sha256(text.encode()).hexdigest(),
                                  provenance={**node.provenance, 'chunk_policy': POLICY,
                                              'original_child_id': node.chunk_id,
                                              'source_char_start': span['char_start'],
                                              'source_char_end': span['char_end'],
                                              'input_tokens': span['tokens'], 'maximum_tokens': maximum}))
    return result

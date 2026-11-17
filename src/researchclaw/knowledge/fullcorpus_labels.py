"""Apply one declared work-identity space to both candidates and public qrels."""
import hashlib
import json

POLICY = 'canonical-work-qrels-v1'


def canonicalize_row(row, id_to_work):
    if row.get('relevance_unit') != 'paper':
        return row
    result = dict(row)
    gold = list(dict.fromkeys(id_to_work.get(g, g) for g in row['gold']))
    positives = set(gold)
    result.update(gold=gold, total_positives=len(gold),
                  labels=[int(work in positives) for work in row['candidate_works']],
                  candidate_recall=len(positives & set(row['candidate_works'])) / len(gold))
    return result


def label_space(index):
    path = index / 'corpus.json'
    digest = hashlib.sha256(path.read_bytes()).hexdigest()
    expected = json.loads((index / 'summary.json').read_text(encoding='utf8'))['index_sha256']['corpus.json']
    if digest != expected:
        raise ValueError('Canonical label mapping requires verified corpus')
    docs = json.loads(path.read_text(encoding='utf8'))
    mapping = {d['id']: d['canonical_work'] for d in docs}
    return mapping, dict(version=POLICY, corpus_sha256=digest,
                        scope='Same automatic normalized-title work identity as frozen candidates, also applied to original qrels; not manual version adjudication. Original questions, documents, candidate order, teacher requests unchanged.')

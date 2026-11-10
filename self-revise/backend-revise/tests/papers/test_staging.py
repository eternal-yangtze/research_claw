import threading
from types import SimpleNamespace

import pytest

from researchclaw.papers.staging import stage_papers


def test_parallel_partial_download_and_explicit_ids(monkeypatch, tmp_path):
    barrier = threading.Barrier(2)
    def download(url, *args, **kwargs):
        barrier.wait(timeout=2)  # Serial implementation cannot pass.
        assert kwargs['deadline_seconds'] == 60
        if '2106' in url:
            raise TimeoutError('slow paper')
        return {'path': str(tmp_path/'vpt.pdf'), 'bytes': 42}
    monkeypatch.setattr('researchclaw.papers.staging.download_pdf', download)
    result = stage_papers(None, '2203.12119 2106.09685', tmp_path)
    assert result['status'] == 'partial'
    assert len(result['downloaded']) == len(result['failures']) == 1


def test_field_query_is_arxiv_only_and_irrelevant_titles_rejected(monkeypatch, tmp_path):
    def search(query, **kwargs):
        assert kwargs['sources'] == ['arxiv']
        return {'papers': [{'title': 'Ti-Audio Tibetan Speech', 'pdf_url': 'https://example.com/wrong'}], 'sources': []}
    def unexpected(*args, **kwargs):
        raise AssertionError('irrelevant PDF must not be downloaded')
    monkeypatch.setattr('researchclaw.papers.staging.download_pdf', unexpected)
    result = stage_papers(SimpleNamespace(search=search), 'ti:"LoRA Low Rank Adaptation"', tmp_path)
    assert result['status'] == 'failed'
    assert not result['downloaded']


def test_slow_stream_deadline_leaves_no_partial_pdf(monkeypatch, tmp_path):
    from researchclaw.papers.connectors import download_pdf
    clock = iter([0, 0, 61])
    monkeypatch.setattr('researchclaw.papers.connectors.time.monotonic', lambda: next(clock))
    class Response:
        headers = {}
        def __enter__(self):
            return self
        def __exit__(self, *args):
            pass
        def read1(self, size):
            return b'%PDF-slow stream'
    monkeypatch.setattr('researchclaw.papers.connectors.urllib.request.urlopen', lambda *a, **k: Response())
    with pytest.raises(TimeoutError):
        download_pdf('https://example.com/slow.pdf', tmp_path)
    assert not list(tmp_path.glob('*.pdf'))

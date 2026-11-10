"""Parallel, bounded PDF acquisition, shared by CLI and Web."""
import re
from concurrent.futures import ThreadPoolExecutor, as_completed

from researchclaw.papers.connectors import download_pdf


def stage_papers(literature, query, output_dir, limit=3, sources=None, progress=lambda value: None):
    # Explicit IDs are deterministic acquisition, not fabricated search results.
    ids = list(dict.fromkeys(re.findall(r'(?<!\d)(\d{4}\.\d{4,5})(?:v\d+)?(?!\d)', query)))
    if ids:
        papers = [{'paper_id': 'arxiv_' + identifier, 'title': 'arXiv ' + identifier,
                   'pdf_url': 'https://arxiv.org/pdf/' + identifier} for identifier in ids]
        statuses = []
    else:
        # Field syntax belongs only to arXiv, not to every search provider.
        selected = ['arxiv'] if re.search(r'\b(?:ti|au|abs|id|all):', query) else sources
        progress({'stage': 'search_started', 'sources': selected or list(literature.connectors)})
        found = literature.search(query, run_id='staged_download', limit_per_source=5,
                                  sources=selected, progress=progress)
        papers, statuses = found['papers'], found['sources']
        terms = set(re.findall(r'[a-z0-9]+', query.lower())) - {'the', 'of', 'and', 'in', 'for', 'ti', 'all'}
        def score(paper):
            title = set(re.findall(r'[a-z0-9]+', paper['title'].lower()))
            return len(terms & title) / max(1, len(terms))
        papers = sorted((p for p in papers if score(p) >= .4), key=score, reverse=True)
    candidates = [p for p in papers if p.get('pdf_url')][:max(1, limit)]
    downloaded, failures = [], []
    def fetch(paper):
        progress({'stage': 'downloading', 'title': paper['title'], 'url': paper['pdf_url']})
        try:
            result = download_pdf(paper['pdf_url'], output_dir, filename=paper['paper_id']+'.pdf',
                                  max_bytes=20*1024*1024, deadline_seconds=60)
            result['title'] = paper['title']
            progress({'stage': 'downloaded', **result})
            return result, None
        except Exception as error:
            failure = {'paper_id': paper['paper_id'], 'error': str(error)}
            progress({'stage': 'download_failed', **failure})
            return None, failure
    with ThreadPoolExecutor(max_workers=3) as pool:
        for future in as_completed([pool.submit(fetch, p) for p in candidates]):
            result, failure = future.result()
            if result:
                downloaded.append(result)
            else:
                failures.append(failure)
    status = 'partial' if downloaded and failures else 'staged' if downloaded else 'failed'
    return {'status': status, 'downloaded': downloaded, 'failures': failures,
            'sources': statuses, 'indexed': False,
            'message': f'成功下载 {len(downloaded)} 篇，失败 {len(failures)} 篇；请确认相关性后入库。' if downloaded else '未下载到相关原文，请检查来源与网络，或使用明确的 arXiv 编号。'}

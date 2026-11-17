"""Source-supported PDF title candidates; never infer work identity from a filename."""
import hashlib
from pathlib import Path
import re
import statistics
import unicodedata

VERSION='pdf-frontmatter-identity-v05.2'
NOISE=re.compile(r'https?://|doi\b|arxiv\s*:|@|copyright|all rights reserved|accepted|proceedings|'
                 r'conference on|transactions on|journal of|contents lists|available online|'
                 r'^review article$|^research article$|^abstract\b|^keywords\b|^\d+\s*$',re.I)

def normalized(text):
    return re.sub(r'[^\w]+','',unicodedata.normalize('NFKC',text).casefold())

def identify_pdf(path):
    import fitz
    path=Path(path)
    with fitz.open(path) as doc:
        metadata=(doc.metadata or {}).get('title','').strip()
        if not len(doc):raise ValueError('Empty PDF')
        page=doc[0];lines=[]
        for block in page.get_text('dict')['blocks']:
            for line in block.get('lines',[]):
                spans=line.get('spans',[])
                text=''.join(s['text'] for s in spans).strip()
                if text:lines.append(dict(text=text,size=max(s['size'] for s in spans),bbox=list(line['bbox'])))
        body=statistics.median([l['size'] for l in lines]) if lines else 10
        candidates=[l for l in lines if l['bbox'][1]<page.rect.height*.48 and l['size']>=body*1.13
            and len(l['text'])>=3 and not NOISE.search(l['text'])]
        groups=[]
        for line in sorted(candidates,key=lambda l:(l['bbox'][1],l['bbox'][0])):
            if (groups and .78<=line['size']/groups[-1][-1]['size']<=1.25
                and -line['size']*.4<=line['bbox'][1]-groups[-1][-1]['bbox'][3]<=line['size']*.9):
                groups[-1].append(line)
            else:groups.append([line])
        groups=[g for g in groups if len(' '.join(l['text'] for l in g).split())>=3]
        group=max(groups,key=lambda g:(max(l['size'] for l in g),sum(len(l['text']) for l in g)),default=[])
        layout=' '.join(l['text'] for l in group)
        literal=' '.join(l['text'] for l in sorted(lines,key=lambda l:(l['bbox'][1],l['bbox'][0])))
        supported=bool(metadata and len(metadata)>=8 and not NOISE.search(metadata)
                       and normalized(metadata) in normalized(literal))
        conflict=bool(layout and supported and normalized(metadata) not in normalized(layout)
                      and normalized(layout) not in normalized(metadata))
        if supported:
            title,status=metadata,('literal_metadata_layout_disagreement' if conflict else 'literal_metadata_candidate')
        elif layout:
            title,status=layout,('layout_metadata_conflict' if conflict else 'layout_candidate')
        else:title,status='Unresolved paper '+hashlib.sha256(path.read_bytes()).hexdigest()[:12],'unresolved'
        return dict(version=VERSION,title=title,status=status,source_sha256=hashlib.sha256(path.read_bytes()).hexdigest(),
            metadata_title=metadata,metadata_supported_on_first_page=supported,
            unsupported_metadata_ignored=bool(metadata and not supported),layout_title=layout,
            title_spans=group,doi_candidates=sorted(set(re.findall(r'10\.\d{4,9}/[^\s<>]+',literal)))[:5],
            filename_used_as_identity=False,human_verified=False,
            warning='PDF hash identifies bytes, not work family; title/DOI candidates never auto-merge versions or become gold.')

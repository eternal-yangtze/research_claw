"""PDF physical-page locators and immutable crops, without invented geometry."""
import hashlib
from pathlib import Path


def locate_quote(path, *, page, quote, output_dir=None):
    import fitz
    path=Path(path)
    digest=hashlib.sha256(path.read_bytes()).hexdigest()
    with fitz.open(path) as document:
        if not 1 <= page <= len(document):
            raise ValueError('Physical PDF page is out of range')
        source=document[page-1]
        boxes=source.search_for(quote) if quote.strip() else []
        locator={'source_hash':digest,'physical_page':page,'page_count':len(document),
                 'quote':quote,'bbox':[list(box) for box in boxes],
                 'geometry':'exact_text_search' if boxes else 'page_fallback_no_verified_bbox'}
        if output_dir:
            out=Path(output_dir).resolve()
            root=Path(__file__).resolve().parents[3]/'self-revisev0.5'
            if not out.is_relative_to(root.resolve()):
                raise ValueError('0.5 crops must stay inside self-revisev0.5')
            out.mkdir(parents=True,exist_ok=True)
            clip=boxes[0] if len(boxes)==1 else source.rect
            image=out/f'{digest[:16]}-p{page}-{hashlib.sha256(quote.encode()).hexdigest()[:8]}.png'
            source.get_pixmap(matrix=fitz.Matrix(1.5,1.5),clip=clip).save(image)
            locator['image']=str(image)
        return locator

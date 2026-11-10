"""Offline typed-node annotations, separate from body paragraph alignment."""
from collections import defaultdict
import unicodedata

from researchclaw.knowledge.coordinate_alignment import spatial_support
from researchclaw.knowledge.rag_pipeline import repair_mojibake


def canonical_text(text):
    # Expand ligatures without deleting mathematical letters or digits.
    text = unicodedata.normalize('NFKC', repair_mojibake(text)).casefold()
    return ''.join(c for c in text if c.isalnum())


def extract_typed_nodes(document, root):
    parents = {child: parent for parent in root.iter() for child in parent}
    by_title = defaultdict(list)
    for section in document['sections']:
        by_title[canonical_text(section['section_title'])].append(section['section_id'])
    def tag(e):
        return e.tag.rsplit('}', 1)[-1]
    def ancestors(e):
        while e in parents:
            e = parents[e]
            yield e
    def section_of(e):
        for a in ancestors(e):
            if tag(a) != 'div':
                continue
            heads = [c for c in a if tag(c) == 'head']
            if heads:
                ids = by_title.get(canonical_text(''.join(heads[0].itertext())), [])
                # Never borrow a broader section when the nearest is ambiguous.
                return ids[0] if len(ids) == 1 else None
        return None
    nodes = []
    for element in root.iter():
        kind = tag(element)
        lineage = list(ancestors(element))
        if kind == 'p' and any(tag(a) == 'abstract' for a in lineage):
            content_type = 'abstract'
        elif kind == 'p' and any(tag(a) == 'back' for a in lineage) and not any(tag(a) == 'listBibl' for a in lineage):
            # Appendices can contain proofs or conclusions omitted by a
            # body-only schema. Keep separate; do not fabricate body IDs.
            content_type = 'back_matter_text'
        elif kind == 'head' and tag(parents[element]) == 'div' and any(tag(a) == 'body' for a in lineage):
            content_type = 'section_heading'
        elif kind == 'formula':
            content_type = 'formula'
        elif kind == 'figure':
            # GROBID also places theorem/proof/code text in figure/table tags.
            # A native tag is source provenance, not a verified semantic label.
            content_type = 'tei_table_unverified' if element.get('type') == 'table' else 'tei_figure_unverified'
        elif kind == 'note' and element.get('place') == 'foot':
            content_type = 'footnote'
        else:
            continue
        coords = []
        for chunk in element.get('coords', '').split(';'):
            parts = chunk.split(',')
            if len(parts) != 5:
                continue
            page, x, y, width, height = map(float, parts)
            if not page.is_integer() or width <= 0 or height <= 0:
                continue
            coords.append({'page':int(page), 'bbox':[x,y,x+width,y+height]})
        text = ' '.join(''.join(element.itertext()).split())
        variants = [canonical_text(text)]
        if content_type == 'section_heading' and element.get('n'):
            variants.append(canonical_text(element.get('n')+' '+text))
        nodes.append({'node_id':f"{document['paper']['paper_id']}:typed:{len(nodes)+1}",
                      'content_type':content_type, 'text':text, 'variants':variants,
                      'coordinates':coords, 'section_id':section_of(element)})
    return nodes


def annotate_typed_nodes(rows, nodes, geometry):
    supported = {g['page'] for g in geometry if g.get('supported',False)}
    by_page = defaultdict(list)
    for node in nodes:
        for page in {c['page'] for c in node['coordinates']}:
            by_page[page].append(node)
    result = []
    for original in rows:
        row = dict(original)
        row['typed_node_id'] = None
        row['typed_content_type'] = 'unknown'
        if row['page'] not in supported or row.get('content_type') == 'furniture':
            result.append(row)
            continue
        value = canonical_text(row['text'])
        boxes = [row['pdf_bbox']] if row.get('pdf_bbox') else row.get('source_instance_boxes', [])
        matches = []
        if len(value) >= 6 and boxes:
            for node in by_page[row['page']]:
                if not any(value in v for v in node['variants']):
                    continue
                coords = [c['bbox'] for c in node['coordinates'] if c['page'] == row['page']]
                # All unresolved source instances must support the same node.
                if all(spatial_support({'bbox':box,'words':row.get('pdf_words',[]) if len(boxes)==1 else []},coords)>=.7 for box in boxes):
                    matches.append(node)
        if len(matches) == 1:
            node = matches[0]
            row.update(typed_node_id=node['node_id'], typed_content_type=node['content_type'],
                       typed_section_id=node['section_id'])
            # Preserve body attribution; typed section may supplement unknown.
            if row['section_id'] is None and node['section_id']:
                row['section_id'] = node['section_id']
                row['section_attribution_source'] = 'tei_ancestor_of_typed_node'
        result.append(row)
    return result

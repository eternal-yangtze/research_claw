"""Conservative source-preserving packing for offline structure experiments."""
import hashlib


def pack_structured_lines(lines, *, maximum=220, target=180, minimum=100):
    """Prefer proven boundaries, keep unknown text, never cross PDF pages.

    Unverified figure/table containers do not create semantic boundaries.
    Words are used only for size limits; no embedding similarity is claimed.
    """
    result, current = [], None
    for row in lines:
        words = row['text'].split()
        node = row.get('paragraph_id')
        if not node and row.get('typed_content_type') in {'abstract','section_heading','formula','footnote','back_matter_text'}:
            node = row.get('typed_node_id')
        for start in range(0,len(words),maximum):
            fragment = words[start:start+maximum]
            section = row.get('section_id')
            size = len(current['words']) if current else 0
            boundary = bool(current and size>=minimum and
                            ((section and current['last_section'] and section!=current['last_section']) or
                             (node and current['last_node'] and node!=current['last_node'] and size>=target)))
            if current is None or current['page']!=row['page'] or size+len(fragment)>maximum or boundary:
                current={'page':row['page'],'words':[],'sources':[],'sections':[],
                         'last_section':section,'last_node':node}
                result.append(current)
            current['words'].extend(fragment)
            current['sections'].append(section)
            current['sources'].append({'page':row['page'],'line_ordinal':row['line_ordinal'],
                                       'word_start':start,'word_end':start+len(fragment)})
            current['last_section'],current['last_node']=section,node
    output=[]
    for c in result:
        sections=set(c['sections'])
        text=' '.join(c['words'])
        output.append({'text':text,'page':c['page'],
                       'section_id':next(iter(sections)) if len(sections)==1 and None not in sections else None,
                       'sources':c['sources'],'text_hash':hashlib.sha256(text.encode()).hexdigest()})
    assert ' '.join(c['text'] for c in output).split()== ' '.join(r['text'] for r in lines).split()
    return output

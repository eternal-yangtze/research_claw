"""Lossless section-tree packing over frozen PDF layout nodes.

Headings are heuristic, not claims about mathematical dependencies. Markdown
is an inspection view; JSON nodes retain source identity and coordinates.
"""
import hashlib
from researchclaw.knowledge.token_safety import source_windows, token_count


def build_tree(structure, tokenizer, *, maximum=500, target=320):
    root = structure.paper_id + ':root'
    sections = [{**s, 'parent_id': s['parent_id'] or root, 'type': 'section'}
                for s in structure.sections]
    valid = {s['node_id'] for s in sections}
    groups = []
    # Parser emits headings at the end. Keep them as evidence, but do not merge
    # those trailing records into body paragraphs or change source reading order.
    for node in structure.nodes:
        parent = node['parent_id'] if node['parent_id'] in valid else root
        heading = ' / '.join(node['heading_path'])
        while heading and token_count(tokenizer, heading+'\n') >= maximum//2:
            heading = heading[:len(heading)//2]
        previous = groups[-1] if groups else None
        join = (previous and previous['parent_id'] == parent
                and previous['region'] == node['region']
                and previous['kind'] in ('paragraph', 'formula', 'paragraph_group')
                and node['kind'] in ('paragraph', 'formula')
                and token_count(tokenizer, heading+'\n'+previous['text']) < target
                and token_count(tokenizer, heading+'\n'+previous['text']+'\n\n'+node['text']) <= maximum)
        if join:
            offset = len(previous['text'])+2
            previous['text'] += '\n\n'+node['text']
            previous['units'].append(node['node_id'])
            previous['kind'] = 'paragraph_group'
            previous['spans'].extend({**s, 'char_start': s['char_start']+offset,
                                     'char_end': s['char_end']+offset} for s in node['mapped_spans'])
        else:
            groups.append(dict(parent_id=parent, section=heading, text=node['text'],
                               region=node['region'], kind=node['kind'],
                               units=[node['node_id']], spans=[dict(s) for s in node['mapped_spans']]))
    children = []
    for gi, group in enumerate(groups):
        for wi, window in enumerate(source_windows(group['text'], tokenizer,
                                  prefix=group['section']+'\n', maximum=maximum)):
            spans = [{**s, 'overlap_start': max(s['char_start'], window['char_start']),
                      'overlap_end': min(s['char_end'], window['char_end'])}
                     for s in group['spans'] if s['char_end'] > window['char_start']
                     and s['char_start'] < window['char_end']]
            children.append(dict(id=f'{structure.paper_id}:tree:g{gi}:w{wi}',
                paper_id=structure.paper_id, canonical_work=structure.paper_id,
                family=structure.paper_id, section=group['section'], text=window['text'],
                embedding_text=group['section']+'\n'+window['text'],
                parent_chunk_id=group['parent_id'], page=spans[0]['page'],
                pdf_sha256=structure.source_hash, source_type='pdf_fulltext',
                input_tokens=window['tokens'], source_spans=spans, source_units=group['units'],
                region=group['region'], group_id=f'{structure.paper_id}:tree:g{gi}',
                source_char_start=window['char_start'], source_char_end=window['char_end']))
    expected = [n['node_id'] for n in structure.nodes]
    actual = [u for g in groups for u in g['units']]
    assert expected == actual, 'Every source node must occur once, in source order'
    tree = dict(root_id=root, source_hash=structure.source_hash,
                nodes=[dict(node_id=root, parent_id=None, type='paper'), *sections],
                leaves=[dict(id=c['id'], parent_id=c['parent_chunk_id'],
                             source_units=c['source_units']) for c in children],
                audit=dict(source_nodes=len(expected), packed_groups=len(groups),
                           source_text_sha256=hashlib.sha256(''.join(n['text'] for n in structure.nodes).encode()).hexdigest(),
                           all_source_nodes_preserved=True))
    return tree, children


def render_markdown(structure, title):
    """Readable original units, with source IDs; no generated scientific text."""
    lines = ['# '+title, '', '<!-- PDF SHA256: '+structure.source_hash+' -->', '']
    last = None
    for node in structure.nodes:
        if node['kind'] == 'heading':
            continue  # Their text is rendered in heading paths below.
        path = tuple(node['heading_path'])
        if path != last:
            shared = 0
            if last:
                while shared < min(len(last), len(path)) and last[shared] == path[shared]:
                    shared += 1
            lines.extend('#'*min(i+2, 6)+' '+h+'\n' for i,h in enumerate(path) if i >= shared)
            last = path
        lines.extend([f'<!-- {node["node_id"]}; pages {node["page_start"]}-{node["page_end"]} -->', node['text'], ''])
    return '\n'.join(lines)

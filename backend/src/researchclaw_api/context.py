"""Legacy character-budget adapter for callers/tests; Web uses harness.context.

Full conversation is stored as individual PostgreSQL rows. This helper does not
implement the new token/state/memory policy and is kept for compatibility only.
"""
import json


def bounded_history(messages, max_chars=48000):
    if len(json.dumps(messages, ensure_ascii=False)) <= max_chars:
        return list(messages), {'compacted': False, 'archived_messages': len(messages)}
    # Keep complete user turns, so tool replies never become orphan messages.
    starts = [i for i, m in enumerate(messages) if m.get('role') == 'user']
    cut = starts[-1] if starts else len(messages)
    for i in reversed(starts):
        if len(json.dumps(messages[i:], ensure_ascii=False)) > max_chars // 2:
            break
        cut = i
    older = messages[:cut]
    excerpts = [m.get('role','')+': '+str(m.get('content', ''))[:1500] for m in older if m.get('role') in {'user','assistant'} and m.get('content')]
    note = {'role': 'system', 'content': 'Older conversation was compacted. These are partial historical user requests, '
            'and assistant replies, not verified evidence. Follow the latest request; retrieve original sources again. Full history is archived.\n' +
            '\n'.join(excerpts)[-max_chars // 3:]}
    policy = [messages[0]] if messages and messages[0].get('role') == 'system' else []
    result = [*policy, note, *messages[cut:]]
    shortened = 0
    # One completed turn can itself contain a very large tool observation.
    # Bound content without dropping tool-call IDs or their paired replies.
    for index,message in enumerate(result):
        if len(json.dumps(result,ensure_ascii=False)) <= max_chars:
            break
        if message.get('role') not in {'tool','assistant'} or not isinstance(message.get('content'),str):
            continue
        excess=len(json.dumps(result,ensure_ascii=False))-max_chars
        content=message['content']
        if len(content)>256:
            keep=max(0,len(content)-excess-150)
            result[index]={**message,'content':content[:keep]+'\n[Long historical content abbreviated; full message remains in conversation archive. Retrieve source evidence again.]'}
            shortened+=1
    return result, {'compacted': True, 'archived_messages': len(messages),
                    'omitted_messages': cut, 'shortened_observations':shortened,
                    'strategy': 'recent-complete-turns+dialogue-excerpts'}

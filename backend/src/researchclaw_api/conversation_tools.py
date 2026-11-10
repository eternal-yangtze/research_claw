"""Bind session capabilities to the existing ToolRegistry, never client owner IDs."""
from researchclaw.harness.runtime import ToolSpec, ToolRisk


CONVERSATION_TOOLS=('read_session_state','update_session_state','propose_user_memory','recall_user_memory','search_conversation_history','read_conversation_message','read_context_artifact')


def register_conversation_tools(registry,store,session_id,owner,project,agent_name,artifact_root=None):
    def register(name,description,properties,required,handler,read_only):
        registry.register(ToolSpec(name,description,
            {'type':'object','properties':properties,'required':required,'additionalProperties':False},
            handler,risk=ToolRisk.LOW if read_only else ToolRisk.NORMAL,read_only=read_only,
            side_effect=not read_only,allowed_agents=frozenset({agent_name})))
    register('read_session_state','Read the current research goal, decisions and pending actions with version.',{},[],
             lambda:store.state(session_id,owner,project),True)
    register('update_session_state','Update working research state, not long-term profile. On goal change old dependent decisions are cleared. Read current version first.',
             {'expected_version':{'type':'integer'},'patch':{'type':'object'}},['expected_version','patch'],
             lambda expected_version,patch:store.update_state(session_id,owner,project,expected_version,patch),False)
    register('propose_user_memory','Propose a stable user preference, never a paper fact. Not active until the user confirms it in memory controls.',
             {'content':{'type':'string'},'key':{'type':'string'},'kind':{'type':'string'}},['content','key'],
             lambda content,key,kind='preference':{'id':store.save_memory(session_id,owner,project,content=content,key=key,kind=kind,
                 source_message=next((m['id'] for m in reversed(store.message_records(session_id,owner,project)) if m.get('role')=='user'),None),session_only=True), 'status':'proposed'},False)
    register('recall_user_memory','Recall confirmed notes from THIS session only. No cross-session profile or history recall. Not scientific evidence.',
             {'query':{'type':'string'}},['query'],lambda query:store.memories(owner,project,query,source_session=session_id),True)
    def search_history(query,limit=6,scope='session'):
        if scope!='session':
            raise ValueError('Cross-session history recall is disabled in the active agent')
        return store.search_conversation_history(session_id,owner,project,query,limit,'session')
    register('search_conversation_history','Search historical statements in THIS session only, using entities/events or alternate terms. Historical results are not confirmed facts. Other sessions are unavailable; shared papers use paper RAG tools.',
             {'query':{'type':'string'},'limit':{'type':'integer','minimum':1,'maximum':10},
              'scope':{'type':'string','enum':['session']}},['query'],search_history,True)
    def read_message(message_id,offset=0,source_session=None):
        if offset<0:
            raise ValueError('Offset must be nonnegative')
        if source_session not in (None,session_id):
            raise ValueError('Cross-session source reads are disabled in the active agent')
        row=store.read_message(session_id,owner,project,message_id)
        content=str(row.get('content',''))
        return {**row,'content':content[offset:offset+4000],'offset':offset,'has_more':offset+4000<len(content)}
    register('read_conversation_message','Read a source message in THIS session only. Paginated, not a filesystem read. Other sessions cannot be read.',
             {'message_id':{'type':'string'},'offset':{'type':'integer'},'source_session':{'type':'string'}},['message_id'],read_message,True)
    def read_artifact(sha256,offset=0):
        import re,json
        if artifact_root is None or not re.fullmatch('[a-f0-9]{64}',sha256) or offset<0:
            raise ValueError('Invalid artifact request')
        store.session(session_id,owner,project)
        root=artifact_root.resolve()
        path=(root/(sha256+'.json')).resolve()
        if not path.is_relative_to(root) or not path.is_file():
            raise ValueError('Unknown context artifact')
        content=str(json.loads(path.read_text(encoding='utf8')).get('content',''))
        return dict(sha256=sha256,content=content[offset:offset+4000],offset=offset,has_more=offset+4000<len(content))
    register('read_context_artifact','Read a paginated externalized tool observation by SHA256 from THIS run; cannot read arbitrary files.',
             {'sha256':{'type':'string'},'offset':{'type':'integer'}},['sha256'],read_artifact,True)

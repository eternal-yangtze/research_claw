"""Lifecycle regressions use isolated PG schemas, not the user's corpus."""
import json
import uuid
import pytest
from fastapi.testclient import TestClient
from researchclaw_api.main import create_app
from researchclaw_api.store import WebStore
from researchclaw_api.conversation import StateConflict
from researchclaw.harness.context import ContextBuilder, ContextManagedModel, ContextOverflow
from researchclaw.harness.runtime import ModelTurn


@pytest.fixture
def store(tmp_path):
    return WebStore(tmp_path)


def add(store,s,owner='alice',project='default',text='Remember the error budget is 0.01'):
    history=store.session(s,owner,project)
    run=uuid.uuid4().hex
    store.create_run(run,s,owner,project,True,text)
    store.finish(run,s,history+[{'role':'user','content':text},{'role':'assistant','content':'Acknowledged'}])
    return store.message_records(s,owner,project)[-2]['id']


def test_append_only_restart_and_source_ids(store):
    s=store.new_session('alice','default')
    mid=add(store,s)
    restored=WebStore(store.root)
    assert restored.message_records(s,'alice','default')[0]['id']==mid
    assert len(restored.session(s,'alice','default'))==2
    with restored.connect() as db:
        assert db.execute('SELECT messages FROM sessions WHERE id=?',(s,)).fetchone()['messages']=='[]'
    run=uuid.uuid4().hex
    restored.create_run(run,s,'alice','default',True,'new')
    with pytest.raises(StateConflict):
        restored.finish(run,s,[{'role':'user','content':'rewritten'}])


def test_session_notes_do_not_replace_other_session_notes(store):
    a=store.new_session('alice','default')
    b=store.new_session('alice','default')
    first=store.save_memory(a,'alice','default',content='epsilon=0.006',key='epsilon',explicit=True,session_only=True)
    store.save_memory(b,'alice','default',content='epsilon=0.1',key='epsilon',explicit=True,session_only=True)
    proposal=store.save_memory(a,'alice','default',content='epsilon=0.004',key='epsilon',session_only=True)
    store.memory_action(proposal,'alice','default','confirm')
    assert [m['content'] for m in store.memories('alice','default',source_session=a)]==['epsilon=0.004']
    assert [m['content'] for m in store.memories('alice','default',source_session=b)]==['epsilon=0.1']
    with store.connect() as db:
        assert db.execute('SELECT status FROM conversation_memories WHERE id=?',(first,)).fetchone()['status']=='superseded'


def test_active_tools_reject_other_sessions_even_with_legacy_dependencies(store):
    from researchclaw.harness.runtime import ToolRegistry
    from researchclaw_api.conversation_tools import register_conversation_tools
    a=store.new_session('alice','default')
    b=store.new_session('alice','default')
    mid=add(store,a,text='PRIVATE Zephyr device is Jetson Orin NX')
    store.save_memory(a,'alice','default',content='PRIVATE Zephyr device is Jetson Orin NX',key='device',explicit=True)
    # Existing legacy dependencies are not authorization for the active tools.
    with store.connect() as db:
        db.execute('INSERT INTO conversation_dependencies(session,source) VALUES(?,?)',(b,a))
    registry=ToolRegistry()
    register_conversation_tools(registry,store,b,'alice','default','test-agent')
    assert registry.get('recall_user_memory').handler(query='Zephyr')==[]
    assert registry.get('search_conversation_history').handler(query='Zephyr')['matches']==[]
    with pytest.raises(ValueError):
        registry.get('search_conversation_history').handler(query='Zephyr',scope='project')
    with pytest.raises(ValueError):
        registry.get('read_conversation_message').handler(message_id=mid,source_session=a)
    with pytest.raises(KeyError):
        registry.get('read_conversation_message').handler(message_id=mid)


def test_chat_injects_only_current_session_notes(tmp_path):
    import time
    class Inspect:
        def complete(self,messages,tools,max_tokens):
            wire=json.dumps(messages)
            assert 'PRIVATE-OTHER-SESSION' not in wire
            assert 'LOCAL-ONLY-SESSION' in wire
            if 'ROLE: Reviewer' in messages[0]['content']:
                return ModelTurn(json.dumps(dict(verdict='PASS',target='worker',failed_criteria=[],reason='checked')))
            return ModelTurn(json.dumps(dict(mode='direct',answer='Local context checked.',tasks=[])))
    store=WebStore(tmp_path)
    a=store.new_session('alice','default'); b=store.new_session('alice','default')
    store.save_memory(a,'alice','default',content='PRIVATE-OTHER-SESSION',key='private',kind='constraint',explicit=True)
    store.save_memory(b,'alice','default',content='LOCAL-ONLY-SESSION',key='local',kind='constraint',explicit=True,session_only=True)
    with TestClient(create_app(workspace=tmp_path,model_factory=Inspect,password='test',secret='x'*40)) as c:
        c.post('/api/v1/auth/login',json={'user_id':'alice','password':'test'})
        scoped=c.get('/api/v1/memories',params={'session_id':b}).json()
        assert len(scoped)==1 and scoped[0]['content']=='LOCAL-ONLY-SESSION'
        run=c.post('/api/v1/chat',json={'session_id':b,'message':'Recall my local constraints.'}).json()
        for _ in range(300):
            result=c.get('/api/v1/runs/'+run['run_id']).json()
            if result['status']!='running':
                break
            time.sleep(.03)
        assert result['status']=='completed',result


def test_project_history_recall_is_scoped_and_deletion_tracked(store):
    source=store.new_session('alice','default')
    mid=add(store,source,text='Zephyr compression prototype runs on Jetson Orin NX.')
    history=store.session(source,'alice','default')
    run=uuid.uuid4().hex
    store.create_run(run,source,'alice','default',True,'Any candidate compressor?')
    store.finish(run,source,history+[{'role':'user','content':'Any candidate compressor?'},
                                   {'role':'assistant','content':'A candidate compressor is SZ3.'}])
    own=store.search_conversation_history(source,'alice','default','candidate compressor SZ3')
    assert any(m['role']=='assistant' and 'SZ3' in m['content'] for m in own['matches'])
    other=store.new_session('bob','default')
    add(store,other,owner='bob',text='Zephyr private competitor uses SECRET-DEVICE.')
    other_project=store.new_session('alice','private')
    add(store,other_project,project='private',text='Zephyr private project uses SECRET-PROJECT.')
    target=store.new_session('alice','default')
    add(store,target,text='Which device runs Zephyr?')
    with pytest.raises(ValueError):
        store.read_recalled_message(target,'alice','default',mid,source)
    result=store.search_conversation_history(target,'alice','default','Zephyr device',scope='project')
    assert any(m['source_message']==mid for m in result['matches'])
    assert 'SECRET' not in json.dumps(result)
    assert store.read_recalled_message(target,'alice','default',mid,source)['content'].endswith('NX.')
    with pytest.raises(KeyError):
        store.read_recalled_message(target,'alice','default',mid,other)
    assert store.memories('alice','default')==[]  # raw evidence not profile promotion
    preview=store.prepare_delete(source,'alice','default')
    assert target in preview['impact']['branches']
    store.confirm_delete(source,'alice','default',preview['token'])
    assert store.session(target,'alice','default')==[]
    assert not store.search_conversation_history(target,'alice','default','Zephyr',scope='project')['matches']


def test_state_optimistic_lock_and_replanning(store):
    s=store.new_session('alice','default')
    first=store.update_state(s,'alice','default',0,{'goal':'compression','decisions':['use A'],'pending_actions':['run A']})
    assert first['state_version']==1
    with pytest.raises(StateConflict):
        store.update_state(s,'alice','default',0,{'goal':'stale'})
    second=store.update_state(s,'alice','default',1,{'goal':'DTW'})
    assert second['decisions']==second['pending_actions']==[]
    with pytest.raises(KeyError):
        store.state(s,'bob','default')
    with pytest.raises(KeyError):
        store.update_state(s,'alice','other',2,{'goal':'hack'})


def test_memory_proposal_confirmation_conflict_expiry_scope(store):
    s=store.new_session('alice','default')
    mid=add(store,s)
    proposed=store.save_memory(s,'alice','default',content='error budget 0.01',key='error-budget',source_message=mid)
    assert store.memories('alice','default','error')==[]
    store.memory_action(proposed,'alice','default','confirm')
    assert store.memories('alice','default','error')[0]['id']==proposed
    new=store.save_memory(s,'alice','default',content='error budget 0.02',key='error-budget',explicit=True)
    assert [m['id'] for m in store.memories('alice','default','error')]==[new]
    assert store.memories('bob','default','error')==[]
    assert store.memories('alice','other','error')==[]
    store.save_memory(s,'alice','default',content='expired',key='expired',explicit=True,expires=1)
    assert not store.memories('alice','default','expired')
    with pytest.raises(ValueError):
        store.save_memory(s,'alice','default',content='invalid',key='bad',source_message='other')


def test_deleted_legacy_memory_is_not_resurrected_on_restart(store):
    s=store.new_session('alice','default')
    store.edit_session(s,'alice','default',memory='Remember my local constraint')
    memory=store.memories('alice','default')[0]
    store.memory_action(memory['id'],'alice','default','delete')
    restored=WebStore(store.root)
    assert restored.memories('alice','default')==[]
    assert restored.sessions('alice','default')[0]['memory']==''


def test_deleted_memory_cannot_be_injected_from_stale_context(store):
    source=store.new_session('alice','default')
    target=store.new_session('alice','default')
    mid=store.save_memory(source,'alice','default',content='private budget',key='budget',explicit=True)
    store.memory_action(mid,'alice','default','delete')
    with pytest.raises(StateConflict):
        store.save_context_snapshot(target,'alice','default',{'memory_ids':[mid]})


def test_branch_and_recalled_memory_cascade_with_confirmation(store):
    s=store.new_session('alice','default')
    mid=add(store,s)
    m=store.save_memory(s,'alice','default',content='error budget 0.01',key='error',explicit=True,source_message=mid)
    branch=store.branch_session(s,'alice','default','branch')
    cross=store.new_session('alice','default')
    store.recall_for_session(cross,'alice','default','error')
    add(store,cross,text='derived discussion')
    run=uuid.uuid4().hex
    with store.connect() as db:
        db.execute("INSERT INTO runs(id,session,owner,status) VALUES(?,?,?,'completed')",(run,s,'alice'))
    artifact=store.root/'users/alice/projects/default/runs'/run/'context.json'
    artifact.parent.mkdir(parents=True)
    artifact.write_text('private',encoding='utf8')
    job_id=uuid.uuid4().hex
    job_dir=store.root/'users/alice/projects/default/web_jobs'/job_id
    job_dir.mkdir(parents=True)
    (job_dir/'output.log').write_text('private task text',encoding='utf8')
    with store.connect() as db:
        db.execute('INSERT INTO web_jobs(id,owner,project,state) VALUES(?,?,?,?)',
                   (job_id,'alice','default',json.dumps({'session_id':s,'status':'completed'})))
    preview=store.prepare_delete(s,'alice','default')
    assert set(preview['impact']['branches'])=={branch,cross}
    add(store,s,text='changed after preview')
    with pytest.raises(StateConflict):
        store.confirm_delete(s,'alice','default',preview['token'])
    preview=store.prepare_delete(s,'alice','default')
    store.confirm_delete(s,'alice','default',preview['token'])
    with pytest.raises(KeyError):
        store.session(s,'alice','default')
    assert store.memories('alice','default','error')==[]
    assert store.session(branch,'alice','default')==store.session(cross,'alice','default')==[]
    assert not artifact.exists()
    assert not job_dir.exists()
    with store.connect() as db:
        assert not db.execute('SELECT id FROM web_jobs WHERE id=?',(job_id,)).fetchone()


def test_active_run_blocks_delete_and_restart_does_not_replay(store):
    s=store.new_session('alice','default')
    run=uuid.uuid4().hex
    store.create_run(run,s,'alice','default',True,'persist me before model call')
    with pytest.raises(StateConflict):
        store.confirm_delete(s,'alice','default',store.prepare_delete(s,'alice','default')['token'])
    restarted=WebStore(store.root)
    assert restarted.run(run,'alice')['status']=='interrupted'
    assert restarted.session(s,'alice','default')[0]['content']=='persist me before model call'


@pytest.mark.parametrize('turns',[20,50,100])
def test_budget_preserves_state_recent_turns_and_sources(turns):
    records=[]
    for i in range(turns):
        records += [{'id':f'u{i}','role':'user','content':f'iteration {i}: '+('neutral observation '*80)},
                    {'id':f'a{i}','role':'assistant','content':'continue carefully '*60}]
    state={'goal':'bounded compression','decisions':['epsilon=0.01'],'state_version':3}
    messages,stats=ContextBuilder(window=4096,output=512,reserve=256).build(records,'What is the current error bound?',state=state)
    assert stats['input_tokens']<=stats['input_budget']
    assert 'epsilon=0.01' in json.dumps(messages)
    assert stats['omitted_messages']>0
    assert messages[-2]['role']=='user' and messages[-1]['role']=='assistant'
    assert len(records)==turns*2


def test_long_input_not_silently_cut():
    with pytest.raises(ContextOverflow):
        ContextBuilder(window=2048,output=512,reserve=256).build([],'long input '*10000)


def test_tool_context_externalized_with_pair_and_full_artifact(tmp_path):
    class Model:
        def complete(self,messages,tools,max_tokens):
            assert messages[1]['tool_calls'][0]['id']==messages[2]['tool_call_id']
            assert 'artifact' in messages[2]['content']
            return ModelTurn('ok')
    messages=[{'role':'user','content':'read'},
              {'role':'assistant','tool_calls':[{'id':'x','type':'function','function':{'name':'read','arguments':'{}'}}]},
              {'role':'tool','tool_call_id':'x','content':'huge '*20000}]
    wrapper=ContextManagedModel(Model(),window=4096,reserve=256,artifact_root=tmp_path)
    assert wrapper.complete(messages,[],512).text=='ok'
    assert json.loads(next(tmp_path.glob('*.json')).read_text(encoding='utf8'))['content']==messages[2]['content']


def test_accumulated_tool_previews_fit_without_losing_pairs_or_originals(tmp_path):
    class Model:
        def complete(self, messages, tools, max_tokens):
            for i, message in enumerate(messages):
                if message['role'] == 'tool':
                    assert messages[i-1]['tool_calls'][0]['id'] == message['tool_call_id']
            assert messages[0]['content'] == 'Preserve policy and current request'
            return ModelTurn('ok')
    messages = [{'role': 'user', 'content': 'Preserve policy and current request'}]
    for i in range(8):
        messages += [{'role': 'assistant', 'tool_calls': [{'id': str(i), 'type': 'function', 'function': {'name': 'read', 'arguments': '{}'}}]},
            {'role': 'tool', 'tool_call_id': str(i), 'content': json.dumps({'chunk_id': str(i), 'text': '原文证据，不能编造。' * 1500})}]
    wrapper = ContextManagedModel(Model(), window=6000, reserve=512, artifact_root=tmp_path)
    assert wrapper.complete(messages, [], 512).text == 'ok'
    assert wrapper.stats[-1]['input_tokens'] <= wrapper.stats[-1]['input_budget']
    assert wrapper.stats[-1]['archived_preview_receipts']
    assert len(list(tmp_path.glob('*.json'))) == 8


def test_tool_pressure_preserves_user_correction_and_archives_old_assistant(tmp_path):
    packet = {'user': 'Explain the new paper', 'current_task': {'id': 't1'}, 'context': [
        {'role': 'system', 'content': 'Current structured session state (data): offline, maximum absolute error'},
        {'role': 'user', 'content': 'Correct my decision: offline, never streaming; not RMSE'},
        {'role': 'assistant', 'content': 'Old assistant speculation ' * 8000}]}
    class Model:
        def complete(self, messages, tools, max_tokens):
            current = json.loads(messages[0]['content'])
            assert current['user'] == packet['user']
            assert any('never streaming' in e['content'] for e in current['context'])
            assert any('maximum absolute error' in e['content'] for e in current['context'])
            assert current['history_omission']['sha256']
            return ModelTurn('ok')
    wrapper = ContextManagedModel(Model(), window=4096, reserve=256, artifact_root=tmp_path)
    assert wrapper.complete([{'role': 'user', 'content': json.dumps(packet)}], [], 512).text == 'ok'
    original = json.loads(next(tmp_path.glob('*.json')).read_text(encoding='utf8'))
    assert json.loads(original['content'])['context'][-1] == packet['context'][-1]


def test_api_confirmation_and_memory_endpoints(tmp_path):
    class Echo:
        def complete(self,messages,tools,max_tokens):
            return ModelTurn('ok')
    with TestClient(create_app(workspace=tmp_path,model_factory=Echo,password='test',secret='x'*40)) as c:
        c.post('/api/v1/auth/login',json={'user_id':'alice','password':'test'})
        s=c.post('/api/v1/sessions',json={}).json()['id']
        assert c.post('/api/v1/sessions/'+s,json={'delete':True}).status_code==409
        response=c.post('/api/v1/memories',json={'session_id':s,'content':'use English','key':'language','explicit':True})
        assert response.status_code==201
        assert c.get('/api/v1/memories').json()[0]['content']=='use English'
        assert c.post('/api/v1/sessions/'+s+'/state',json={'expected_version':0,'patch':{'goal':'DTW'}}).status_code==200
        preview=c.post('/api/v1/sessions/'+s+'/prepare-delete',json={}).json()
        assert c.post('/api/v1/sessions/'+s,json={'delete':True,'confirmation_token':preview['token']}).status_code==200
        assert c.get('/api/v1/memories').json()==[]


def test_web_reuses_skill_tools_and_persists_agent_state(tmp_path):
    import time
    from researchclaw.harness.runtime import ToolCall
    class ToolModel:
        def complete(self,messages,tools,max_tokens):
            if 'ROLE: Planner' in messages[0]['content']:
                return ModelTurn(json.dumps(dict(mode='tasks',answer='',tasks=[dict(id='t1',goal='Discuss compression and update state',depends_on=[],
                    allowed_tools=['read_session_state','update_session_state','propose_user_memory','load_skill','invoke_skill','retrieve_evidence'],
                    acceptance_criteria=['Update state and retrieve evidence'],requires_evidence=False)])),usage={'total_tokens':10})
            if 'ROLE: Reviewer' in messages[0]['content']:
                return ModelTurn(json.dumps(dict(verdict='PASS',target='worker',failed_criteria=[],reason='honest partial evidence response')),usage={'total_tokens':10})
            names={t['function']['name'] for t in tools}
            assert {'load_skill','invoke_skill','update_session_state','propose_user_memory'}<=names
            assert 'implement_experiment' not in names and 'download_pdf' not in names
            results=[json.loads(m['content']) for m in messages if m['role']=='tool']
            n=len(results)
            if n==0:
                call=ToolCall('state','read_session_state',{})
            elif n==1:
                call=ToolCall('update','update_session_state',{'expected_version':results[-1]['output']['state_version'],'patch':{'goal':'bounded compression'}})
            elif n==2:
                call=ToolCall('proposal','propose_user_memory',{'content':'Prefers bounded compression','key':'research-interest','kind':'profile'})
            elif n==3:
                call=ToolCall('load','load_skill',{'name':'evidence-retrieve'})
            elif n==4:
                assert results[-1]['status']=='success',results[-1]
                call=ToolCall('invoke','invoke_skill',{'name':'evidence-retrieve','sha256':results[-1]['output']['sha256'],'arguments':{'question':'error bounded compression'}})
            else:
                assert results[-1]['status']=='success',results[-1]
                return ModelTurn(json.dumps(dict(status='complete',answer='Evidence checked; memory awaits confirmation.',citations=[],uncertainties=[])),usage={'total_tokens':10})
            return ModelTurn(tool_calls=(call,),usage={'total_tokens':10})
    with TestClient(create_app(workspace=tmp_path,model_factory=ToolModel,password='test',secret='x'*40)) as c:
        c.post('/api/v1/auth/login',json={'user_id':'alice','password':'test'})
        sid=c.post('/api/v1/sessions',json={}).json()['id']
        run=c.post('/api/v1/chat',json={'session_id':sid,'message':'Discuss error bounded compression'}).json()
        for _ in range(300):
            result=c.get('/api/v1/runs/'+run['run_id']).json()
            if result['status']!='running':
                break
            time.sleep(.03)
        assert result['status']=='completed',result
        assert c.get('/api/v1/sessions/'+sid+'/state').json()['goal']=='bounded compression'
        memories=c.get('/api/v1/memories').json()
        assert len(memories)==1 and memories[0]['status']=='proposed'
        history=c.get('/api/v1/sessions/'+sid).json()
        # Role-private transcripts live in the run checkpoint, not the main conversation.
        trace=json.loads((tmp_path/'users/alice/projects/default/runs'/run['run_id']/'agent.json').read_text(encoding='utf8'))
        assert trace['engine']=='langgraph-single-v1'
        assert len(trace['tasks']['t1']['observations'])==5
        project_root=tmp_path/'users/alice/projects/default'
        assert not list((project_root/'artifacts/evidence').glob('*.json'))
        assert list((project_root/'runs'/run['run_id']/'artifacts/evidence').glob('*.json'))
        from researchclaw.application.workflow import ResearchClawApp
        project_db=ResearchClawApp(project_root).store.database
        with project_db.connect(read_only=True) as db:
            assert db.execute('SELECT count(*) FROM checkpoints WHERE thread_id=?',(run['run_id'],)).fetchone()[0]>0
        preview=c.post('/api/v1/sessions/'+sid+'/prepare-delete',json={}).json()
        assert c.post('/api/v1/sessions/'+sid,json={'delete':True,'confirmation_token':preview['token']}).status_code==200
        assert not (project_root/'runs'/run['run_id']).exists()
        with project_db.connect(read_only=True) as db:
            for table in ('agent_checkpoints','checkpoints','checkpoint_blobs','checkpoint_writes'):
                column='run_id' if table=='agent_checkpoints' else 'thread_id'
                assert db.execute('SELECT count(*) FROM '+table+' WHERE '+column+'=?',(run['run_id'],)).fetchone()[0]==0

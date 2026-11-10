"""Role separation, evidence guard, replanning and restart contracts."""
import json
from dataclasses import replace
import pytest

from researchclaw.harness.research_agent import ResearchAgentRuntime, AgentJournal, validate_plan, WORKER
from researchclaw.harness.runtime import ToolSpec, ToolRegistry, ToolExecutor, ToolPolicy, TraceStore, ModelTurn, ToolCall, ToolRisk, AgentSpec, Session


def plan(tools=None, evidence=True):
    return dict(mode='tasks',answer='',tasks=[dict(id='t1',goal='Compare error guarantees',depends_on=[],
        allowed_tools=tools or ['search_knowledge'],acceptance_criteria=['Use evidence'],requires_evidence=evidence)])


def artifact(status='complete',citations=None):
    return dict(status=status,answer='Evidence [c1]: absolute error bound, subject to input assumptions.',citations=['c1'] if citations is None else citations,uncertainties=[])


def review(verdict='PASS',target='worker'):
    return dict(verdict=verdict,target=target,failed_criteria=[] if verdict=='PASS' else ['Need evidence'],reason='Checked evidence')


class Script:
    def __init__(self, turns): self.turns=list(turns); self.roles=[]
    def complete(self,messages,tools,max_tokens):
        role=messages[0]['content']
        self.roles.append(role)
        if 'ROLE: Planner' in role or 'ROLE: Reviewer' in role:
            assert tools==[]
        value=self.turns.pop(0)
        return value if isinstance(value,ModelTurn) else ModelTurn(json.dumps(value),usage={'total_tokens':10})


def setup(tmp_path,model,**kwargs):
    registry=ToolRegistry()
    dispatched=[]
    def search(query):
        dispatched.append(query)
        return [{'chunk_id':'c1','text':'Absolute error bound under bounded input assumptions.'}]
    registry.register(ToolSpec('search_knowledge','search',{'type':'object','properties':{'query':{'type':'string'}},'required':['query'],'additionalProperties':False},search,allowed_agents=frozenset({WORKER})))
    registry.register(ToolSpec('delete_everything','danger',{'type':'object'},lambda: dispatched.append('DELETED'),risk=ToolRisk.HIGH,allowed_agents=frozenset({WORKER})))
    traces=TraceStore(tmp_path/'runs')
    executor=ToolExecutor(registry,ToolPolicy(registry),traces)
    return ResearchAgentRuntime(model,executor,registry,traces,artifact_root=tmp_path,**kwargs),dispatched


def call(name='search_knowledge',args=None):
    return ModelTurn(tool_calls=(ToolCall('call1',name,{'query':'bounded compression'} if args is None else args),))


def test_roles_evidence_and_private_threads(tmp_path):
    model=Script([plan(),call(),artifact(),review()])
    runtime,dispatched=setup(tmp_path,model)
    session=Session('s','research')
    result=runtime.run(session,'Compare bounds','r1')
    assert result['status']=='complete' and dispatched==['bounded compression']
    assert result['model_calls']==4 and result['tool_calls']==1
    assert [m['role'] for m in session.messages]==['user','assistant']
    assert 'ROLE: Reviewer' in model.roles[-1]


@pytest.mark.parametrize('mode', ['direct','clarify'])
def test_conversation_and_clarification_do_not_force_tools(tmp_path,mode):
    runtime,dispatched=setup(tmp_path,Script([dict(mode=mode,answer='Which error metric?',tasks=[]),review()]))
    assert runtime.run(Session('s','x'),'hello')['status']==mode
    assert dispatched==[]


def test_host_overrides_reviewer_false_pass(tmp_path):
    runtime,_=setup(tmp_path,Script([plan(),artifact(citations=['invented']),review(),artifact('partial',[]),review()]))
    result=runtime.run(Session('s','x'),'scientific question')
    assert result['status']=='partial'
    events=json.loads((tmp_path/'agent.json').read_text(encoding='utf8'))['events']
    assert any(e['event']=='reviewed' and e['payload']['review']['reason']=='Host evidence guard overrides PASS' for e in events)


def test_hidden_tool_denied_even_model_fabricates_call(tmp_path):
    runtime,dispatched=setup(tmp_path,Script([plan(evidence=False),call('delete_everything',{}),artifact('partial',[]),review()]))
    result=runtime.run(Session('s','x'),'delete')
    assert dispatched==[] and result['tasks']['p0-t1']['observations'][0]['result']['status']=='denied'


def test_high_risk_flag_itself_requires_host_confirmation(tmp_path):
    runtime,dispatched=setup(tmp_path,Script([]))
    agent=AgentSpec(WORKER,'Worker',('delete_everything',))
    result=runtime.executor.execute('r',Session('s','x'),agent,ToolCall('x','delete_everything',{}))
    assert result.status=='denied' and dispatched==[]


def test_planner_cannot_execute_tools(tmp_path):
    runtime,dispatched=setup(tmp_path,Script([call('delete_everything',{}),dict(mode='clarify',answer='Please specify.',tasks=[]),review()]))
    assert runtime.run(Session('s','x'),'research')['status']=='clarify'
    assert dispatched==[]


@pytest.mark.parametrize('mutation', ['cycle','missing','duplicate','unauthorized','criteria','type'])
def test_invalid_plan_rejected(mutation):
    p=plan()
    t=p['tasks'][0]
    if mutation=='cycle': t['depends_on']=['t1']
    if mutation=='missing': t['depends_on']=['t2']
    if mutation=='duplicate': p['tasks'].append(dict(t))
    if mutation=='unauthorized': t['allowed_tools']=['shell']
    if mutation=='criteria': t['acceptance_criteria']=[]
    if mutation=='type': t['requires_evidence']='true'
    with pytest.raises(ValueError): validate_plan(p,{'search_knowledge':{}})


def test_replan_is_bounded_and_can_change_tools(tmp_path):
    runtime,_=setup(tmp_path,Script([plan(evidence=False),artifact('replan',[]),review('REVISE','planner'),
        dict(mode='clarify',answer='Please specify the new objective.',tasks=[]),review()]))
    assert runtime.run(Session('s','x'),'change idea')['status']=='clarify'


def test_tool_budget_halts_without_extra_dispatch(tmp_path):
    runtime,dispatched=setup(tmp_path,Script([plan(),call()]),max_tool_calls=0)
    assert runtime.run(Session('s','x'),'question')['status']=='partial'
    assert dispatched==[]


def test_checkpoint_idempotent_terminal_and_input_binding(tmp_path):
    from researchclaw.core.postgres import PostgresDatabase
    journal=AgentJournal(PostgresDatabase(tmp_path))
    runtime,_=setup(tmp_path,Script([dict(mode='direct',answer='Hello',tasks=[]),review()]),journal=journal)
    first=runtime.run(Session('s','x'),'hello','stable')
    assert runtime.run(Session('s','x'),'hello','stable',resume=True)==first
    with pytest.raises(ValueError,match='different session/input'):
        runtime.run(Session('other','x'),'hello','stable',resume=True)


def test_restart_never_replays_partial_tool(tmp_path):
    from researchclaw.core.postgres import PostgresDatabase
    journal=AgentJournal(PostgresDatabase(tmp_path))
    class Interrupted(Script):
        def complete(self,messages,tools,max_tokens):
            if not self.turns: raise RuntimeError('provider interrupted')
            return super().complete(messages,tools,max_tokens)
    runtime,dispatched=setup(tmp_path,Interrupted([plan(),call()]),journal=journal)
    with pytest.raises(RuntimeError): runtime.run(Session('s','x'),'question','interrupted')
    runtime.model=Script([artifact(),review()])
    result=runtime.run(Session('s','x'),'question','interrupted',resume=True)
    assert result['status']=='complete' and dispatched==['bounded compression']
    assert result['model_calls']==5  # Interrupted provider admission still counts.


def test_recovery_reuses_accepted_dag_node(tmp_path):
    from researchclaw.core.postgres import PostgresDatabase
    journal=AgentJournal(PostgresDatabase(tmp_path))
    p=plan()
    p['tasks'].append(dict(id='t2',goal='Synthesize',depends_on=['t1'],allowed_tools=[],
        acceptance_criteria=['Cite original upstream evidence'],requires_evidence=True))
    class Interrupted(Script):
        def complete(self,*args):
            if not self.turns: raise RuntimeError('provider interrupted')
            return super().complete(*args)
    runtime,dispatched=setup(tmp_path,Interrupted([p,call(),artifact(),review()]),journal=journal)
    with pytest.raises(RuntimeError): runtime.run(Session('s','x'),'question','dag')
    runtime.model=Script([artifact(),review()])
    result=runtime.run(Session('s','x'),'question','dag',resume=True)
    assert result['status']=='complete' and dispatched==['bounded compression']
    assert len(runtime.model.roles)==2 and 'Planner' not in runtime.model.roles[0]
    saved=json.loads(journal.load('dag')['payload'])
    assert any(e['event']=='task.reused' for e in saved['events'])


def test_recovery_refuses_ambiguous_mutation_and_tampered_artifact(tmp_path):
    from researchclaw.core.postgres import PostgresDatabase
    from researchclaw.harness.research_agent import digest
    journal=AgentJournal(PostgresDatabase(tmp_path))
    s=Session('s','x')
    binding=digest([s.session_id,s.metadata,s.messages,'question'])
    state=dict(recovery_version=1,tasks={},pending_tool=dict(tool='update_session_state'),
        limits=dict(model=24,tools=18,seconds=240))
    journal.save('ambiguous','s',binding,state)
    runtime,dispatched=setup(tmp_path,Script([]),journal=journal)
    with pytest.raises(ValueError,match='Ambiguous'):
        runtime.run(s,'question','ambiguous',resume=True)
    state.pop('pending_tool')
    state['tasks']={'p0-t1':{'artifact':artifact(),'artifact_sha256':'tampered'}}
    journal.save('tampered','s',binding,state)
    with pytest.raises(ValueError,match='hash changed'):
        runtime.run(s,'question','tampered',resume=True)
    assert dispatched==[]


def test_reviewer_cannot_execute_a_tool(tmp_path):
    runtime,dispatched=setup(tmp_path,Script([plan(evidence=False),artifact('partial',[]),call('delete_everything',{}),review()]))
    assert runtime.run(Session('s','x'),'question')['status']=='partial'
    assert dispatched==[]


def test_cancel_stops_before_model_or_tool(tmp_path):
    import threading
    event=threading.Event(); event.set()
    runtime,dispatched=setup(tmp_path,Script([]))
    assert runtime.run(Session('s','x'),'question',cancellation=event)['status']=='cancelled'
    assert dispatched==[]


def test_missing_evidence_cannot_complete_even_reviewer_passes(tmp_path):
    runtime,_=setup(tmp_path,Script([plan(),artifact(citations=[]),review(),artifact(citations=[]),review()]))
    result=runtime.run(Session('s','x'),'question')
    assert result['status']=='partial' and '尚未通过复核' in result['answer']


def test_dependent_task_not_dispatched_after_partial(tmp_path):
    p=plan(evidence=False)
    p['tasks'].append(dict(id='t2',goal='Use upstream conclusions',depends_on=['t1'],allowed_tools=['search_knowledge'],
                           acceptance_criteria=['Accepted upstream evidence'],requires_evidence=True))
    runtime,dispatched=setup(tmp_path,Script([p,artifact('partial',[]),review()]))
    result=runtime.run(Session('s','x'),'question')
    assert result['tasks']['p0-t2']['status']=='blocked' and dispatched==[]


def test_failed_schema_is_observed_and_correctable(tmp_path):
    runtime,dispatched=setup(tmp_path,Script([plan(),call(args={'wrong':'value'}),call(),artifact(),review()]))
    result=runtime.run(Session('s','x'),'question')
    assert result['status']=='complete' and len(dispatched)==1
    assert result['tasks']['p0-t1']['observations'][0]['result']['status']=='error'


def test_worker_last_step_requires_a_deliverable(tmp_path):
    turns=[plan()]+[call(args={'query':f'bounded method {i}'}) for i in range(7)]+[artifact(),review()]
    runtime,dispatched=setup(tmp_path,Script(turns))
    result=runtime.run(Session('s','x'),'question')
    assert result['status']=='complete' and len(dispatched)==7


def test_review_handoff_is_bounded_source_text_not_full_tool_logs():
    from researchclaw.harness.research_agent import review_packet
    observations=[dict(tool='get_chunk',arguments={'chunk_id':'c1'},result=dict(status='success',output=dict(chunk_id='c1',text='source '*10000,vector_json='numeric '*10000)))]
    packet=review_packet(observations,['c1'])
    assert len(json.dumps(packet))<5000
    assert packet['source_excerpts'][0]['incomplete']
    assert 'vector_json' not in json.dumps(packet)


def test_unpublished_library_never_triggers_legacy_embedding(tmp_path,monkeypatch):
    from researchclaw.application.workflow import ResearchClawApp
    from researchclaw.application.paper_library import PaperLibrary
    from researchclaw.core.models import Paper,Chunk
    app=ResearchClawApp(tmp_path)
    app.store.upsert_paper(Paper('p1','paper'))
    app.store.replace_chunks('p1',[Chunk('c1','p1','Methods','source text')])
    def forbidden(*args,**kwargs): raise AssertionError('Online reindex/legacy retrieval attempted')
    monkeypatch.setattr(app.retriever,'search',forbidden)
    library=PaperLibrary(app)
    assert library.search('compression')==[]
    row=library.get_chunk('c1')
    assert row['text']=='source text' and 'vector_json' not in row and 'file_path' not in row


def test_budget_stop_returns_honest_partial_without_new_paid_call(tmp_path):
    from researchclaw.coding.budget import BudgetAdmissionError
    class Limited(Script):
        def complete(self,messages,tools,max_tokens):
            if not self.turns: raise BudgetAdmissionError('token_budget_not_admitted')
            return super().complete(messages,tools,max_tokens)
    runtime,_=setup(tmp_path,Limited([plan(),call()]))
    result=runtime.run(Session('s','x'),'question')
    assert result['status']=='partial' and '尚未完成完整复核' in result['answer'] and '[c1]' in result['answer']


def test_worker_history_compaction_keeps_protocol_pairs_and_source_ids():
    from researchclaw.harness.research_agent import bound_worker_history
    messages=[dict(role='assistant',content='',tool_calls=[dict(id='old')]),
        dict(role='tool',tool_call_id='old',content=json.dumps(dict(status='success',output=[dict(chunk_id='c1',text='text '*3000)]))),
        dict(role='assistant',content='',tool_calls=[dict(id='new')]),
        dict(role='tool',tool_call_id='new',content='newest exact result')]
    bound_worker_history(messages)
    assert messages[-1]['content']=='newest exact result'
    assert messages[1]['tool_call_id']=='old' and 'c1' in messages[1]['content'] and len(messages[1]['content'])<2000


def test_dependent_worker_inherits_verified_sources_without_tool_permission(tmp_path):
    p=plan()
    p['tasks'].append(dict(id='t2',goal='Synthesize the accepted evidence',depends_on=['t1'],allowed_tools=[],
                           acceptance_criteria=['Cite upstream evidence'],requires_evidence=True))
    runtime,dispatched=setup(tmp_path,Script([p,call(),artifact(),review(),artifact(),review()]))
    result=runtime.run(Session('s','x'),'question')
    assert result['status']=='complete' and len(dispatched)==1
    second=result['tasks']['p0-t2']
    assert second['observations']==[] and second['inherited_observations'][0]['tool']=='search_knowledge'
    assert second['dependency_hashes']['t1']==result['tasks']['p0-t1']['artifact_sha256']


def test_local_replan_preserves_accepted_node_and_repairs_only_successor(tmp_path):
    p=plan()
    p['tasks'].append(dict(id='t2',goal='Original synthesis',depends_on=['t1'],allowed_tools=[],
        acceptance_criteria=['Use upstream'],requires_evidence=True))
    revised=json.loads(json.dumps(p))
    revised['tasks'][1]['goal']='Revised synthesis with corrected scope'
    runtime,dispatched=setup(tmp_path,Script([p,call(),artifact(),review(),artifact('replan'),
        review('REVISE','planner'),revised,artifact(),review()]))
    result=runtime.run(Session('s','x'),'question')
    assert result['status']=='complete' and len(dispatched)==1
    assert result['tasks']['p1-t1']['artifact_sha256']==result['tasks']['p0-t1']['artifact_sha256']


def test_recovery_refuses_changed_external_state_or_budget(tmp_path):
    from researchclaw.core.postgres import PostgresDatabase
    journal=AgentJournal(PostgresDatabase(tmp_path))
    scope={'version':0}
    class Interrupted(Script):
        def complete(self,*args):
            if not self.turns: raise RuntimeError('interrupted')
            return super().complete(*args)
    runtime,_=setup(tmp_path,Interrupted([plan(),call()]),journal=journal,recovery_scope=lambda:scope)
    with pytest.raises(RuntimeError): runtime.run(Session('s','x'),'question','r')
    runtime.max_tool_calls+=1
    with pytest.raises(ValueError,match='budget limits'):
        runtime.run(Session('s','x'),'question','r',resume=True)
    runtime.max_tool_calls-=1
    scope['version']=1
    with pytest.raises(ValueError,match='state/history changed'):
        runtime.run(Session('s','x'),'question','r',resume=True)


@pytest.mark.parametrize('mutation',[False,True])
def test_crash_between_effect_and_observation_replays_only_reads(tmp_path,mutation):
    from researchclaw.core.postgres import PostgresDatabase
    journal=AgentJournal(PostgresDatabase(tmp_path))
    name='update_session_state' if mutation else 'search_knowledge'
    args={} if mutation else {'query':'bounded compression'}
    runtime,effects=setup(tmp_path,Script([plan([name],evidence=not mutation),call(name,args)]),journal=journal)
    if mutation:
        runtime.registry.register(ToolSpec(name,'state write',{'type':'object','properties':{},'additionalProperties':False},
            lambda:effects.append('STATE_CHANGED'),allowed_agents=frozenset({WORKER})))
    execute=runtime.executor.execute
    def crash_after_effect(*a,**kw):
        execute(*a,**kw)
        raise SystemExit('Injected process crash after effect, before observation persistence')
    runtime.executor.execute=crash_after_effect
    with pytest.raises(SystemExit): runtime.run(Session('s','x'),'question','crash')
    assert len(effects)==1
    runtime.executor.execute=execute
    runtime.model=Script([call(),artifact(),review()])
    if mutation:
        with pytest.raises(ValueError,match='Ambiguous'):
            runtime.run(Session('s','x'),'question','crash',resume=True)
        assert len(effects)==1
    else:
        result=runtime.run(Session('s','x'),'question','crash',resume=True)
        assert result['status']=='complete' and len(effects)==2


def test_failed_branch_does_not_block_independent_node(tmp_path):
    p=plan(evidence=False)
    p['tasks'].append(dict(id='independent',goal='Ask for clarification',depends_on=[],allowed_tools=[],
        acceptance_criteria=['Ask clearly'],requires_evidence=False))
    runtime,_=setup(tmp_path,Script([p,artifact('blocked',[]),review('BLOCKED','human'),
        artifact('complete',[]),review()]))
    result=runtime.run(Session('s','x'),'question')
    assert result['status']=='partial'
    assert result['tasks']['p0-t1']['status']=='blocked'
    assert result['tasks']['p0-independent']['status']=='accepted'
    checkpoint=json.loads((tmp_path/'agent.json').read_text(encoding='utf8'))
    assert checkpoint['ready_queue']==[] and len(checkpoint['graph']['nodes'])==2

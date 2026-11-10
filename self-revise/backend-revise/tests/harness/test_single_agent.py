"""Actual LangGraph/PostgreSQL execution, task DAG and crash recovery."""
import json
import pytest

from researchclaw.harness.single_agent import LangGraphResearchAgent
from researchclaw.harness.research_agent import AgentJournal
from researchclaw.core.postgres import PostgresDatabase
from researchclaw.harness.runtime import Session,ModelTurn,ToolCall,ToolSpec,ToolRegistry,ToolExecutor,ToolPolicy,TraceStore


class Model:
    def __init__(self,turns): self.turns=list(turns)
    def complete(self,messages,tools,max_tokens):
        value=self.turns.pop(0)
        if isinstance(value,BaseException): raise value
        return value if isinstance(value,ModelTurn) else ModelTurn(json.dumps(value),usage={'total_tokens':10})


def artifact(citations=None,status='complete'):
    return dict(status=status,answer='Evidence-based draft [c1].',citations=['c1'] if citations is None else citations)


def call(name='search_knowledge',args=None):
    return ModelTurn(tool_calls=(ToolCall('c'+name,name,{'query':'bounds'} if args is None else args),))


def setup(tmp_path,turns,**kwargs):
    registry=ToolRegistry(); effects=[]
    registry.register(ToolSpec('search_knowledge','search',{'type':'object','properties':{'query':{'type':'string'}},'required':['query']},
        lambda query:effects.append(query) or [{'chunk_id':'c1','text':'Absolute error under stated assumptions.'}]))
    traces=TraceStore(tmp_path/'runs')
    journal=AgentJournal(PostgresDatabase(tmp_path))
    runtime=LangGraphResearchAgent(Model(turns),ToolExecutor(registry,ToolPolicy(registry),traces),registry,traces,
        journal=journal,artifact_root=tmp_path,**kwargs)
    return runtime,effects


def test_native_graph_saves_steps_in_postgresql_and_reuses_terminal(tmp_path):
    runtime,effects=setup(tmp_path,[call(),artifact()])
    result=runtime.run(Session('s','x'),'question','r')
    assert result['status']=='complete' and effects==['bounds']
    assert (tmp_path/'control-graph.json').is_file()
    with runtime.journal.database.connect(read_only=True) as db:
        assert db.execute("SELECT count(*) FROM checkpoints WHERE thread_id='r'").fetchone()[0]>=5
    assert runtime.run(Session('s','x'),'question','r',resume=True)==result


def test_bibliography_location_is_citable_without_claiming_candidate_fulltext(tmp_path):
    runtime, _ = setup(tmp_path, [call('get_paper_references', {'paper_id': 'p1'}), artifact(citations=['bib_1'])])
    runtime.registry.register(ToolSpec('get_paper_references', 'Read original bibliography',
        {'type': 'object', 'properties': {'paper_id': {'type': 'string'}}, 'required': ['paper_id']},
        lambda paper_id: {'excerpts': [{'chunk_id': 'bib_1', 'paper_id': paper_id, 'page': 17,
            'text': 'Xie et al. Maximum error-bounded piecewise linear representation.',
            'source_type': 'bibliography', 'scientific_findings_verified': False}]}))
    result = runtime.run(Session('s', 'x'), 'Recommend a reference, not its scientific findings', 'r')
    assert result['status'] == 'complete'
    assert result['tasks']['t1']['artifact']['citations'] == ['bib_1']


def test_single_agent_generates_and_executes_dependency_dag(tmp_path):
    tasks=[dict(id='a',goal='Retrieve facts',depends_on=[],allowed_tools=['search_knowledge'],acceptance_criteria=['Read source'],requires_evidence=True),
           dict(id='b',goal='Synthesize',depends_on=['a'],allowed_tools=[],acceptance_criteria=['Use source'],requires_evidence=True)]
    runtime,effects=setup(tmp_path,[call('set_task_graph',{'tasks':tasks}),call(),call('submit_task',artifact()),artifact()])
    result=runtime.run(Session('s','x'),'question','r')
    assert result['status']=='complete' and effects==['bounds']
    state=json.loads(runtime.journal.load('r')['payload'])
    assert state['graph']['edges']==[{'from':'a','to':'b'}]
    assert all(t['status']=='accepted' for t in state['tasks'].values())


def test_single_recovery_preserves_observations_and_calls(tmp_path):
    runtime,effects=setup(tmp_path,[call(),RuntimeError('provider disconnected')])
    with pytest.raises(RuntimeError): runtime.run(Session('s','x'),'question','r')
    runtime.model=Model([artifact()])
    result=runtime.run(Session('s','x'),'question','r',resume=True)
    assert result['status']=='complete' and effects==['bounds'] and result['model_calls']==3


def test_false_citation_is_rejected_and_correctable(tmp_path):
    runtime,_=setup(tmp_path,[call(),artifact(['invented']),artifact()])
    assert runtime.run(Session('s','x'),'question','r')['status']=='complete'


def test_repeated_evidence_reads_reserve_a_tool_free_synthesis_turn(tmp_path):
    runtime, effects = setup(tmp_path, [])
    class RepeatedReader:
        def complete(self, messages, tools, max_tokens):
            if tools:
                return call(args={'query': 'bounds-' + str(len(effects))})
            assert 'synthesis reserve' in messages[-1]['content']
            return ModelTurn(json.dumps(artifact()))
    runtime.model = RepeatedReader()
    result = runtime.run(Session('s', 'x'), 'Explain the available evidence', 'r')
    assert result['status'] == 'complete'
    assert len(effects) == 8


def test_large_original_evidence_keeps_citable_json_and_full_artifact(tmp_path):
    runtime, _ = setup(tmp_path, [])
    original='Maximum absolute reconstruction error is bounded. ' * 1000
    runtime.registry.register(ToolSpec('get_chunk','read source',
        {'type':'object','properties':{'chunk_id':{'type':'string'}},'required':['chunk_id']},
        lambda chunk_id:{'chunk_id':chunk_id,'paper_id':'p1','page':3,'text':original}))
    class EvidenceReader:
        def complete(self,messages,tools,max_tokens):
            if messages[-1]['role']!='tool':
                return call('get_chunk',{'chunk_id':'c1'})
            packet=json.loads(messages[-1]['content'])
            assert packet['output'][0]['chunk_id']=='c1'
            assert packet['output'][0]['page']==3
            assert packet['output'][0]['text']==original[:1200]
            full=json.loads(__import__('pathlib').Path(packet['artifact']).read_text(encoding='utf8'))
            assert json.loads(full['content'])['output']['text']==original
            assert __import__('pathlib').Path(packet['artifact']).stem==packet['sha256']
            return ModelTurn(json.dumps(artifact()))
    runtime.model=EvidenceReader()
    assert runtime.run(Session('s','x'),'Explain the bound','r')['status']=='complete'


def test_model_timeout_is_not_reported_as_token_exhaustion(tmp_path):
    runtime,_=setup(tmp_path,[TimeoutError('provider read timeout')])
    result=runtime.run(Session('s','x'),'question','r')
    assert result['status']=='partial'
    assert '模型服务响应超时' in result['answer']
    assert '累计模型预算不足' not in result['answer']


def test_cycle_is_rejected_without_executing_tools(tmp_path):
    tasks=[dict(id='a',goal='cycle',depends_on=['a'],allowed_tools=[],acceptance_criteria=['x'],requires_evidence=False)]
    runtime,effects=setup(tmp_path,[call('set_task_graph',{'tasks':tasks}),artifact([])])
    result=runtime.run(Session('s','x'),'question','r')
    assert result['status']=='complete' and effects==[]
    assert 'Cycle' in (tmp_path/'agent.json').read_text(encoding='utf8')


def test_budget_stop_and_cancellation_are_explicit(tmp_path):
    runtime,effects=setup(tmp_path,[call()],max_tool_calls=0)
    assert runtime.run(Session('s','x'),'question','r')['status']=='partial'
    assert effects==[]


@pytest.mark.parametrize('mutation',[False,True])
def test_ambiguous_effect_boundary_retries_only_immutable_reads(tmp_path,mutation):
    name='update_session_state' if mutation else 'search_knowledge'
    args={} if mutation else {'query':'bounds'}
    runtime,effects=setup(tmp_path,[call(name,args)])
    if mutation:
        runtime.registry.register(ToolSpec(name,'write state',{'type':'object','properties':{}},lambda:effects.append('CHANGED')))
    execute=runtime.executor.execute
    def crash(*a,**kw):
        execute(*a,**kw)
        raise SystemExit('Crash after effect before checkpoint')
    runtime.executor.execute=crash
    with pytest.raises(SystemExit): runtime.run(Session('s','x'),'question','r')
    assert len(effects)==1
    runtime.executor.execute=execute
    runtime.model=Model([artifact()])
    if mutation:
        with pytest.raises(ValueError,match='Ambiguous'):
            runtime.run(Session('s','x'),'question','r',resume=True)
        assert len(effects)==1
    else:
        assert runtime.run(Session('s','x'),'question','r',resume=True)['status']=='complete'
        assert len(effects)==2


def test_local_replanning_preserves_accepted_contracts(tmp_path):
    tasks=[dict(id='a',goal='Read',depends_on=[],allowed_tools=['search_knowledge'],acceptance_criteria=['Cite'],requires_evidence=True),
           dict(id='b',goal='Draft',depends_on=['a'],allowed_tools=[],acceptance_criteria=['Use evidence'],requires_evidence=True)]
    changed=json.loads(json.dumps(tasks));changed[0]['goal']='Tamper accepted goal'
    runtime,effects=setup(tmp_path,[call('set_task_graph',{'tasks':tasks}),call(),artifact(),
        call('set_task_graph',{'tasks':changed}),artifact()])
    result=runtime.run(Session('s','x'),'question','r')
    assert result['status']=='complete' and effects==['bounds']
    assert result['tasks']['a']['task']['goal']=='Read'
    assert 'Accepted node contracts must remain unchanged' in (tmp_path/'agent.json').read_text(encoding='utf8')


def test_concurrent_first_checkpoint_setup_in_one_project(tmp_path):
    from concurrent.futures import ThreadPoolExecutor
    a,_=setup(tmp_path,[artifact([])])
    b,_=setup(tmp_path,[artifact([])])
    a.artifact_root=tmp_path/'a';a.artifact_root.mkdir()
    b.artifact_root=tmp_path/'b';b.artifact_root.mkdir()
    with ThreadPoolExecutor(max_workers=2) as pool:
        futures=[pool.submit(runtime.run,Session(identity,'x'),'hello',identity)
                 for runtime,identity in ((a,'a'),(b,'b'))]
        assert all(f.result()['status']=='complete' for f in futures)

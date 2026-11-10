"""One conversational agent, LangGraph control plane, model-generated task DAG.

The native graph checkpoints every model/tool/validation/scheduling step. A
PostgreSQL intent journal covers the narrower tool-effect/checkpoint crash gap.
Neither checkpointing nor model self-checking authorizes a side effect.
"""
from __future__ import annotations

import json
import re
import time
import uuid
from typing import TypedDict

from researchclaw.core.artifacts import write_json
from researchclaw.harness.runtime import AgentSpec, ToolCall
from researchclaw.harness.research_agent import (
    COMMON, SAFE_TOOLS, WORKER, RECOVERY_READS, HarnessLimitError,
    digest, validate_plan, paper_evidence, bound_worker_history, review_packet,
)
from researchclaw.harness.context import ContextOverflow
from researchclaw.coding.budget import BudgetAdmissionError


POLICY=COMMON+"""
ROLE: ResearchAgent. You are ONE research agent with a host-owned LangGraph executor.
Decide the next useful action from the current user request, session context and observations.
No mandatory research stages. Answer greetings directly, clarify missing intent, update session
state when requested, search local paper evidence for scientific facts. Preserve corrections,
negation and error metrics; use precise English searches for the English paper library.
Use genuine MCP when requested. Skills require discover/load/invoke with the observed hash.
Downloads/ingestion/experiments require request_action, NEVER claim they already executed.
For a short question the default one-node DAG is sufficient: do NOT plan unnecessarily.
For genuinely dependent goals you MAY call set_task_graph to generate at most three tasks.
Retain accepted contracts verbatim when replanning; alter only unfinished tasks, at most once.
Use submit_task to deliver an intermediate task, or return final JSON for the current task:
{"status":"complete|partial|blocked","answer":"answer in user's language",
 "citations":["exact observed chunk IDs"],"uncertainties":["limits"]}.
Return that final object directly, without prose outside it. Put the exact source IDs
used in the citations array as well as the answer. For surveys and experiment drafts,
prefer a compact memo (about 1200 Chinese characters, or similar length), covering all
requested items; do not generate an implementation manual unless the user asks for one.
Only cite returned paper-content IDs; metadata and search expressions are not scientific proof.
Accepted upstream artifacts and original excerpts are supplied by the host. Do not reread
evidence merely because a new task began. Never fabricate results or claim global novelty.
Once sufficient evidence is present, synthesize it rather than searching indefinitely.
For an initial research-direction survey, do not first inspect counts or update state:
start with 2-3 complementary English search_knowledge calls in one batch (e.g. bounded
piecewise linear approximation, predictive quantization, lossless compression). Choose
queries appropriate to the user's intent; these are examples, not mandatory stages.
Use their returned original excerpts to give a preliminary 2-4 direction map with actual
paper titles, feasibility on the stated hardware, and evidence gaps. A survey is not an
exhaustive literature review: do not read entire papers or prove corpus completeness.
One focused follow-up batch may resolve a material gap; then deliver the useful map.
Avoid ceremonial skill/state/catalog calls unless needed for the current request.
When the user describes a research intention, explain what the retrieved library covers and
what remains unknown; matches do not prove completeness. For ideas distinguish observed overlap,
corpus-relative novelty and experimental hypotheses. Suggest feasible baselines and controls,
never claim experiments ran. Use get_paper_references when asked for specific bibliography leads;
verify names from extracted text and disclose missing references. Give channel-specific search
expressions when evidence is missing. Downloaded PDFs remain staged until another user confirmation
publishes them. Users may interrupt, recalibrate an idea and change papers at any point.
For a request to recommend one bibliography item and search expressions, a visible title in the
original bibliography is enough to identify a LEAD, not its scientific findings. Do not expand
this into repeatedly searching/reading the candidate's full scientific contents unless requested.
Use the bounded bibliography excerpt, label it as unread, and explain any incomplete title.
Cite the returned bib_ chunk_id for the original reference-list location. This proves the
bibliographic entry exists, NOT that the candidate's methods/findings have been read or verified.
For a question about ONLY one paper, locate its paper_id, then constrain follow-up
search_knowledge calls with paper_id. Do not repeatedly search unrelated papers to prove
that the target lacks a guarantee. State 'not established by the inspected evidence' when
appropriate. Once task/method/evaluation excerpts are sufficient, synthesize; do not seek
exhaustive proof of a negative across the whole library.
"""


def schema(name,description,properties,required):
    return {'type':'function','function':{'name':name,'description':description,
        'parameters':{'type':'object','properties':properties,'required':required,'additionalProperties':False}}}


TASK_SCHEMA={'type':'object','properties':{
    'id':{'type':'string'},'goal':{'type':'string'},'depends_on':{'type':'array','items':{'type':'string'}},
    'allowed_tools':{'type':'array','items':{'type':'string'}},
    'acceptance_criteria':{'type':'array','items':{'type':'string'}},'requires_evidence':{'type':'boolean'}},
    'required':['id','goal','depends_on','allowed_tools','acceptance_criteria','requires_evidence'],
    'additionalProperties':False}
ARTIFACT_SCHEMA={'status':{'type':'string','enum':['complete','partial','blocked']},
    'answer':{'type':'string'},'citations':{'type':'array','items':{'type':'string'}},
    'uncertainties':{'type':'array','items':{'type':'string'}}}
CONTROL_TOOLS=[
    schema('set_task_graph','Generate/revise a small task DAG only when dependencies justify it.',
           {'tasks':{'type':'array','items':TASK_SCHEMA,'minItems':1,'maxItems':3}},['tasks']),
    schema('submit_task','Deliver the current task artifact for deterministic host checks.',
           ARTIFACT_SCHEMA,['status','answer','citations'])]


class GraphEnvelope(TypedDict):
    state: dict


class LangGraphResearchAgent:
    def __init__(self,model,executor,registry,traces,*,journal,artifact_root=None,
                 max_model_calls=24,max_tool_calls=18,timeout_seconds=240,recovery_scope=None):
        self.model,self.executor,self.registry,self.traces=model,executor,registry,traces
        self.journal,self.artifact_root=journal,artifact_root
        self.max_model_calls,self.max_tool_calls,self.timeout_seconds=max_model_calls,max_tool_calls,timeout_seconds
        self.recovery_scope=recovery_scope

    def run(self,session,user_message,run_id=None,*,resume=False,cancellation=None):
        from langgraph.graph import StateGraph, START, END
        from langgraph.checkpoint.postgres import PostgresSaver
        run_id=run_id or uuid.uuid4().hex
        binding=digest([session.session_id,session.metadata,session.messages,user_message])
        limits={'model':self.max_model_calls,'tools':self.max_tool_calls,'seconds':self.timeout_seconds}
        old=self.journal.load(run_id)
        saved=json.loads(old['payload']) if old else None
        if old and (old['session_id']!=session.session_id or old['input_hash']!=binding):
            raise ValueError('Run belongs to different session/input')
        if old and not resume:
            raise ValueError('Existing run requires explicit recovery')
        if resume:
            if not saved or saved.get('engine')!='langgraph-single-v1':
                raise ValueError('No compatible single-agent checkpoint')
            if self.recovery_scope and saved.get('scope_digest')!=digest(self.recovery_scope()):
                raise ValueError('Session state/history changed; start a new turn')
            if saved.get('result'):
                session.messages.extend(saved['conversation_delta'])
                return saved['result']
            if saved['limits']!=limits:
                raise ValueError('Recovery cannot change host budget limits')
            if saved.get('pending_tool',{}).get('tool') not in RECOVERY_READS|{None}:
                raise ValueError('Ambiguous non-read tool outcome; inspect effects instead of replaying')
            if saved.get('recovery_count',0)>=2:
                raise ValueError('Recovery attempt limit reached')
            for entry in saved['tasks'].values():
                if entry.get('artifact') and digest(entry['artifact'])!=entry.get('artifact_sha256'):
                    raise ValueError('Checkpoint artifact hash changed')
        agent=AgentSpec(WORKER,POLICY,tuple(t for t in SAFE_TOOLS if self.registry.get(t)),max_tokens=2500)
        catalog={s.name:s for s in self.registry.visible(agent)}
        default=dict(id='t1',goal=user_message,depends_on=[],allowed_tools=list(catalog),
                     acceptance_criteria=['Answer the current request; distinguish sources, assumptions and unexecuted experiments'],requires_evidence=False)
        state=saved or dict(engine='langgraph-single-v1',recovery_version=1,limits=limits,phase='reason',status='executing',
            tasks={'t1':dict(task=default,status='ready',observations=[])},order=['t1'],current='t1',
            observations=[],messages=[{'role':'system','content':POLICY},{'role':'user','content':json.dumps({
                'user':user_message,'context':session.messages,'current_task':default},ensure_ascii=False)}],
            tool_queue=[],model_calls=0,tool_calls=0,usage={},events=[],replans=0,validation_failures=0,
            input_snapshot=dict(session_id=session.session_id,task_id=session.task_id,messages=session.messages.copy(),
                metadata=session.metadata.copy(),user_message=user_message))
        started=time.monotonic()-state.get('elapsed_seconds',0)
        if saved:
            state['recovery_count']=state.get('recovery_count',0)+1

        def save(event,payload=None):
            state['elapsed_seconds']=time.monotonic()-started
            if self.recovery_scope:
                state['scope_digest']=digest(self.recovery_scope())
            nodes=[]
            for identity in state['order']:
                entry=state['tasks'][identity]
                if entry['status'] in {'pending','ready'}:
                    deps=[state['tasks'][d]['status'] for d in entry['task']['depends_on']]
                    entry['status']='ready' if all(s=='accepted' for s in deps) else 'blocked' if any(s in {'blocked','partial'} for s in deps) else 'pending'
                nodes.append(dict(id=identity,status=entry['status'],depends_on=entry['task']['depends_on']))
            state['graph']={'nodes':nodes,'edges':[{'from':d,'to':i} for i in state['order'] for d in state['tasks'][i]['task']['depends_on']]}
            state['ready_queue']=[n['id'] for n in nodes if n['status']=='ready']
            state['plans']=[{'mode':'tasks','tasks':[state['tasks'][i]['task'] for i in state['order']]}]
            state['events'].append(dict(event=event,payload=payload or {}))
            self.traces.emit(run_id,'agent.'+event,session.session_id,payload or {})
            self.journal.save(run_id,session.session_id,binding,state)
            if self.artifact_root:
                write_json(self.artifact_root/'agent.json',state)

        def check_limits():
            if cancellation is not None and cancellation.is_set():
                raise InterruptedError('User cancelled')
            if time.monotonic()-started>=self.timeout_seconds:
                raise HarnessLimitError('Agent time budget exhausted')

        def sources():
            return paper_evidence([o['result'].get('output') for o in state['observations']
                if o['result']['status']=='success' and o['tool'] in {
                    'search_knowledge','get_chunk','get_paper_references','retrieve_evidence','answer_with_evidence','mcp_read_chunk','invoke_skill'}])

        def reason(_):
            check_limits()
            if state['model_calls']>=self.max_model_calls:
                raise HarnessLimitError('Agent model budget exhausted')
            entry=state['tasks'][state['current']]
            entry['status']='running'
            bound_worker_history(state['messages'])
            allowed=entry['task']['allowed_tools']
            tools=[catalog[t].model_schema() for t in allowed]+CONTROL_TOOLS
            # Reserve a synthesis turn before repeated evidence reads consume the
            # entire ledger. This is per DAG node, not a fixed research workflow.
            evidence_reads=sum(o['tool'] in {'search_knowledge','get_chunk','mcp_read_chunk',
                'retrieve_evidence','read_context_artifact'} and o['result']['status']=='success'
                for o in entry['observations'])
            if state['model_calls']>=self.max_model_calls-1 or state['tool_calls']>=self.max_tool_calls or evidence_reads>=8:
                tools=[]
                state['messages'].append({'role':'user','content':'Host synthesis reserve: deliver an answer using the evidence already read. Explicitly distinguish supported facts from unestablished claims; mark partial if essential evidence is missing. No more tools for this task.'})
            state['model_calls']+=1
            save('model.started',{'role':'ResearchAgent','call':state['model_calls']})
            remaining=max(.1,self.timeout_seconds-(time.monotonic()-started))
            turn=(self.model.complete_bounded(state['messages'],tools,2500,timeout=remaining)
                  if hasattr(self.model,'complete_bounded') else self.model.complete(state['messages'],tools,2500))
            check_limits()
            for k,v in turn.usage.items():
                if isinstance(v,int): state['usage'][k]=state['usage'].get(k,0)+v
            state['messages'].append({'role':'assistant','content':turn.text,**({'tool_calls':[
                {'id':c.call_id,'type':'function','function':{'name':c.name,'arguments':json.dumps(c.arguments)}} for c in turn.tool_calls]} if turn.tool_calls else {})})
            if turn.tool_calls:
                state['tool_queue']=[dict(id=c.call_id,tool=c.name,arguments=c.arguments) for c in turn.tool_calls]
                state['phase']='tools'
            else:
                try:
                    candidate=json.loads(re.sub(r'^```(?:json)?\s*|\s*```$','',turn.text.strip()))
                except ValueError:
                    candidate={'status':'complete','answer':turn.text,'citations':[]}
                state['candidate']=candidate
                state['phase']='validate'
            save('model.completed')
            return {'state':state}

        def tools(_):
            check_limits()
            item=state['tool_queue'][0]
            name,args=item['tool'],item['arguments']
            state['tool_calls']+=1
            if state['tool_calls']>self.max_tool_calls:
                raise HarnessLimitError('Agent tool budget exhausted')
            current=state['tasks'][state['current']]
            if name in {'set_task_graph','submit_task'}:
                try:
                    if len(state['tool_queue'])!=1:
                        raise ValueError('Graph control must be the only tool in a batch')
                    if name=='set_task_graph':
                        if state['replans']>=2:
                            raise ValueError('One initial explicit graph and one local replan allowed')
                        ordered=validate_plan({'mode':'tasks','tasks':args['tasks']},catalog)
                        existing={i:e for i,e in state['tasks'].items() if e['status']=='accepted'}
                        contracts={t['id']:t for t in ordered}
                        if any(contracts.get(i)!=e['task'] for i,e in existing.items()):
                            raise ValueError('Accepted node contracts must remain unchanged')
                        state['tasks']={t['id']:existing.get(t['id']) or dict(task=t,status='pending',observations=[]) for t in ordered}
                        state['order']=[t['id'] for t in ordered]
                        state['replans']+=1
                        state['phase']='advance'
                        output={'status':'planned','tasks':state['order']}
                    else:
                        state['candidate']=args
                        state['candidate_call_id']=item['id']
                        state['tool_queue']=[]
                        state['phase']='validate'
                        save('artifact.submitted')
                        return {'state':state}
                    result={'status':'success','output':output}
                except (ValueError,KeyError,TypeError) as error:
                    result={'status':'error','error_message':str(error)}
            else:
                signature=digest([name,args])
                previous=next((o for o in state['observations'] if digest([o['tool'],o['arguments']])==signature),None)
                if previous:
                    result=previous['result']
                else:
                    state['pending_tool']=dict(tool=name,arguments=args)
                    save('tool.dispatching',state['pending_tool'])
                    scoped=AgentSpec(WORKER,POLICY,tuple(current['task']['allowed_tools']),max_tokens=2500)
                    result=self.executor.execute(run_id,session,scoped,ToolCall(item['id'],name,args),
                        deadline_at=started+self.timeout_seconds,cancellation=cancellation).to_dict()
                    observation=dict(tool=name,arguments=args,result=result)
                    state['observations'].append(observation)
                    current['observations'].append(observation)
                    state.pop('pending_tool',None)
            raw=json.dumps(result,ensure_ascii=False,default=str)
            evidence=paper_evidence(result.get('output')) if name in {
                'search_knowledge','get_chunk','mcp_read_chunk','retrieve_evidence','answer_with_evidence'} else {}
            if evidence and len(raw)>9000 and self.artifact_root:
                # Keep valid structured evidence, not a sliced JSON preview. Full
                # parent pages stay in the trace/artifact, outside every model turn.
                artifact_id=digest([name,args,result])
                path=self.artifact_root/'tool-artifacts'/(artifact_id+'.json')
                # Same envelope consumed by read_context_artifact's paginated reader.
                write_json(path,{'role':'tool','content':raw})
                excerpts=[]
                for identity,source in list(evidence.items())[:8]:
                    text=next((source.get(k) for k in ('text','excerpt','quote','snippet') if source.get(k)), '')
                    excerpts.append({**{k:source.get(k) for k in ('paper_id','title','paper_title','page','section')},
                        'chunk_id':identity,'text':text[:1200],'incomplete':len(text)>1200})
                raw=json.dumps({'status':result['status'],'output':excerpts,'artifact':str(path),'sha256':artifact_id,
                    'instruction':'Exact visible excerpts are citable evidence. Read a specific chunk only for missing details; no need to reload already sufficient evidence.'},ensure_ascii=False)
            elif len(raw)>20000:
                raw=json.dumps({'status':result['status'],'incomplete':True,'preview':raw[:16000],
                    'instruction':'Read precise chunks. Omitted text is not evidence.'},ensure_ascii=False)
            state['messages'].append({'role':'tool','tool_call_id':item['id'],'content':raw})
            state['tool_queue'].pop(0)
            if state['phase']!='advance': state['phase']='tools' if state['tool_queue'] else 'reason'
            save('tool.observed',{'tool':name,'status':result['status']})
            return {'state':state}

        def validate(_):
            check_limits()
            candidate=state.pop('candidate')
            entry=state['tasks'][state['current']]
            issues=[]
            if not isinstance(candidate,dict) or not isinstance(candidate.get('answer'),str) or not candidate['answer'].strip():
                issues.append('Nonempty answer required')
                candidate={'answer':'','citations':[],'status':'partial'}
            candidate.setdefault('status','complete')
            candidate.setdefault('citations',[])
            evidence=sources()
            if candidate['status'] not in {'complete','partial','blocked'}:
                issues.append('Status must be complete, partial or blocked')
            if not isinstance(candidate['citations'],list) or any(not isinstance(c,str) or c not in evidence for c in candidate['citations']):
                issues.append('Citations must be successfully observed paper-content IDs')
            if candidate['status']=='complete' and entry['task']['requires_evidence'] and not candidate['citations']:
                issues.append('Evidence-required task lacks citations')
            if candidate['status']=='complete' and evidence and not candidate['citations']:
                issues.append('Paper evidence was read; cite the exact sources used')
            call_id=state.pop('candidate_call_id',None)
            if issues:
                state['validation_failures']+=1
                feedback=json.dumps({'status':'error','issues':issues},ensure_ascii=False)
                state['messages'].append({'role':'tool','tool_call_id':call_id,'content':feedback} if call_id else {'role':'user','content':'Correct your artifact: '+feedback})
                if state['validation_failures']>=2:
                    raise HarnessLimitError('Artifact validation limit reached')
                state['phase']='reason'
            else:
                entry.update(status='accepted' if candidate['status']=='complete' else candidate['status'],
                    artifact=candidate,artifact_sha256=digest(candidate))
                if call_id:
                    state['messages'].append({'role':'tool','tool_call_id':call_id,'content':'{"status":"accepted_by_host_rules"}'})
                state['phase']='advance'
            save('artifact.checked',{'task':state['current'],'issues':issues})
            return {'state':state}

        def advance(_):
            check_limits()
            save('scheduler.updated')
            if state['ready_queue']:
                state['current']=state['ready_queue'][0]
                task=state['tasks'][state['current']]['task']
                state['messages'].append({'role':'user','content':json.dumps({'current_task':task,
                    'accepted_dependencies':{d:{'artifact':state['tasks'][d]['artifact'],
                        'artifact_sha256':state['tasks'][d]['artifact_sha256']} for d in task['depends_on']},
                    'evidence':review_packet(state['observations'],[])},ensure_ascii=False)})
                state['phase']='reason'
            else:
                state['phase']='finalize'
            save('task.scheduled',{'task':state['current']})
            return {'state':state}

        def finalize(_):
            parents={d for e in state['tasks'].values() for d in e['task']['depends_on']}
            leaves=[e for i,e in state['tasks'].items() if i not in parents and e.get('artifact')]
            answers=[e['artifact']['answer'] for e in leaves]
            status='complete' if all(e['status']=='accepted' for e in state['tasks'].values()) else 'partial'
            answer='\n\n'.join(answers) or '前置证据或任务尚未完成，请缩小问题或补充输入。'
            finish(answer,status)
            return {'state':state}

        def finish(answer,status):
            state['status']=status
            state['phase']='done'
            result=dict(run_id=run_id,engine=state['engine'],answer=answer,status=status,tasks=state['tasks'],
                model_calls=state['model_calls'],tool_calls=state['tool_calls'],usage=state['usage'],
                duration_seconds=round(time.monotonic()-started,3))
            state['result']=result
            state['conversation_delta']=[{'role':'user','content':user_message},{'role':'assistant','content':answer}]
            session.messages.extend(state['conversation_delta'])
            save('completed',{'status':status})
            return result

        graph=StateGraph(GraphEnvelope)
        actions={'reason':reason,'tools':tools,'validate':validate,'advance':advance,'finalize':finalize}
        for name,action in actions.items(): graph.add_node(name,action)
        route=lambda value:END if value['state']['phase']=='done' else value['state']['phase']
        targets={name:name for name in actions}|{END:END}
        graph.add_conditional_edges(START,route,targets)
        for name in actions: graph.add_conditional_edges(name,route,targets)
        try:
            save('recovered' if resume else 'started',{'engine':state['engine']})
            if resume:
                save('task.reused',{'tasks':[i for i,e in state['tasks'].items() if e['status']=='accepted']})
            with self.journal.database.connect(autocommit=True) as conn:
                saver=PostgresSaver(conn.raw)
                # PostgreSQL CREATE TABLE IF NOT EXISTS is not race-free for
                # concurrent first users of the same project/type namespace.
                setup_key=self.journal.database.schema+':langgraph-setup'
                # A blocking advisory-lock SELECT can itself hold a snapshot
                # which CREATE INDEX CONCURRENTLY in setup waits for. Poll the
                # nonblocking form so waiting connections retain no snapshot.
                while not conn.execute('SELECT pg_try_advisory_lock(hashtext(?))',(setup_key,)).fetchone()[0]:
                    check_limits()
                    time.sleep(.05)
                try:
                    saver.setup()
                finally:
                    conn.execute('SELECT pg_advisory_unlock(hashtext(?))',(setup_key,))
                compiled=graph.compile(checkpointer=saver)
                if self.artifact_root:
                    write_json(self.artifact_root/'control-graph.json',compiled.get_graph().to_json())
                compiled.invoke({'state':state},{'configurable':{'thread_id':run_id},'recursion_limit':160})
            return state['result']
        except InterruptedError:
            return finish('已取消本轮。已记录的操作不会自动重放；可以修改目标后继续。','cancelled')
        except (BudgetAdmissionError,ContextOverflow,TimeoutError) as error:
            excerpts=[]
            for identity,source in list(sources().items())[:3]:
                text=next((source.get(k) for k in ('text','excerpt','quote','snippet') if source.get(k)), '')
                excerpts.append(f'[{identity}] 未完成综合的原文摘录：{text[:700]}')
            accepted=[e['artifact']['answer'] for e in state['tasks'].values() if e['status']=='accepted']
            state['guard_stop']={'type':type(error).__name__,'reason':str(error)}
            reason=('模型服务响应超时' if isinstance(error,TimeoutError) else
                    '模型上下文容量不足' if isinstance(error,ContextOverflow) else '本轮累计模型预算不足')
            return finish('\n\n'.join(accepted+[reason+'，以下是尚未综合的部分证据，本轮未完成。']+excerpts),'partial')
        except Exception as error:
            state['status']='interrupted'
            save('interrupted',{'error_type':type(error).__name__,'reason':str(error)})
            raise


def research_runtime(*args,engine=None,**kwargs):
    """Default single-agent runtime; old multi-agent is explicit comparison only."""
    from researchclaw.harness.research_agent import ResearchAgentRuntime
    if engine in {None,'langgraph-single-v1'}:
        return LangGraphResearchAgent(*args,**kwargs)
    if engine=='multi':
        return ResearchAgentRuntime(*args,**kwargs)
    raise ValueError('Unknown research runtime engine')

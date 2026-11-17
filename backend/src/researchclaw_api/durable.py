"""PostgreSQL outbox and fenced Redis Streams delivery for conversational runs.

Redis carries identifiers only. PostgreSQL owns admission, leases and events.
Expired running jobs become interrupted: ambiguous paid/effect calls are never
automatically repeated. Queued jobs survive Redis loss through outbox repair.
"""
from contextlib import contextmanager
import hashlib
import json
import logging
import threading
import time
import uuid

SCHEMA='''
CREATE TABLE durable_tasks (
 run_id TEXT PRIMARY KEY REFERENCES runs(id) ON DELETE CASCADE,
 owner TEXT NOT NULL, payload TEXT NOT NULL, state TEXT NOT NULL DEFAULT 'queued',
 token TEXT, lease_until DOUBLE PRECISION NOT NULL DEFAULT 0,
 cancel_requested BOOLEAN NOT NULL DEFAULT false,
 deliveries INTEGER NOT NULL DEFAULT 0, published DOUBLE PRECISION NOT NULL DEFAULT 0,
 created DOUBLE PRECISION NOT NULL, updated DOUBLE PRECISION NOT NULL);
CREATE INDEX durable_task_pending ON durable_tasks(state,updated);
CREATE TABLE durable_events (
 seq BIGINT GENERATED ALWAYS AS IDENTITY PRIMARY KEY,
 run_id TEXT NOT NULL REFERENCES runs(id) ON DELETE CASCADE,
 event TEXT NOT NULL, payload TEXT NOT NULL, created DOUBLE PRECISION NOT NULL);
CREATE INDEX durable_event_run ON durable_events(run_id,seq);
CREATE TABLE durable_requests (
 owner TEXT NOT NULL, request_key TEXT NOT NULL, fingerprint TEXT NOT NULL,
 run_id TEXT NOT NULL REFERENCES runs(id) ON DELETE CASCADE,
 PRIMARY KEY(owner,request_key));
'''


class LeaseLost(RuntimeError):
    pass


class QueueFull(ValueError):
    pass


class DurableQueue:
    def __init__(self,store,url,*,workers=4,capacity=256,per_owner=16,lease_seconds=60):
        import redis
        if not 1<=workers<=64 or capacity<workers or not 1<=per_owner<=capacity or lease_seconds<3:
            raise ValueError('Invalid durable worker/admission configuration')
        self.store=store;self.workers=workers;self.capacity=capacity;self.per_owner=per_owner;self.lease_seconds=lease_seconds
        self.redis=redis.Redis.from_url(url,decode_responses=True,socket_timeout=3,socket_connect_timeout=3)
        self.stream='researchclaw:'+store.database.schema+':runs';self.group='workers-v1'
        self.stop=threading.Event();self.threads=[];self.local=threading.local()
        store.database.migrate('012-durable-harness',SCHEMA)

    @staticmethod
    def event(db,run,event,payload=None):
        db.execute('INSERT INTO durable_events(run_id,event,payload,created) VALUES(?,?,?,?)',
                   (run,event,json.dumps(payload or {},ensure_ascii=False),time.time()))

    def admit(self,db,run,owner,payload):
        # The same database transaction also inserts the run and user input.
        db.execute('SELECT pg_advisory_xact_lock(hashtext(?))',(self.store.database.schema+':admission',))
        counts=db.execute("SELECT count(*) AS total,count(*) FILTER(WHERE owner=?) AS owned FROM durable_tasks WHERE state IN ('queued','running')",(owner,)).fetchone()
        if counts['total']>=self.capacity or counts['owned']>=self.per_owner:raise QueueFull('Queue capacity reached; retry later')
        now=time.time()
        db.execute('INSERT INTO durable_tasks(run_id,owner,payload,created,updated) VALUES(?,?,?,?,?)',(run,owner,json.dumps(payload),now,now))
        self.event(db,run,'run.queued',{'queue_version':1})

    def publish(self,run):
        self.redis.xadd(self.stream,{'run_id':run})
        with self.store.connect() as db:db.execute('UPDATE durable_tasks SET published=? WHERE run_id=?',(time.time(),run))

    def start(self,execute):
        self.redis.ping()
        self.ensure_group()
        self.execute=execute
        for i in range(self.workers):
            thread=threading.Thread(target=self._consume,args=(uuid.uuid4().hex,),name='rc-stream-'+str(i),daemon=True)
            thread.start();self.threads.append(thread)
        thread=threading.Thread(target=self._maintain,name='rc-outbox',daemon=True);thread.start();self.threads.append(thread)

    def ensure_group(self):
        try:self.redis.xgroup_create(self.stream,self.group,id='0',mkstream=True)
        except Exception as error:
            if 'BUSYGROUP' not in str(error):raise

    def close(self):
        self.stop.set()
        for thread in self.threads:thread.join(timeout=4)
        # Active calls retain their bounded provider timeout. Process loss is
        # handled by lease expiry, without resetting another instance's jobs.

    def repair(self):
        now=time.time()
        with self.store.connect() as db:
            lost=db.execute("SELECT run_id FROM durable_tasks WHERE state='running' AND lease_until<? FOR UPDATE SKIP LOCKED",(now,)).fetchall()
            for row in lost:
                run=row['run_id']
                db.execute("UPDATE durable_tasks SET state='interrupted',token=NULL,updated=? WHERE run_id=?",(now,run))
                db.execute("UPDATE runs SET status='interrupted',error='Worker lease expired; explicit recovery required' WHERE id=? AND status='running'",(run,))
                self.event(db,run,'run.interrupted',{'reason':'lease_expired','automatic_replay':False})
            pending=db.execute("SELECT run_id FROM durable_tasks WHERE state='queued' AND published<? ORDER BY created LIMIT 256",(now-5,)).fetchall()
        self.ensure_group()
        for row in pending:self.publish(row['run_id'])
        # Redis PEL is a delivery aid, never an execution lock. Clear abandoned
        # delivery records; queued jobs remain recoverable from PostgreSQL.
        abandoned=self.redis.xautoclaim(self.stream,self.group,'outbox-repair',int(self.lease_seconds*2000),count=100)
        for message_id,_ in abandoned[1]:
            self.redis.xack(self.stream,self.group,message_id);self.redis.xdel(self.stream,message_id)

    def _maintain(self):
        while not self.stop.is_set():
            try:self.repair()
            except Exception as error:logging.warning('Durable outbox unavailable: %s',type(error).__name__)
            self.stop.wait(1)

    def claim(self,run):
        with self.store.connect() as db:
            row=db.execute('SELECT * FROM durable_tasks WHERE run_id=? FOR UPDATE',(run,)).fetchone()
            if not row or row['state']!='queued':return None
            if row['cancel_requested']:
                db.execute("UPDATE durable_tasks SET state='cancelled',updated=? WHERE run_id=?",(time.time(),run))
                db.execute("UPDATE runs SET status='cancelled' WHERE id=?",(run,));self.event(db,run,'run.cancelled');return None
            token=uuid.uuid4().hex
            db.execute("UPDATE durable_tasks SET state='running',token=?,lease_until=?,deliveries=deliveries+1,updated=? WHERE run_id=?",(token,time.time()+self.lease_seconds,time.time(),run))
            self.event(db,run,'run.started',{'delivery':row['deliveries']+1})
            return token,json.loads(row['payload'])

    @contextmanager
    def guard(self,run,token):
        with self.store.connect() as db:
            row=db.execute('SELECT * FROM durable_tasks WHERE run_id=? FOR UPDATE',(run,)).fetchone()
            if not row or row['state']!='running' or row['token']!=token or row['lease_until']<time.time():raise LeaseLost('Run lease lost; stale output rejected')
            yield db

    def assert_active(self,run,token):
        with self.guard(run,token):pass

    def heartbeat(self,run,token):
        with self.guard(run,token) as db:
            db.execute('UPDATE durable_tasks SET lease_until=?,updated=? WHERE run_id=?',(time.time()+self.lease_seconds,time.time(),run))
            return db.execute('SELECT cancel_requested FROM durable_tasks WHERE run_id=?',(run,)).fetchone()['cancel_requested']

    def _consume(self,consumer):
        while not self.stop.is_set():
            try:
                batches=self.redis.xreadgroup(self.group,consumer,{self.stream:'>'},count=1,block=1000)
                for _,entries in batches:
                    for message_id,fields in entries:
                        self._deliver(fields['run_id'])
                        self.redis.xack(self.stream,self.group,message_id)
                        self.redis.xdel(self.stream,message_id)
            except Exception as error:
                logging.warning('Durable worker observation: %s',type(error).__name__);self.stop.wait(.5)

    def _deliver(self,run):
        claimed=self.claim(run)
        if not claimed:return
        token,payload=claimed;cancel=threading.Event();done=threading.Event()
        def beat():
            while not done.wait(self.lease_seconds/4):
                try:
                    if self.heartbeat(run,token):cancel.set()
                except Exception:
                    cancel.set();return
        thread=threading.Thread(target=beat,daemon=True);thread.start()
        try:
            self.local.fence=(run,token)
            self.execute(run,payload,cancel)
            with self.guard(run,token) as db:
                status=db.execute('SELECT status FROM runs WHERE id=?',(run,)).fetchone()['status']
                if status=='running':raise RuntimeError('Executor returned without terminal settlement')
                db.execute('UPDATE durable_tasks SET state=?,token=NULL,updated=? WHERE run_id=?',(status,time.time(),run))
                self.event(db,run,'run.'+status)
        except LeaseLost:pass
        except Exception as error:
            try:
                with self.guard(run,token) as db:
                    db.execute("UPDATE runs SET status='interrupted',error=? WHERE id=?",('Execution interrupted: '+type(error).__name__,run))
                    db.execute("UPDATE durable_tasks SET state='interrupted',token=NULL,updated=? WHERE run_id=?",(time.time(),run))
                    self.event(db,run,'run.interrupted',{'error_type':type(error).__name__})
            except LeaseLost:pass
        finally:
            self.local.fence=None;done.set();thread.join(timeout=2)

    def cancel(self,run):
        with self.store.connect() as db:
            db.execute('UPDATE durable_tasks SET cancel_requested=true WHERE run_id=?',(run,))
            self.event(db,run,'run.cancel_requested')

    def requeue(self,db,run):
        row=db.execute('SELECT payload FROM durable_tasks WHERE run_id=? FOR UPDATE',(run,)).fetchone()
        if not row:raise ValueError('No durable task for this run')
        payload=json.loads(row['payload']);payload['resume']=True
        db.execute("UPDATE durable_tasks SET state='queued',payload=?,token=NULL,cancel_requested=false,published=0,updated=? WHERE run_id=?",(json.dumps(payload),time.time(),run))
        self.event(db,run,'run.recovery_queued')

    def events(self,run,after=0):
        with self.store.connect() as db:
            return [dict(seq=r['seq'],event=r['event'],payload=json.loads(r['payload']),created=r['created']) for r in db.execute('SELECT * FROM durable_events WHERE run_id=? AND seq>? ORDER BY seq LIMIT 256',(run,after))]


class FencedJournal:
    """Persist runtime checkpoints and queue audit events under one lease lock."""
    def __init__(self,inner,queue,fence):self.inner=inner;self.queue=queue;self.fence=fence;self.database=inner.database
    def load(self,*args):return self.inner.load(*args)
    def save(self,*args):
        with self.queue.guard(*self.fence) as db:
            self.inner.save(*args)
            state=args[-1];event=state.get('events',[{}])[-1]
            self.queue.event(db,self.fence[0],'checkpoint.saved',{'checkpoint_sha256':hashlib.sha256(json.dumps(state,sort_keys=True,ensure_ascii=False).encode()).hexdigest(),
                'phase':state.get('phase'), 'status':state.get('status'),'runtime_event':event.get('event')})

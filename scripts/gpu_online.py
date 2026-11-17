"""Manage this project's private GPU service; never print credentials."""
import argparse
import json
from pathlib import Path
import shlex
import sys

ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT/'src'))
from researchclaw.knowledge.gpu_client import GPUClient, connect_gpu, GPUServiceError

REMOTE_CONTROL = r'''
import fcntl,json,os,signal,subprocess,sys
from pathlib import Path
root=Path(sys.argv[1]); action=sys.argv[2]; port=int(sys.argv[3])
state=root/'self-revisev0.5/workspace/gpu-online';state.mkdir(parents=True,exist_ok=True)
guard=(state/'control.lock').open('a');fcntl.flock(guard,fcntl.LOCK_EX)
try:
    pid=int((state/'pid').read_text())
    cmd=Path('/proc/'+str(pid)+'/cmdline').read_bytes().split(b'\0')
    live=any(v.endswith(b'/serve_gpu_models.py') for v in cmd) and Path('/proc/'+str(pid)+'/cwd').resolve()==root.resolve()
except (OSError,ValueError):live=False;pid=None
if action=='start':
    if live:print(json.dumps({'state':'already_running_or_warming','pid':pid}))
    else:
        logs=root/'self-revisev0.5/artifacts/online-gpu';logs.mkdir(parents=True,exist_ok=True)
        env=dict(os.environ,PYTHONDONTWRITEBYTECODE='1',PYTHONUNBUFFERED='1')
        with (logs/'service.log').open('ab') as log:
            proc=subprocess.Popen([str(root/'.venv/bin/python'),'-B',str(root/'scripts/serve_gpu_models.py'),'--port',str(port)],cwd=root,env=env,stdout=log,stderr=log,stdin=subprocess.DEVNULL,start_new_session=True)
        (state/'pid').write_text(str(proc.pid))
        print(json.dumps({'state':'starting','pid':proc.pid}))
elif action=='stop':
    if live:os.kill(pid,signal.SIGTERM)
    print(json.dumps({'state':'stop_requested' if live else 'not_running','pid':pid}))
elif action=='process':print(json.dumps({'state':'running_or_warming' if live else 'not_running','pid':pid}))
elif action=='logs':
    p=root/'self-revisev0.5/artifacts/online-gpu/service.log'
    print(json.dumps({'log':p.read_bytes()[-10000:].decode('utf8',errors='replace') if p.exists() else ''}))
'''


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('action',choices=['start','status','stop','logs'])
    parser.add_argument('--report',type=Path)
    args=parser.parse_args()
    if args.report and not args.report.resolve().is_relative_to((ROOT/'self-revisev0.5').resolve()):
        parser.error('Reports must stay inside self-revisev0.5')
    client=GPUClient()
    try:
        if args.action=='status':
            result=client.request('health')
        else:
            ssh=connect_gpu()
            try:
                with ssh.open_sftp() as sftp:root=sftp.normalize('.')+'/ResearchClaw'
                command=' '.join(shlex.quote(x) for x in ['python3','-c',REMOTE_CONTROL,root,args.action,str(client.port)])
                _,out,err=ssh.exec_command(command,timeout=30)
                raw=out.read();code=out.channel.recv_exit_status()
                if code:raise GPUServiceError('Remote service control failed ('+str(code)+')')
                result=json.loads(raw)
            finally:ssh.close()
        if args.report:
            args.report.parent.mkdir(parents=True,exist_ok=True)
            args.report.write_text(json.dumps(result,ensure_ascii=False,indent=2),encoding='utf8')
        print(json.dumps(result,ensure_ascii=True,indent=2))
    except GPUServiceError as exc:
        print(str(exc),file=sys.stderr);return 1
    finally:client.close()
    return 0


if __name__=='__main__':raise SystemExit(main())

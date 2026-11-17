"""Back up and restore this project's GPU runtime without migrating credentials."""
import argparse
import hashlib
import json
from pathlib import Path
import shlex
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / 'src'))
from researchclaw.knowledge.gpu_client import connect_gpu

OUT = ROOT / 'self-revisev0.5/artifacts/gpu-migration-20261115'


def execute(client, code, timeout=120):
    stdin, stdout, stderr = client.exec_command('python3 -c ' + shlex.quote(code), timeout=timeout)
    stdin.close()
    raw = stdout.read().decode('utf8')
    error = stderr.read().decode('utf8')
    if stdout.channel.recv_exit_status():
        raise RuntimeError(error[-3000:])
    return json.loads(raw)


INSPECT = r'''
import json,os,subprocess
from pathlib import Path
h=Path.home()
paths=[h/'ResearchClaw',h/'.cache/huggingface',h/'work']
print(json.dumps({'paths':[{ 'path':str(p),'real':str(p.resolve()),'exists':p.exists(),
'size':subprocess.getoutput('timeout 15 du -sh '+str(p)) if p.exists() else None} for p in paths],
'project':sorted(p.name for p in (h/'ResearchClaw').iterdir()) if (h/'ResearchClaw').exists() else [],
'work':sorted(p.name for p in (h/'work').iterdir()),'python':os.readlink('/proc/self/exe')}))
'''


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('action', choices=['inspect', 'backup', 'restore', 'switch', 'progress', 'verify', 'smoke', 'start-new', 'stop-old', 'cloud-copy', 'evidence', 'cleanup', 'audit'])
    args = parser.parse_args()
    OUT.mkdir(parents=True, exist_ok=True)
    if args.action == 'audit':
        import tarfile
        from dotenv import dotenv_values
        load = lambda name: json.loads((OUT / name).read_text(encoding='utf8'))
        backup, restore, smoke, active = [load(n) for n in ['backup.json', 'restore.json', 'smoke.json', 'active-status.json']]
        assert backup['sha256'] == restore['archive_sha256'] == load('cloud-copy.json')['manifest']['sha256']
        with (OUT / 'evidence.tar.gz').open('rb') as handle:
            assert hashlib.file_digest(handle, 'sha256').hexdigest() == backup['local_evidence']['sha256']
        with tarfile.open(OUT / 'evidence.tar.gz') as archive:
            names = archive.getnames()
            assert all(n.startswith('ResearchClaw/') and '..' not in Path(n).parts for n in names)
            assert not any(n.endswith('/gpu-online/token') or '/.env' in n for n in names)
        assert smoke['passed'] and active['status'] == 'ready' and active['pid'] == smoke['health']['pid']
        config = dotenv_values(ROOT / '.env.gpu.local', interpolate=False)
        assert config['GPU_SSH_PORT'] == '14960'
        result = {'status': 'complete', 'active_host': config['GPU_SSH_HOST'], 'active_port': 14960,
            'full_runtime_bytes': backup['bytes'], 'full_runtime_sha256': backup['sha256'],
            'local_evidence_bytes': backup['local_evidence']['bytes'], 'local_evidence_members': len(names),
            'cuda_smoke_passed': True, 'active_pid': active['pid'], 'old_service': load('stop-old.json'),
            'cleanup': load('cleanup.json'), 'full_runtime_local_copy': False,
            'scope': 'runtime backup on old/new instances; separately verified source/evidence archive local; no production PG/Qdrant migration'}
        (OUT / 'audit.json').write_text(json.dumps(result, indent=2), encoding='utf8')
        files = [p for p in OUT.iterdir() if p.is_file() and p.name != 'evidence-manifest.json']
        files.append(Path(__file__))
        manifest = {}
        for path in files:
            with path.open('rb') as handle:
                manifest[str(path.relative_to(ROOT))] = hashlib.file_digest(handle, 'sha256').hexdigest()
        (OUT / 'evidence-manifest.json').write_text(json.dumps(manifest, indent=2), encoding='utf8')
        print(json.dumps(result, indent=2))
        return
    if args.action == 'smoke':
        from researchclaw.knowledge.gpu_client import GPUClient, RemoteEmbedding, RemoteReranker, BGE_REVISION
        from researchclaw.knowledge.embedding_contract import validate_vectors
        gpu = GPUClient(config_path=ROOT / '.env.gpu.next.local')
        try:
            health = gpu.request('health')
            assert health['status'] == 'ready' and health['device'] == 'cuda:0'
            vectors = RemoteEmbedding('BAAI/bge-small-en-v1.5', revision=BGE_REVISION, client=gpu).encode(['The error is bounded by epsilon.', 'The error is bounded by epsilon.'])
            validate_vectors(vectors, expected=2)
            cosine = sum(x * y for x, y in zip(*vectors))
            assert cosine > .999
            scores = RemoteReranker(client=gpu).score('What is the capital of France?', ['Paris is the capital of France.', 'Bananas are yellow fruits.'])
            assert scores[0] > scores[1]
            qwen_diagnostics = dict(gpu.last_diagnostics)
            minilm = RemoteReranker(family='minilm', client=gpu).score('What is the capital of France?', ['Paris is the capital of France.', 'Bananas are yellow fruits.'])
            assert minilm[0] > minilm[1]
            result = {'health': health, 'embedding_count': len(vectors), 'dimension': len(vectors[0]),
                'duplicate_cosine': cosine, 'qwen_scores': scores, 'qwen_diagnostics': qwen_diagnostics,
                'minilm_scores': minilm, 'minilm_diagnostics': gpu.last_diagnostics, 'passed': True}
            (OUT / 'smoke.json').write_text(json.dumps(result, indent=2), encoding='utf8')
            print(json.dumps(result, indent=2))
        finally:
            gpu.close()
        return
    if args.action == 'switch':
        import shutil
        old = ROOT / '.env.gpu.previous.local'
        if not old.exists():
            shutil.copy2(ROOT / '.env.gpu.local', old)
        shutil.copy2(ROOT / '.env.gpu.next.local', ROOT / '.env.gpu.local')
        print('Active GPU connection switched; previous private configuration retained.')
        return
    config = ROOT / ('.env.gpu.next.local' if args.action in {'restore', 'progress', 'verify', 'start-new', 'cloud-copy', 'cleanup'} else
                     '.env.gpu.previous.local' if args.action == 'stop-old' else '.env.gpu.local')
    client = connect_gpu(config)
    try:
        if args.action == 'cleanup':
            assert json.loads((OUT / 'smoke.json').read_text())['passed']
            result = execute(client, r'''
import json
from pathlib import Path
p=Path('/home/featurize/work/ResearchClaw-gpu-backup-20261115/runtime.tar')
if p.exists():
 assert p.is_file() and p.stat().st_size==15284101120,'Unexpected target; refusing deletion'
 p.unlink()
print(json.dumps({'removed_failed_shared_archive':True,'valid_archives_retained':True}))
''')
            partial = OUT / 'runtime.tar.gz.partial'
            assert partial.resolve().is_relative_to(OUT.resolve())
            if partial.exists():
                result['removed_local_partial_bytes'] = partial.stat().st_size
                partial.unlink()
        elif args.action == 'cloud-copy':
            from dotenv import dotenv_values
            credentials = dotenv_values(ROOT / '.env.gpu.local', interpolate=False)
            source = connect_gpu(ROOT / '.env.gpu.local')
            try:
                key = source.get_transport().get_remote_server_key()
                payload = {'host': credentials['GPU_SSH_HOST'], 'port': int(credentials['GPU_SSH_PORT']),
                    'username': credentials['GPU_SSH_USER'], 'password': credentials['GPU_SSH_PASSWORD'],
                    'key_type': key.get_name(), 'key': key.get_base64()}
                payload['manifest'] = execute(source, "import json;from pathlib import Path;print((Path.home()/'gpu-migration-20261115/manifest.json').read_text())")
            finally:
                source.close()
            code = r'''
import json,sys,hashlib,os,subprocess,tempfile
from pathlib import Path
cfg=json.load(sys.stdin)
dest=Path.home()/'runtime-migration-20261115.tar.gz'
manifest=cfg['manifest']
with tempfile.TemporaryDirectory(prefix='researchclaw-migration-') as temp:
 temp=Path(temp);ask=temp/'askpass';known=temp/'known_hosts'
 ask.write_text('#!/bin/sh\nprintf \'%s\\n\' "$RESEARCHCLAW_MIGRATION_PASSWORD"\n');ask.chmod(0o700)
 known.write_text('[%s]:%s %s %s\n'%(cfg['host'],cfg['port'],cfg['key_type'],cfg['key']))
 env=dict(os.environ,SSH_ASKPASS=str(ask),SSH_ASKPASS_REQUIRE='force',DISPLAY=':0',RESEARCHCLAW_MIGRATION_PASSWORD=cfg['password'])
 cmd=['scp','-P',str(cfg['port']),'-o','UserKnownHostsFile='+str(known),'-o','StrictHostKeyChecking=yes','-o','ConnectTimeout=20',
 cfg['username']+'@'+cfg['host']+':'+manifest['archive'],str(dest)]
 p=subprocess.run(cmd,env=env,stdin=subprocess.DEVNULL,capture_output=True,text=True,timeout=1800)
 if p.returncode:raise RuntimeError('Pinned cloud SCP failed: '+p.stderr[-1000:])
assert hashlib.file_digest(dest.open('rb'),'sha256').hexdigest()==manifest['sha256']
print(json.dumps({'copied':True,'archive':str(dest),'manifest':manifest}))
'''
            stdin, stdout, stderr = client.exec_command('python3 -c ' + shlex.quote(code), timeout=1900)
            stdin.write(json.dumps(payload)); stdin.flush(); stdin.channel.shutdown_write()
            raw = stdout.read().decode('utf8'); error = stderr.read().decode('utf8')
            if stdout.channel.recv_exit_status():
                raise RuntimeError(error[-1500:].replace(payload['password'], '[REDACTED]'))
            result = json.loads(raw)
        elif args.action in {'start-new', 'stop-old'}:
            from gpu_online import REMOTE_CONTROL
            with client.open_sftp() as sftp:
                root = sftp.normalize('.') + '/ResearchClaw'
            code = 'import sys\nsys.argv=' + repr(['control', root, 'start' if args.action == 'start-new' else 'stop', '8766']) + '\n' + REMOTE_CONTROL
            result = execute(client, code)
        elif args.action == 'progress':
            result = {'shared_archive': 'not trusted: prior NFS close I/O error'}
            result['new_archive'] = execute(client, "import json;from pathlib import Path;p=Path.home()/'runtime-migration-20261115.tar.gz';print(json.dumps({'bytes':p.stat().st_size if p.exists() else 0}))")
            old = connect_gpu(ROOT / '.env.gpu.local')
            try:
                result['old_local_disk'] = execute(old, r'''
import json
from pathlib import Path
p=Path.home()/'gpu-migration-20261115'
print(json.dumps({f.name:f.stat().st_size for f in p.iterdir() if f.is_file()} if p.exists() else {}))
''')
            finally:
                old.close()
            result['local_files'] = {p.name: p.stat().st_size for p in OUT.glob('*.gz')}
        elif args.action == 'verify':
            result = execute(client, r'''
import json,subprocess
from pathlib import Path
root=Path.home()/'ResearchClaw'
code="import torch,transformers,sentence_transformers,json; print(json.dumps({'torch':torch.__version__,'transformers':transformers.__version__,'sentence_transformers':sentence_transformers.__version__,'cuda':torch.cuda.is_available(),'gpu':torch.cuda.get_device_name(0),'matrix_sum':float((torch.ones((128,128),device='cuda')@torch.ones((128,128),device='cuda')).sum())}))"
p=subprocess.run([str(root/'.venv/bin/python'),'-c',code],capture_output=True,text=True,timeout=90)
if p.returncode:raise RuntimeError(p.stderr[-2000:])
print(p.stdout)
''')
        elif args.action == 'inspect':
            old = execute(client, INSPECT)
            other = connect_gpu(ROOT / '.env.gpu.next.local')
            try:
                new = execute(other, INSPECT)
            finally:
                other.close()
            result = {'old': old, 'new': new}
        elif args.action == 'evidence':
            result = execute(client, r'''
import hashlib,json,subprocess
from pathlib import Path
h=Path.home();dest=h/'gpu-migration-20261115';p=dest/'evidence.tar.gz'
subprocess.run(['tar','-czf',str(p),'--exclude=ResearchClaw/self-revisev0.5/model-cache',
'--exclude=ResearchClaw/self-revisev0.5/cache','--exclude=ResearchClaw/self-revisev0.5/workspace/gpu-online/token',
'--exclude=ResearchClaw/self-revisev0.5/workspace/gpu-online/pid','-C',str(h),
'ResearchClaw/scripts','ResearchClaw/src','ResearchClaw/self-revisev0.5'],check=True)
manifest=json.loads((dest/'manifest.json').read_text())
manifest['local_evidence']={'archive':str(p),'sha256':hashlib.file_digest(p.open('rb'),'sha256').hexdigest(),'bytes':p.stat().st_size}
print(json.dumps(manifest))
''', timeout=600)
            evidence = result['local_evidence']
            with client.open_sftp() as sftp:
                target = OUT / 'evidence.tar.gz'
                if target.exists():
                    raise RuntimeError('Evidence archive exists; refusing overwrite')
                sftp.get(evidence['archive'], str(target))
            with target.open('rb') as handle:
                assert hashlib.file_digest(handle, 'sha256').hexdigest() == evidence['sha256']
            evidence['local_verified'] = True
            result['complete_runtime_cloud_copy'] = json.loads((OUT / 'cloud-copy.json').read_text(encoding='utf8'))
            (OUT / 'backup.json').write_text(json.dumps(result, indent=2), encoding='utf8')
        elif args.action == 'backup':
            result = execute(client, r'''
import hashlib,json,subprocess
from pathlib import Path
h=Path.home(); root=h/'ResearchClaw'
dest=h/'gpu-migration-20261115';dest.mkdir(mode=0o700,exist_ok=True)
archive=dest/'runtime.tar.gz'
if archive.exists():raise RuntimeError('Backup already exists; inspect before reuse')
# Project artifacts/code/venv plus fixed model caches; no SSH or provider secrets.
cmd=['tar','-I','gzip -1','-cf',str(archive),'--exclude=ResearchClaw/self-revisev0.5/workspace/gpu-online/token',
'--exclude=ResearchClaw/self-revisev0.5/workspace/gpu-online/pid',
'--exclude=ResearchClaw/.env*','--exclude=ResearchClaw/.git','-C',str(h),'ResearchClaw']
if (h/'.cache/huggingface').exists():cmd.append('.cache/huggingface')
subprocess.run(cmd,check=True)
digest=hashlib.file_digest(archive.open('rb'),'sha256').hexdigest()
manifest={'archive':str(archive),'bytes':archive.stat().st_size,'sha256':digest,
'contents':'project code, cloud venv, jobs, workspaces, artifacts and HuggingFace cache; token/pid/secrets excluded'}
(dest/'manifest.json').write_text(json.dumps(manifest,indent=2))
print(json.dumps(manifest))
''', timeout=900)
            # Keep a complete independent local copy, rather than trusting shared storage.
            with client.open_sftp() as sftp:
                target = OUT / 'runtime.tar.gz'
                if target.exists():
                    raise RuntimeError('Local runtime archive exists; refusing overwrite')
                sftp.get(result['archive'], str(target))
            with target.open('rb') as handle:
                assert hashlib.file_digest(handle, 'sha256').hexdigest() == result['sha256']
            result['local_runtime_verified'] = True
            evidence = execute(client, r'''
import hashlib,json,subprocess
from pathlib import Path
h=Path.home(); dest=h/'gpu-migration-20261115'
p=dest/'evidence.tar.gz'
subprocess.run(['tar','-czf',str(p),'--exclude=ResearchClaw/self-revisev0.5/model-cache',
'--exclude=ResearchClaw/self-revisev0.5/cache',
'--exclude=ResearchClaw/self-revisev0.5/workspace/gpu-online/token',
'--exclude=ResearchClaw/self-revisev0.5/workspace/gpu-online/pid',
'-C',str(h),'ResearchClaw/scripts','ResearchClaw/src','ResearchClaw/self-revisev0.5'],check=True)
print(json.dumps({'archive':str(p),'sha256':hashlib.file_digest(p.open('rb'),'sha256').hexdigest(),'bytes':p.stat().st_size}))
''', timeout=900)
            with client.open_sftp() as sftp:
                target = OUT / 'evidence.tar.gz'
                if target.exists():
                    raise RuntimeError('Local archive exists; refusing overwrite')
                sftp.get(evidence['archive'], str(target))
            with target.open('rb') as handle:
                assert hashlib.file_digest(handle, 'sha256').hexdigest() == evidence['sha256']
            result['local_evidence'] = {**evidence, 'local_verified': True}
        else:
            if (OUT / 'backup.json').exists():
                manifest = json.loads((OUT / 'backup.json').read_text(encoding='utf8'))
            else:
                manifest = json.loads((OUT / 'cloud-copy.json').read_text(encoding='utf8'))['manifest']
            with client.open_sftp() as sftp:
                shared = manifest['archive']
                try:
                    sftp.stat(shared)
                except OSError:
                    shared = sftp.normalize('.') + '/runtime-migration-20261115.tar.gz'
                    try:
                        sftp.stat(shared)
                    except OSError:
                        sftp.put(str(OUT / 'runtime.tar.gz'), shared)
            result = execute(client, '''
import hashlib,json,subprocess
from pathlib import Path
archive=Path(%r)
assert hashlib.file_digest(archive.open('rb'),'sha256').hexdigest()==%r
root=Path.home()/'ResearchClaw'
if root.exists():raise RuntimeError('Target project exists; refusing overwrite')
subprocess.run(['tar','-xf',str(archive),'-C',str(Path.home())],check=True)
print(json.dumps({'restored':True,'archive_sha256':%r,'root':str(root)}))
''' % (shared, manifest['sha256'], manifest['sha256']), timeout=900)
        (OUT / (args.action + '.json')).write_text(json.dumps(result, ensure_ascii=False, indent=2), encoding='utf8')
        print(json.dumps(result, ensure_ascii=True, indent=2))
    finally:
        client.close()


if __name__ == '__main__':
    main()

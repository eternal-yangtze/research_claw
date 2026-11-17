"""Connect to the GPU instance and perform read-only diagnostics, from local CMD."""
from __future__ import annotations

import argparse
import base64
import getpass
import hashlib
import json
import os
import shlex
from pathlib import Path
import sys
import time
import uuid
import re
import posixpath

ROOT = Path(__file__).resolve().parents[1]
OUTPUT_ROOT = ROOT / 'self-revisev0.5'


def output_path(path):
    resolved = path.resolve()
    if not resolved.is_relative_to(OUTPUT_ROOT.resolve()):
        raise ValueError('0.5 outputs must stay inside self-revisev0.5')
    return resolved


def job_action(client, args):
    """Explicit-file SFTP sync and detached tasks; never uploads a directory tree."""
    job_id = args.job_id or ('job-' + uuid.uuid4().hex[:16])
    if not re.fullmatch(r'[A-Za-z0-9_-]{1,80}', job_id):
        raise ValueError('Invalid job ID')
    with client.open_sftp() as sftp:
        base = sftp.normalize('.') + '/ResearchClaw'
        def mkdir(path):
            import stat
            try:
                sftp.mkdir(path)
            except OSError:
                if not stat.S_ISDIR(sftp.lstat(path).st_mode):
                    raise ValueError('Remote path is not a real directory')
        def parents(path):
            current = base
            mkdir(current)
            for component in posixpath.relpath(path, base).split('/'):
                if component != '.':
                    current += '/' + component
                    mkdir(current)
        remote_jobs = base + '/self-revisev0.5/jobs'
        parents(remote_jobs)
        job = remote_jobs + '/' + job_id
        if args.action == 'sync':
            if not args.manifest:
                raise ValueError('--manifest is required')
            manifest = json.loads(output_path(args.manifest).read_text(encoding='utf8'))
            def remote_digest(path):
                _, out, err = client.exec_command('sha256sum -- ' + shlex.quote(path), timeout=30)
                value = out.read().decode('ascii').split()
                code = out.channel.recv_exit_status()
                if code or not value:
                    return None
                return value[0]
            synced = []
            for item in manifest['files']:
                rel = Path(item)
                if rel.is_absolute() or '..' in rel.parts:
                    raise ValueError('Relative paths without traversal required')
                source = (ROOT / rel).resolve()
                if not source.is_relative_to(ROOT) or any(p.startswith('.') for p in rel.parts):
                    raise ValueError('Hidden/external files cannot be uploaded')
                allowed = (rel.parts[0] in {'src', 'scripts'} and source.suffix == '.py') or (
                    rel.parts[0] == 'self-revisev0.5' and source.suffix in {'.json', '.jsonl', '.pdf', '.py', '.txt', '.tsv', '.zip'})
                if not allowed or not source.is_file() or source.stat().st_size > 100 * 1024**2:
                    raise ValueError('File not allowed or too large: ' + str(rel))
                data = source.read_bytes()
                digest = hashlib.sha256(data).hexdigest()
                target = base + '/' + rel.as_posix()
                parents(posixpath.dirname(target))
                changed = True
                changed = remote_digest(target) != digest
                if changed:
                    with sftp.open(target + '.upload', 'wb') as handle:
                        handle.set_pipelined(True)
                        handle.write(data)
                    if remote_digest(target + '.upload') != digest:
                        raise ValueError('Upload checksum mismatch')
                    sftp.posix_rename(target + '.upload', target)
                synced.append({'path': rel.as_posix(), 'sha256': digest, 'uploaded': changed})
            return {'remote_root': base, 'files': synced}
        if args.action == 'submit':
            if not args.spec:
                raise ValueError('--spec is required')
            spec = json.loads(output_path(args.spec).read_text(encoding='utf8'))
            if not isinstance(spec.get('argv'), list) or not all(isinstance(v, str) for v in spec['argv']):
                raise ValueError('spec argv must be a string list')
            spec['timeout'] = min(14400, max(1, int(spec.get('timeout', 3600))))
            sftp.mkdir(job)  # Existing job IDs are immutable; never replay automatically.
            with sftp.open(job + '/spec.json', 'w') as handle:
                handle.write(json.dumps(spec))
            command = ('nohup python3 ' + shlex.quote(base + '/scripts/gpu_job_worker.py') + ' ' +
                       shlex.quote(job) + ' </dev/null >/dev/null 2>&1 &')
            _, stdout, stderr = client.exec_command(command)
            if stdout.channel.recv_exit_status():
                raise RuntimeError(stderr.read().decode())
            return {'job_id': job_id, 'state': 'submitted', 'remote_dir': job}
        if args.action == 'cancel':
            with sftp.open(job + '/cancel', 'w') as handle:
                handle.write('cancel requested')
            return {'job_id': job_id, 'state': 'cancel_requested'}
        if args.action == 'fetch':
            if not args.remote_file:
                raise ValueError('--remote-file relative to self-revisev0.5 is required')
            rel = Path(args.remote_file)
            if rel.is_absolute() or '..' in rel.parts:
                raise ValueError('Invalid fetch path')
            remote_file = base + '/self-revisev0.5/' + rel.as_posix()
            with sftp.open(remote_file, 'rb') as handle:
                size=handle.stat().st_size
                if size>128*1024*1024:
                    raise ValueError('Fetch artifact exceeds 128 MiB bounded transfer budget')
                handle.prefetch(size,max_concurrent_requests=16)
                data = handle.read()
            _, out, _ = client.exec_command('sha256sum -- ' + shlex.quote(remote_file), timeout=30)
            expected = out.read().decode('ascii').split()
            if out.channel.recv_exit_status() or not expected or hashlib.sha256(data).hexdigest() != expected[0]:
                raise ValueError('Fetch checksum mismatch; remote artifact may still be changing')
            target = output_path(OUTPUT_ROOT / 'artifacts' / 'fetched' / rel)
            target.parent.mkdir(parents=True, exist_ok=True)
            target.write_bytes(data)
            return {'path': str(target), 'sha256': hashlib.sha256(data).hexdigest(), 'bytes': len(data)}
        result = {'job_id': job_id}
        for name in (['status.json'] if args.action == 'status' else ['stdout.log', 'stderr.log']):
            try:
                with sftp.open(job + '/' + name, 'rb') as handle:
                    handle.seek(max(0, handle.stat().st_size - 16000))
                    value = handle.read().decode('utf8', errors='replace')
                    result[name] = json.loads(value) if name.endswith('.json') else value
            except OSError:
                result[name] = 'not available yet'
        return result

REMOTE_CHECK = r'''
import importlib.util, json, os, platform, shutil, subprocess, sys
def run(args):
    try:
        p = subprocess.run(args, capture_output=True, text=True, timeout=20)
        return {'exit_code': p.returncode, 'stdout': p.stdout.strip(), 'stderr': p.stderr.strip()}
    except Exception as e:
        return {'error': type(e).__name__ + ': ' + str(e)}
result = {'hostname': platform.node(), 'user': run(['id', '-un']),
          'python': sys.version, 'python_executable': sys.executable,
          'home': os.path.expanduser('~'), 'cwd': os.getcwd(),
          'cpu_count': os.cpu_count(), 'memory': run(['free', '-h']),
          'disk': run(['df', '-h']),
          'task_processes': run(['ps', '-u', str(os.getuid()), '-o', 'pid,ppid,stat,etime,comm']),
          'gpu': run(['nvidia-smi', '--query-gpu=name,memory.total,driver_version', '--format=csv,noheader']),
          'commands': {x: shutil.which(x) for x in ['python', 'python3', 'git', 'nvidia-smi', 'tmux', 'docker']}}
try:
    if importlib.util.find_spec('torch'):
        import torch
        result['torch'] = {'version': torch.__version__, 'cuda_build': torch.version.cuda,
                           'cuda_available': torch.cuda.is_available()}
        if torch.cuda.is_available():
            x = torch.tensor([1., 2., 3.], device='cuda')
            result['torch']['gpu_name'] = torch.cuda.get_device_name(0)
            result['torch']['cuda_tensor_test'] = (x * 2).tolist()
            result['torch']['bf16_supported'] = torch.cuda.is_bf16_supported()
    else:
        result['torch'] = {'installed': False}
except Exception as e:
    result['torch'] = {'error': type(e).__name__ + ': ' + str(e)}
print(json.dumps(result, ensure_ascii=False, indent=2))
'''


def fingerprint(key):
    return 'SHA256:' + base64.b64encode(hashlib.sha256(key.asbytes()).digest()).decode().rstrip('=')


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('action', choices=['check', 'sync-check', 'sync', 'submit', 'status', 'logs', 'fetch', 'cancel'])
    parser.add_argument('--manifest', type=Path)
    parser.add_argument('--spec', type=Path)
    parser.add_argument('--job-id')
    parser.add_argument('--remote-file')
    parser.add_argument('--config', type=Path, default=ROOT / '.env.gpu.local')
    parser.add_argument('--accept-new-host-key', action='store_true',
                        help='Trust first observed host key and pin locally; never accept changed keys.')
    parser.add_argument('--report', type=Path, help='Save non-secret diagnostic JSON locally')
    args = parser.parse_args()
    if args.report:
        try:
            args.report = output_path(args.report)
        except ValueError as exc:
            parser.error(str(exc))
    try:
        import paramiko
        from dotenv import dotenv_values
    except ImportError:
        parser.error('Activate project .venv; install: python -m pip install -e ".[gpu-remote]"')
    if not args.config.is_file():
        parser.error('Config missing. Copy .env.gpu.example to .env.gpu.local and fill it in.')
    config = dotenv_values(args.config, interpolate=False)
    def setting(name, default=''):
        return os.environ.get(name, config.get(name) or default)
    host = setting('GPU_SSH_HOST')
    user = setting('GPU_SSH_USER')
    if not host or not user:
        parser.error('GPU_SSH_HOST and GPU_SSH_USER are required')
    try:
        port = int(setting('GPU_SSH_PORT', '22'))
        if not 1 <= port <= 65535:
            raise ValueError()
    except ValueError:
        parser.error('GPU_SSH_PORT must be 1..65535')
    key_file = setting('GPU_SSH_KEY_FILE')
    password = setting('GPU_SSH_PASSWORD') or None
    if not password and not key_file:
        password = getpass.getpass('SSH password (hidden): ')
    known_hosts = ROOT / '.env.gpu.known_hosts'
    client = paramiko.SSHClient()
    client.load_system_host_keys()
    if known_hosts.exists():
        client.load_host_keys(str(known_hosts))

    class PinFirstKey(paramiko.MissingHostKeyPolicy):
        def missing_host_key(self, ssh, hostname, key):
            fp = fingerprint(key)
            if not args.accept_new_host_key:
                raise paramiko.SSHException(
                    f'Unknown SSH host key {fp}. Compare with provider; '
                    'use --accept-new-host-key once to pin the first observed key.')
            ssh.get_host_keys().add(hostname, key.get_name(), key)
            ssh.save_host_keys(str(known_hosts))
            print(f'First observed host key pinned (TOFU): {fp}', file=sys.stderr)

    client.set_missing_host_key_policy(PinFirstKey())
    try:
        client.connect(hostname=host, port=port, username=user, password=password,
                       key_filename=str(Path(key_file).expanduser()) if key_file else None,
                       allow_agent=False, look_for_keys=False,
                       timeout=15, banner_timeout=20, auth_timeout=20)
        transport = client.get_transport()
        transport.set_keepalive(30)
        if args.action not in {'check', 'sync-check'}:
            result = job_action(client, args)
            output = json.dumps(result, ensure_ascii=True, indent=2)
            if password:
                output = output.replace(password, '[REDACTED]')
            print(output)
            if args.report:
                args.report.parent.mkdir(parents=True, exist_ok=True)
                args.report.write_text(output + '\n', encoding='utf8')
            return 0
        payload = REMOTE_CHECK
        transfer = None
        if args.action == 'sync-check':
            # Fixed minimal payload: no recursive project upload or credentials.
            probe = (ROOT / 'scripts' / 'gpu_cuda_probe.py').read_bytes()
            digest = hashlib.sha256(probe).hexdigest()
            with client.open_sftp() as sftp:
                remote_home = sftp.normalize('.')
                remote_root = remote_home + '/self-revisev0.5'
                for directory in (remote_root, remote_root + '/workspace'):
                    try:
                        sftp.mkdir(directory)
                    except OSError:
                        import stat
                        if not stat.S_ISDIR(sftp.lstat(directory).st_mode):
                            raise ValueError('Remote workspace must be a real directory')
                remote_file = remote_root + '/workspace/cuda-probe-' + digest + '.py'
                with sftp.open(remote_file, 'wb') as handle:
                    handle.write(probe)
                with sftp.open(remote_file, 'rb') as handle:
                    if hashlib.sha256(handle.read()).hexdigest() != digest:
                        raise ValueError('SFTP payload checksum mismatch')
            transfer = {'sha256': digest, 'remote_file': remote_file}
            command = 'python3 ' + shlex.quote(remote_file)
            payload = ''
        else:
            command = 'python3 -'
        stdin, stdout, stderr = client.exec_command(command, timeout=70)
        stdin.write(payload)
        stdin.flush()
        stdin.channel.shutdown_write()
        channel = stdout.channel
        chunks, errors = [], []
        deadline = time.monotonic() + 70
        while True:
            if time.monotonic() > deadline:
                channel.close()
                raise TimeoutError('Remote diagnostics exceeded 70 seconds')
            if channel.recv_ready():
                chunks.append(channel.recv(65536))
                continue
            if channel.recv_stderr_ready():
                errors.append(channel.recv_stderr(65536))
                continue
            if channel.exit_status_ready():
                break
            time.sleep(0.05)
        code = channel.recv_exit_status()
        raw = b''.join(chunks).decode('utf-8', errors='replace')
        if code:
            raise RuntimeError(f'Remote check failed (exit {code}): ' +
                               b''.join(errors).decode('utf-8', errors='replace'))
        result = {'ssh_connected': True, 'host': host, 'port': port, 'user': user,
                  'host_key_fingerprint': fingerprint(transport.get_remote_server_key()),
                  'remote': json.loads(raw)}
        if transfer:
            result['transfer'] = transfer
        output = json.dumps(result, ensure_ascii=False, indent=2)
        print(output)
        if args.report:
            args.report.parent.mkdir(parents=True, exist_ok=True)
            args.report.write_text(output + '\n', encoding='utf-8')
        return 0
    except paramiko.AuthenticationException:
        print('SSH authentication failed; check the local password/key configuration.', file=sys.stderr)
    except paramiko.BadHostKeyException:
        print('SSH host key changed; connection refused. Verify instance identity before updating pin.', file=sys.stderr)
    except (paramiko.SSHException, OSError, ValueError, RuntimeError) as exc:
        message = str(exc)
        if password:
            message = message.replace(password, '[REDACTED]')
        print(f'{type(exc).__name__}: {message}', file=sys.stderr)
    finally:
        client.close()
    return 1


if __name__ == '__main__':
    raise SystemExit(main())

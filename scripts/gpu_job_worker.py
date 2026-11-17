"""Detached, bounded remote process supervisor. Invoked by gpu_remote.py."""
import argparse
import json
import os
from pathlib import Path
import signal
import subprocess
import time


def main():
    p = argparse.ArgumentParser()
    p.add_argument('job_dir', type=Path)
    args = p.parse_args()
    job = args.job_dir.resolve()
    spec = json.loads((job / 'spec.json').read_text())
    status = {'state': 'running', 'supervisor_pid': os.getpid(), 'started': time.time()}
    def save():
        temp = job / 'status.tmp'
        temp.write_text(json.dumps(status))
        temp.replace(job / 'status.json')
    save()
    try:
        with (job / 'stdout.log').open('wb') as out, (job / 'stderr.log').open('wb') as err:
            env = dict(os.environ, PYTHONUNBUFFERED='1', PYTHONDONTWRITEBYTECODE='1',
                       HF_HOME=str(job.parents[1] / 'model-cache'),
                       XDG_CACHE_HOME=str(job.parents[1] / 'cache'),
                       PIP_CACHE_DIR=str(job.parents[1] / 'cache/pip'),
                       PYTHONPATH=str(job.parents[2] / 'src'))
            child = subprocess.Popen(spec['argv'], cwd=job.parents[2], env=env,
                                     stdout=out, stderr=err, start_new_session=True)
            status['pid'] = child.pid
            save()
            deadline = time.monotonic() + spec['timeout']
            while child.poll() is None:
                if (job / 'cancel').exists() or time.monotonic() > deadline:
                    os.killpg(child.pid, signal.SIGTERM)
                    try:
                        child.wait(timeout=10)
                    except subprocess.TimeoutExpired:
                        os.killpg(child.pid, signal.SIGKILL)
                    status['state'] = 'cancelled' if (job / 'cancel').exists() else 'timed_out'
                    break
                time.sleep(.5)
            status['exit_code'] = child.wait()
            if status['state'] == 'running':
                status['state'] = 'completed' if child.returncode == 0 else 'failed'
    except Exception as exc:
        status.update(state='failed', error=type(exc).__name__ + ': ' + str(exc))
    status['finished'] = time.time()
    save()


if __name__ == '__main__':
    main()

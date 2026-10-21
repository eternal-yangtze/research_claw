"""Frozen job dispatcher, validation-only tuning, complete-study aggregation."""
import argparse
import copy
import hashlib
import json
import statistics
from collections import defaultdict
from pathlib import Path

from train import run_job
from researchops.experiment_support import spec_hash


def summarize(records, *, complete, smoke):
    groups = defaultdict(list)
    for job, result in records:
        if job['stage'] == 'validation_tuning':
            continue
        key = (job['stage'], job['dataset_id'], job['method'], job['fraction'], job.get('rank'), job.get('dropout'))
        groups[key].append((job, result))
    rows = []
    for key, values in groups.items():
        row = dict(zip(('stage', 'dataset_id', 'method', 'fraction', 'rank', 'dropout'), key))
        row['seeds'] = [job['seed'] for job, _ in values]
        row['metrics'] = {}
        for metric in ('accuracy', 'macro_f1', 'ece'):
            numbers = [result.get('test_metrics', result['metrics'])[metric] for _, result in values]
            row['metrics'][metric] = {'mean': statistics.mean(numbers),
                'sample_std': statistics.stdev(numbers) if len(numbers) > 1 else None}
        row['trainable_parameters'] = statistics.mean(r['trainable_parameters'] for _, r in values)
        rows.append(row)
    success = None
    if complete and not smoke:
        success = True
        for dataset in {row['dataset_id'] for row in rows}:
            relevant = {row['method']: row for row in rows if row['stage'] == 'main'
                and row['fraction'] == 1.0 and row['dataset_id'] == dataset}
            full, lora = relevant['full_finetune'], relevant['lora']
            success &= (len(set(lora['seeds'])) == 3 and len(set(full['seeds'])) == 3
                and lora['metrics']['accuracy']['mean'] >= full['metrics']['accuracy']['mean'] - .02
                and lora['trainable_parameters'] / full['trainable_parameters'] <= .10)
    return {'complete': complete, 'mode': 'smoke' if smoke else 'full', 'research_success': success, 'groups': rows}


def main(argv=None):
    parser = argparse.ArgumentParser()
    parser.add_argument('--spec', required=True)
    parser.add_argument('--output', required=True)
    parser.add_argument('--smoke', action='store_true')
    parser.add_argument('--resume', action='store_true')
    parser.add_argument('--max-jobs', type=int)
    args = parser.parse_args(argv)
    spec = json.loads(Path(args.spec).read_text(encoding='utf-8'))
    jobs = spec.get('jobs', [])
    if not jobs:
        raise ValueError('Missing frozen jobs')
    root = Path(args.output)
    root.mkdir(parents=True, exist_ok=True)
    records, tuning = [], defaultdict(list)
    for original in jobs[:args.max_jobs]:
        job, current = copy.deepcopy(original), copy.deepcopy(spec)
        tuning_key = (job['dataset_id'], job['method'])
        if not isinstance(job['lr'], (int, float)):
            candidates = tuning[tuning_key]
            if not candidates:
                raise ValueError('Missing validation tuning results')
            job['lr'] = min(candidates, key=lambda pair: (-pair[1], pair[0]))[0]
        if job['stage'] == 'ablation':
            current['methods']['lora'].update({key: job[key] for key in ('rank', 'alpha', 'dropout') if key in job})
        target = root / job['job_id']
        result_path = target / 'result.json'
        result = None
        job_digest = hashlib.sha256(json.dumps(job, sort_keys=True).encode()).hexdigest()
        if args.resume and result_path.is_file():
            prior = json.loads(result_path.read_text(encoding='utf-8'))
            if (prior.get('spec_hash') == spec_hash(current) and prior.get('job_hash') == job_digest
                and prior.get('mode') == ('smoke' if args.smoke else 'full')
                and prior.get('training_complete') and (target / 'checkpoint.pt').is_file()):
                result = prior
        if result is None:
            result = run_job(current, job, target, smoke=args.smoke, max_steps=3 if args.smoke else None, resume=args.resume)
        records.append((job, result))
        if job['stage'] == 'validation_tuning':
            tuning[tuning_key].append((job['lr'], result['metrics']['accuracy']))
    complete = (len(records) == len(jobs) and not args.smoke
        and all(result.get('training_complete') and (job['stage'] == 'validation_tuning' or result.get('test_evaluated'))
                for job, result in records))
    summary = summarize(records, complete=complete, smoke=args.smoke)
    (root / 'summary.json').write_text(json.dumps(summary, indent=2), encoding='utf-8')
    return summary


if __name__ == '__main__':
    main()

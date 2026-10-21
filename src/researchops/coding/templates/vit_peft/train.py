"""Audited integration repair of the LLM trainer; resumable deterministic batches."""
import hashlib
import json
import math
import time
from pathlib import Path

import numpy as np
import torch
from torch.utils.data import default_collate

from models import build_model
from metrics import classification_metrics
from researchops.experiment_support import ManifestDataset, seed_everything, spec_hash, save_checkpoint, load_checkpoint


def _batches(data, indices, batch_size, cursor=0):
    for offset in range(cursor * batch_size, len(indices), batch_size):
        yield default_collate([data[index] for index in indices[offset:offset + batch_size]])


def _predict(net, data, batch_size, device, smoke):
    net.eval()
    logits, labels = [], []
    with torch.no_grad():
        for index, (x, y) in enumerate(_batches(data, list(range(len(data))), batch_size)):
            if smoke and index >= 4:
                break
            logits.append(net(x.to(device)).cpu())
            labels.append(y)
    return torch.cat(logits), torch.cat(labels)


def run_job(spec, job, output_dir, *, smoke=False, max_steps=None, resume=False):
    if job['stage'] == 'validation_tuning' and job.get('evaluate_test'):
        raise ValueError('Test access prohibited during validation tuning')
    out = Path(output_dir)
    out.mkdir(parents=True, exist_ok=True)
    seed_everything(int(job['seed']))
    dataset = next(d for d in spec['datasets'] if d['dataset_id'] == job['dataset_id'])
    cfg = spec['training']
    net = build_model(spec, job['method'], dataset['class_count'], smoke=smoke)
    device = torch.device('cuda' if torch.cuda.is_available() and not smoke else 'cpu')
    net.to(device)
    data_cfg = {'mean': (.485, .456, .406), 'std': (.229, .224, .225)}
    if not smoke:
        from timm.data import resolve_model_data_config
        data_cfg.update(resolve_model_data_config(net.backbone))
    transform_args = {key: data_cfg[key] for key in ('mean', 'std')}
    trainset = ManifestDataset(dataset, 'train', fraction=float(job['fraction']), smoke=smoke, **transform_args)
    valset = ManifestDataset(dataset, 'val', smoke=smoke, allow_test=False, **transform_args)
    batch = 2 if smoke else int(cfg['batch_size'])
    epochs = int(cfg['max_epochs'])
    steps_epoch = math.ceil(len(trainset) / batch)
    warm = max(0, steps_epoch * int(cfg['warmup_epochs']))
    total = max(1, steps_epoch * epochs)
    groups = {True: [], False: []}
    for name, parameter in net.named_parameters():
        if parameter.requires_grad:
            groups[parameter.ndim < 2 or 'prompt' in name].append(parameter)
    optimizer = torch.optim.AdamW([{'params': params, 'weight_decay': 0.0 if exclude else cfg['weight_decay']}
        for exclude, params in groups.items() if params], lr=float(job['lr']), betas=tuple(cfg['betas']), eps=cfg['eps'])
    def schedule(step):
        if warm and step < warm:
            return (step + 1) / warm
        progress = min(1.0, max(0.0, (step - warm) / max(1, total - warm)))
        return .01 + .99 * .5 * (1 + math.cos(math.pi * progress))
    scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer, schedule)
    digest = spec_hash(spec)
    job_digest = hashlib.sha256(json.dumps(job, sort_keys=True).encode()).hexdigest()
    checkpoint = out / 'checkpoint.pt'
    best_path = out / 'best_model.pt'
    step, epoch, cursor, best, best_epoch, stale = 0, 0, 0, -1., -1, 0
    if resume and checkpoint.is_file():
        state = load_checkpoint(checkpoint, net, optimizer, scheduler, spec_hash=digest, job_id=job['job_id'])
        extra = state['extra']
        if extra.get('job_hash') != job_digest:
            raise ValueError('Resolved job configuration changed')
        step, epoch, cursor = state['global_step'], extra['epoch'], extra['cursor']
        best, best_epoch, stale = extra['best'], extra['best_epoch'], extra['stale']
    if device.type == 'cuda':
        torch.cuda.reset_peak_memory_stats()
        torch.cuda.synchronize()
    began = time.perf_counter()
    def save():
        save_checkpoint(checkpoint, net, optimizer, scheduler, spec_hash=digest, job_id=job['job_id'], global_step=step,
            extra={'epoch': epoch, 'cursor': cursor, 'best': best, 'best_epoch': best_epoch,
                   'stale': stale, 'job_hash': job_digest})
    while epoch < epochs and stale < int(cfg['early_stopping']['patience']):
        if max_steps is not None and step >= max_steps:
            break
        # Dedicated permutation RNG makes cursor resume independent of model/data augmentation RNG.
        generator = torch.Generator().manual_seed(int(job['seed']) * 100000 + epoch)
        order = torch.randperm(len(trainset), generator=generator).tolist()
        net.train()
        for x, y in _batches(trainset, order, batch, cursor):
            optimizer.zero_grad(set_to_none=True)
            loss = torch.nn.functional.cross_entropy(net(x.to(device)), y.to(device))
            if not torch.isfinite(loss):
                raise FloatingPointError('Nonfinite training loss')
            loss.backward()
            norm = torch.nn.utils.clip_grad_norm_(net.parameters(), cfg['grad_clip_norm'])
            if not torch.isfinite(norm):
                raise FloatingPointError('Nonfinite gradient')
            optimizer.step()
            scheduler.step()
            step += 1
            cursor += 1
            save()
            if max_steps is not None and step >= max_steps:
                break
        val_logits, val_labels = _predict(net, valset, batch, device, smoke)
        score = classification_metrics(val_logits, val_labels, dataset['class_count'])['accuracy']
        if score > best:
            best, best_epoch, stale = score, epoch, 0
            torch.save({key: value.detach().cpu() for key, value in net.state_dict().items()}, best_path)
        elif cursor >= steps_epoch:
            stale += 1
        if cursor >= steps_epoch:
            epoch, cursor = epoch + 1, 0
        save()
        if max_steps is not None and step >= max_steps:
            break
    if not best_path.is_file():
        raise ValueError('No validated checkpoint available')
    net.load_state_dict(torch.load(best_path, map_location='cpu', weights_only=True))
    logits, labels = _predict(net, valset, batch, device, smoke)
    np.save(out / 'val_logits.npy', logits.numpy())
    np.save(out / 'val_labels.npy', labels.numpy())
    result = {'spec_hash': digest, 'job_hash': job_digest, 'job_id': job['job_id'], 'global_step': step,
        'mode': 'smoke' if smoke else 'full', 'test_evaluated': False,
        'training_complete': epoch >= epochs or stale >= int(cfg['early_stopping']['patience']),
        'trainable_parameters': sum(p.numel() for p in net.parameters() if p.requires_grad),
        'metrics': classification_metrics(logits, labels, dataset['class_count']), 'best_epoch': best_epoch,
        'preprocessing': transform_args, 'loader_policy': 'deterministic in-process batches; resumable sample cursor',
        'research_result': not smoke}
    if not smoke and job.get('evaluate_test') and result['training_complete']:
        testset = ManifestDataset(dataset, 'test', smoke=False, allow_test=True, **transform_args)
        tx, ty = _predict(net, testset, batch, device, False)
        np.save(out / 'test_logits.npy', tx.numpy())
        np.save(out / 'test_labels.npy', ty.numpy())
        result.update(test_evaluated=True, test_metrics=classification_metrics(tx, ty, dataset['class_count']))
    if device.type == 'cuda':
        torch.cuda.synchronize()
    result.update(training_seconds=time.perf_counter() - began,
        peak_memory=torch.cuda.max_memory_allocated() if device.type == 'cuda' else None)
    (out / 'result.json').write_text(json.dumps(result, indent=2), encoding='utf-8')
    return result

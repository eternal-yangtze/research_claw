# Frozen ViT adaptation experiment

Runtime: Python 3.12, torch 2.6, torchvision 0.21, timm 1.0.27, numpy 2.2.6; install these versions and make `researchops.experiment_support` importable. No sklearn or scipy is needed.

Run engineering smoke checks with `python run_experiment.py --spec inputs/execution_spec.json --output artifacts/smoke --smoke --max-jobs 5`. Smoke uses random tiny ViT weights and real manifest train/validation images at 32 pixels, batch size 2 and no workers. It checks execution on CPU; its scores provide no research evidence.

Expose the full frozen study with `python run_experiment.py --spec inputs/execution_spec.json --output artifacts/full`. Full execution requires the exact pretrained timm backbone and accessible pinned weights; missing weights are an explicit error. No full study is run as part of verification.

DTD and Oxford-IIIT Pet come from the frozen dataset manifests and fraction files. Their supplied SHA256 hashes identify the prepared allocations; official test allocations are preserved. Respect the original dataset terms. Test data must never enter tuning or smoke.

Each job writes checkpoint.pt, result.json, val_logits.npy and val_labels.npy. After failure, rerun the command with `--resume`; checkpoint identity must match the job and spec. A max_steps argument in the Python API limits total optimizer updates across resumes.

Report accuracy, macro F1 over every configured class, 15-bin ECE, trainable parameters, synchronized training time and peak CUDA allocation (null on CPU). Full study aggregation uses three seeds with sample standard deviation. The frozen hypothesis requires LoRA accuracy within 0.02 of full fine tuning and a trainable parameter ratio at most 0.10 on both datasets at full fraction. Smoke never supports this hypothesis.

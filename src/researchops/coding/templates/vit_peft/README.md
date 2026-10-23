# Contract-driven ViT adaptation project

Runtime: Python 3.12, torch 2.6, torchvision 0.21, timm 1.0.27, numpy 2.2.6; install these versions and make `researchops.experiment_support` importable. No sklearn or scipy is needed.

Start a compact CPU run with `python run_experiment.py --spec inputs/execution_spec.json --output artifacts/preview --smoke --max-jobs 5`. This mode uses random tiny-ViT weights and manifest train/validation images at 32 pixels, batch size 2 and no workers. It exercises the implementation and produces execution artifacts; scientific conclusions use the full study mode below.

Execute the full research contract with `python run_experiment.py --spec inputs/execution_spec.json --output artifacts/full`. Full execution uses the specified pretrained timm backbone and pinned weights, preserving the supplied study design.

DTD and Oxford-IIIT Pet come from the dataset manifests and fraction files. Their supplied SHA256 hashes identify the prepared allocations. Model selection uses train/validation allocations; final evaluation allocations remain separate. Respect the original dataset terms.

Each job writes checkpoint.pt, result.json, val_logits.npy and val_labels.npy. After failure, rerun the command with `--resume`; checkpoint identity must match the job and spec. A max_steps argument in the Python API limits total optimizer updates across resumes.

Report accuracy, macro F1 over every configured class, 15-bin ECE, trainable parameters, synchronized training time and peak CUDA allocation (null on CPU). Full study aggregation uses three seeds with sample standard deviation. The supplied research contract compares LoRA accuracy with full fine tuning within 0.02, with a trainable parameter ratio at most 0.10 on both datasets at full fraction; assess this hypothesis using the full study's actual results.

"""Small standalone CUDA execution probe; no corpus or application credentials."""
import json
import torch


def main():
    if not torch.cuda.is_available():
        raise RuntimeError('CUDA required; CPU fallback is disabled')
    torch.cuda.reset_peak_memory_stats()
    start = torch.cuda.Event(enable_timing=True)
    end = torch.cuda.Event(enable_timing=True)
    matrix = torch.eye(512, device='cuda') * 2
    start.record()
    result = matrix @ matrix
    end.record()
    torch.cuda.synchronize()
    assert result.device.type == 'cuda'
    assert torch.allclose(result, torch.eye(512, device='cuda') * 4)
    print(json.dumps({'cuda_available': True, 'device': str(result.device),
                      'gpu': torch.cuda.get_device_name(0), 'torch': torch.__version__,
                      'cuda_build': torch.version.cuda, 'shape': list(result.shape),
                      'matmul_ms': start.elapsed_time(end),
                      'peak_memory_bytes': torch.cuda.max_memory_allocated(),
                      'correct': True}))


if __name__ == '__main__':
    main()

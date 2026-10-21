import numpy as np


def classification_metrics(logits, labels, num_classes):
    scores = np.asarray(logits, dtype=np.float64)
    target = np.asarray(labels, dtype=np.int64).reshape(-1)
    if scores.ndim != 2 or scores.shape != (target.size, num_classes):
        raise ValueError('logits must have shape [n, num_classes] and match labels')
    pred = scores.argmax(axis=1)
    accuracy = float(np.mean(pred == target)) if target.size else 0.0
    f1s = []
    for c in range(num_classes):
        tp = int(np.sum((pred == c) & (target == c)))
        fp = int(np.sum((pred == c) & (target != c)))
        fn = int(np.sum((pred != c) & (target == c)))
        denom = 2 * tp + fp + fn
        f1s.append(2 * tp / denom if denom else 0.0)
    shifted = scores - scores.max(axis=1, keepdims=True) if target.size else scores
    probs = np.exp(shifted)
    probs /= probs.sum(axis=1, keepdims=True) if target.size else 1
    confidence = probs.max(axis=1) if target.size else np.array([])
    correct = pred == target
    ece = 0.0
    for i in range(15):
        low, high = i / 15, (i + 1) / 15
        mask = (confidence >= low) & ((confidence < high) if i < 14 else (confidence <= high))
        if mask.any():
            ece += float(mask.mean()) * abs(float(correct[mask].mean()) - float(confidence[mask].mean()))
    return {'accuracy': accuracy, 'macro_f1': float(np.mean(f1s)), 'ece': float(ece)}

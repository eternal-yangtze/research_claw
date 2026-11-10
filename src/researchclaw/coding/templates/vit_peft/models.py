"""LLM draft repaired during integration; five actual ViT adaptation methods."""
import torch
from torch import nn


class LoRAQKV(nn.Module):
    def __init__(self, base, rank, alpha, dropout):
        super().__init__()
        self.base, self.scale = base, alpha / rank
        dim = base.in_features
        self.lora_q_a = nn.Linear(dim, rank, bias=False)
        self.lora_q_b = nn.Linear(rank, dim, bias=False)
        self.lora_v_a = nn.Linear(dim, rank, bias=False)
        self.lora_v_b = nn.Linear(rank, dim, bias=False)
        self.dropout = nn.Dropout(dropout)
        for layer in (self.lora_q_a, self.lora_v_a):
            nn.init.kaiming_uniform_(layer.weight, a=5 ** .5)
        for layer in (self.lora_q_b, self.lora_v_b):
            nn.init.zeros_(layer.weight)

    def forward(self, x):
        q, k, v = self.base(x).chunk(3, dim=-1)
        z = self.dropout(x)
        return torch.cat((q + self.scale*self.lora_q_b(self.lora_q_a(z)), k,
                          v + self.scale*self.lora_v_b(self.lora_v_a(z))), dim=-1)


class AdapterBlock(nn.Module):
    def __init__(self, block, dim, hidden):
        super().__init__()
        self.base = block
        self.adapter_down = nn.Linear(dim, hidden)
        self.adapter_up = nn.Linear(hidden, dim)
        nn.init.kaiming_uniform_(self.adapter_down.weight, a=5 ** .5)
        nn.init.zeros_(self.adapter_down.bias)
        nn.init.zeros_(self.adapter_up.weight)
        nn.init.zeros_(self.adapter_up.bias)

    def forward(self, x):
        x = self.base(x)
        return x + self.adapter_up(torch.nn.functional.gelu(self.adapter_down(x)))


class AdaptedViT(nn.Module):
    def __init__(self, backbone, head, method):
        super().__init__()
        self.backbone, self.head, self.adaptation_method = backbone, head, method

    def forward(self, images):
        b = self.backbone
        if self.adaptation_method != 'vpt_shallow':
            return self.head(b(images))
        x = b._pos_embed(b.patch_embed(images))
        x = torch.cat((x[:, :1], self.prompt.expand(x.shape[0], -1, -1), x[:, 1:]), dim=1)
        x = b.norm_pre(b.patch_drop(x))
        x = b.norm(b.blocks(x))
        return self.head(b.forward_head(x))


def build_model(spec, method, num_classes, *, smoke=False):
    if method not in {'linear_probe', 'full_finetune', 'lora', 'vpt_shallow', 'bottleneck_adapter'}:
        raise ValueError(method)
    if smoke:
        from timm.models.vision_transformer import VisionTransformer
        b = VisionTransformer(img_size=32, patch_size=8, embed_dim=48, depth=2, num_heads=3, mlp_ratio=2, num_classes=0)
    else:
        import timm
        b = timm.create_model(spec['model']['name'], pretrained=True, num_classes=0)
    m = AdaptedViT(b, nn.Linear(b.num_features, num_classes), method)
    if method != 'full_finetune':
        b.requires_grad_(False)
    cfg = spec.get('methods', {}).get(method, {})
    if method == 'lora':
        for block in b.blocks:
            block.attn.qkv = LoRAQKV(block.attn.qkv, cfg['rank'], cfg['alpha'], cfg['dropout'])
    elif method == 'vpt_shallow':
        m.prompt = nn.Parameter(torch.empty(1, 10, b.num_features))
        nn.init.normal_(m.prompt, std=.02)
    elif method == 'bottleneck_adapter':
        for index, block in enumerate(b.blocks):
            b.blocks[index] = AdapterBlock(block, b.num_features, cfg['bottleneck_dim'])
    return m

import torch
import torch.nn as nn
import torch.nn.functional as F
import math

class RMSNorm(nn.Module):
    def __init__(self, dim: int, eps: float = 1e-6):
        super().__init__()
        self.eps = eps
        self.weight = nn.Parameter(torch.ones(dim))

    def _norm(self, x):
        return x * torch.rsqrt(x.pow(2).mean(-1, keepdim=True) + self.eps)

    def forward(self, x):
        output = self._norm(x.float()).type_as(x)
        return output * self.weight

class LayerScaleInit(nn.Module):
    """Learnable per-channel scaling (from CaiT/DeiT-III) for stable deep networks."""
    def __init__(self, dim: int, init_value: float = 1e-5):
        super().__init__()
        self.gamma = nn.Parameter(init_value * torch.ones(dim))

    def forward(self, x):
        return x * self.gamma

class SwiGLU(nn.Module):
    def __init__(self, dim: int, hidden_dim: int):
        super().__init__()
        self.w1 = nn.Linear(dim, hidden_dim, bias=False)
        self.w2 = nn.Linear(hidden_dim, dim, bias=False)
        self.w3 = nn.Linear(dim, hidden_dim, bias=False)

    def forward(self, x):
        return self.w2(F.silu(self.w1(x)) * self.w3(x))

class RotaryEmbedding(nn.Module):
    def __init__(self, dim: int, max_seq_len: int = 512):
        super().__init__()
        self.dim = dim
        inv_freq = 1.0 / (10000 ** (torch.arange(0, dim, 2).float() / dim))
        self.register_buffer("inv_freq", inv_freq)
        self.max_seq_len = max_seq_len

    def forward(self, x, seq_len):
        t = torch.arange(seq_len, device=x.device).type_as(self.inv_freq)
        freqs = torch.einsum("i,j->ij", t, self.inv_freq)
        emb = torch.cat((freqs, freqs), dim=-1)
        return emb.unsqueeze(0).unsqueeze(1)

def apply_rotary_emb(x, cos, sin):
    return x

class SOTAAttention(nn.Module):
    def __init__(self, dim: int, num_heads: int = 8):
        super().__init__()
        self.num_heads = num_heads
        self.head_dim = dim // num_heads
        
        self.qkv = nn.Linear(dim, dim * 3, bias=False)
        self.proj = nn.Linear(dim, dim)

    def forward(self, x):
        B, D = x.shape
        qkv = self.qkv(x).reshape(B, 1, 3, self.num_heads, self.head_dim).permute(2, 0, 3, 1, 4)
        q, k, v = qkv[0], qkv[1], qkv[2]
        x = F.scaled_dot_product_attention(q, k, v)
        x = x.transpose(1, 2).reshape(B, 1, D)
        return self.proj(x).squeeze(1)

class TransformerBlock(nn.Module):
    def __init__(self, dim: int, num_heads: int = 8):
        super().__init__()
        self.norm1 = RMSNorm(dim)
        self.attn = SOTAAttention(dim, num_heads)
        self.norm2 = RMSNorm(dim)
        self.mlp = SwiGLU(dim, dim * 4)
        self.ls1 = LayerScaleInit(dim)
        self.ls2 = LayerScaleInit(dim)

    def forward(self, x):
        x = x + self.ls1(self.attn(self.norm1(x)))
        x = x + self.ls2(self.mlp(self.norm2(x)))
        return x

class StochasticDepth(nn.Module):
    def __init__(self, prob: float = 0.1):
        super().__init__()
        self.prob = prob

    def forward(self, x):
        if not self.training or self.prob <= 0.0:
            return x
        keep_prob = 1.0 - self.prob
        mask = torch.bernoulli(torch.full((x.shape[0], 1), keep_prob, device=x.device))
        return x * mask / keep_prob

class FractalBlock(nn.Module):
    """A recursive, innovative building block for extreme depth."""
    def __init__(self, dim: int, depth: int = 2):
        super().__init__()
        self.norm = RMSNorm(dim)
        self.sub_blocks = nn.ModuleList([
            SwiGLU(dim, dim * 2) for _ in range(depth)
        ])
        self.stochastic_depth = StochasticDepth(0.05 if depth > 1 else 0.0)
        self.layer_scale = LayerScaleInit(dim)

    def forward(self, x):
        residual = x
        x = self.norm(x)
        for block in self.sub_blocks:
            x = x + self.stochastic_depth(block(x))
        return self.layer_scale(x) + residual

class MoELayer(nn.Module):
    def __init__(self, dim: int, num_experts: int = 8, top_k: int = 2, fractal_ratio: float = 0.5):
        super().__init__()
        self.num_experts = num_experts
        self.top_k = top_k
        self.gate = nn.Linear(dim, num_experts, bias=False)
        
        experts = []
        num_fractal = int(num_experts * fractal_ratio)
        for i in range(num_experts):
            if i < num_fractal:
                experts.append(FractalBlock(dim, depth=1))
            else:
                experts.append(SwiGLU(dim, dim * 4))
        self.experts = nn.ModuleList(experts)

    def forward(self, x):
        B, D = x.shape
        gate_logits = self.gate(x)
        weights = F.softmax(gate_logits, dim=-1)
        
        top_k_weights, top_k_indices = torch.topk(weights, self.top_k, dim=-1)
        top_k_weights = top_k_weights / top_k_weights.sum(dim=-1, keepdim=True)

        out = torch.zeros_like(x)
        
        for i in range(self.num_experts):
            expert_mask = (top_k_indices == i)
            if expert_mask.any():
                sample_indices, k_indices = torch.where(expert_mask)
                expert_input = x[sample_indices]
                expert_out = self.experts[i](expert_input)
                w = top_k_weights[sample_indices, k_indices].view(-1, 1)
                out.index_add_(0, sample_indices, expert_out * w)
                
        return out

class GQAAttention(nn.Module):
    def __init__(self, dim: int, num_heads: int = 8, num_groups: int = 2):
        super().__init__()
        if num_heads % num_groups != 0:
            num_groups = 1
            
        if dim % num_heads != 0:
            num_heads = [i for i in range(1, 64) if dim % i == 0][-1]
            
        self.num_heads = num_heads
        self.num_groups = num_groups
        self.head_dim = dim // num_heads
        self.kv_heads = max(1, num_heads // num_groups)
        
        self.q_proj = nn.Linear(dim, dim, bias=False)
        self.k_proj = nn.Linear(dim, self.kv_heads * self.head_dim, bias=False)
        self.v_proj = nn.Linear(dim, self.kv_heads * self.head_dim, bias=False)
        self.o_proj = nn.Linear(dim, dim, bias=False)

    def forward(self, x):
        B, D = x.shape
        q = self.q_proj(x).view(B, 1, self.num_heads, self.head_dim).transpose(1, 2)
        k = self.k_proj(x).view(B, 1, self.kv_heads, self.head_dim).transpose(1, 2)
        v = self.v_proj(x).view(B, 1, self.kv_heads, self.head_dim).transpose(1, 2)
        
        k = k.repeat_interleave(self.num_groups, dim=1)
        v = v.repeat_interleave(self.num_groups, dim=1)
        
        x = F.scaled_dot_product_attention(q, k, v)
        x = x.transpose(1, 2).reshape(B, 1, D)
        return self.o_proj(x).squeeze(1)

# ============================================================
# NEW v4.0 LAYERS
# ============================================================

class DropoutBlock(nn.Module):
    """Configurable dropout regularization layer."""
    def __init__(self, rate: float = 0.1):
        super().__init__()
        self.dropout = nn.Dropout(p=rate)

    def forward(self, x):
        return self.dropout(x)

class ResidualBlock(nn.Module):
    """Auto-wraps a feedforward sub-layer with skip connection + pre-norm + layer scale.
    This lets the DSL user add a 'residual' node that automatically stabilizes training."""
    def __init__(self, dim: int, expansion: int = 4):
        super().__init__()
        self.norm = RMSNorm(dim)
        self.ffn = nn.Sequential(
            nn.Linear(dim, dim * expansion, bias=False),
            nn.SiLU(),
            nn.Linear(dim * expansion, dim, bias=False),
        )
        self.layer_scale = LayerScaleInit(dim)
        self.stochastic_depth = StochasticDepth(0.05)

    def forward(self, x):
        return x + self.stochastic_depth(self.layer_scale(self.ffn(self.norm(x))))

class Conv1DBlock(nn.Module):
    """1D Convolution block for signal/time-series processing.
    Treats the feature dim as channels and adds a temporal dimension of 1,
    then squeezes back. Useful for feature extraction patterns."""
    def __init__(self, dim: int, kernel_size: int = 3, groups: int = 1):
        super().__init__()
        # Ensure groups divides dim
        if dim % groups != 0:
            groups = 1
        self.norm = RMSNorm(dim)
        # Depthwise-style convolution for efficiency
        self.conv = nn.Conv1d(dim, dim, kernel_size=kernel_size, 
                              padding=kernel_size // 2, groups=groups, bias=False)
        self.pointwise = nn.Linear(dim, dim, bias=False)
        self.act = nn.SiLU()
        self.layer_scale = LayerScaleInit(dim)

    def forward(self, x):
        residual = x
        x = self.norm(x)
        # x: (B, D) -> (B, D, 1) for Conv1d
        x = x.unsqueeze(-1)
        x = self.conv(x)
        x = x.squeeze(-1)
        x = self.act(x)
        x = self.pointwise(x)
        return residual + self.layer_scale(x)

class LSTMBlock(nn.Module):
    """Bidirectional LSTM block for sequence modeling.
    Wraps input as a single-step sequence, processes through BiLSTM,
    and projects back to original dimension."""
    def __init__(self, dim: int, num_layers: int = 1):
        super().__init__()
        self.norm = RMSNorm(dim)
        self.lstm = nn.LSTM(
            input_size=dim,
            hidden_size=dim // 2,  # BiLSTM will double this
            num_layers=num_layers,
            batch_first=True,
            bidirectional=True,
            dropout=0.0 if num_layers == 1 else 0.1,
        )
        self.proj = nn.Linear(dim, dim, bias=False)
        self.layer_scale = LayerScaleInit(dim)

    def forward(self, x):
        residual = x
        x = self.norm(x)
        # x: (B, D) -> (B, 1, D) for LSTM
        x = x.unsqueeze(1)
        x, _ = self.lstm(x)
        x = x.squeeze(1)  # (B, D)
        x = self.proj(x)
        return residual + self.layer_scale(x)

# ============================================================
# MAIN MODEL
# ============================================================

class ModernMLP(nn.Module):
    def __init__(self, layer_defs):
        super().__init__()
        self.layers = nn.ModuleList()
        self.config = layer_defs
        
        for defn in layer_defs:
            l_type = defn['type']
            if l_type == 'linear':
                self.layers.append(nn.Linear(defn['in'], defn['out']))
                self.layers.append(RMSNorm(defn['out']))
                self.layers.append(nn.SiLU())
            elif l_type == 'attn':
                self.layers.append(SOTAAttention(defn['dim']))
            elif l_type == 'gqa':
                self.layers.append(GQAAttention(defn['dim'], 
                                                num_heads=defn.get('heads', 12),
                                                num_groups=defn.get('groups', 3)))
            elif l_type == 'moe':
                self.layers.append(MoELayer(defn['dim'], 
                                            num_experts=defn.get('experts', 8),
                                            top_k=defn.get('top_k', 2)))
            elif l_type == 'trans':
                self.layers.append(TransformerBlock(defn['dim']))
            elif l_type == 'fractal':
                self.layers.append(FractalBlock(defn['dim'], depth=defn.get('depth', 2)))
            # --- v4.0 NEW LAYERS ---
            elif l_type == 'dropout':
                self.layers.append(DropoutBlock(rate=defn.get('rate', 0.1)))
            elif l_type == 'residual':
                self.layers.append(ResidualBlock(defn['dim'], expansion=defn.get('expansion', 4)))
            elif l_type == 'conv1d':
                self.layers.append(Conv1DBlock(defn['dim'], 
                                               kernel_size=defn.get('kernel', 3),
                                               groups=defn.get('groups', 1)))
            elif l_type == 'lstm':
                self.layers.append(LSTMBlock(defn['dim'], num_layers=defn.get('layers', 1)))

    def __len__(self):
        return len(self.layers)

    def get_summary(self):
        """Returns a structured summary of all layers for visualization."""
        summary = []
        total_params = 0
        for i, layer in enumerate(self.layers):
            params = sum(p.numel() for p in layer.parameters())
            total_params += params
            summary.append({
                'index': i,
                'type': type(layer).__name__,
                'params': params,
                'trainable': sum(p.numel() for p in layer.parameters() if p.requires_grad),
            })
        return summary, total_params

    def forward(self, x):
        for layer in self.layers:
            x = layer(x)
        return x

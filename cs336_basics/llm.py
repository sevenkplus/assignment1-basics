import math
import torch
import torch.nn as nn
import einx

class Linear(nn.Module):
    def __init__(self, d_in, d_out, device=None, dtype=None):
        super().__init__()
        self.weight = nn.Parameter(torch.empty((d_out, d_in), device=device, dtype=dtype))
        sigma = math.sqrt(2/(d_in+d_out))
        nn.init.trunc_normal_(self.weight, std=sigma, a=-3*sigma, b=3*sigma)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return einx.dot("d_out d_in, ... d_in -> ... d_out", self.weight, x)

class Embedding(nn.Module):
    def __init__(self, n_emb, d_emb, device=None, dtype=None):
        super().__init__()
        self.weight = nn.Parameter(torch.empty((n_emb, d_emb), device=device, dtype=dtype))
        nn.init.trunc_normal_(self.weight, a=-3, b=3)

    def forward(self, token_ids: torch.Tensor) -> torch.Tensor:
        return einx.get_at("[n_emb] d_emb, ... -> ... d_emb", self.weight, token_ids)

class RMSNorm(nn.Module):
    def __init__(self, d: int, eps: float = 1e-5, device=None, dtype=None):
        super().__init__()
        self.d = d
        self.eps = eps
        self.weight = nn.Parameter(torch.ones(d, device=device, dtype=dtype))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        in_dtype = x.dtype
        x = x.to(dtype=torch.float32)

        # rms = ((x**2).mean(dim=-1) + self.eps).sqrt()
        # result = x / rms.unsqueeze(-1) * self.g

        rms = (einx.mean("... [d]", x**2) + self.eps).sqrt()

        result = einx.elementwise(
            "... d, ..., d -> ... d",
            x, rms, self.weight,
            op=lambda x, y, z: x/y*z
        )

        result = result.to(dtype=in_dtype)
        return result

class SiLU(nn.Module):
    def __init__(self):
        super().__init__()

    def forward(self, x):
        return x * torch.sigmoid(x)

class SwiGLU(nn.Module):
    def __init__(self, d_model, d_ff, device=None, dtype=None):
        super().__init__()
        self.d_model = d_model
        self.d_ff = d_ff

        self.w1 = Linear(d_model, d_ff, device=device, dtype=dtype)
        self.w2 = Linear(d_ff, d_model, device=device, dtype=dtype)
        self.w3 = Linear(d_model, d_ff, device=device, dtype=dtype)

        self.silu = SiLU()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.w2(
            einx.multiply(
                "[...]",
                self.silu(self.w1(x)),
                self.w3(x)
            ),
        )

class RoPE(nn.Module):
    def __init__(self, theta: float, d_k: int, max_seq_len: int, device=None):
        super().__init__()

        def rot_mat(i, k):
            th = i/theta**(2*k/d_k)
            return [
                [math.cos(th), -math.sin(th)],
                [math.sin(th), math.cos(th)]
            ]

        r = torch.tensor(
            [[rot_mat(i, k) for k in range(d_k//2)] for i in range(max_seq_len)],
            device=device
        )

        self.register_buffer("r", r, persistent=False)

        # shape: max_seq_len, dk/2, 2, 2

    def forward(self, x: torch.Tensor, pos: torch.Tensor) -> torch.Tensor:
        return einx.dot(
            "... seq_len (dk d2), seq_len dk d1 d2 -> ... seq_len (dk d1)",
            x,
            einx.get_at(
                "[max_seq_len] dk d1 d2, ... -> ... dk d1 d2",
                self.r, pos
            ),
            d1=2,
            d2=2
        )

def scaled_dot_product_attention(Q, K, V, mask):
    attn = einx.dot(
        "... nq d_k, ... nk d_k -> ... nq nk",
        Q, K
    )
    d_k = Q.shape[-1]
    attn /= math.sqrt(d_k)
    if mask is not None:
        if len(mask.shape) == 2:
            attn = einx.where("nq nk, ... nq nk,", mask, attn, -torch.inf)
        else:
            attn = einx.where("..., ...,", mask, attn, -torch.inf)
    softmax_attn = einx.softmax("... nq [nk]", attn)
    res = einx.dot(
        "... nq nk, ... nk d_v -> ... nq d_v",
        softmax_attn, V
    )
    return res

class MultiheadSelfAttention(nn.Module):
    def __init__(self, d_model, num_heads, rope=None, device=None, dtype=None):
        super().__init__()

        self.q_proj = Linear(d_model, d_model, device=device, dtype=dtype)
        self.k_proj = Linear(d_model, d_model, device=device, dtype=dtype)
        self.v_proj = Linear(d_model, d_model, device=device, dtype=dtype)
        self.output_proj = Linear(d_model, d_model, device=device, dtype=dtype)

        self.num_heads = num_heads
        self.rope = rope

    def forward(self, x, pos=None):
        seq_len = x.shape[-2]
        Q = einx.rearrange(
            "... seq_len (nh hd_k) -> ... nh seq_len hd_k",
            self.q_proj(x),
            nh = self.num_heads
        )
        if self.rope is not None and pos is not None:
            Q = self.rope(Q, pos)
        K = einx.rearrange(
            "... seq_len (nh hd_k) -> ... nh seq_len hd_k",
            self.k_proj(x),
            nh = self.num_heads
        )
        if self.rope is not None and pos is not None:
            K = self.rope(K, pos)
        V = einx.rearrange(
            "... seq_len (nh hd_v) -> ... nh seq_len hd_v",
            self.v_proj(x),
            nh = self.num_heads
        )
        mask = torch.tril(torch.ones((seq_len, seq_len), dtype=torch.bool))
        mh = einx.rearrange(
            "... nh seq_len hd_v -> ... seq_len (nh hd_v)",
            scaled_dot_product_attention(Q, K, V, mask)
        )
        return self.output_proj(mh)

class TransformerBlock(nn.Module):
    def __init__(self, d_model: int, num_heads: int, d_ff: int, max_seq_len: int, theta: float,
                 device=None, dtype=None):
        super().__init__()

        rope = RoPE(theta, d_model // num_heads, max_seq_len, device=device)
        self.attn = MultiheadSelfAttention(d_model, num_heads, rope=rope)
        self.ffn = SwiGLU(d_model, d_ff, device=device, dtype=dtype)
        self.ln1 = RMSNorm(d_model, device=device, dtype=dtype)
        self.ln2 = RMSNorm(d_model, device=device, dtype=dtype)

    def forward(self, x):
        seq_len = x.shape[-2]
        pos = torch.arange(seq_len)
        x += self.attn(self.ln1(x), pos)
        x += self.ffn(self.ln2(x))
        return x

class Transformer(nn.Module):
    def __init__(self, vocab_size: int, context_length: int, d_model: int, num_layers: int, num_heads: int, d_ff: int, rope_theta: float,
                 device=None, dtype=None):
        super().__init__()

        self.token_embeddings = Embedding(vocab_size, d_model, device=device, dtype=dtype)
        self.layers = nn.Sequential(*[TransformerBlock(d_model, num_heads, d_ff, context_length, rope_theta, device=device, dtype=dtype) for _ in range(num_layers)])
        self.ln_final = RMSNorm(d_model, device=device, dtype=dtype)
        self.lm_head = Linear(d_model, vocab_size, device=device, dtype=dtype)

    def forward(self, token_ids):
        x = self.token_embeddings(token_ids)
        x = self.layers(x)
        x = self.ln_final(x)
        x = self.lm_head(x)
        return x

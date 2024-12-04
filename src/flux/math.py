import torch
from torch import Tensor
from torch.nn.functional import scaled_dot_product_attention
from einops import rearrange
from triton.ops import attention as attention_triton


def _xformers_flash_hopper(q, k, v, compile: bool):
    import xformers
    import xformers.ops

    if compile:
        xformers_flash = torch.compile(
            xformers.ops.fmha.flash.FwOp,
            fullgraph=True,
            backend="inductor",
        )
    else:
        xformers_flash = xformers.ops.fmha.flash.FwOp()
    softmax_scale = q.size(-1) ** -0.5

    return xformers.ops.fmha.memory_efficient_attention_forward(  # noqa: E731
        q,
        k,
        v,
        scale=softmax_scale,
        op=xformers_flash,  # type: ignore
    )


def attention(q: Tensor, k: Tensor, v: Tensor, pe: Tensor, method: str) -> Tensor:
    q, k = apply_rope(q, k, pe)

    if method == "xformers_compiled_flash" or method == "xformers_flash":
        q = q.permute(0, 2, 1, 3)  # B, H, S, D
        k = k.permute(0, 2, 1, 3)  # B, H, S, D
        v = v.permute(0, 2, 1, 3)  # B, H, S, D

        if "compiled" in method:
            x = _xformers_flash_hopper(q, k, v, compile=True).permute(0, 2, 1, 3)
        else:
            x = _xformers_flash_hopper(q, k, v, compile=False).permute(0, 2, 1, 3)
    elif method == "torch_sdpa":
        x = scaled_dot_product_attention(q, k, v)
    elif method == "triton_attention":
        softmax_scale = q.size(-1) ** -0.5
        x = attention_triton(q, k, v, True, softmax_scale)
    else:
        raise ValueError(f"Unknown method {method}")

    x = rearrange(x, "B H L D -> B L (H D)")
    assert x is not None, "x is None"
    return x


def rope(pos: Tensor, dim: int, theta: int) -> Tensor:
    assert dim % 2 == 0
    scale = torch.arange(0, dim, 2, dtype=torch.float64, device=pos.device) / dim
    omega = 1.0 / (theta**scale)
    out = torch.einsum("...n,d->...nd", pos, omega)
    out = torch.stack([torch.cos(out), -torch.sin(out), torch.sin(out), torch.cos(out)], dim=-1)
    out = rearrange(out, "b n d (i j) -> b n d i j", i=2, j=2)
    return out.float()


def apply_rope(xq: Tensor, xk: Tensor, freqs_cis: Tensor) -> tuple[Tensor, Tensor]:
    xq_ = xq.float().reshape(*xq.shape[:-1], -1, 1, 2)
    xk_ = xk.float().reshape(*xk.shape[:-1], -1, 1, 2)
    xq_out = freqs_cis[..., 0] * xq_[..., 0] + freqs_cis[..., 1] * xq_[..., 1]
    xk_out = freqs_cis[..., 0] * xk_[..., 0] + freqs_cis[..., 1] * xk_[..., 1]
    return xq_out.reshape(*xq.shape).type_as(xq), xk_out.reshape(*xk.shape).type_as(xk)

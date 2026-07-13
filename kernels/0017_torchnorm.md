# torchnorm — Fused RMSNorm, FusedAddRMSNorm, LayerNorm

**Link:** https://github.com/liodon-ai/torchnorm/blob/main/torchnorm/_triton.py

**Author:** [Liodon AI](https://github.com/liodon-ai) (py-ai-dev)

**Tags:** Normalization, RMSNorm, LayerNorm, Training, LLM

**Description:** <br/>
Standalone Triton kernel library for normalisation layers, available as the `torchnorm` PyPI package. Kernels:

- **`fused_rms_norm`** — single-pass fused forward for RMSNorm (used by LLaMA, Mistral, Gemma, Qwen, Falcon). Reads each element once; computes RMS and applies weight in one pass.
- **`fused_rms_norm_backward`** — exact backward: `dx = rms_inv * (w*dy - x_hat * mean(w*dy*x_hat))`. Weight gradient accumulated with `tl.atomic_add` across rows.
- **`fused_add_rms_norm`** — fuses residual add + RMSNorm into one kernel. Returns `(out, updated_residual)`. Eliminates the intermediate `x + residual` allocation that appears in every transformer pre-norm block.
- **`fused_layer_norm`** — LayerNorm forward+backward with optional bias.

All kernels auto-dispatch: Triton on CUDA, `F.rms_norm`/`F.layer_norm` fallback elsewhere.

**Minimal Usage:**
```python
pip install "torchnorm[triton]"

import torch
from torchnorm import RMSNorm, FusedAddRMSNorm

# Drop-in for LLaMA-style norm
norm = RMSNorm(dim=4096).cuda()
out = norm(x)  # fused Triton on CUDA

# Fuse residual add + norm — one kernel instead of two ops
norm = FusedAddRMSNorm(dim=4096).cuda()
hidden, residual = norm(x, residual)
```

**Triton Version:** ≥ 2.1

**Other Notes:**<br/>
Clean reference for the RMSNorm backward in Triton, including the `mean(w*dy*x_hat)` correction term that is easy to drop. The `tl.atomic_add` pattern for weight gradient accumulation across rows is a useful idiom. `FusedAddRMSNorm` is particularly useful in pre-norm transformers: it saves one memory round-trip per layer by fusing the residual add that precedes every norm. Companion to [0016_torchembed_RoPE](0016_torchembed_RoPE.md).

**Id in triton index:** 0017

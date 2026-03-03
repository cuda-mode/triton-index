# Flash Attention Forward (SM120 Gluon — TMA + MMAv2)

**Link:** https://github.com/triton-lang/kernels/blob/main/kernels/flash_attention_sm120.py

**Author:** Second Nature Computing (https://joinsecondnature.com)

**Tags:** attention, flash attention, SM120, Gluon, TMA, MMAv2, BF16, FP8, RTX 5090, DGX Spark, GB10

**Description:** <br/>Flash attention forward pass targeting SM120 GPUs (DGX Spark, RTX 5090, GB10) using Triton's experimental Gluon framework. SM120 lacks tcgen05/TMEM/WGMMA but has MMAv2 (`mma.sync.aligned`) tensor cores and TMA (Tensor Memory Accelerator) for async data movement. Uses TMA for all global memory access (Q/K/V loads and output store), double-buffered K pipelining with mbarrier phase tracking, and online softmax with exp2-based rescaling. Supports BF16 and FP8 (E5M2), causal and non-causal, arbitrary sequence lengths including non-square (Sq != Sk).

**Minimal Usage:**
```py
import torch
from kernels.flash_attention_sm120 import attention_forward_sm120

B, H, S, D = 2, 16, 1024, 64
q = torch.randn(B, H, S, D, device="cuda", dtype=torch.bfloat16)
k = torch.randn(B, H, S, D, device="cuda", dtype=torch.bfloat16)
v = torch.randn(B, H, S, D, device="cuda", dtype=torch.bfloat16)
sm_scale = 1.0 / (D ** 0.5)

out = attention_forward_sm120(q, k, v, sm_scale, causal=True)
```

**Triton Version:** Requires Triton with Gluon support (triton.experimental.gluon)

**Other Notes:**<br/>First open-source Gluon flash attention kernel for SM120. Originally submitted as [triton-lang/triton PR #9600](https://github.com/triton-lang/triton/pull/9600), redirected here. Also available at [triton-lang/kernels PR #20](https://github.com/triton-lang/kernels/pull/20). Requires SM12x GPU — will skip gracefully on other architectures.

**Id in triton index:** 16

# torchembed — Fused RoPE (rotate-half & adjacent-pairs)

**Link:** https://github.com/liodon-ai/torchembed/blob/main/torchembed/_triton.py

**Author:** [Liodon AI](https://github.com/liodon-ai) (py-ai-dev)

**Tags:** Embedding, Positional-Encoding, RoPE, Rotary-Embeddings, Attention

**Description:** <br/>
Full-library Triton kernel collection for embedding layers, available as the `torchembed` PyPI package. Core kernels:

- **`fused_rope_rotate_half`** — fused forward+backward for HuggingFace-convention RoPE (used by LLaMA, Mistral, Gemma, Qwen). Single-pass; no intermediate `cos`/`sin` tensor allocations. Integrated into HuggingFace Transformers `apply_rotary_pos_emb` (PR #47278).
- **`fused_rope_adjacent`** — same for torchtune/adjacent-pairs convention (rotates paired adjacent elements rather than the second half). Integrated into torchtune.

Both handle arbitrary `(batch, seq, heads, head_dim)` layout and support BF16/FP16/FP32. ~4–7× speedup over standard PyTorch `rotate_half` at typical training seq lengths on A100/H100.

Additional kernels: patch embedding (ViT), categorical embedding (tabular ML), Fourier features, ALiBi attention bias.

**Minimal Usage:**
```python
pip install "torchembed[triton]"

import torch
from torchembed import RotaryEmbedding

rope = RotaryEmbedding(dim=128, convention="rotate_half").cuda()
q = torch.randn(2, 512, 8, 128, device="cuda", dtype=torch.bfloat16)
k = torch.randn(2, 512, 8, 128, device="cuda", dtype=torch.bfloat16)
q_rot, k_rot = rope(q, k)  # fused Triton kernel dispatched automatically
```

**Triton Version:** ≥ 2.1

**Other Notes:**<br/>
Good reference for writing a library that handles two incompatible conventions (rotate-half vs adjacent-pairs) in a single codebase. The auto-dispatch pattern (`x.is_cuda and triton_available`) is clean and reusable. Complements [0017_torchnorm](0017_torchnorm.md) which covers the normalization side.

**Id in triton index:** 0016

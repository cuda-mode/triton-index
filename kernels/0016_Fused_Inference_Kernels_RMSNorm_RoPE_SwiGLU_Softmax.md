# Fused Inference Kernels: RMSNorm, RoPE, SwiGLU, Softmax
**Link:** https://github.com/ragulk143/inference-kernels
**Author:** Ragul K
**Tags:** RMSNorm, RoPE, SwiGLU, Softmax, Inference, Fusion
**Description:** A small, from-scratch collection of fused Triton kernels for the four operations that run in every transformer layer: RMSNorm, RoPE, SwiGLU, and Softmax. Each kernel is benchmarked against a PyTorch baseline and verified with Nsight Compute profiling (memory throughput, compute throughput, occupancy), not just wall-clock timing. Includes an honest documented failed optimization attempt on the RoPE kernel (a "fix" for strided memory access that made performance 3.5x worse instead of better) and a comparison against torch.compile's automatic fusion.
**Triton Version:** Triton 3.1.0
**Id in triton index:** 0016

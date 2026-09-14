# Winograd Convolution

A PyTorch implementation and numerical study of the Winograd
**F(2×2, 3×3)** minimal filtering algorithm for CNN convolution.

For each 2×2 output tile, direct 3×3 convolution requires 36 scalar
multiplications per channel pair, while the Winograd transform performs 16
element-wise multiplications after transforming the input and kernel. The lab
also demonstrates why fewer multiplications do not automatically guarantee a
faster high-level Python implementation.

## What is included

- transform matrices and tiled `conv2d` implementation;
- comparison with `torch.nn.functional.conv2d`;
- timing on several image sizes;
- floating-point error checks, including difficult inputs and FP16;
- an `nn.Module` wrapper and a small neural-network example;
- optional OpenCV real-time demos.

## Core showcase

Run these commands from the repository root:

```bash
python3 winograd-convolution/lab_showcase.py matrices
python3 winograd-convolution/lab_showcase.py check
python3 winograd-convolution/lab_showcase.py bench
python3 winograd-convolution/lab_showcase.py badcases
```

On Apple Silicon, the showcase automatically selects MPS when it is available.
Use `--device cpu` to force CPU execution, or add `--fp16` after `bench` to
compare half precision.

Additional examples:

```bash
python3 winograd-convolution/sanity_check.py
python3 winograd-convolution/mini_net_demo.py
python3 winograd-convolution/realtime_demo_v2.py
```

The real-time examples require a webcam and OpenCV.

## Files

- `winograd_f23.py` — compact functional implementation.
- `winograd_module.py` — reusable `torch.nn.Module` wrapper.
- `lab_showcase.py` — matrices, correctness, timing, and numerical edge cases.
- `sanity_check.py` — focused equivalence check.
- `mini_net_demo.py` — integration into a small network.
- `realtime_demo.py`, `realtime_demo_v2.py` — camera demonstrations.

## Reference

The transform construction follows the Winograd/Cook–Toom formulation used in
Andrew Lavin's [`wincnn`](https://github.com/andravin/wincnn) reference
implementation. That upstream repository is kept separately and is not copied
into this project.

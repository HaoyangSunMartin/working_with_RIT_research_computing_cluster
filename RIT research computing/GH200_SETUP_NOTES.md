# GH200 Setup Notes

This file summarizes the useful findings and failure modes encountered while bringing up local-model inference on the `grace` partition with a `gh200`.

## What Worked

- Partition and GPU request:
  - `#SBATCH --partition=grace`
  - `#SBATCH --gres=gpu:gh200:1`
- A working GH200-compatible Spack environment:
  - `default-nlp-aarch64-26012701`
- Torch on GH200 in that Spack env:
  - Python: `3.11.7`
  - `torch==2.1.1`
  - `torch.version.cuda == 12.3`
  - `torch.cuda.is_available() == True`
- The node architecture is ARM:
  - `uname -m -> aarch64`
- DeepSeek model load worked on GH200:
  - `deepseek-ai/DeepSeek-R1-Distill-Qwen-32B`

## GH200 Hardware Facts

- Example node:
  - `gh-00.rc.rit.edu`
- GPU reported by `nvidia-smi`:
  - `NVIDIA GH200 480GB`
- GPU memory visible to Torch/CUDA:
  - about `96 GB`

Important:
- The `480GB` label is platform branding for GH200, not the usual "single GPU VRAM available to your model".
- For model-fitting decisions in this workflow, use the memory visible to CUDA/Torch, which was about `96 GB`.

## Why GH200 Is Different

- GH200 nodes are `aarch64`, not `x86_64`.
- Existing x86 Conda environments should not be assumed to work on GH200.
- Prefer a GH200-compatible Spack environment first.

## Spack Lessons

### Safe Mental Model

- `spack env activate <env>`:
  - enters a managed software world by changing `PATH`, library resolution, and related shell state
- `spack load <pkg>`:
  - overlays an installed package into the current shell

Operational rule:
- treat `spack env activate` as the last Spack command whenever possible

### Important Failure Mode

Running `spack` again after activating the GH200 Spack env caused Spack itself to fail with an OpenSSL mismatch:

- error shape:
  - `libcrypto.so.3: version OPENSSL_3.4.0 not found`

What this means:
- after env activation, the shell was using libraries from the env
- then `spack` itself ran inside that modified environment
- Spack's own Python runtime picked up incompatible OpenSSL libraries

Conclusion:
- avoid `spack load ...` after `spack env activate ...` in this setup

## Apptainer on GH200

Loading Apptainer worked when done before activating the Spack env:

```bash
spack load apptainer
spack env activate default-nlp-aarch64-26012701
```

Verified:
- `apptainer --version` worked after that sequence

## Accelerate Problem and Workaround

### Problem

DeepSeek-R1 model loading with:
- `device_map="auto"`

required `accelerate`, but the GH200 NLP Spack env did not include it.

The missing-package error was:
- `Using low_cpu_mem_usage=True or a device_map requires Accelerate`

### First Attempt That Failed

Trying to add Accelerate with:

```bash
spack load /vdggrhg
```

failed due to package conflicts:
- `py-accelerate@1.1.1` conflicted with currently loaded runtime packages

### Diagnostic Result

Two orderings were tested:

1. `spack env activate ...` then `spack load /vdggrhg`
   - failed
2. `spack load /vdggrhg` then `spack env activate ...`
   - worked in a controlled diagnostic shell for Python import of `accelerate`

But the full `spack load /vdggrhg` approach was still too fragile for the actual DeepSeek job because of runtime conflicts.

### Working Workaround

Instead of loading the full package, only inject Accelerate's Python package path:

1. find the install prefix:

```bash
spack location -i /vdggrhg
```

2. locate its `site-packages`
3. prepend that path to `PYTHONPATH`

This allowed Python to import `accelerate` without overlaying the full Spack runtime stack.

## DeepSeek-R1 GH200 Smoke Test Result

A smoke test was successfully run with:
- `deepseek-ai/DeepSeek-R1-Distill-Qwen-32B`
- GH200
- GH200 Spack env
- Apptainer loaded
- Accelerate injected via `PYTHONPATH`

What passed:
- tokenizer load
- model download
- model load
- one short generation

What did not pass:
- output control was weak
- the model produced reasoning text instead of only SQL

This means:
- infrastructure path: good
- model load path: good
- prompt/output shaping still needs work

## Warnings Seen

These appeared during the GH200 DeepSeek test:

- `Unable to register cuFFT factory`
- `Unable to register cuDNN factory`
- `Unable to register cuBLAS factory`
- `Sliding Window Attention is enabled but not implemented for sdpa`
- generation config warnings about `temperature` and `top_p` while `do_sample=False`

Interpretation:
- these were warnings, not hard blockers
- the main blocker had been missing `accelerate`, which is now worked around

## Recommended Bring-Up Order on GH200

1. Submit a minimal Torch smoke test on `grace`
2. Use a GH200-compatible Spack env
3. Load any extra Spack tools like `apptainer` before env activation
4. Avoid further `spack` commands after env activation
5. If a missing Python package exists in Spack but conflicts as a full load, try injecting only its `site-packages` via `PYTHONPATH`
6. Only after Torch + CUDA + model load work, attempt full benchmark inference

## Practical Commands

Check GH200 GPU info inside a job or interactive session:

```bash
nvidia-smi --query-gpu=name,memory.total,driver_version --format=csv,noheader
```

Useful GH200-compatible Spack env found during setup:

```bash
spack env activate default-nlp-aarch64-26012701
```

Known Accelerate install used for the workaround:

```bash
spack location -i /vdggrhg
```

## Bottom Line

- GH200 is a viable path for larger local models in this project
- the main challenge is not CUDA availability
- the main challenge is software-environment management on `aarch64`
- Spack ordering matters
- full package overlays can conflict
- path-based Python-package injection can be a practical workaround when full `spack load` is too disruptive

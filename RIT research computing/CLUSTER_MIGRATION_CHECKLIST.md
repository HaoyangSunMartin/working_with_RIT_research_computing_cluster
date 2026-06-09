# Cluster Migration Checklist

This checklist captures the working pattern from the Spider2 cluster setup so it can be reused in another project on the same cluster.

## Recommended split

Use these layers deliberately:

- `Spack` for cluster-provided system tools such as `apptainer` and, when needed, `cuda`
- `Conda` for the host Python environment and model/runtime libraries
- `Apptainer` for the task or execution sandbox

For GPU-backed offline inference, keep the model process on the host Conda environment and use the container only for sandboxed task execution.

## Clean-shell rule

Always start from a clean login shell or clean batch shell.

Why:

- `spack load` can modify `PATH`
- `conda activate` can modify `PATH`
- mixing them in a dirty shell can silently switch the active Python interpreter

If things look inconsistent, restart from a fresh shell instead of debugging shell state indefinitely.

## Spack usage

Prefer exact Spack specs over generic package names.

Example:

```bash
spack load /kcushwg
```

This is more reliable than:

```bash
spack load apptainer
```

Use Spack for:

- `apptainer`
- optionally `cuda` if you need a cluster-compatible CUDA runtime/toolkit in the shell

## Conda usage

After any Spack load, reactivate Conda.

Recommended pattern:

```bash
source /home/hs6750/miniforge3/etc/profile.d/conda.sh
conda deactivate >/dev/null 2>&1 || true
conda deactivate >/dev/null 2>&1 || true
conda activate Spider2
hash -r
```

Do not rely on `which python` before reactivation.

Use the Conda interpreter explicitly:

```bash
PYTHON_CMD="$CONDA_PREFIX/bin/python"
```

This is more robust than using whatever `python` happens to be first on `PATH`.

## Python package installation

Install packages using the environment's Python:

```bash
python -m pip install <package>
```

Avoid bare `pip install ...`, because `pip` may belong to the wrong interpreter if Spack and Conda are both in play.

## Apptainer usage

Validate the image separately before long benchmark runs.

Example:

```bash
apptainer exec /path/to/image.sif python3 -c "import pandas, duckdb; print('ok')"
```

Good practice:

- compare the image timestamp to the current definition file
- rebuild if the image is older than the recipe

For task-sandbox containers, keep GPU passthrough off unless the container itself needs GPU execution:

```bash
SPIDER_AGENT_APPTAINER_NV=0
```

Only use `--nv` when the containerized process actually needs direct GPU access.

## GPU inference pattern

For offline inference:

- run the LLM from the host Conda environment
- allocate the GPU to the Slurm job
- do not force the task sandbox container to also consume GPU resources unless needed

Validate CUDA on the actual GPU node, not the login node:

```bash
python -c "import torch; print(torch.__version__); print(torch.version.cuda); print(torch.cuda.is_available())"
```

Installing `torch` on the login node is fine. CUDA validation must still happen on the GPU node.

## Cluster-preflight checks

Before launching a real run, check:

```bash
which apptainer
"$PYTHON_CMD" --version
"$PYTHON_CMD" -c "import torch; print(torch.cuda.is_available())"
apptainer exec /path/to/image.sif python3 -c "import pandas, duckdb, snowflake.connector; from google.cloud import bigquery; print('ok')"
```

If any of these fail, fix them before burning GPU time on a long job.

## Slurm job design

Prefer multiple shorter jobs over one very long job when queue wait is a concern.

Good pattern:

- split benchmark tasks into chunks
- give each chunk its own suffix
- merge outputs later for evaluation

Benefits:

- shorter requested wall time
- easier retry of failed chunks
- lower cost of one bad job

## Output management

Use unique run suffixes for every experiment.

This avoids mixing:

- old outputs
- new outputs
- partial reruns

For chunked runs, use a consistent family of suffixes, for example:

- `myrun-000_025`
- `myrun-025_050`
- `myrun-050_075`

## Common failure modes

### Spack overwrote Conda Python

Symptom:

- wrong Python version in logs
- missing packages that are installed in Conda

Fix:

- reactivate Conda after Spack load
- use `"$CONDA_PREFIX/bin/python"` explicitly

### Container image drift

Symptom:

- package is in Dockerfile/definition file but missing at runtime

Fix:

- rebuild the image
- add an import-based preflight check

### CUDA mismatch

Symptom:

- `torch.cuda.is_available() == False` on GPU node
- CUDA/driver warning from PyTorch

Fix:

- install a CUDA-compatible PyTorch build for the cluster
- optionally load a matching CUDA runtime via Spack
- validate on the GPU node

### Missing container packages

Symptom:

- `ModuleNotFoundError` inside task execution

Fix:

- add the missing package to the image recipe
- rebuild and revalidate imports

### Agent tries unsupported CLIs

Symptom:

- `command not found` for tools such as `bq`, `snowsql`, or `jq`

Fix:

- either install those tools in the image
- or change prompts/tooling so the agent does not rely on them

## Migration recipe

When porting this setup to a new project:

1. Build the container recipe for the task sandbox.
2. Create a host bootstrap script that:
   - loads Spack tools
   - reactivates Conda
   - resolves the interpreter explicitly
3. Add container import preflight checks.
4. Add host CUDA checks if using offline inference.
5. Keep outputs isolated by suffix.
6. Prefer chunked Slurm submission for large runs.
7. Add a merge step if evaluation expects a single submission folder.

## Practical default

If unsure, use this default pattern:

- `Spack` for `apptainer`
- `Conda` for Python and model libraries
- explicit `PYTHON_CMD="$CONDA_PREFIX/bin/python"`
- `Apptainer` for sandboxing only
- no `--nv` for the task container unless required
- preflight checks before benchmark runs

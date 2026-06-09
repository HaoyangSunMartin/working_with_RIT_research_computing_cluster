# Working with Cluster 101

This note is a practical checklist for using the RIT Research Computing cluster with this project. It is based on the local notes under `knowledge/RIT research computing/` and the project sync script `sync_with_cluster.sh`.

## 1. Connect to the Cluster

The login host used by the RIT Research Computing documentation is:

```bash
sporcsubmit.rc.rit.edu
```

Use your RIT username in place of `<rit_username>`.

### First Login

Before SSH keys work, you may need to log in once using your password and Duo:

```bash
ssh <rit_username>@sporcsubmit.rc.rit.edu
```

### Create a Local SSH Key

Run this on your local computer, not on the cluster:

```bash
ssh-keygen -t rsa
```

Use the default file path unless you have a specific reason not to. Use a passphrase.

### Copy the Key to the Cluster

```bash
ssh-copy-id -i ~/.ssh/id_rsa.pub <rit_username>@sporcsubmit.rc.rit.edu
```

If `ssh-copy-id` is unavailable:

```bash
cat ~/.ssh/id_rsa.pub | ssh <rit_username>@sporcsubmit.rc.rit.edu "mkdir -p ~/.ssh && cat >> ~/.ssh/authorized_keys"
```

### Configure the Project SSH Alias

The project sync script expects the SSH alias `RITCluster`. Add this to your local `~/.ssh/config`:

```sshconfig
Host RITCluster
    HostName sporcsubmit.rc.rit.edu
    User <rit_username>
    IdentityFile ~/.ssh/id_rsa
```

Then test:

```bash
ssh RITCluster
```

If SSH complains about a stale host key:

```bash
ssh-keygen -R sporcsubmit.rc.rit.edu
```

## 2. Basic Slurm Commands

Common commands:

```bash
sinfo
squeue -u "$USER"
sbatch path/to/job.slurm
scancel <job_id>
sacct -j <job_id>
```

For GH200 jobs, the project notes used:

```bash
#SBATCH --partition=grace
#SBATCH --gres=gpu:gh200:1
```

Check GPU information inside a GPU job:

```bash
nvidia-smi --query-gpu=name,memory.total,driver_version --format=csv,noheader
```

## 3. Storage Expectations

Use your home directory for SSH keys, Conda/Miniforge environments, configuration files, and small project state. Use shared project storage for large datasets, generated databases, model caches, and cluster results when possible.

Useful checks:

```bash
df -h ~
du -sh /path/to/folder
```

This project often uses shared paths such as:

```bash
/shared/rc/llm4lctk/bird
```

## 4. Set Up Miniforge

The RIT documentation describes local Conda-style installs as user-managed. The safest pattern is to install Miniforge in your home directory and always activate it explicitly in batch scripts.

### Choose the Installer

For normal x86_64 login or compute nodes:

```bash
Miniforge3-Linux-x86_64.sh
```

For GH200 ARM nodes:

```bash
Miniforge3-Linux-aarch64.sh
```

Important: GH200 nodes are `aarch64`, so x86 Conda environments should not be assumed to work there. For GH200, the local notes found a working cluster-provided Spack environment, described below.

### Install Miniforge

On the cluster:

```bash
mkdir -p ~/software
wget -O ~/software/Miniforge3-Linux-x86_64.sh \
  https://github.com/conda-forge/miniforge/releases/latest/download/Miniforge3-Linux-x86_64.sh
bash ~/software/Miniforge3-Linux-x86_64.sh -b -p ~/miniforge3
source ~/miniforge3/etc/profile.d/conda.sh
conda --version
```

Create a project environment:

```bash
conda create -n BIRD python=3.12 -y
conda activate BIRD
python -m pip install --upgrade pip
```

In batch scripts, do not rely on interactive shell initialization. Source Conda directly:

```bash
source /home/<rit_username>/miniforge3/etc/profile.d/conda.sh
conda activate BIRD
PYTHON_CMD="$CONDA_PREFIX/bin/python"
```

If you load Spack tools first, reactivate Conda afterward:

```bash
source /home/<rit_username>/miniforge3/etc/profile.d/conda.sh
conda deactivate >/dev/null 2>&1 || true
conda deactivate >/dev/null 2>&1 || true
conda activate BIRD
hash -r
PYTHON_CMD="$CONDA_PREFIX/bin/python"
```

Use:

```bash
python -m pip install <package>
```

Avoid bare `pip install ...`, because `pip` can point at the wrong Python when Spack and Conda are both active.

## 5. Install PyTorch and TensorFlow

The exact GPU package build should match the cluster CUDA/driver environment. Install packages in the Conda environment, then validate them on the node type where they will run.

### CPU-Only Install

```bash
source ~/miniforge3/etc/profile.d/conda.sh
conda activate BIRD
python -m pip install torch torchvision torchaudio tensorflow
```

### GPU Install Pattern

If you need CUDA-backed PyTorch or TensorFlow, first check the available CUDA tools:

```bash
spack find -l cuda
```

If needed, load a specific CUDA build by version/hash:

```bash
spack load cuda@<version> /<spack_hash>
```

Then reactivate Conda and install the CUDA-compatible framework packages:

```bash
source ~/miniforge3/etc/profile.d/conda.sh
conda activate BIRD
python -m pip install --upgrade pip
python -m pip install torch torchvision torchaudio
python -m pip install tensorflow
```

For PyTorch GPU wheels, use the wheel index recommended by the current PyTorch installation instructions for the CUDA version available on the cluster. Do not assume a login-node check is enough; CUDA availability must be verified on a GPU node.

### Verify PyTorch

Run this inside the intended job environment:

```bash
python - <<'PY'
import torch
print("torch", torch.__version__)
print("torch cuda", torch.version.cuda)
print("cuda available", torch.cuda.is_available())
if torch.cuda.is_available():
    print("device", torch.cuda.get_device_name(0))
PY
```

### Verify TensorFlow

```bash
python - <<'PY'
import tensorflow as tf
print("tensorflow", tf.__version__)
print("gpus", tf.config.list_physical_devices("GPU"))
PY
```

### GH200-Specific Notes

The local GH200 notes found this working setup:

```bash
spack load apptainer
spack env activate default-nlp-aarch64-26012701
```

That environment reported:

```text
Python 3.11.7
torch 2.1.1
torch.version.cuda == 12.3
torch.cuda.is_available() == True
```

Operational rule for GH200: load standalone Spack tools such as `apptainer` before activating the GH200 Spack environment, then avoid further `spack load` calls after `spack env activate ...`. The notes observed OpenSSL/library conflicts when running more Spack commands after entering that environment.

## 6. Apptainer and Containers

The recommended split for this project is:

- `Spack` for cluster-provided tools such as `apptainer` and possibly `cuda`
- `Conda` or `Miniforge` for host Python and model/runtime packages
- `Apptainer` for the controlled database/task container

Load Apptainer:

```bash
spack load apptainer
```

Validate an image before a long run:

```bash
apptainer exec /path/to/image.sif python3 -c "import pandas; print('ok')"
```

Only use GPU passthrough when the containerized process needs the GPU. For database/control-plane jobs, GPU passthrough is usually unnecessary.

## 7. Sync Project Changes with `rsync`

Use `rsync` to keep a local project directory and a cluster project directory synchronized. The main pattern is:

```bash
rsync [options] <local_project_dir>/ <cluster_alias>:<remote_project_dir>/
```

The trailing slash matters. `local_project_dir/` means "copy the contents of this directory", not the directory itself.

### Recommended Upload Pattern

Use a dry run first:

```bash
rsync -ani --delete \
  --exclude-from=<upload_exclude_file> \
  <local_project_dir>/ \
  <cluster_alias>:<remote_project_dir>/
```

Then run the actual upload:

```bash
rsync -a --delete \
  --exclude-from=<upload_exclude_file> \
  <local_project_dir>/ \
  <cluster_alias>:<remote_project_dir>/
```

Recommended meaning of the options:

- `-a` preserves the directory structure, permissions, timestamps, and symlinks.
- `-n` performs a dry run.
- `-i` itemizes changes, which makes the dry run easier to inspect.
- `--delete` makes the remote copy mirror the local copy.
- `--exclude-from=<file>` prevents caches, large outputs, local-only files, and generated artifacts from being uploaded.

Use `--delete` only for the upload direction when the cluster copy should mirror the local code. Always inspect the dry run before the real upload.

### Recommended Download Pattern

For downloading cluster-generated results, avoid `--delete` by default:

```bash
rsync -ani \
  --filter="merge <download_filter_file>" \
  <cluster_alias>:<remote_project_dir>/ \
  <local_project_dir>/
```

Then:

```bash
rsync -a \
  --filter="merge <download_filter_file>" \
  <cluster_alias>:<remote_project_dir>/ \
  <local_project_dir>/
```

This pattern lets you pull result files back without deleting local files unexpectedly.

### Suggested Project Pattern

Maintain two sync rule files in the project root:

- An upload exclude file, for files that should not be pushed to the cluster.
- A download filter file, for files that should be pulled back from the cluster.

Common upload exclusions include:

- `.git/`
- Python caches such as `__pycache__/`
- local virtual environments
- large local result folders
- temporary files
- logs that should stay local

Common download targets include:

- cluster result folders
- Slurm logs
- generated metrics
- generated figures or summaries

The script `sync_with_cluster.sh` in this repository is just a convenience wrapper around this pattern.

## 8. Recommended Batch-Script Environment Pattern

Use a clean shell, load cluster tools first, then reactivate Conda:

```bash
#!/usr/bin/env bash
set -euo pipefail

cd /home/<rit_username>/Projects/bird

spack load apptainer

source /home/<rit_username>/miniforge3/etc/profile.d/conda.sh
conda deactivate >/dev/null 2>&1 || true
conda deactivate >/dev/null 2>&1 || true
conda activate BIRD
hash -r

PYTHON_CMD="$CONDA_PREFIX/bin/python"
"$PYTHON_CMD" --version
```

For GPU jobs, validate CUDA inside the allocated GPU job:

```bash
"$PYTHON_CMD" - <<'PY'
import torch
print(torch.__version__)
print(torch.version.cuda)
print(torch.cuda.is_available())
PY
```

## 9. Troubleshooting Checklist

### SSH Alias Fails

Check:

```bash
ssh RITCluster
```

If this fails, inspect `~/.ssh/config` and confirm the `Host RITCluster` block.

### Wrong Python Is Active

Check:

```bash
which python
python --version
echo "$CONDA_PREFIX"
```

Fix by sourcing Conda and reactivating the environment after any Spack command.

### Torch Cannot See GPU

Verify that the command is running on a GPU node, not the login node:

```bash
nvidia-smi
python -c "import torch; print(torch.cuda.is_available())"
```

If CUDA is unavailable on a GPU node, check CUDA/PyTorch compatibility and any Spack CUDA module that was loaded.

### Apptainer Not Found

```bash
spack load apptainer
which apptainer
apptainer --version
```

If generic `spack load apptainer` is unstable, prefer the exact Spack hash once identified:

```bash
spack load /<spack_hash>
```

### Sync Would Delete Too Much

Run the dry-run upload first:

```bash
rsync -ani --delete \
  --exclude-from=<upload_exclude_file> \
  <local_project_dir>/ \
  <cluster_alias>:<remote_project_dir>/
```

Inspect the listed files carefully. If needed, add paths to the upload exclude file before running the real upload.

## Source Notes

This guide was derived from:

- `knowledge/RIT research computing/SSH Tutorial _ RIT Research Computing Documentation.pdf`
- `knowledge/RIT research computing/Software Tutorial _ RIT Research Computing Documentation.pdf`
- `knowledge/RIT research computing/Storage Tutorial _ RIT Research Computing Documentation.pdf`
- `knowledge/RIT research computing/Slurm Tutorial 1_ Getting Started _ RIT Research Computing Documentation.pdf`
- `knowledge/RIT research computing/GH200_SETUP_NOTES.md`
- `knowledge/RIT research computing/CLUSTER_MIGRATION_CHECKLIST.md`
- `sync_with_cluster.sh`

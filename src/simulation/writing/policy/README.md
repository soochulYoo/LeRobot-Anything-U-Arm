# policy — force-aware flow policies with a compliance action

The four policy structures of CoFA (Fig. 1) in Flax NNX, trained on the
stiffness-protocol demonstrations (`../demos/protocol_v1`) and evaluated
closed-loop in the writing simulator. Structure (d), cross-conditioning, is the
model of interest.

| case | name | what couples action and force |
|---|---|---|
| a | `ft_input` | current force is only an input to the action flow (baseline) |
| b | `uni_dir` | a future-force predictor, from the observation alone, feeds the action flow |
| c | `unified` | one flow over [action \| future force], one flow time |
| **d** | **`cross_cond`** | separate action and future-force flows with independent times; at every Euler step each sees the other's current state |

**The action includes stiffness.** One sample predicts a 1 s chunk at 10 Hz:

| | shape | content |
|---|---|---|
| action | (10, 6) | x_d offset from the current x_d (3, m) + log stiffness along the paper u, v and normal n (3) |
| future force | (10, 3) | contact force over the same 10 frames |
| observation | | top and wrist cameras (64×64); x_d, spring deflection, velocity, log K, qpos; the goal path (32 points, relative to x_d); the last 10 frames of contact force (GRU) |

At run time the chunk's first `n_exec` = 3 set-points (0.3 s) are executed,
then the policy samples again. The force history is the only place the paper's
hidden height and tilt show up, just as μ is hidden in the toy task.

## Files

| file | what it is |
|---|---|
| `data.py` | demos → samples, normalization, image shift augmentation |
| `model.py` | the four structures (a–d), `Config`, `build()` |
| `train.py` | training plus open-loop validation on held-out demos, in mm / N / level accuracy |
| `rollout.py` | closed-loop evaluation on unseen randomizations; success, force, and the stiffness the policy chose vs. the demos |
| `compare.py` | every run's open- and closed-loop numbers in one table, with a ranking |
| `slurm/train_{a,b,c,d}_*.sbatch` | one script per structure, 3 seeds as an array |
| `slurm/_train.sh` | their shared body |
| `slurm/eval_policy.sbatch` | closed-loop evaluation of one structure, 3 seeds |
| `slurm/compare.sbatch` | the comparison table, once the evaluations are in |
| `slurm/submit_all.sh` | the whole chain: train -> evaluate -> compare |
| `requirements.txt` | what the server environment needs on top of ManiSkill |

## On the server

**1. Environments -- two of them, and why.** `jax` 0.9.1 requires `numpy >= 2`,
and the ManiSkill venv pins `numpy` 1.26.4 / `scipy` 1.13.1; installing JAX
there would upgrade both under a working simulator. So:

| what | where | holds |
|---|---|---|
| training | `/scratch2/soochul/venvs/jaxgpu` | `jax[cuda12]` 0.9.1, flax 0.12.5, optax, h5py |
| evaluation | `/scratch2/soochul/venvs/writingsim` | the same JAX (CPU build) **plus** sapien / mani_skill, so the rollout has both |

Both are defaults in the Slurm scripts and both take a `VENV=` override. To
rebuild them:

```bash
uv venv --python 3.12 /scratch2/soochul/venvs/jaxgpu
VIRTUAL_ENV=/scratch2/soochul/venvs/jaxgpu uv pip install \
  "jax[cuda12]==0.9.1" "flax==0.12.5" "optax==0.2.6" h5py numpy

uv venv --python 3.12 /scratch2/soochul/venvs/writingsim
VIRTUAL_ENV=/scratch2/soochul/venvs/writingsim uv pip install "torch==2.13.0" \
  -e /scratch2/soochul/ManiSkill "sapien==3.0.3" "gymnasium==1.3.0" \
  "jax==0.9.1" "flax==0.12.5" "optax==0.2.6" h5py scipy matplotlib imageio transforms3d
```

**A gotcha worth knowing.** The cluster's `cuda/12.8` module puts
`/opt/ohpc/pub/apps/cuda/12.8/lib64` on `LD_LIBRARY_PATH` ahead of the CUDA
libraries pip installed beside `jaxlib`. JAX then fails to load cuSPARSE, prints
one warning, and trains on the CPU ten times slower. `_train.sh` strips those
entries; if you run `train.py` by hand, check the first log line says
`jax 0.9.1 on [CudaDevice(id=0)]`.

**2. Data.** `demos/` is gitignored, so an upload through git does not carry
it. Copy the 150 episodes (≈98 MB); the training cache is rebuilt on the
server in about 15 s:

```bash
rsync -av --include='*/' --include='*.h5' --include='attempts.jsonl' --exclude='*' \
  src/simulation/writing/demos/protocol_v1/  <server>:<repo>/src/simulation/writing/demos/protocol_v1/
```

`attempts.jsonl` is needed by the evaluation, which uses it to size episode
lengths.

**3. Train, evaluate and compare in one chain.** Submit from
`src/simulation/writing`, so that it becomes `SLURM_SUBMIT_DIR`:

```bash
cd src/simulation/writing
mkdir -p logs
bash policy/slurm/submit_all.sh                    # 12 training + 12 eval + 1 compare
sbatch policy/slurm/train_d_cross_cond.sbatch      # or one structure, seeds 0-2
```

`submit_all.sh` chains the jobs with Slurm dependencies:

```
train_<s>.sbatch (array 0-2)  --aftercorr-->  eval_policy.sbatch (array 0-2)
                                                     |
                                    all four --afterany--> compare.sbatch
```

`aftercorr` pairs the arrays task by task, so seed k is evaluated the moment
seed k has trained and a failed seed does not hold up the others; the final
compare job uses `afterany`, so a partial sweep still produces a table.
`NO_EVAL=1` submits training alone.

Overrides at submission: `STEPS=50000`, `DATA=...`, `EXTRA_ARGS="--no-vision"`,
`--array=0` for a single seed. Checkpoints go to
`policy/runs/<name>/seed<k>/model.pkl` (with `train_log.json`); logs go to
`logs/writing_<case>_<name>_<job>_<seed>.log`.

**4. Evaluate** (closed loop, 20 unseen randomizations per text and seed):

```bash
NAME=cross_cond sbatch policy/slurm/eval_policy.sbatch
NAME=ft_input   sbatch policy/slurm/eval_policy.sbatch
```

This writes `policy/runs/<name>/seed<k>/eval/results.json` and
`rollouts.png`. The GPU is for SAPIEN's cameras; the policy runs on CPU JAX
inside each of the 12 workers, so their GPU memory doesn't collide.

## What to expect

The validation line in the log is open loop, on the last 5 demos of each text:

```
[cross_cond]   2000  loss 0.6332  grad norm mean 1.767 max 2.226  41 ms/step
[cross_cond]  10000  loss 0.3076  pos1 0.50 mm  posH 3.39 mm  level acc xy 97.5% z 98.3%  ft 0.33 N  val 0.3081
```

The cheaper line every `--log-every` steps carries the **gradient norm** (mean
and max since the last line, before any clipping), which is where an unstable
structure shows itself first; `--clip N` turns on global-norm clipping if one
ever needs it. A non-finite loss or norm stops the run with a message rather
than filling the log with NaNs.

| field | meaning |
|---|---|
| `pos1` / `posH` | x_d error at the first / last step of the 1 s chunk |
| `level acc` | predicted stiffness rounded to the nearest protocol level (500 / 1000 / 3000 N/m), against the recorded level |
| `ft` | future contact-force error (not for `ft_input`) |
| `val` | the flow loss on the same held-out frames, beside the training loss: 135 training episodes is little data, so the gap is worth watching |

Open loop cannot rank these structures on its own -- a chunk can be close in
millimetres and still fail closed loop, most often by choosing a normal
stiffness that does not forgive the paper's unknown height and tilt. That is
what `compare.py` ranks on:

```bash
python3 policy/compare.py --out policy/runs/COMPARISON.md
```

The line above is structure (d) at step 10 000 of a run on a local CPU, which
was stopped there. It is a sanity reference for the server runs, not a result:
nothing has been evaluated closed-loop yet.

## Tested here

- All four structures train and save.
- `_train.sh` runs outside Slurm with the array index as seed, and the cache
  builds atomically.
- `eval_policy.sbatch` refuses a missing checkpoint.
- `rollout.py` completed a closed-loop pass with a 300-step checkpoint.

Not tested: the scripts under a real `sbatch`, and JAX on a GPU.

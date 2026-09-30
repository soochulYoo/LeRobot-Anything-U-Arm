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
| `slurm/train_{a,b,c,d}_*.sbatch` | one script per structure, 3 seeds as an array |
| `slurm/_train.sh` | their shared body |
| `slurm/eval_policy.sbatch` | closed-loop evaluation of one structure, 3 seeds |
| `slurm/submit_all.sh` | submits all four training scripts |
| `requirements.txt` | what the server environment needs on top of ManiSkill |

## On the server

**1. Environment.** The scripts activate `/scratch2/soochul/ManiSkill/.venv`
(override with `VENV=`). Training needs JAX with CUDA; evaluation also needs
the simulator, which that venv already has:

```bash
source /scratch2/soochul/ManiSkill/.venv/bin/activate
pip install -r src/simulation/writing/policy/requirements.txt
python -c "import jax; print(jax.devices())"      # should list a CUDA device
```

**2. Data.** `demos/` is gitignored, so an upload through git does not carry
it. Copy the 150 episodes (≈98 MB); the training cache is rebuilt on the
server in about 15 s:

```bash
rsync -av --include='*/' --include='*.h5' --include='attempts.jsonl' --exclude='*' \
  src/simulation/writing/demos/protocol_v1/  <server>:<repo>/src/simulation/writing/demos/protocol_v1/
```

`attempts.jsonl` is needed by the evaluation, which uses it to size episode
lengths.

**3. Train.** Submit from `src/simulation/writing`, so that it becomes
`SLURM_SUBMIT_DIR`:

```bash
cd src/simulation/writing
mkdir -p logs
sbatch policy/slurm/train_d_cross_cond.sbatch      # one structure, seeds 0-2
bash policy/slurm/submit_all.sh                     # or all four (12 jobs)
```

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
[cross_cond]  10000  loss 0.3076  pos1 0.50 mm  posH 3.39 mm  level acc xy 97.5% z 98.3%  ft 0.33 N
```

| field | meaning |
|---|---|
| `pos1` / `posH` | x_d error at the first / last step of the 1 s chunk |
| `level acc` | predicted stiffness rounded to the nearest protocol level (500 / 1000 / 3000 N/m), against the recorded level |
| `ft` | future contact-force error (not for `ft_input`) |

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

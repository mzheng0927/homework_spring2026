# Homework 1: Imitation Learning

## Goal of the homework

In Push-T, a circular agent must push a T-shaped block into a fixed goal region.
The policy observes

```text
[agent_x, agent_y, block_x, block_y, block_angle]
```

and outputs target positions `[target_agent_x, target_agent_y]` for the agent.
Positions use the environment's `0` to `512` coordinate system, and the block
angle is in radians.

This homework is **offline imitation learning (behavior cloning)**. Instead of
learning by trial and error from rewards, the policy learns to reproduce actions
from previously collected expert demonstrations:

```text
expert state -> policy -> predicted action chunk
                         compared with expert action chunk -> loss
```

Your main tasks are to implement the two policy choices in `model.py` and the
optimization loop in `train.py`:

- `MSEPolicy` directly predicts an action chunk and trains with mean squared
  error. It is the deterministic behavior-cloning baseline.
- `FlowMatchingPolicy` learns to transform noise into an action chunk. This can
  represent multiple plausible ways to act from a state instead of averaging
  them into one prediction.

During evaluation, the policy runs in the simulator: it predicts a chunk,
executes those actions, observes a new state, and repeats. The reward measures
how much of the T-shaped block overlaps the goal, with success at about 95%
coverage. Reward is therefore an evaluation signal in this homework, not the
training target. `eval/mean_reward` is the mean of the maximum reward reached in
each of 100 evaluation episodes.

## What the data looks like

The training script downloads the Push-T demonstrations to
`data/pusht/pusht_cchi_v7_replay.zarr`. Zarr is an on-disk collection of named
NumPy-like arrays. The arrays used by this homework are:

| Zarr array | Shape | Meaning |
| --- | ---: | --- |
| `data/state` | `(25,650, 5)` | Agent position `(x, y)` and T-block pose `(x, y, angle)` at each time step |
| `data/action` | `(25,650, 2)` | Expert target agent position `(x, y)` at the same time step |
| `meta/episode_ends` | `(206,)` | Exclusive cumulative end index of each demonstration episode |

All states and actions are `float32`; episode ends are `int64`. The archive also
contains images, block keypoints, and contact counts, but the provided state-based
loader does not use them.

The first aligned state/action pair is approximately:

```text
state  = [222.0, 97.0, 222.99, 381.60, 3.008]
action = [233.0, 71.0]
```

Training uses action chunking rather than predicting only one action. With the
default `chunk_size=8`, `PushtChunkDataset` turns the demonstrations into 24,208
sliding-window samples. One sample contains:

```text
state_t:       shape (5,)
action_chunk: shape (8, 2) = [action_t, ..., action_(t+7)]
```

A batch therefore has shapes `(batch_size, 5)` and `(batch_size, 8, 2)`. Windows
are stopped at episode boundaries, so an action chunk never mixes two different
demonstrations. The loader also normalizes every state and action feature using
the dataset mean and standard deviation; predicted actions are converted back to
the original coordinate system before simulator evaluation.

## Setup

This project uses `uv` for package management. `uv` is a Python package and environment manager from [Astral](https://astral.sh). It replaces tools like
`pip`, `pipx`, `conda`, and `virtualenv` with a single, simple interface. It is also much faster than prior tools.

### Installing `uv`

Run the following in your terminal:

```bash
curl -LsSf https://astral.sh/uv/install.sh | sh
```

After installation, open a new terminal so `uv` is on your `PATH`.

### Always use `uv run`

Do **not** run `python` or `pip` directly. Always run scripts through `uv run` so dependencies
and environments are handled automatically. If you want to add a new dependency, you can use `uv add`. This will add the dependency to `pyproject.toml`, update `uv.lock`, and install the package into your virtual environment.

Example:

```bash
uv run src/hw1_imitation/train.py --help
```

This should work out of the box with the provided starter code.

## Weights & Biases (wandb) login

These assignments use [Weights & Biases (WandB)](https://wandb.ai) for experiment tracking. WandB is a tool for logging and visualizing machine learning experiments. It is free for academic use. Before running a training script, you will need to log in to WandB using your API key.

```bash
uv run wandb login
```

Follow the prompt to paste your API key.

## Using Modal

**Note that Modal is likely not necessary for this assignment. In testing, training was much faster on a local laptop CPU than on Modal. However, you may need to use Modal in future assignments, so if you want to get set up, here are the instructions:**

First, create a Modal account. You should recieve $30 in free credits, which will be plenty for this assignment. Then, you can train on Modal with the following command:

```bash
uv run modal run src/hw1_imitation/modal_train.py
```

This will build a Modal container and launch training remotely. You can pass the same flags as the local training script. If you are logged into WandB locally, your API key will be automatically forwarded to the Modal container.

Logs and checkpoints will be saved to a Modal volume called `hw1-imitation-volume`. To inspect the logs, you can use:

```bash
uv run modal volume ls hw1-imitation-volume exp
```

Then, you can download the logs and checkpoints to your local machine using a command like the following:

```bash
uv run modal volume get hw1-imitation-volume exp/<experiment_name>
```

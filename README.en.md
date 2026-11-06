# LaMOO-BMAS

[中文](README.md) | [English](README.en.md)

**Batched Multi-objective Adaptive Search**

Provide measured parameter configurations and their metrics; the optimizer recommends the next batch to evaluate. Repeating **ask → evaluate → tell** updates the performance models and the measured Pareto frontier.

LaMOO-BMAS is independent of the underlying architecture. Your evaluator runs the experiments and computes metrics. The optimizer proposes configurations, allocates budgets, persists state, and identifies tradeoffs. Evaluations can use functions, batched matrices, job queues, or remote services.

The distribution is `lamoo-bmas`; the import is `pareto_search`. It depends on NumPy, SciPy, and scikit-learn and fits its search models on CPU.

## Installation

```sh
git clone https://github.com/SKYdayDUCK/LaMOO-BMAS.git
cd LaMOO-BMAS
python -m pip install .
```

Or install directly:

```sh
python -m pip install "git+https://github.com/SKYdayDUCK/LaMOO-BMAS.git"
```

## Inputs and outputs

| Interface | Input | Output |
|---|---|---|
| `Optimizer(...)` | Parameter bounds, objectives, scales, budget, shared seeds, optional history and priorities | Persistent optimizer |
| `ask(8)` | Requested batch size | Up to eight configurations with stable IDs |
| External evaluator | One configuration and one seed | Metrics in the frozen objective order |
| `tell(batch_id, rows)` | Configurations, per-seed metrics, stable batch ID | Saved observations and updated state |
| `frontier()` | Complete observations | Nondominated configurations on shared-seed means |
| `representative()` | Complete observations and optional priorities | One configuration with its full metric vector |
| `status()` | Search state | Allocation, budget use, observation count, and stopping reason |

All objectives are minimized. Negate a maximization objective, such as accuracy, before submitting it. Keep this transformation and the objective order fixed. `scale` specifies the magnitude of each metric for search normalization.

## Minimal integration

```python
from pareto_search import Optimizer

search = Optimizer(
    "results/search",
    space={
        "learning_rate": {"low": 0.001, "high": 1.0, "log": True},
        "window": {"low": 16, "high": 512, "integer": True},
    },
    objectives=["regret", "latency"],
    budget=64,
    batch_size=8,
    seeds=[42, 123],
    target=[0.0, 0.0],
    scale=[100.0, 50.0],
    context={"scenario": "v1"},
)

while points := search.ask(8):
    # Supply evaluate(params, seed), returning [regret, latency].
    # You can submit this batch to a matrix evaluator or job service.
    rows = [
        {
            "candidate": point,
            "objectives": search.objectives,
            "minimize": True,
            "per_seed": [
                {"seed": seed, "values": evaluate(point["params"], seed)}
                for seed in search.seeds
            ],
        }
        for point in points
    ]
    search.tell("batch:" + ":".join(p["id"] for p in points), rows)

print(search.frontier())
print(search.representative())
print(search.status())
```

`budget=64` counts new configurations. With two seeds per configuration, it permits up to 128 screening runs. `batch_size=8` caps each proposal batch. The last batch shrinks when the remaining budget or candidate pool is smaller. `run_budget` additionally limits configuration–seed runs.

Parameter spaces are Cartesian products of continuous linear, continuous logarithmic, and integer axes. Logarithmic bounds must be positive, and integer bounds must be integers. Supply at least two distinct seeds; their variation informs observation noise.

## Objective priorities and a joint representative

Version 0.2.0 adds optional objective levels, weights, and fixed references:

```python
search = Optimizer(
    "results/priority",
    space={"learning_rate": {"low": 0.001, "high": 1.0, "log": True}},
    objectives=["regret", "latency", "worst_regret", "p95_latency", "switch_rate"],
    budget=64,
    seeds=[42, 123],
    scale=[100, 50, 150, 90, 1],
    priority={
        "levels": [[0, 1], [2, 3], [4]],
        "weights": [0.70, 0.25, 0.05],
        "reference": [100, 50, 150, 90, 0.4],
        "reducers": ["max", "max", "mean"],
        "required_levels": 2,
    },
)
# After evaluations through the ask/tell loop:
print(search.representative())
```

`levels` lists groups in priority order. Indices refer to `objectives`, and each objective belongs to exactly one level. Nonnegative `weights` sum to one. `reducers` supports `max` and `mean`, defaulting to `max` for each level. `reference` contains fixed per-objective reference values. `required_levels` selects how many leading levels determine attainment tiers; it defaults to the first two. Set it to zero to rank directly by weighted score.

An objective passes when its mean is strictly below its reference. The example first ranks configurations into three tiers: both leading levels pass; only the first passes; the first has a failing objective. Within a tier, lower weighted score wins. For a shared-seed mean `f[j]`, compute `r[j] = f[j] / max(abs(reference[j]), 1e-12)` and `v[j] = r[j] / (1 + abs(r[j]))`. The example score is `0.70*max(v[0:2]) + 0.25*max(v[2:4]) + 0.05*v[4]`. The transformation preserves ordering for negative objectives too.

Priorities guide proposals as well as final selection. Joint ranks supply partition labels; the joint nondominated set on the required levels guides frontier-neighborhood filtering. Regional and global LCB slots use predicted tiers and weighted scores, while distance-based exploration slots remain active. `frontier()` continues to report nondominance across all objectives. `representative()` returns one complete configuration and all its metrics, plus `tier`, `score`, `levels_passed`, and weighted `contributions`. It returns `None` until a configuration has all shared seeds.

Priority settings are frozen in the protocol and must match on resume. Omitting `priority` retains the existing objective-group rotation and proposal rule. In that mode, `representative()` minimizes the maximum objective after normalization by `target` and `scale`.

## Two runnable examples

### Online policy: regret versus switching

[examples/bandit_tradeoff.py](examples/bandit_tradeoff.py) simulates four arms over 400 steps, changing the best arm every 100 steps. It searches learning rate and exploration probability, measuring cumulative dynamic pseudo-regret and action switching rate.

```sh
python examples/bandit_tradeoff.py --output results/bandit --budget 16
```

Environment and action sampling have separate random streams. Every configuration uses the same seeds. Replace `evaluate(params, seed)` to integrate your environment.

### Classification: validation error versus model size

[examples/model_tradeoff.py](examples/model_tradeoff.py) uses scikit-learn's bundled Wine dataset. It searches the number and maximum depth of random-forest trees, measuring validation error and the total number of fitted tree nodes.

```sh
python examples/model_tradeoff.py --output results/model --budget 16
```

Each seed fixes a training/validation split. The search uses validation metrics; an independent test set can evaluate the final selected configuration.

Both examples produce `frontier.json` and `observations.json`. The latter can initialize a later search. [examples/minimal.py](examples/minimal.py) demonstrates the interface with two simple functions.

## Cold starts, history, and recovery

Cold starts are the default. Initial proposals cover the parameter range; complete observations enable model fitting. Pass historical rows in the `tell` format through `history=rows` at construction. For example:

```sh
python examples/bandit_tradeoff.py --output results/bandit-next --budget 16 --history results/bandit/observations.json
```

- Historical parameters, objectives, context, and seeds should match the experiment. Configurations with all screening seeds train the models; extra seeds are retained. Historical configurations are tracked separately from the new-configuration budget.
- Resume with the same directory and constructor settings, including the same initial history.
- Parameter order, objectives, scales, budget, context, history hash, and optional priorities are frozen in `PROTOCOL.json`. Use a new directory for a new protocol.
- `ask` persists reservations. Until all shared seeds arrive, repeated calls return the pending batch. An external executor can reuse its completed evaluation receipts.
- `tell` is idempotent by batch ID and rolls back in-memory changes after validation or persistence failures.
- One controller owns each directory. Evaluation workers may run concurrently and return results to that controller.

## How proposals are generated

```mermaid
flowchart LR
    A[Parameter bounds and objectives] --> B[Cold start or history]
    B --> C[Normalization and per-objective models]
    C --> D[Partition and region selection]
    D --> E[Batch proposals]
    E --> F[External evaluation]
    F --> G[Per-seed metrics]
    G --> C
    G --> H[Measured Pareto frontier]
    H --> D
```

1. Normalize coordinates, taking logarithms for log-scaled axes. Transform objectives using the fixed target and scale.
2. After four complete configurations, fit one Gaussian process per objective, using shared-seed variation as observation noise.
3. Form dominance-count labels and learn RBF-SVM partitions. Sufficient observations enable recursive partitioning. Sampled hypervolume and exploration terms select a region; measured frontier neighborhoods refine regional candidates. With priorities enabled, labels instead follow the joint ranking.
4. Generate a Sobol candidate pool and remove observed or reserved equivalent points.
5. For a full eight-point batch, use up to four regional LCB points, two global objective-gap points, and two distance-exploration points. Distance penalties spread the batch. Priority mode ranks the first six slots by predicted tiers and weighted scores.
6. Update the measured frontier from shared-seed means. A configuration dominates another when it is no worse on all objectives and strictly better on at least one.

Without priorities, groups of at most three objectives rotate between batches, alongside all-objective selection and frontier tracking. For example, `groups=[[0,1,2],[1,3]]` covers four objectives. Hypervolume is estimated inside a sampling box frozen from the initial complete observations.

For example, configurations A `(regret, latency)=(20,10)` and B `(25,8)` both remain on the frontier; C `(28,12)` is dominated by A. Predictions guide evaluation locations, while the frontier consists of measured configurations.

## Multiple algorithms and stopping

Create one `Optimizer` per algorithm, each with its own parameter space, budget, and state. An executor can combine their `ask` batches and return results through separate `tell` calls. `method="grid"` provides one-axis grid proposals; `method="fixed"` maintains fixed observations.

`ask` returns an empty list when the configuration budget, run budget, or effective candidate pool is exhausted. A caller may also stop using an external validation rule. Statistical confirmation and final testing are managed by the evaluator.

## Method sources

LaMOO-BMAS draws on LaMOO's dominance-based partitioning, Pareto guidance, and region exploration. Its independent implementation uses global Gaussian processes, batched LCB proposals, and objective-group rotation. History initialization, budget reservations, shrinking final batches, and persistent recovery connect it to external experiments.

- [LaMOO repository](https://github.com/aoiang/LaMOO)
- [Multi-objective Optimization by Learning Space Partition — ICLR 2022](https://openreview.net/forum?id=FlwzVjfMryn)
- [Multi-Objective Neural Architecture Search by Learning Search Space Partitions — JMLR 2024](https://www.jmlr.org/papers/v25/23-1013.html)

## Tests and license

```sh
python -m pip install pytest
python -m pytest tests -q
```

Tests cover cold starts, history, objective dimensions, parameter types, seed completeness, idempotence, persistence rollback, recovery, budget shrinkage, and optional priority selection.

Released under the [MIT License](LICENSE).

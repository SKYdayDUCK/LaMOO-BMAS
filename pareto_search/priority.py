"""Frozen objective levels and weighted selection of complete configurations."""
import copy
import numpy as np


def validate_priority(policy, size):
    if policy is None:
        return None
    policy = copy.deepcopy(policy)
    allowed = {'levels', 'weights', 'reference', 'reducers', 'required_levels'}
    if not isinstance(policy, dict) or set(policy) - allowed:
        raise ValueError('Unknown objective priority settings')
    levels = policy.get('levels', [])
    if not levels or any(not isinstance(g, list) or not g for g in levels):
        raise ValueError('Priority levels must be nonempty index lists')
    indices = [j for group in levels for j in group]
    if any(type(j) is not int for j in indices) or sorted(indices) != list(range(size)):
        raise ValueError('Priority levels must cover every objective exactly once')
    weights = np.asarray(policy.get('weights', []), dtype=float)
    reference = np.asarray(policy.get('reference', []), dtype=float)
    if weights.shape != (len(levels),) or not np.isfinite(weights).all() or np.any(weights < 0) or not np.isclose(weights.sum(), 1.):
        raise ValueError('Priority weights must be nonnegative and sum to one')
    if reference.shape != (size,) or not np.isfinite(reference).all():
        raise ValueError('Provide one finite priority reference per objective')
    reducers = policy.get('reducers', ['max'] * len(levels))
    if len(reducers) != len(levels) or any(r not in {'max', 'mean'} for r in reducers):
        raise ValueError('Priority reducers must be max or mean, one per level')
    required = policy.get('required_levels', min(2, len(levels)))
    if type(required) is not int or not 0 <= required <= len(levels):
        raise ValueError('Invalid number of required priority levels')
    return dict(levels=levels, weights=weights.tolist(), reference=reference.tolist(),
                reducers=list(reducers), required_levels=required)


def components(values, policy):
    values = np.asarray(values, dtype=float)
    reference = np.asarray(policy['reference'])
    ratios = values / np.maximum(np.abs(reference), 1e-12)
    # Signed extension of r/(1+r), retaining order for negative objectives.
    bounded = ratios / (1. + np.abs(ratios))
    terms, passed = [], []
    for group, weight, reducer in zip(policy['levels'], policy['weights'], policy['reducers']):
        reduce = np.max if reducer == 'max' else np.mean
        terms.append(weight * reduce(bounded[:, group], axis=1))
        passed.append(np.all(values[:, group] < reference[group], axis=1))
    passed = np.asarray(passed).T
    required = policy['required_levels']
    tier = required - np.cumprod(passed[:, :required], axis=1).sum(axis=1)
    return dict(tier=tier, score=np.asarray(terms).sum(axis=0),
                passed=passed, contributions=np.asarray(terms).T, ratios=ratios)


def order(values, policy, penalty=None):
    stats = components(values, policy)
    score = stats['score'] if penalty is None else stats['score'] + penalty
    return np.lexsort((np.arange(len(score)), stats['ratios'].mean(axis=1), score, stats['tier']))

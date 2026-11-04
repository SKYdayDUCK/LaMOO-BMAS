"""Small, persistent ask/tell multi-objective search; all objectives minimized.

LaMOO-inspired engineering adaptation: dominance/SVM partitions and a global
GP with batched LCB selection. No experiment engine or GPU dependency.
"""
from __future__ import annotations
import copy
import math
from pathlib import Path
import numpy as np
from scipy.stats import qmc
from sklearn.gaussian_process import GaussianProcessRegressor
from sklearn.gaussian_process.kernels import Matern
from sklearn.svm import SVC
from .storage import stable_hash, read_json, write_json, freeze_json


def pareto_mask(y):
    y = np.asarray(y)
    return np.array([not np.any(np.all(y <= row, axis=1) & np.any(y < row, axis=1)) for row in y], dtype=bool)


class Optimizer:
    def __init__(self, root, *, space, objectives, budget=64, batch_size=8,
                 seeds=(0, 1), history=None, target=None, scale=None,
                 groups=None, context=None, name='default', method='lamoo',
                 run_budget=None, exit_reason=None):
        self.root = Path(root)
        self.space = copy.deepcopy(space)
        self.objectives = list(objectives)
        self.seeds = list(seeds)
        self.budget, self.batch_size = budget, batch_size
        self.method, self.name = method, name
        self.context = context
        self.run_budget, self.exit_reason = run_budget, exit_reason
        if method not in {'lamoo', 'grid', 'fixed'}:
            raise ValueError('Unknown search method')
        if type(budget) is not int or budget < 0 or type(batch_size) is not int or not 1 <= batch_size <= 8:
            raise ValueError('Invalid configuration budget or proposal batch')
        if run_budget is not None and (type(run_budget) is not int or run_budget < 0):
            raise ValueError('Invalid screening run budget')
        if len(self.seeds) < 2 or len(set(self.seeds)) != len(self.seeds) or any(type(s) is not int or s < 0 for s in self.seeds):
            raise ValueError('Provide at least two distinct nonnegative integer seeds')
        if not self.objectives or len(set(self.objectives)) != len(self.objectives) or not all(isinstance(o,str) and o for o in self.objectives):
            raise ValueError('Provide distinct objective names')
        for key, axis in self.space.items():
            lo, hi = axis['low'], axis['high']
            if not np.isfinite([lo,hi]).all() or not lo < hi or (axis.get('log',False) and lo <= 0):
                raise ValueError('Invalid search bounds: '+key)
            if axis.get('integer',False) and (int(lo) != lo or int(hi) != hi):
                raise ValueError('Integer axes require integer bounds')
        if not space or (method == 'grid' and len(space) != 1):
            raise ValueError('Provide a nonempty space; grid requires one axis')
        size = len(self.objectives)
        self.target = np.array([0.]*size if target is None else target,dtype=float)
        self.scale = np.array([1.]*size if scale is None else scale,dtype=float)
        if self.target.shape != (size,) or self.scale.shape != (size,) or not np.isfinite(self.target).all() or not np.isfinite(self.scale).all() or np.any(self.scale <= 0):
            raise ValueError('Invalid frozen objective scales')
        self.groups = [list(g) for g in (groups if groups is not None else [list(range(i,min(i+3,size))) for i in range(0,size,3)])]
        if not self.groups or any(not g or len(g)>3 or len(set(g)) != len(g) or any(type(j) is not int or not 0 <= j < size for j in g) for g in self.groups) or set(j for g in self.groups for j in g) != set(range(size)):
            raise ValueError('Objective groups must cover every objective, at most three per group')
        self.root.mkdir(parents=True,exist_ok=True)
        protocol = dict(space=self.space,axis_order=list(self.space), objectives=self.objectives,
            budget=budget,batch_size=batch_size,seeds=self.seeds,target=self.target.tolist(),
            scale=self.scale.tolist(),groups=self.groups,context=context,name=name,method=method,
            run_budget=run_budget,exit_reason=exit_reason,
            history_sha256=stable_hash(history or []),version='pareto-search-v1')
        freeze_json(self.root/'PROTOCOL.json',protocol)
        self.path = self.root/'STATE.json'
        if self.path.exists():
            self.state = read_json(self.path)
        else:
            self.state = dict(observations={},batches={},asked=0,reserved=[],rounds=[],status='active')
            self.tell('initial:'+stable_hash(history or []),history or [],historical=True)
        if 'hv_box' not in self.state and len(self.training()[0]):
            self.state['initial_progress'] = self.progress_snapshot()
            write_json(self.path,self.state)

    def coordinates(self, candidate):
        if set(candidate['params']) != set(self.space):
            raise ValueError('Candidate parameters differ from the search space')
        return np.array([float(candidate['params'][key]) for key in self.space])

    def key(self, candidate):
        return stable_hash([self.name,[format(v,'.12g') for v in self.coordinates(candidate)]])

    def decode(self, x):
        params = {}
        for unit,(name,axis) in zip(x,self.space.items()):
            lo,hi = axis['low'],axis['high']
            value = math.exp(math.log(lo)+float(unit)*math.log(hi/lo)) if axis.get('log',False) else lo+float(unit)*(hi-lo)
            params[name] = int(round(value)) if axis.get('integer',False) else value
        candidate = dict(params=params)
        candidate['id'] = self.key(candidate)[:16]
        return candidate

    def encode(self, candidate):
        values = self.coordinates(candidate)
        encoded = []
        for value,axis in zip(values,self.space.values()):
            lo,hi = axis['low'],axis['high']
            if not math.isfinite(value) or (axis.get('log',False) and value <= 0):
                raise ValueError('Invalid candidate coordinate')
            if axis.get('integer',False) and int(value) != value:
                raise ValueError('Noninteger candidate coordinate')
            encoded.append((math.log(value)-math.log(lo))/math.log(hi/lo) if axis.get('log',False) else (value-lo)/(hi-lo))
        return np.array(encoded)

    def _complete(self, candidate):
        row = self.state['observations'].get(self.key(candidate),{})
        return all(str(s) in row.get('per_seed',{}) for s in self.seeds)

    def tell(self, batch_id, rows, historical=False):
        """Persist a whole result batch atomically; repeated IDs are idempotent.

        rows: [{candidate, objectives, minimize: True, per_seed: [{seed,values}]}]
        Historical observations are supplied at construction, or explicitly
        imported with historical=True; they do not consume new-config budget.
        """
        if not isinstance(batch_id,str) or not batch_id:
            raise ValueError('Provide a nonempty batch ID')
        digest = stable_hash(rows)
        if batch_id in self.state['batches']:
            if self.state['batches'][batch_id] != digest:
                raise ValueError('Conflicting repeat of optimizer tell')
            return
        updated = copy.deepcopy(self.state)
        for row in rows:
            candidate = row['candidate']
            self.encode(candidate)
            if row['objectives'] != self.objectives or row['minimize'] is not True:
                raise ValueError('Objective contract differs')
            key = self.key(candidate)
            if not historical and key not in self.state['reserved'] and key not in self.state['observations']:
                raise ValueError('Unasked candidate; import history explicitly')
            old = updated['observations'].setdefault(key,dict(candidate=copy.deepcopy(candidate),per_seed={},historical=historical))
            if not row['per_seed']:
                raise ValueError('Empty observation')
            for sample in row['per_seed']:
                values = np.asarray(sample['values'],dtype=float)
                if type(sample['seed']) is not int or sample['seed'] < 0 or values.shape != (len(self.objectives),) or not np.isfinite(values).all():
                    raise ValueError('Invalid seed or objective values')
                seed = str(sample['seed'])
                if seed in old['per_seed'] and not np.allclose(old['per_seed'][seed],values,rtol=1e-12,atol=1e-12):
                    raise ValueError('Conflicting observation for an effective configuration/seed')
                old['per_seed'][seed] = values.tolist()
        previous = self.state
        self.state = updated
        try:
            self.state['batches'][batch_id] = digest
            if not historical and rows and len(self.training()[0]) and any(s['seed'] in self.seeds for r in rows for s in r['per_seed']):
                self.state.setdefault('progress',[]).append(self.progress_snapshot())
            write_json(self.path,self.state)
        except Exception:
            self.state = previous
            raise

    def frontier(self):
        rows,_,values = self.training()
        return [dict(candidate=copy.deepcopy(rows[i]['candidate']),values=values[i].tolist()) for i in np.flatnonzero(pareto_mask(values))]

    def status(self):
        n,reason = self.allocation()
        return dict(status=reason,next_batch=n,reserved=len(self.state['reserved']),
                    budget=self.budget,observed=len(self.training()[0]))

    def progress_snapshot(self):
        rows, x, raw = self.training()
        if not len(rows):
            raise ValueError("No complete paired observations")
        y = (raw-self.target)/self.scale
        if 'hv_box' not in self.state:
            self.state['hv_box'] = dict(lo=np.minimum(-1.,y.min(axis=0)-.2).tolist(),
                                       hi=np.maximum(1.,y.max(axis=0)+.2).tolist())
        box = self.state['hv_box']
        hv = []
        for columns in self.groups:
            cols = list(columns)
            lo, hi = np.array(box['lo'])[cols], np.array(box['hi'])[cols]
            grid = qmc.Sobol(len(cols),scramble=True,seed=0).random_base2(12)*(hi-lo)+lo
            hv.append(float(np.mean(np.any(np.all(y[:,None,cols] <= grid[None,:,:],axis=2),axis=0))))
        return dict(hv=hv, best=y.min(axis=0).tolist(), gap=float(np.min(np.max(y,axis=1))),
                    reserved=len(self.state['reserved']))

    def training(self):
        rows = [r for r in self.state['observations'].values()
                if all(str(seed) in r['per_seed'] for seed in self.seeds)]
        x = np.array([self.encode(r['candidate']) for r in rows]).reshape(-1,len(self.space))
        y = np.array([np.mean([r['per_seed'][str(s)] for s in self.seeds], axis=0) for r in rows]).reshape(-1,len(self.objectives))
        return rows, x, y

    def allocation(self):
        pending = self.state.get('pending')
        if pending and not all(self._complete(p) for p in pending):
            return len(pending), 'resume_pending'
        reason = self.exit_reason
        if reason:
            return 0, reason
        if self.method == 'fixed':
            return 0, 'fixed_reference'
        if self.state.get('space_exhausted'):
            return 0, 'candidate_space_exhausted'
        if self.state.get('plateau_status') == 'plateau_verified':
            return 0, 'plateau_verified'
        remaining = self.budget - len(self.state['reserved'])
        if remaining <= 0:
            return 0, 'budget_complete_screening'
        if self.run_budget is not None:
            runs_left = self.run_budget - len(self.state['reserved']) * len(self.seeds)
            remaining = min(remaining, runs_left // len(self.seeds))
            if remaining <= 0:
                return 0, 'run_budget_complete_screening'
        if self.method == 'grid':
            keys = {self.key(self.decode([v])) for v in np.linspace(0., 1., self.budget)}
            remaining = min(remaining, len(keys - set(self.state['observations']) - set(self.state['reserved'])))
            if not remaining:
                return 0, 'finite_grid_complete'
        return min(self.batch_size, remaining), 'active'

    def _regions(self, x, y, pool, grid):
        leaves = []
        def split(ids, slots, depth):
            if not len(slots):
                return
            if len(ids) >= 12 and depth < 3:
                yy = y[ids]
                counts = np.array([np.sum(np.all(yy <= row, axis=1) & np.any(yy < row, axis=1)) for row in yy])
                labels = counts > np.median(counts)
                if len(np.unique(labels)) == 2:
                    svm = SVC(kernel='rbf', gamma='scale', C=10).fit(x[ids], labels)
                    train_labels, pool_labels = svm.predict(x[ids]), svm.predict(pool[slots])
                    if min(np.sum(train_labels == k) for k in (False, True)) >= 3:
                        for k in (False, True):
                            split(ids[train_labels == k], slots[pool_labels == k], depth + 1)
                        return
            if len(slots):
                hv = np.mean(np.any(np.all(y[ids, None, :] <= grid[None, :, :], axis=2), axis=0))
                ucb = hv + .2 * np.sqrt(2 * np.log(max(2, len(x))) / len(ids))
                leaves.append((float(ucb), slots))
        split(np.arange(len(x)), np.arange(len(pool)), 0)
        return leaves

    def ask(self, n):
        previous = self.state
        self.state = copy.deepcopy(previous)
        try:
            return copy.deepcopy(self._ask(n))
        except Exception:
            self.state = previous
            raise

    def _ask(self, n):
        if type(n) is not int or not 1 <= n <= 8:
            raise ValueError('ask requires 1..8 candidate slots')
        # Repeated ask after a process crash returns the same pending proposals.
        pending = self.state.get('pending')
        if pending and not all(self._complete(p) for p in pending):
            if len(pending) > n:
                raise ValueError('Pending proposals exceed requested allocation')
            return pending
        allowed, reason = self.allocation()
        n = min(n, allowed)
        if not n:
            return []
        d = len(self.space)
        iteration = self.state['asked']
        seed = int(stable_hash([self.context, self.name, iteration])[:8], 16)
        rng = np.random.default_rng(seed)
        pool = qmc.Sobol(d, scramble=True, seed=seed).random_base2(11)
        if self.method == 'grid':
            if d != 1:
                raise ValueError('Grid mode currently requires one effective coordinate')
            pool = np.linspace(0., 1., self.budget).reshape(-1, 1)
        candidates, positions, seen = [], [], set(self.state['observations']) | set(self.state['reserved'])
        for point in pool:
            proposal = self.decode(point)
            key = self.key(proposal)
            if key not in seen:
                candidates.append(proposal); positions.append(self.encode(proposal)); seen.add(key)
        n = min(n, len(candidates))
        if not n:
            self.state['space_exhausted'] = True
            write_json(self.path,self.state)
            return []
        pool = np.array(positions)
        rows, x, raw_y = self.training()
        columns = list(self.groups[iteration % len(self.groups)])
        selected, routes = [], []
        if self.method == 'grid' or len(rows) < 4:
            # Cover the range first, then fill the largest remaining gaps.
            for _ in range(n):
                reference = np.vstack([x, pool[selected]]) if len(x) else pool[selected]
                dist = np.min(np.linalg.norm(pool[:, None] - reference[None, :], axis=2), axis=1) if len(reference) else np.ones(len(pool))
                dist[selected] = -np.inf
                selected.append(int(np.argmax(dist))); routes.append('grid' if self.method == 'grid' else 'global_cold_start')
        else:
            y = (raw_y - self.target) / self.scale
            # Fixed CPU GP; per-seed variance is retained as observation noise.
            variance = np.array([np.var([r['per_seed'][str(s)] for s in self.seeds], axis=0, ddof=1) / len(self.seeds) for r in rows]) / self.scale**2
            mu, sigma = [], []
            for j in range(len(self.objectives)):
                response_scale = max(float(np.std(y[:, j])), 1e-6)
                gp = GaussianProcessRegressor(kernel=Matern(length_scale=.35, nu=2.5),
                    optimizer=None, normalize_y=True,
                    alpha=np.maximum(variance[:, j] / response_scale**2, 1e-6))
                gp.fit(x, y[:, j])
                m, s = gp.predict(pool, return_std=True); mu.append(m); sigma.append(s)
            lcb = np.array(mu).T - np.array(sigma).T
            # Frozen normalized box defined by initial history, not each champion.
            initial = self.state.get('hv_box')
            if initial is None:
                initial = dict(lo=np.minimum(-1., y.min(axis=0)-.2).tolist(),
                               hi=np.maximum(1., y.max(axis=0)+.2).tolist())
                self.state['hv_box'] = initial
            lo, hi = np.array(initial['lo'])[columns], np.array(initial['hi'])[columns]
            grid = qmc.Sobol(len(columns), scramble=True, seed=0).random_base2(10) * (hi-lo) + lo
            leaves = self._regions(x, y[:, columns], pool, grid)
            region_slots = max(leaves, key=lambda r:r[0])[1] if leaves else np.arange(len(pool))
            # An SVM extrapolates indefinitely. Regional exploitation stays
            # near measured projected-front points; two global slots explore
            # the rest of the domain explicitly instead of mislabelling it.
            front_x = x[pareto_mask(y[:, columns])]
            nearby = np.min(np.linalg.norm(pool[:, None] - front_x[None, :], axis=2), axis=1) <= .25
            local_slots = region_slots[nearby[region_slots]]
            if len(local_slots) >= min(4, n):
                region_slots = local_slots
            for slot in range(n):
                if slot < min(4, n):
                    weights = rng.dirichlet(np.ones(len(columns)))
                    score = np.max(weights * lcb[:, columns], axis=1)
                    allowed_slots = region_slots; route = 'partition_lcb'
                elif slot < min(6, n):
                    score = np.max(lcb, axis=1); allowed_slots = np.arange(len(pool)); route = 'all_objective_gap'
                else:
                    reference = np.vstack([x, pool[selected]])
                    score = -np.min(np.linalg.norm(pool[:, None] - reference[None, :], axis=2), axis=1)
                    allowed_slots = np.arange(len(pool)); route = 'global_exploration'
                if selected:
                    distance = np.min(np.linalg.norm(pool[:, None] - pool[selected][None, :], axis=2), axis=1)
                    score = score + .2 * np.exp(-distance**2 / .01)
                valid = [i for i in allowed_slots if i not in selected]
                if not valid:
                    valid = [i for i in range(len(pool)) if i not in selected]
                index = min(valid, key=lambda i: score[i])
                selected.append(index); routes.append(route)
        proposals = [candidates[i] for i in selected]
        self.state['asked'] += 1
        self.state['reserved'].extend(self.key(p) for p in proposals)
        self.state['pending'] = proposals
        self.state['rounds'].append(dict(iteration=iteration, objective_group=columns, routes=routes,
            candidates=[p['id'] for p in proposals], known_configurations=len(rows)))
        write_json(self.path, self.state)
        return proposals

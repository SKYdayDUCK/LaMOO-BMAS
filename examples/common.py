"""Shared CLI and external evaluation loop for runnable examples."""
import argparse
import json
from pathlib import Path
from pareto_search import Optimizer


def run_example(space, objectives, scale, evaluate, scene):
    parser = argparse.ArgumentParser()
    parser.add_argument('--output', type=Path, required=True)
    parser.add_argument('--budget', type=int, default=16)
    parser.add_argument('--history', type=Path)
    args = parser.parse_args()
    history = json.loads(args.history.read_text(encoding='utf-8')) if args.history else None
    search = Optimizer(args.output, space=space, objectives=objectives, scale=scale,
        budget=args.budget, seeds=[42, 123], history=history, context={'example': scene})
    while points := search.ask(8):
        # This is the boundary to replace with a job service or matrix executor.
        rows = [dict(candidate=p, objectives=objectives, minimize=True,
            per_seed=[dict(seed=s, values=evaluate(p['params'], s)) for s in search.seeds]) for p in points]
        search.tell('batch:' + ':'.join(p['id'] for p in points), rows)
    result = dict(objectives=objectives, status=search.status(), frontier=search.frontier())
    (args.output/'frontier.json').write_text(json.dumps(result,indent=2),encoding='utf-8')
    history = [dict(candidate=row['candidate'], objectives=objectives, minimize=True,
        per_seed=[dict(seed=int(seed),values=values) for seed,values in row['per_seed'].items()])
        for row in search.state['observations'].values()]
    (args.output/'observations.json').write_text(json.dumps(history,indent=2),encoding='utf-8')
    print(json.dumps(result,indent=2))

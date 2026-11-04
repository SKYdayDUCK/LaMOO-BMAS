"""Run after installing the package with `pip install .`.

Replace evaluate() with an external experiment, simulator or job service.
"""
import argparse
import json
from pathlib import Path
from pareto_search import Optimizer


def evaluate(params,seed):
    # Deterministic toy functions demonstrate integration.
    x = params['x']
    return [x*x,(x-1.)**2]


def main():
    p = argparse.ArgumentParser()
    p.add_argument('--output',type=Path,required=True)
    p.add_argument('--history',type=Path,help='Optional JSON observation list; omitted means cold start')
    args = p.parse_args()
    history = json.loads(args.history.read_text(encoding='utf-8')) if args.history else None
    search = Optimizer(args.output,space={'x':dict(low=-1.,high=2.)},
                       objectives=['loss','cost'],budget=16,seeds=[42,123],history=history)
    while points := search.ask(8):
        rows = [dict(candidate=p,objectives=search.objectives,minimize=True,
            per_seed=[dict(seed=s,values=evaluate(p['params'],s)) for s in search.seeds]) for p in points]
        # Stable key across restart; same pending batch is returned by ask().
        search.tell('batch:'+':'.join(p['id'] for p in points),rows)
    print(json.dumps(dict(status=search.status(),frontier=search.frontier()),indent=2))


if __name__ == '__main__':
    main()

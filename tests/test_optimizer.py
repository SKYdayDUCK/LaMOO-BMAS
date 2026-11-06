import copy
import json
import subprocess
import sys
from pathlib import Path

import numpy as np
import pytest

sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
from pareto_search import Optimizer


def config():
    return dict(space={'rate':dict(low=.01,high=1.,log=True),
                       'steps':dict(low=2,high=8,integer=True)},
                objectives=['loss','cost'],seeds=[42,123],budget=10)


def result(points,seeds=(42,123)):
    return [dict(candidate=p,objectives=['loss','cost'],minimize=True,
        per_seed=[dict(seed=s,values=[p['params']['rate']+s/10000.,float(p['params']['steps'])]) for s in seeds]) for p in points]


def test_cold_start_generic_objectives_resume_and_shrink(tmp_path):
    s = Optimizer(tmp_path,**config())
    assert s.training()[0] == []
    points = s.ask(8)
    assert len(points) == 8
    # Partial seed upload must not allow the next batch to silently advance.
    s.tell('first-seed',result(points,(42,)))
    assert s.ask(8) == points
    recovered = Optimizer(tmp_path,**config())
    recovered.tell('second-seed',result(points,(123,)))
    assert len(recovered.frontier()) > 0
    remaining = recovered.ask(8)  # exercises GP with two objectives
    assert len(remaining) == 2
    recovered.tell('tail',result(remaining))
    assert recovered.ask(8) == []
    assert recovered.status()['status'] == 'budget_complete_screening'


def test_optional_history_and_frozen_resume(tmp_path):
    empty = Optimizer(tmp_path/'source',**config())
    history = result(empty.ask(4))
    warm = Optimizer(tmp_path/'warm',**config(),history=history)
    assert warm.status()['observed'] == 4
    assert warm.status()['reserved'] == 0
    assert len(warm.ask(8)) == 8
    with pytest.raises(ValueError,match='Frozen'):
        Optimizer(tmp_path/'warm',**config())


def test_rejected_tell_is_atomic_and_retries_are_idempotent(tmp_path):
    s = Optimizer(tmp_path,**config())
    points = s.ask(2)
    valid = result(points)
    invalid = copy.deepcopy(valid)
    invalid[1]['per_seed'][1]['values'] = [1.]
    before = copy.deepcopy(s.state)
    with pytest.raises(ValueError,match='values'):
        s.tell('batch',invalid)
    assert s.state == before
    assert json.loads(s.path.read_text(encoding='utf-8')) == before
    s.tell('batch',valid)
    after = copy.deepcopy(s.state)
    s.tell('batch',valid)
    assert s.state == after
    with pytest.raises(ValueError,match='Conflicting'):
        s.tell('batch',result(points,(42,)))


def test_failed_ask_write_rolls_back_and_retry_persists(tmp_path,monkeypatch):
    import pareto_search.core as core
    s = Optimizer(tmp_path,**config())
    before = copy.deepcopy(s.state)
    write = core.write_json
    def fail(*args):
        raise OSError('disk full')
    monkeypatch.setattr(core,'write_json',fail)
    with pytest.raises(OSError,match='disk full'):
        s.ask(8)
    assert s.state == before
    assert json.loads(s.path.read_text(encoding='utf-8')) == before
    monkeypatch.setattr(core,'write_json',write)
    points = s.ask(8)
    recovered = Optimizer(tmp_path,**config())
    assert recovered.ask(8) == points


def test_returned_objects_do_not_mutate_optimizer_state(tmp_path):
    s = Optimizer(tmp_path,**config())
    points = s.ask(2)
    original = copy.deepcopy(points)
    points[0]['params']['rate'] = 999.
    assert s.ask(2) == original
    s.tell('batch',result(original))
    front = s.frontier()
    before = copy.deepcopy(s.state)
    front[0]['candidate']['params']['rate'] = 999.
    assert s.state == before


@pytest.mark.parametrize('count',[1,4])
def test_objective_dimension_and_linear_negative_axis(tmp_path,count):
    s = Optimizer(tmp_path,space={'x':dict(low=-2.,high=2.)},
                  objectives=[str(j) for j in range(count)],budget=9)
    points = s.ask(8)
    s.tell('batch',[dict(candidate=p,objectives=s.objectives,minimize=True,
        per_seed=[dict(seed=seed,values=[(p['params']['x']-j/2)**2 for j in range(count)]) for seed in s.seeds]) for p in points])
    assert np.isfinite(s.ask(8)[0]['params']['x'])


def test_standalone_import_has_no_campaign_or_torch_dependency(tmp_path):
    package = str(Path(__file__).resolve().parents[1])
    code = 'import sys; sys.path.insert(0,'+repr(package)+'); import pareto_search; assert "torch" not in sys.modules'
    subprocess.run([sys.executable,'-I','-c',code],cwd=tmp_path,check=True,capture_output=True)


def test_priority_selects_one_configuration_and_guides_batches(tmp_path):
    policy = dict(levels=[[0,1],[2,3]],weights=[.7,.3],reference=[100.,50.,100.,80.])
    options = dict(space={'x':dict(low=0.,high=1.)}, objectives=['loss','latency','tail','p95'],
                   priority=policy,budget=5,batch_size=4)
    source = Optimizer(tmp_path/'source',**options)
    points = source.ask(4)
    values = [[90,60,70,60],[95,45,110,70],[99,49,99,79],[80,40,90,75]]
    rows = [dict(candidate=p,objectives=source.objectives,minimize=True,
                 per_seed=[dict(seed=s,values=v) for s in source.seeds]) for p,v in zip(points,values)]
    source.tell('first',rows)
    champion = source.representative()
    assert champion['candidate'] == points[3]
    assert champion['values'] == values[3]
    assert champion['tier'] == 0 and champion['levels_passed'] == [True,True]
    expected = .7*(.8/1.8)+.3*((75/80)/(1+75/80))
    assert champion['score'] == pytest.approx(expected)
    champion['values'][0] = -1000
    assert source.representative()['values'] == values[3]
    # History supplies observations while preserving the new-config budget.
    warm = Optimizer(tmp_path/'warm',**options,history=rows)
    proposed = warm.ask(4)
    assert len(proposed) == 4
    assert warm.state['rounds'][-1]['routes'] == [
        'priority_partition_lcb','priority_partition_lcb','priority_global_lcb','global_exploration']
    restored = Optimizer(tmp_path/'warm',**options,history=rows)
    assert restored.ask(4) == proposed
    assert restored.representative()['candidate'] == points[3]
    changed = copy.deepcopy(options)
    changed['priority']['weights'] = [.8,.2]
    with pytest.raises(ValueError,match='Frozen'):
        Optimizer(tmp_path/'warm',**changed,history=rows)


def test_priority_tiers_precede_score_and_negative_values_keep_order(tmp_path):
    s = Optimizer(tmp_path,space={'x':dict(low=0.,high=1.)},objectives=['quality','cost'],
        priority=dict(levels=[[0],[1]],weights=[.7,.3],reference=[-.7,50.]))
    points = s.ask(4)
    values = [[-.99,100.],[-.8,40.],[-.9,40.],[-.6,1.]]
    s.tell('batch',[dict(candidate=p,objectives=s.objectives,minimize=True,
        per_seed=[dict(seed=seed,values=v) for seed in s.seeds]) for p,v in zip(points,values)])
    assert s.representative()['values'] == [-.9,40.]


@pytest.mark.parametrize('change',[
    {'levels':[[0],[0]]}, {'weights':[.7,.7]}, {'reference':[1.]},
    {'reducers':['sum','mean']}, {'required_levels':3}])
def test_invalid_priority_is_rejected(tmp_path,change):
    priority = dict(levels=[[0],[1]],weights=[.7,.3],reference=[1.,1.])
    priority.update(change)
    with pytest.raises(ValueError):
        Optimizer(tmp_path,**config(),priority=priority)


def test_default_protocol_remains_compatible(tmp_path):
    s = Optimizer(tmp_path,**config())
    assert 'priority' not in json.loads((tmp_path/'PROTOCOL.json').read_text(encoding='utf-8'))
    assert s.representative() is None


def test_priority_weights_change_model_guided_proposals(tmp_path, monkeypatch):
    import pareto_search.core as core

    class TradeoffGP:
        def __init__(self, **kwargs):
            pass

        def fit(self, x, y):
            self.increasing = y[-1] > y[0]
            return self

        def predict(self, x, return_std=False):
            mean = x[:, 0] if self.increasing else 1. - x[:, 0]
            return mean, np.zeros(len(x))

    monkeypatch.setattr(core, 'GaussianProcessRegressor', TradeoffGP)
    options = dict(space={'x': dict(low=0., high=1.)}, objectives=['loss', 'cost'],
                   seeds=[0, 1], budget=4, batch_size=4)
    history = [dict(candidate=dict(params={'x': x}), objectives=options['objectives'],
                    minimize=True, per_seed=[dict(seed=s, values=[x, 1.-x]) for s in options['seeds']])
               for x in (0., .33, .67, 1.)]
    proposed = []
    for name, weights in [('loss', [.9, .1]), ('cost', [.1, .9])]:
        search = Optimizer(tmp_path/name, **options, history=history,
            priority=dict(levels=[[0], [1]], weights=weights, reference=[1., 1.], required_levels=0))
        proposed.append(search.ask(4)[0]['params']['x'])
    assert proposed[0] < .05
    assert proposed[1] > .95

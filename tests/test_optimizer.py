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

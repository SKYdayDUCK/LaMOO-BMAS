"""Tune a small EXP3-style policy for regret and action switching.

Self-contained synthetic illustration of an external evaluation function.
"""
import numpy as np
from common import run_example


def evaluate(params, seed):
    arms, steps = 4, 400
    environment = np.random.default_rng(seed)
    actions = np.random.default_rng(seed+10000)
    means = np.full((steps,arms),.3)
    for phase in range(4):
        means[phase*100:(phase+1)*100,phase] = .8
    rewards = environment.random((steps,arms)) < means
    log_weights = np.zeros(arms)
    regret, switches, previous = 0., 0, None
    for t in range(steps):
        weights = np.exp(log_weights-log_weights.max())
        p = (1-params['exploration'])*weights/weights.sum()+params['exploration']/arms
        action = actions.choice(arms,p=p)
        loss = 1.-float(rewards[t,action])
        log_weights[action] -= params['learning_rate']*loss/p[action]
        regret += means[t].max()-means[t,action]
        switches += int(previous is not None and previous != action)
        previous = action
    return [float(regret),float(switches/(steps-1))]


if __name__ == '__main__':
    run_example(
        space={'learning_rate':dict(low=.005,high=.5,log=True),
               'exploration':dict(low=.005,high=.3,log=True)},
        objectives=['cumulative_pseudo_regret','switch_rate'],scale=[200.,1.],
        evaluate=evaluate,scene='rotating-four-arm-demo-v1')

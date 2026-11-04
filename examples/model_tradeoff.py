"""Tune validation error versus fitted model size, with no dataset download."""
from sklearn.datasets import load_wine
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from common import run_example


def evaluate(params, seed):
    x,y = load_wine(return_X_y=True)
    train_x,valid_x,train_y,valid_y = train_test_split(x,y,test_size=.3,stratify=y,random_state=seed)
    model = RandomForestClassifier(n_estimators=params['trees'],max_depth=params['depth'],
                                   random_state=seed,n_jobs=1)
    model.fit(train_x,train_y)
    error = 1.-model.score(valid_x,valid_y)
    size = sum(tree.tree_.node_count for tree in model.estimators_)
    return [float(error),float(size)]


if __name__ == '__main__':
    run_example(space={'trees':dict(low=8,high=48,integer=True),
                       'depth':dict(low=2,high=10,integer=True)},
        objectives=['validation_error','tree_nodes'],scale=[1.,1000.],
        evaluate=evaluate,scene='sklearn-wine-validation-demo-v1')

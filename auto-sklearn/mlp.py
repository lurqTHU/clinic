from ConfigSpace.configuration_space import ConfigurationSpace
from ConfigSpace.hyperparameters import CategoricalHyperparameter, \
    UniformIntegerHyperparameter, UniformFloatHyperparameter

import sklearn.metrics
import autosklearn.classification
import autosklearn.pipeline.components.classification
from autosklearn.pipeline.components.base \
    import AutoSklearnClassificationAlgorithm
from autosklearn.pipeline.constants import DENSE, SIGNED_DATA, UNSIGNED_DATA, \
    PREDICTIONS

import argparse
import numpy as np

import sys
sys.path.append('../')
from dataset import partition_dataset
import os
from tools import calculate_95CI, analyze_roc


# Create MLP classifier component for auto-sklearn.
class MLPClassifier(AutoSklearnClassificationAlgorithm):
    def __init__(self, hidden_layer_depth, num_nodes_per_layer,
                 activation, alpha, solver, max_iter,
                 learning_rate, learning_rate_init, 
                 random_state=None,
                 ):
        self.hidden_layer_depth = hidden_layer_depth
        self.num_nodes_per_layer = num_nodes_per_layer
        self.activation = activation
        self.alpha = alpha
        self.solver = solver
        self.max_iter = max_iter
        self.learning_rate = learning_rate
        self.learning_rate_init = learning_rate_init
        self.random_state = random_state

    def fit(self, X, y):
        self.num_nodes_per_layer = int(self.num_nodes_per_layer)
        self.hidden_layer_depth = int(self.hidden_layer_depth)
        self.alpha = float(self.alpha)
        self.max_iter = int(self.max_iter)
        self.learning_rate_init = float(self.learning_rate_init)


        print('Using hyperparameters:', self.hidden_layer_depth,
              self.num_nodes_per_layer, self.alpha, self.solver,
              self.activation, self.max_iter, self.learning_rate,
              self.learning_rate_init, self.random_state)

        from sklearn.neural_network import MLPClassifier
        hidden_layer_sizes = tuple(self.num_nodes_per_layer \
                                   for i in range(self.hidden_layer_depth))

        self.estimator = MLPClassifier(
                            hidden_layer_sizes=hidden_layer_sizes,
                            activation=self.activation,
                            alpha=self.alpha,
                            solver=self.solver,
                            max_iter=self.max_iter,
                            learning_rate=self.learning_rate,
                            learning_rate_init=self.learning_rate_init,
                            random_state=self.random_state)
        self.estimator.fit(X, y)
        return self

    def predict(self, X):
        if self.estimator is None:
            raise NotImplementedError()
        return self.estimator.predict(X)

    def predict_proba(self, X):
        if self.estimator is None:
            raise NotImplementedError()
        return self.estimator.predict_proba(X)

    @staticmethod
    def get_properties(dataset_properties=None):
        return {'shortname':'MLP Classifier',
                'name': 'MLP CLassifier',
                'handles_regression': False,
                'handles_classification': True,
                'handles_multiclass': True,
                'handles_multilabel': False,
                'is_deterministic': False,
                # Both input and output must be tuple(iterable)
                'input': [DENSE, SIGNED_DATA, UNSIGNED_DATA],
                'output': [PREDICTIONS]
                }

    @staticmethod
    def get_hyperparameter_search_space(dataset_properties=None):
        cs = ConfigurationSpace()
        hidden_layer_depth = UniformIntegerHyperparameter(
            name="hidden_layer_depth", 
            lower=1, upper=3, default_value=1
        )
        num_nodes_per_layer = UniformIntegerHyperparameter(
            name="num_nodes_per_layer", 
            lower=2, upper=16, default_value=8
        )
        activation = CategoricalHyperparameter(
            name="activation", 
            choices=['logistic', 'tanh', 'relu'],
            default_value='relu'
        )
        alpha = UniformFloatHyperparameter(
            name="alpha", lower=0.0001, 
            upper=0.2, default_value=0.0001
        )
        solver = CategoricalHyperparameter(
            name="solver", 
            choices=['sgd', 'adam'], 
            default_value='sgd'
        )
        max_iter = CategoricalHyperparameter(
            name="max_iter", 
            choices=[200, 300, 400, 500],
            default_value=500
        )
        learning_rate = CategoricalHyperparameter(
            name="learning_rate", 
            choices=['constant', 'invscaling', 'adaptive'],
            default_value='constant'
        )
        learning_rate_init = CategoricalHyperparameter(
            name="learning_rate_init", 
            choices=[0.1, 0.01, 0.001, 0.0001], 
            default_value=0.01
        )
   
        cs.add_hyperparameters(
            [hidden_layer_depth, num_nodes_per_layer,
             activation, alpha, solver, max_iter,
             learning_rate, learning_rate_init])
        return cs


def main(args):
    os.environ['OMP_NUM_THREADS']='1'
    # Add MLP classifier component to auto-sklearn.
    autosklearn.pipeline.components.classification.add_classifier(MLPClassifier)
    cs = MLPClassifier.get_hyperparameter_search_space()
    print(cs)

    data_path = '../dataset/update.xlsx'
    feats, targets,\
      masks = partition_dataset(data_path, target_name=args.target_name, 
                                use_icon=args.use_icon, 
                                ratio=0.8, seed=args.seed)
    
    X_trainval, X_train, X_val, X_test =\
        feats[masks[0]], feats[masks[1]], feats[masks[2]], feats[masks[3]] 
    y_trainval, y_train, y_val, y_test =\
        targets[masks[0]].squeeze(), targets[masks[1]].squeeze(),\
        targets[masks[2]].squeeze(), targets[masks[3]].squeeze()

    # Fit MLP classifier to the data.
    clf = autosklearn.classification.AutoSklearnClassifier(
        time_left_for_this_task=1800,
        per_run_time_limit=30,
        include_estimators=['MLPClassifier'],# 'MLPClassifier''libsvm_svc', 'random_forest'],
        include_preprocessors=['no_preprocessing'],
        ensemble_size=1,
        ensemble_nbest=1,
        ensemble_memory_limit=4096,
        ml_memory_limit=4096,
        initial_configurations_via_metalearning=0,
        resampling_strategy='cv',
        resampling_strategy_arguments={'folds': 4,
                                       'shuffle': True},
    )

    clf.fit(X=X_trainval.copy(), y=y_trainval.copy().squeeze())
   
    all_train_acc = []
    all_val_acc = []
    all_test_acc = []
    for i in range(100):
        clf.refit(X=X_trainval.copy(), y=y_trainval.copy().squeeze())   
        all_train_acc.append(sklearn.metrics.accuracy_score(clf.predict(X_train), y_train))
        all_val_acc.append(sklearn.metrics.accuracy_score(clf.predict(X_val), y_val))
        all_test_acc.append(sklearn.metrics.accuracy_score(clf.predict(X_test), y_test))
   
    print('Train Acc:', calculate_95CI(np.array(all_train_acc))) 
    print('Val Acc:', calculate_95CI(np.array(all_val_acc)))
    print('Test Acc:', calculate_95CI(np.array(all_test_acc)))
    print(clf.sprint_statistics())
    print(clf.show_models())


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Auto Sklearn')
    parser.add_argument('--target_name', choices=['vas', 'sas', 'qol'], 
                        default='vas', type=str)
    parser.add_argument('--use_icon', default=True, type=bool)
    parser.add_argument('--seed', default=0, type=int)
    args = parser.parse_args()
    main(args) 

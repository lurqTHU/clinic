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
            choices=['identity', 'logistic', 'tanh', 'relu'],
            default_value='relu'
        )
        alpha = UniformFloatHyperparameter(
            name="alpha", lower=0.0001, 
            upper=0.2, default_value=0.0001
        )
        solver = CategoricalHyperparameter(
            name="solver", 
            choices=['lbfgs', 'sgd', 'adam'], 
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
    # Add MLP classifier component to auto-sklearn.
    autosklearn.pipeline.components.classification.add_classifier(MLPClassifier)
    cs = MLPClassifier.get_hyperparameter_search_space()
    print(cs)

    data_path = '../dataset/update.xlsx'
    feats, targets,\
      masks = partition_dataset(data_path, target_name=args.target_name, 
                                use_icon=args.use_icon, 
                                ratio=0.8, seed=args.seed)
    
    X_train, X_test = feats[masks[0]], feats[masks[3]] 
    y_train, y_test = targets[masks[0]].squeeze(),\
                      targets[masks[3]].squeeze()

    # Fit MLP classifier to the data.
    clf = autosklearn.classification.AutoSklearnClassifier(
        time_left_for_this_task=1800,
        per_run_time_limit=10,
        include_estimators=['MLPClassifier'],
        include_preprocessors=['no_preprocessing'],
        ensemble_size=1,
        initial_configurations_via_metalearning=0
    )
    clf.fit(X=X_train, y=y_train, X_test=X_test, 
            y_test=y_test)

    # Print test accuracy and statistics.
    y_pred = clf.predict(X_test)
    print("accuracy: ", sklearn.metrics.accuracy_score(y_pred, y_test))
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

import argparse
import numpy as np

from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
from timeit import default_timer as timestamp

parser = argparse.ArgumentParser(description = 'Logistic Regression')

parser.add_argument('-X',
                    type = str,
                    required = True,
                    help = 'The generated samples.')
parser.add_argument('-y',
                    type = str,
                    required = True,
                    help = 'The integer labels for class membership of each sample.')
parser.add_argument('--test-size',
                    type = float,
                    default = 0.25,
                    help = 'The proportion of the dataset to include in the test split.')
parser.add_argument('--random-state',
                    type = int,
                    help = 'The seed used by the random number generator.')
parser.add_argument('--tol',
                    type = float,
                    default = 1e-4,
                    help = 'Tolerance for stopping criteria.')
parser.add_argument('-C',
                    type = float,
                    default = 1.0,
                    help = 'Inverse of regularization strength; must be a positive float.')
parser.add_argument('--max-iter',
                    type = int,
                    default = 100,
                    help = 'Maximum number of iterations taken for the solvers to converge.')
parser.add_argument('--verbose',
                    type = int,
                    default = 0,
                    help = 'Set verbose to any positive number for verbosity.')

args = parser.parse_args()

X = np.load(args.X)
y = np.load(args.y)

X_train, X_test, y_train, y_test = train_test_split(X, y,
                                                    test_size = args.test_size,
                                                    random_state = args.random_state)

for run in ['cpu', 'fpga']:
    if run == 'cpu':
        from sklearn.linear_model import LogisticRegression
    if run == 'fpga':
        from inaccel.sklearn.linear_model import LogisticRegression

    print ('---')
    print ('Logistic Regression fit on "{}":'.format(run))

    start = timestamp()
    model = LogisticRegression(tol = args.tol,
                               C = args.C,
                               random_state = args.random_state,
                               max_iter = args.max_iter,
                               verbose = args.verbose).fit(X_train, y_train)
    stop = timestamp()

    print('time=%.3f' % (stop - start))

    predictions = model.predict(X_test)

    print('accuracy=%.3f' % accuracy_score(y_test, predictions))

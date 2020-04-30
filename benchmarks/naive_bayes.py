import argparse
import numpy as np

from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
from timeit import default_timer as timestamp

parser = argparse.ArgumentParser(description = 'Naive Bayes')

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
parser.add_argument('--var-smoothing',
                    type = float,
                    default = 1e-9,
                    help = 'Portion of the largest variance of all features that is added to variances for calculation stability.')

args = parser.parse_args()

X = np.load(args.X)
y = np.load(args.y)

X_train, X_test, y_train, y_test = train_test_split(X, y,
                                                    test_size = args.test_size,
                                                    random_state = args.random_state)

for run in ['cpu', 'fpga']:
    if run == 'cpu':
        from sklearn.naive_bayes import GaussianNB
    if run == 'fpga':
        from inaccel.sklearn.naive_bayes import GaussianNB

    model = GaussianNB(var_smoothing = args.var_smoothing).fit(X_train, y_train)

    print ('---')
    print ('Naive Bayes predict on "{}":'.format(run))

    start = timestamp()
    predictions = model.predict(X_test)
    stop = timestamp()

    print('time=%.3f' % (stop - start))

    print('accuracy=%.3f' % accuracy_score(y_test, predictions))

import argparse
import numpy as np

from sklearn.model_selection import train_test_split
from sklearn.metrics import homogeneity_score
from timeit import default_timer as timestamp

parser = argparse.ArgumentParser(description = 'K-Means')

parser.add_argument('-X',
                    type = str,
                    required = True,
                    help = 'The generated samples.')
parser.add_argument('-y',
                    type = str,
                    required = True,
                    help = 'The integer labels for cluster membership of each sample.')
parser.add_argument('--test-size',
                    type = float,
                    default = 0.25,
                    help = 'The proportion of the dataset to include in the test split.')
parser.add_argument('--random-state',
                    type = int,
                    help = 'The seed used by the random number generator.')
parser.add_argument('--clusters',
                    type = int,
                    default = 8,
                    help = 'The number of clusters to form as well as the number of centroids to generate.')
parser.add_argument('--init',
                    type = int,
                    default = 10,
                    help = 'Number of time the k-means algorithm will be run with different centroid seeds.')
parser.add_argument('--max-iter',
                    type = int,
                    default = 300,
                    help = 'Maximum number of iterations of the k-means algorithm for a single run.')
parser.add_argument('--tol',
                    type = float,
                    default = 1e-4,
                    help = 'Relative tolerance with regards to inertia to declare convergence.')
parser.add_argument('--verbose',
                    type = int,
                    default = 0,
                    help = 'Verbosity mode.')

args = parser.parse_args()

X = np.load(args.X)
y = np.load(args.y)

X_train, X_test, y_train, y_test = train_test_split(X, y,
                                                    test_size = args.test_size,
                                                    random_state = args.random_state)

for run in ['cpu', 'fpga']:
    if run == 'cpu':
        from sklearn.cluster import KMeans
    if run == 'fpga':
        from inaccel.sklearn.cluster import KMeans

    print ('---')
    print ('K-Means fit on "{}":'.format(run))

    start = timestamp()
    model = KMeans(n_clusters = args.clusters,
                   n_init = args.init,
                   max_iter = args.max_iter,
                   tol = args.tol,
                   verbose = args.verbose,
                   random_state = args.random_state).fit(X_train)
    stop = timestamp()

    print('time=%.3f' % (stop - start))

    predictions = model.predict(X_test)

    print('homogeneity=%.3f' % homogeneity_score(y_test, predictions))

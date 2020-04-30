import argparse
import numpy as np

from sklearn.datasets import make_blobs, make_classification
from sklearn.preprocessing import scale

def classification(args):
    X, y = make_classification(n_samples = args.samples,
                               n_features = args.informative + args.redundant,
                               n_informative = args.informative,
                               n_redundant = args.redundant,
                               n_classes = args.classes,
                               random_state = args.random_state)

    np.save(args.X, scale(X))
    np.save(args.y, y)

def clustering(args):
    X, y = make_blobs(n_samples = args.samples,
                      n_features = args.features,
                      centers = args.centers,
                      random_state = args.random_state)

    np.save(args.X, scale(X))
    np.save(args.y, y)

if __name__ == '__main__':
    parser = argparse.ArgumentParser()

    subparsers = parser.add_subparsers(dest = 'generator')
    subparsers.required = True

    classification_parser = subparsers.add_parser('classification')
    classification_parser.set_defaults(function = classification)
    classification_parser.add_argument('--samples',
                                       type = int,
                                       default = 100,
                                       help = 'The number of samples.')
    classification_parser.add_argument('--informative',
                                       type = int,
                                       default = 10,
                                       help = 'The number of informative features.')
    classification_parser.add_argument('--redundant',
                                       type = int,
                                       default = 10,
                                       help = 'The number of redundant features.')
    classification_parser.add_argument('--classes',
                                       type = int,
                                       default = 2,
                                       help = 'The number of classes (or labels) of the classification problem.')
    classification_parser.add_argument('--random-state',
                                       type = int,
                                       help = 'Determines random number generation for dataset creation.')
    classification_parser.add_argument('-X',
                                       type = str,
                                       required = True,
                                       help = 'The generated samples.')
    classification_parser.add_argument('-y',
                                       type = str,
                                       required = True,
                                       help = 'The integer labels for class membership of each sample.')

    clustering_parser = subparsers.add_parser('clustering')
    clustering_parser.set_defaults(function = clustering)
    clustering_parser.add_argument('--samples',
                                   type = int,
                                   default = 100,
                                   help = 'The total number of points equally divided among clusters.')
    clustering_parser.add_argument('--features',
                                   type = int,
                                   default = 2,
                                   help = 'The number of features for each sample.')
    clustering_parser.add_argument('--centers',
                                   type = int,
                                   default = 3,
                                   help = 'The number of centers to generate.')
    clustering_parser.add_argument('--random-state',
                                   type = int,
                                   help = 'Determines random number generation for dataset creation.')
    clustering_parser.add_argument('-X',
                                   type = str,
                                   required = True,
                                   help = 'The generated samples.')
    clustering_parser.add_argument('-y',
                                   type = str,
                                   required = True,
                                   help = 'The integer labels for cluster membership of each sample.')

    args = parser.parse_args()
    args.function(args)

from sklearn.utils import shuffle
from sklearn.model_selection import GroupKFold
from numpy.random import RandomState
import numpy as np
import sys
import pdb

def rand_gkf(groups, n_splits, random_state=None, shuffle_groups=False):
    '''Randomly divides groups in contrast to GroupKFold which is not random and gives the same results each time it's run.  This must have shuffle true to randomize.  Probably will give imbalanced folds...but what else can you do?'''
    ix = np.array(range(len(groups)))
    unique_groups = np.unique(groups)
    if shuffle_groups:
        prng = RandomState(random_state)
        prng.shuffle(unique_groups)
    splits = np.array_split(unique_groups, n_splits)
    train_test_indices = []

    for split in splits:
        mask = [el in split for el in groups]
        train = ix[np.invert(mask)]
        test = ix[mask]
        train_test_indices.append((train, test))
    return train_test_indices

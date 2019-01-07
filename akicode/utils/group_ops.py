'''Used to sample at patient level. Helper for groupkf'''

import numpy as np
class GroupShuffler:

    '''
    gs = GroupShuffler(random_state=11, verbose=10)
    gs.shuffle(X, y, groups)
    '''

    def __init__(self, random_state=None, verbose=0):
        self.random_state = random_state
        self.verbose = verbose

    def shuffle(self, X, y, groups):

        index = range(len(groups))
        rng = np.random.RandomState(self.random_state)
        rng.shuffle(index)
        if self.verbose > 0:
            print "shuffled index", index[0:100]
        return X[index], y[index], groups[index]

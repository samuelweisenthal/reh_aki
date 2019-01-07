'''Helper functions for pickle'''

import pandas
import numpy as np
import pdb
import pickle

def save_obj(obj, name):
    #http://stackoverflow.com/questions/19201290/how-to-save-a-dictionary-to-a-file-in-python
    with open(name + '.pkl', 'wb') as f:
        pickle.dump(obj, f, pickle.HIGHEST_PROTOCOL)

def load_obj(name):
    #http://stackoverflow.com/questions/19201290/how-to-save-a-dictionary-to-a-file-in-python
    with open(name + '.pkl', 'rb') as f:
        return pickle.load(f)

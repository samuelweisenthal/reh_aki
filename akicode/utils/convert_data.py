'''Helper functions for put data into corret shape'''

import numpy as np
import copy

def flatten(data):

    flat_admits = []

    for pt in data:

        for ms,y in zip(*pt):

            flat_admits.append(([ms],np.array([y])))

    return flat_admits


def flatten_mem(data):

    flat_mem_admits = []

    for pt in data:
        pt_ms = []
        for admit,outcome in zip(pt[0],pt[1]):

            for m in admit:
                
                pt_ms.append(m)

            flat_mem_admits.append(([copy.deepcopy(pt_ms)],np.array([outcome]))) 

    return flat_mem_admits


def agg(data):
    
    a = map(lambda x:([np.array(np.mean(x[0]))],x[1]),data)
    b = map(lambda x:([np.array(np.sum(x[0]))],x[1]),data)
    c = map(lambda x:([np.array(np.max(x[0]))],x[1]),data)
    
    return a,b,c #mean, sum, max

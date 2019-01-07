'''For preprocessing meds. Just kind of a wrapper to wrang.py'''


import pandas
import re
import collections
import numpy
import pickle
import wrang
import misc
import copy
import pdb


def make_med_dict(m):

	m = copy.deepcopy(m) #bad use of memory, but just working with prior code

	m = wrang.process_meds(m)


	g = lambda s: s.rstrip()


	m['DESCRIPTION'] = m['DESCRIPTION'].astype(str)


	m['DESCRIPTION'] = m['DESCRIPTION'].apply(g)


	a_dict={}
	for el in  ['THERA_CLASS_C','PHARM_CLASS_C','PHARM_SUBCLASS_C']:
	    for mid,mdf in m.groupby(el):
	        a_dict['_UNIQUE_ID_MED_'+el+'_'+str(mid)]= 'ONE_OF:'+';'.join(list(mdf['DESCRIPTION'].unique()))
	        #print ';'.join(list(mdf['DESCRIPTION'].unique()))


	return a_dict



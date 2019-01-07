'''One hot encodes medications -- was large file, 
splits it up to make more manageable.  Still presents
problems without fairly large (500gb+) amount of RAM'''

import pandas
import re
import collections
import numpy
import wrang
import copy


def ohe_med(m):


	m = copy.deepcopy(m)


	def process_meds(m):
	    
	    ow = m['DESCRIPTION'].str.extract('^(.+?)[0-9]') #http://chrisalbon.com/python/pandas_regex_to_create_columns.html
	    m['DESCRIPTION']= ow
	    m['THERA_CLASS_C'] = m['THERA_CLASS_C'].astype(str)
	    m['PHARM_CLASS_C'] = m['PHARM_CLASS_C'].astype(str)
	    m['PHARM_SUBCLASS_C']=m['PHARM_SUBCLASS_C'].astype(str)
	    
	    return m

	m = process_meds(m)


	numberSplits=50
	blocks=[]
	for block in numpy.array_split(m,numberSplits):#split into blocks--otherwise large sparse matrices-> memory error
	    ablock=pandas.get_dummies(block).groupby('ADMIT_ID').sum()#when we have blocks, we can collapse them by index (correspondng to an admitid)
	    print ablock.info()
	    blocks.append(ablock)#put all the blocks in a list
	fullFrame=pandas.concat(blocks).replace(numpy.NAN,0)#concat and replace nan with 0 (if didn't have procedure in block, will be nan--> no one in the block had it)
	print 
	print 
	print "finished"
	print fullFrame.info()
	fullFrame=fullFrame.groupby(fullFrame.index).sum()#sum again because array_split split some admissions
	print fullFrame.info()
	#

	fullFrame.rename(columns={c:'MED_' + c for c in fullFrame.columns},inplace=True)#label


	fullFrame.info()



	fullFrame = fullFrame.reset_index()


	return fullFrame


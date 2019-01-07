'''Given all of the preprocessed data, this script produces X, y, and groups for training.
fof.py then accesses the output.
Mostly calls other scripts that one hot encode, name features, get outcomes, etc.
'''

import pdb


import pandas
import wrang
import numpy as np
import sklearn
import pathOps
from extractF import concatGroupVar, extractFeatures, joinFeatures
from defineOutcomes import get_expected_dx, any_in
from saver import saveSampled, saveDumm, saveRaw
from sampler import independentSampler
from load_data import collate_datasets, load_data
from drop_rare import rem_rare_wrap


def preprocess(runTag, 
	path_to_admin, path_to_labs, path_to_creatinine_dx_ind, path_to_meds, path_to_and_or_creat, 
	dx_path, loc_path, cms32Px_path, cptProd_path, drgs_path, meds_path,
	path_to_exclusion,
	precDRG, precDX, precicdPx, preccptPx,
	use_labs=1, use_meds=1, use_admin=1, useAKI_dx=0, cat_too=1,
	loadAllSamp=0, PathToallSamp=None, loadDummSamp=0, PathToDummSamp=None,
	targetDxs=['584.5','584.6','584.7','584.8','584.9'],
	memoryless=0,
	exclude_ED=0,
	predDx=1,
	writeDatasets=0, nrows=None):


	if loadAllSamp==1:
	    
	    print "Loading samples from disk",PathToallSamp
	    allSamp = load_data(PathToallSamp)
	    
	else:
	    print "Creating samples..."
	    raw_data = collate_datasets(path_to_admin, 
	                                  path_to_labs, 
	                                  path_to_meds, 
	                                  path_to_creatinine_dx_ind, 
	                                  path_to_and_or_creat, 
	                                  use_labs, 
	                                  use_meds,
	                                  path_to_exclusion=path_to_exclusion,
	                                  exclude_ED=exclude_ED, nrows=nrows)
	    

	    dxs = [col for col in raw_data.columns if 'DIAGNOSES' in col]
	    (positiveSamples,negativeSamples) = get_expected_dx(raw_data,
	                                                        'PATIENT_NUM',
	                                                        'ADMIT_ID',
	                                                         dxs,targetDxs,
	                                                         memoryless=memoryless)

	    negativeSamples = pandas.concat(negativeSamples)
	    positiveSamples = pandas.concat(positiveSamples)
	    allSamp=pandas.concat([positiveSamples,negativeSamples],axis=0)
	    allSamp.reset_index(drop=True,inplace=True)#need to reset index
	    #clear these to free memory
	    negativeSamples=[]
	    positiveSamples =[]
	    raw_data=[]


	if writeDatasets==1: saveRaw(allSamp,runTag)



	allSamp.info()


	# <h2>Extract features


	if loadDummSamp == 1:
	    print "Loading dummSamp from disk",PathToDummSamp
	    DummSamp = pandas.read_csv(PathToDummSamp)#Since all of DummSamp is OHE, no need to specify dtypes
	else:
	    print "Creating dummSamp"
	    DummSamp = joinFeatures(extractFeatures(allSamp,'sampleNo',
	                                                     precDRG, precDX, precicdPx,preccptPx,
	                                                     use_labs,use_meds,use_admin,useAKI_dx,
	                                                     cat_too,
	                                                     #dx
	                                                     dx_path = dx_path,
	                                                     #locations
	                                                     loc_path = loc_path,
	                                                     #icd9 px
	                                                     cms32Px_path = cms32Px_path,
	                                                     #cpt px
	                                                     cptProd_path = cptProd_path,
	                                                     #drg
	                                                     drgs_path = drgs_path,
	                                                     #meds
	                                                     meds_path = meds_path))

	    DummSamp.info()
	    allSamp = []




	if writeDatasets==1: saveDumm(DummSamp,runTag,precDRG,
	                              precDX,precicdPx,preccptPx)



	print "Positive samples",len(DummSamp[DummSamp['target']==1])
	print "Negative samples",len(DummSamp[DummSamp['target']==0])
	print "Unique patients that generated positive samples",len(DummSamp[DummSamp['target']==1]['PATIENT_NUM'].unique())
	print "Unique patients generated negative samples",len(DummSamp[DummSamp['target']==0]['PATIENT_NUM'].unique())
	print "Total unique patients",len(DummSamp['PATIENT_NUM'].unique())


	# <h2>Drop rare features


	DummSamp.info()





	DummSamp = rem_rare_wrap(thresh=100, df=DummSamp)



	DummSamp.info()


	# <h2>Get rid of target, sample no, and patient num



	y=DummSamp['target']
	groups = DummSamp['PATIENT_NUM']
	X=DummSamp[[el for el in DummSamp.columns if el != 'target' and el !='sampleNo' and el !='PATIENT_NUM']]


	# <h2>Return for access by nested CV (n_cv)


	return X, y, groups

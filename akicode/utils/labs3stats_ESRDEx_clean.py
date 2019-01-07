'''Finds patients with ESRD to exclude.'''

import pandas
from sklearn.decomposition import NMF
from sklearn.preprocessing import Imputer
import numpy as np
from defineOutcomes import any_in
from load_data import load_data
import dem
reload(dem)

def exclude(admits, dxs):


	dx = dem.get_pr(admits, 'DIAGNOSES', {'AKI':'584',
	                                  'RenalFailUnspec':'586',
	                                 'DialStat':'V451', 
	                                 'AdjDialCath':'V56', 
	                                 'DialReac':'E8791', 
	                                 'BrokDial':'E8742', 
	                                 'SterDial':'E8722', 
	                                 'ForDial':'E8712', 
	                                 'cutDial':'E8702', 
	                                 'infectDial':'99673', 
	                                 'mechDial': '99656', 
	                                 'cloudDial':'7925', 
	                                 'HypDial':'45821', 
	                                 'CKD':'585', 
	                                 'CKDEndStage':'5856', 
	                                 'CHF': '428', 
	                                 'Rhab':'72888', 
	                                 'Diab':'250', 
	                                 'Shock':'785', 
	                                 'AcuteLiver':'570', 
	                                 'ChronLiver':'571', 
	                                 'OtherLiv':'573'})


	i_px = dem.get_pr(admits,'ICD9_PROCEDURES',{'OtherRenTrans':'5569', 
	                                        'RenTrans':'V420',
	                                        'RenRej':'5553', 
	                                        'DialPx1':'3995', 
	                                        'DialPx2':'5498',
	                                         'RenalAuto':'5561'}) #don't see renal auto..


	# In[64]:


	df = pandas.concat([i_px, dx, dxs], axis=1)


	# <h1>Exclude <s>dialysis</s> end stage CKD (decided not to exclude all dialysis because dialysis is sometimes a temporary measure, so patients might have it and then recover and therefore be at risk of AKI). To go back and exclude dialysis as well, simply uncomment the first line below and comment the second. Note that unfortunately it's still dialex in the file name, kept as such because multiple scripts already find it by filename (to self: don't hardcode paths in future)


	df['dial_or_es'] = df[['CKDEndStage:5856']].sum(axis=1)


	df['dial_or_es_bin'] = 1*(df['dial_or_es']>0)

	#this has admit id as the index
	to_ex_dial = pandas.concat([admits['PATIENT_NUM'],df[['dial_or_es_bin', 'OtherRenTrans:5569']]], axis=1)


	import pdb
	doex=1 #already did this and saved
	if doex ==1:
	    to_ex_dial['keep'] = np.NAN
	    ind_res_df = to_ex_dial.copy()
	    ind_res_df = ind_res_df.reset_index()
	    for ptnum, admits in ind_res_df.groupby('PATIENT_NUM'):
		#pdb.set_trace()
	        #print "pt", ptnum
	        keep = 1
	        for adid, admit in admits.groupby('ADMIT_ID'):
	            to_ex_dial.loc[adid, 'keep'] = keep
	            #print adid
	            #print admit['dial_or_es_bin']
	            if admit['dial_or_es_bin'].values[0] == 1:
	                keep = 0
	            if admit['OtherRenTrans:5569'].values[0] == 1:
	                keep = 1
	    #misses last one? no it's missing pt i
	
	return to_ex_dial

#note that this is actually CKD exclusion


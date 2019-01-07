'''Older script; used to check if patient had AKI dx
Output is used a few times in preprocessing scripts.
Is combined with diagnosis via creatinine labs.  Also makes
the contingency table for AKI.
'''

import pandas
from sklearn.preprocessing import Imputer
import numpy as np
from defineOutcomes import any_in
from load_data import load_data


def make_df(admits_only, from_labs):


	from_labs = from_labs.set_index('ADMIT_ID')


	admits_only = admits_only.set_index('ADMIT_ID')

	joined_akibylab_admits = admits_only.join(from_labs)


	cols = joined_akibylab_admits.columns
	dxs = [c for c in cols if 'DIAG' in c]
	pxs = [c for c in cols if 'PROCEDUR' in c]


	np.nansum(joined_akibylab_admits['AKI_by_creat'])


	joined_akibylab_admits = joined_akibylab_admits.reset_index()




	joined_akibylab_admits = joined_akibylab_admits.sort_values(['PATIENT_NUM','ADMIT_ID'])




	joined_akibylab_admits = joined_akibylab_admits[joined_akibylab_admits['AGE_ON_ADMISSION']>=18].copy()



	np.nansum(joined_akibylab_admits['AKI_by_creat'])



	sum(from_labs['AKI_by_creat'])


	targetDxs = ['584.5', '584.6', '584.7', '584.8', '584.9']




	joined_akibylab_admits['AKI_by_admi'] = 0




	for i in joined_akibylab_admits.index:
	    if any_in(targetDxs,joined_akibylab_admits.loc[[i]][dxs].values[0]):
	        joined_akibylab_admits.loc[i,'AKI_by_admi']=1



	joined_akibylab_admits[['AKI_by_creat', 'AKI_by_admi']].apply(np.nansum)#these numbers same as manuscript




	joined_akibylab_admits['or']=(
	    (joined_akibylab_admits['AKI_by_creat']==1) | (joined_akibylab_admits['AKI_by_admi']==1))*1
	joined_akibylab_admits['and']=(
	    (joined_akibylab_admits['AKI_by_creat']==1) & (joined_akibylab_admits['AKI_by_admi']==1))*1




	joined_akibylab_admits[['or', 'and', 'AKI_by_creat', 'AKI_by_admi']].apply(np.nansum)#also confirms manuscript



	return joined_akibylab_admits[['ADMIT_ID','or', 'and', 'AKI_by_admi']]



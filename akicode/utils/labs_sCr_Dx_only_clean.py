'''Make diagnosis by sCr levels.'''

import pandas
from sklearn.decomposition import NMF
from sklearn.preprocessing import Imputer
import numpy as np
from matplotlib import pyplot as plt
from matplotlib import pyplot as plt
from defineOutcomes import any_in
from load_data import load_data
from diagnoser import dx_fold, dx_abs


def get_scr_dx(l):

	l['RESULT_VALUE'] = l['RESULT_VALUE'].str.replace('[^0-9.]','')
	l['RESULT_VALUE'] = l['RESULT_VALUE'].convert_objects(convert_numeric=True)
	l['RESULT_VALUE'] = l['RESULT_VALUE'].astype(float)


	# <h1> Creatinines

	# Look for greater than 1.5-fold rise (RIFLE definition of risk, which is apparently a stage of AKI (?) ) in creatinine over 7 or fewer days. Used UpToDate, which referenced "Acute renal failure - definition, outcome measures, animal models, fluid therapy and information technology needs: the Second International Consensus Conference of the Acute Dialysis Quality Initiative (ADQI) Group." Bellomo et al. 2004.  It is believed that adminstrative codes are specific but not sensitive for AKI (see, eg, "Validity of International Classification of Diseases, Ninth Revision, Clinical Modification Codes for Acute Renal Failure" by Waikar in 2006).  This makes sense-- AKI is often missed or not coded, but it's probably very rarely coded when there isn't AKI at all.  To complete the KDIGO modifications, would also need to check whether there is an increase in serum creatinine (sCr) by >=0.5 within 48 hours. As baseline, we use the first documented sCr.

	# Convert random collection date to a datetime object and sort by it


	l['RNDM_COLLECTION_DATE']= pandas.to_datetime(l['RNDM_COLLECTION_DATE'])
	l = l.sort_values(['ADMIT_ID','RNDM_COLLECTION_DATE'])
	t = l[l['TEST_NAME']=='CREAT'][['ADMIT_ID','RNDM_COLLECTION_DATE','RESULT_VALUE','ABNORMAL_FLAG','REF_RANGE']]


	# Double check data types



	t.dtypes


	# Some sCr are missing at random.  Forward fill, backward fill, then use mean to fill (for ones with only one measurement).  Aren't missing any admit id s or random collection dates


	tgfilled = t.groupby('ADMIT_ID').fillna(method = 'ffill')



	tgfilled['ADMIT_ID']=t['ADMIT_ID']



	print "# missing val",len(tgfilled[tgfilled['RESULT_VALUE'].isnull()])


	# So, after forward fill, we have 121 missing. (because they were the first). So, now back fill (we are using this to determine the response variable, so it's ok).

	tgallfilled = tgfilled.groupby('ADMIT_ID').fillna(method='bfill')


	# After backfill, still have missing values (because some measurements are singular).  It's moot because these people won't be flagged by the diagnoser anyway.  Not sure whether to keep them here or not. I think I will keep them just because t will be used later for statistics perhaps.  Impute them also, using the mean of the dataset.



	print "# missing val",len(tgallfilled[tgallfilled['RESULT_VALUE'].isnull()])



	tgallfilled['ADMIT_ID']=tgfilled['ADMIT_ID']



	tgallfilled2 = tgallfilled.groupby('ADMIT_ID').fillna(value=np.nanmean(tgallfilled['RESULT_VALUE']))



	print "# missing val",len(tgallfilled2[tgallfilled2['RESULT_VALUE'].isnull()])



	tgallfilled2['ADMIT_ID'] = tgallfilled['ADMIT_ID']


	# Now, we've filled them all.

	# Re-order columns since diagnoser depends on order of columns



	tgallfilled = tgallfilled[t.columns]


	# Diagnose



	fold_dx_i = dx_fold(1.5,'7 days',tgallfilled)



	abs_dx_i = dx_abs(0.3,'48 hours',tgallfilled)


	# How many are diagnosed by fold and how many by absolute?



	print sum(fold_dx_i['AKI'])
	print sum(abs_dx_i['AKI'])



	from_labs = (fold_dx_i+abs_dx_i!=0)*1

	from_labs = from_labs.reset_index()

	from_labs=from_labs.rename(columns={'index':'ADMIT_ID','AKI':'AKI_by_creat'})

	return from_labs

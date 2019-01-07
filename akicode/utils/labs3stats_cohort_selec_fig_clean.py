
'''Make histogram in cohort selection figure and computes some statistics used in cohort selection table (cases, controls, # patients in each).  Reads the file that indicates who to exclude because of ESKD as well as the main demographic billing data file.  Was ipython notebook originally.'''


import pandas
from sklearn.decomposition import NMF
from sklearn.preprocessing import Imputer
import numpy as np
from defineOutcomes import any_in
from load_data import load_data
from matplotlib import pyplot as plt
import dem
reload(dem)
from load_data import load_data


def labs_cohort_sel_mk_dx(dial, all_samp, dxs):


	dxs['creat'] = dxs['or'] - dxs['AKI_by_admi'] + dxs['and']


	len(set(dial[
	    dial['PATIENT_NUM'].isin(dial[dial['keep']==0]['PATIENT_NUM'])
	    &
	    dial['PATIENT_NUM'].isin(dial[dial['OtherRenTrans:5569']==1]['PATIENT_NUM'])
	    
	    ]['PATIENT_NUM']))



	print "Hospitalizations", len(all_samp)
	print "Patients", len(all_samp['PATIENT_NUM'].unique())


	# <h2>Age >=18


	all_samp = all_samp[all_samp['AGE_ON_ADMISSION']>=18]
	print "Hospitalizations", len(all_samp)
	print "Patients", len(all_samp['PATIENT_NUM'].unique())


	# Exclude ESRD (says dialysisEx, means ESRD)



	dial = dial.set_index('ADMIT_ID')
	dial= dial['keep']

	all_samp = all_samp.set_index('ADMIT_ID')


	joined_with_ex = all_samp.join(dial)
	joined_with_ex = joined_with_ex.join(dxs)


	joined_with_ex = joined_with_ex[joined_with_ex['keep'] == 1] 

	joined_with_ex = joined_with_ex.reset_index()

	print "Hospitalizations", len(joined_with_ex) 
	print "Patients", len(joined_with_ex['PATIENT_NUM'].unique())



	print "aki+ in all adult with ex", joined_with_ex['or'].sum()
	print "aki+ % in all adult with ex", joined_with_ex['or'].sum()/float(len(joined_with_ex))*100


	# Plot # HOSPITALIZATIONS (not rehosp) per patient.  This is not the figure in the cohort selection diagram quite yet, since it also contains primary hospitalization for each patient

	# <h2> If comment in would be the histogram in the cohort selection diagram.

	import nice_plots
	reload(nice_plots)
	#nice_plots.nice_hist(joined_with_ex['PATIENT_NUM'].value_counts(), 
	#                     fname='nhos.pdf', xlab='Hospitalizations per Patient', 
	#                     ylab='Log Frequency', yscal='log')
	print "Not plotting hist for cohort selection."



	admits_per_pt = joined_with_ex['PATIENT_NUM'].value_counts()



	pts_w_mult_hosp = admits_per_pt[admits_per_pt>1].index



	h_from_rehosp = joined_with_ex[joined_with_ex['PATIENT_NUM'].isin(pts_w_mult_hosp)]
	print "hospitalizations", len(h_from_rehosp)
	print "patients", len(pts_w_mult_hosp)



	rehosp_coho = h_from_rehosp['ADMIT_ID']



	print "Hospitalizations", len(h_from_rehosp) - len(h_from_rehosp['PATIENT_NUM'].unique()) #subtract one for each patient (the first one)
	print "Patients", len(h_from_rehosp['PATIENT_NUM'].unique()) #same as before, we haven't changed the # patients


	# (Out of curiosity, how many were diagnosed by sCr?)



	h_from_rehosp['creat'].sum()


	# Now just take all of the rehospitalizations



	sorted_rehosp = h_from_rehosp.sort_values(['PATIENT_NUM', 'ADMIT_ID'])


	not_primary_hosp = [] #this is slow, should be better way to do this than looping.
	for ptid, admits in sorted_rehosp.groupby('PATIENT_NUM'):
	    not_primary_hosp.append(admits[1:])


	n_p_h_df = pandas.concat(not_primary_hosp, axis=0)



	to_save = n_p_h_df[
	    ['ADMIT_ID', 'PATIENT_NUM', 'or', 'and', 'creat', 'AKI_by_admi']
	    ]

	# These numbers appear in cohort selection diagram

	print "Rehospitalizations Cases", len(n_p_h_df[n_p_h_df['or']==1]) # No. cases
	print "Patients Cases", len(n_p_h_df[n_p_h_df['or']==1]['PATIENT_NUM'].unique()) # No. patients in cases
	print "Rehospitalizations Controls", len(n_p_h_df[n_p_h_df['or']==0]) # No. controls
	print "Patients controls", len(n_p_h_df[n_p_h_df['or']==0]['PATIENT_NUM'].unique()) # No. patients in controls 

	return to_save

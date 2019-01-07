import pdb
import pandas
import sys
sys.path.insert(0, '../utils')
import misc

#nrows of data to take
NROWSSS = None

pathToData='/homedirec/user/DATA/ADMISSIONS.csv'
pathToSaveAdmissions='ADMISSIONS_split_px_dx.csv'
pathToLabs = '/homedirec/user/DATA/LABS'
meds_path = '/homedirec/user/DATA/MEDS'

path_to_save_dxs_to = 'dx_from_labs_fixed.csv'
pathToSaveCreatTo = 'or_and_byCreat_byAdm.csv'
exfn = 'DialEx.csv' #put final cohort with esrd excluded
to_save_dx_et_al = 'dxs_et_al.csv'
dummyMeds_path = 'dummyMeds.csv'
med_dict_name_path = 'med_name_dictionary.pkl'
#to write to (also write into pwd for access by slurm)
runTag = 'test'
pth_X = 'X_aki_full_dialE_'+runTag+'.csv'
pth_y = 'y_aki_full_dialE_'+runTag+'.csv'
pth_groups = 'groups_aki_full_dialE_'+runTag+'.csv'

from split_admissions_clean import split_dx_px
all_w_i = pandas.read_csv(pathToData, nrows=NROWSSS) ##########
split_df = split_dx_px(all_w_i)
split_df.to_csv(pathToSaveAdmissions)

from labs_sCr_Dx_only_clean import get_scr_dx
labs = pandas.read_csv(pathToLabs, parse_dates=True, nrows=NROWSSS) #######
dx_from_labs_fixed = get_scr_dx(labs)
dx_from_labs_fixed.to_csv(path_to_save_dxs_to,index=False)


from make_or_and_byCreat_byAdm_clean import make_df
or_and_byCreat_byAdm = make_df(split_df, dx_from_labs_fixed) #potential issue bc used
or_and_byCreat_byAdm.to_csv(pathToSaveCreatTo, index=False)

from labs3stats_ESRDEx_clean import exclude
from dem import get_admits_for_dem
admits = get_admits_for_dem(pathToData, nrows=NROWSSS)
to_ex_dial = exclude(admits, or_and_byCreat_byAdm.set_index('ADMIT_ID'))
to_ex_dial.to_csv(exfn) #note that this is actually CKD exclusion

from labs3stats_cohort_selec_fig_clean import labs_cohort_sel_mk_dx
dx_et_al = labs_cohort_sel_mk_dx(dial=to_ex_dial.reset_index(), 
	all_samp=split_df, dxs=or_and_byCreat_byAdm.set_index('ADMIT_ID'))
dx_et_al.to_csv(to_save_dx_et_al)


from med2_clean import ohe_med
m = pandas.read_csv(meds_path, usecols=['ADMIT_ID','THERA_CLASS_C','PHARM_CLASS_C','PHARM_SUBCLASS_C','DESCRIPTION'], nrows=NROWSSS) #####
dummy_meds = ohe_med(m)
dummy_meds.to_csv(dummyMeds_path)


from medications_clean import make_med_dict
med_dict = make_med_dict(m)
misc.save_obj(med_dict, med_dict_name_path)

### at this point, load saved data.  Would
### be much nicer to just keep the data live, but would require fixing
### many functions to take dataframes instead of paths. TODO, but
### not critical for reproducibility

#save intermediary datasets (they can get large, esp all samples)
writeDatasets = 0; 
#runTag = 'test'#written set is much smaller than RAM realization in panda

#files
#I one-hot-encoded px and dx codes. Else, same as original in same folder

path_to_admin = pathToSaveAdmissions
path_to_labs = pathToLabs
path_to_creatinine_dx_ind = path_to_save_dxs_to
path_to_meds = dummyMeds_path
path_to_and_or_creat = pathToSaveCreatTo
path_to_exclusion = exfn


#To name (these are all in pwd)
#diagnoses
dx_path = '../crosswalks/CMS32_DESC_LONG_SHORT_DX.csv'
#locations
loc_path = '../crosswalks/locCode ICU.csv'
#icd9 px
cms32Px_path = '../crosswalks/CMS32_DESC_LONG_SHORT_SG.csv'
#cpt px
cptProd_path = '../crosswalks/cpt product list.csv'
#drg
drgs_path = '../crosswalks/drg2mdcxw2014.csv'

#meds
meds_path = med_dict_name_path #'med_name_dictionary.pkl'
## med_dict

#precisions
precDRG = 3
precDx = 3
precicdPx = 3
preccptPx = 3

#what to predict. diagnoses to analyze. these aren't really used besides for codes.
targetDxs = ['584.5','584.6','584.7','584.8','584.9']#all AKI
#what groups of features to use. Actually for meds was done later on.
predDx = 1
use_labs   = 1
use_meds   = 1
use_admin  = 1
useAKI_dx = 0
cat_too = 1 #cat_too =1 if want to use categorical as well as numeric
#load or make the samples new? If load, give PathToallSamp
loadAllSamp = 0
PathToallSamp = None
#load or extract features from the samples? If load, give PathToDummSamp
loadDummSamp = 0
PathToDummSamp = None
#use only most recent, or all hospitalizations?
memoryless = 0
exclude_ED = 0

from findOutFeatWrangLearn_clean import preprocess

X, y, groups = preprocess(runTag=runTag,
        path_to_admin=path_to_admin, path_to_labs=path_to_labs, 
	path_to_creatinine_dx_ind=path_to_creatinine_dx_ind, path_to_meds=path_to_meds, 
	path_to_and_or_creat=path_to_and_or_creat,
        dx_path=dx_path, loc_path=loc_path, cms32Px_path=cms32Px_path, 
	cptProd_path=cptProd_path, drgs_path=drgs_path, meds_path=meds_path,
        path_to_exclusion=path_to_exclusion,
        precDRG=precDRG, precDX=precDx, precicdPx=precicdPx, preccptPx=preccptPx,
        use_labs=use_labs, use_meds=use_meds, use_admin=use_admin, useAKI_dx=useAKI_dx, 
	cat_too=cat_too,
        loadAllSamp=loadAllSamp, PathToallSamp=PathToallSamp, 
	loadDummSamp=loadDummSamp, PathToDummSamp=PathToDummSamp,
        targetDxs=targetDxs,
        memoryless=memoryless,
        exclude_ED=exclude_ED,
        predDx=predDx,
        writeDatasets=writeDatasets, 
	nrows=NROWSSS)

X.to_csv(pth_X)
y.to_csv(pth_y)
groups.to_csv(pth_groups)

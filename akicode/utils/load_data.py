'''Helper functions to load and collate datasets.  
A bit complicated because
data comes from multiple sources (labs, demographics, etc)'''

import pandas
import numpy as np
import pdb

def load_data(pathToData, nrows=None):
    '''Loads data so doesn't convert codes to floats'''
    codes,other_str,meds,labs,other_cols = get_data_groups(pandas.read_csv(pathToData,nrows=1))
    non_string_cols = meds + labs + other_cols
    string_cols = codes+other_str
    non_strings  = pandas.read_csv(pathToData, usecols=non_string_cols, nrows=nrows)
    strings = pandas.read_csv(pathToData, usecols=string_cols, dtype='str', nrows=nrows)
    strings = strings.astype(str)
    return pandas.concat([strings,non_strings],axis=1)

def get_data_groups(allSamp):
    '''Gets groups in data so can then take care not to switch types (eg, convert codes to floats)'''
    cols = allSamp.columns
    codes = [c for c in cols if ('DIAGNOSES' in c or 'ICD9_PROCEDURE' in c or 'CPT4_PROCEDURE' in c or 'DRG1' in c or 'DRG2' in c)]
    other_str = ['MARITAL_CD','ETHNIC_CD','SEX_CD','INSURANCE','DISCHARGE_DISPOSITION_1','DISCHARGE_DISPOSITION_2','L1', 'L2', 'L3', 'L4', 'L5', 'L6', 'L7', 'L8', 'L9', 'L10', 'L11', 'L12', 'L13', 'L14', 'L15', 'L16', 'L17', 'L18', 'L19', 'L20', 'L21', 'L22', 'L23', 'L24', 'L25']
    meds = [c for c in cols if '_UNIQUE_ID_MED' in c]
    labs = [c for c in cols if '_LAB_' in c]
    other_cols = [c for c in cols if (c not in codes and c not in other_str and c not in meds and c not in labs)]
    return (codes,other_str,meds,labs,other_cols)

def collate_datasets(path_to_admin, path_to_labs, path_to_meds, 
                     path_to_creatine_dx_ind, path_to_and_or_creat, 
                     use_labs, use_meds,  
                     path_to_exclusion='/homedirec/user/DialEx.csv', 
                     exclude_ED = 0, nrows=None):
    
    print "Loading admininstrative data..."
    my_dataset = load_data(path_to_admin, nrows=nrows)
    
    #exclude hosp from patients known to be on dialysis and not transplanted (see labs3stats.ipynb in /home/user/pred
    dial = pandas.read_csv(path_to_exclusion)
    dial = dial.set_index('ADMIT_ID')
    dial = dial['keep']
    my_dataset = my_dataset.set_index('ADMIT_ID')
    joined_with_ex = my_dataset.join(dial)
    joined_with_ex = joined_with_ex[joined_with_ex['keep'] == 1] 
    joined_with_ex = joined_with_ex.reset_index()
    my_dataset = joined_with_ex
    
    my_dataset = my_dataset[my_dataset['AGE_ON_ADMISSION']>=18]#exclude pediatric
    if exclude_ED ==1:
        print "Excluding ED"
        my_dataset = my_dataset[~(((my_dataset['L1']=='ED')|(my_dataset['L1']=='EDB')) & (my_dataset['L2']=='nan'))]
        print "new length", len(my_dataset) #should be 118661 hosp and 72915 pts, have excluded 78385 hosp and 49693 patients
        print "num pt", len(my_dataset[~(((my_dataset['L1']=='ED')|(my_dataset['L1']=='EDB')) & (my_dataset['L2']=='nan'))]['PATIENT_NUM'].unique())

    cols =my_dataset.columns
    dxs=[el for el in cols if 'DIAGNOSES' in el]
    
    #this will be used later, but easier to 'allocate' now
    my_dataset['sampleNo'] = np.nan
    my_dataset['target']= np.nan
    #these two ways of sorting give the same overmy_dataset results, but different dataframes apparently. Not sure which to choose...or if it matters
    my_dataset.sort_values(['PATIENT_NUM','ADMIT_ID'],inplace=True,axis=0)
    #my_dataset.sort(['PATIENT_NUM','RNDM_ADMIT_DATE'],inplace=True,axis=0)#don't sort by admit date, there's a nan

    if use_labs==1:
        print "Loading labs..."
        l = pandas.read_csv(path_to_labs, nrows=nrows)

        l['RESULT_VALUE'] = l['RESULT_VALUE'].str.replace('[^0-9.]','')
        l['RESULT_VALUE'] = l['RESULT_VALUE'].convert_objects(convert_numeric=True)
        f = {'RESULT_VALUE':['min','mean','max','sum','var']}#prod leads to inf in feature vector
        gr = l[['ADMIT_ID','TEST_NAME','RESULT_VALUE','ABNORMAL_FLAG']].groupby(['ADMIT_ID','TEST_NAME']).agg(f)
        gr.columns = ['_'.join(col).strip() for col in gr.columns.values]
        gr = gr.unstack()
        gr.columns = ['_'.join(el) for el in gr.columns.values]
        gr.rename(columns = {c:'UNIQUE_LAB_' + c for c in gr.columns}, inplace = True)
        
        #Do some feature extraction here. Normally, only in extractF.py, but this is a better place for it
        l['ABNORMAL_FLAG']=l['ABNORMAL_FLAG'].fillna('nan')
        l['ABNORMAL_LABS'] = l['ABNORMAL_FLAG'].str.cat(l['TEST_NAME'],sep=',')
        abno_la = pandas.concat([l[['ADMIT_ID']],pandas.get_dummies(l['ABNORMAL_LABS'])],axis=1).groupby('ADMIT_ID').sum()
        abno_la.columns=['ABNORMAL_LAB_FLAG: '+c for c in abno_la.columns]
        gr = gr.join(abno_la,how='outer')#both gr and abno_la already have admit id as index
        
        #add in creatinine dxs #look at this later
        ind=pandas.read_csv(path_to_creatine_dx_ind)
        ind=ind.set_index('ADMIT_ID')
        my_dataset = my_dataset.set_index('ADMIT_ID')
        my_dataset = my_dataset.join(ind,how='left')
        my_dataset = my_dataset.join(gr,how='left')#should start doing 'outer' because now can dx based on labs, except the age isn't there anymore...
        my_dataset = my_dataset.reset_index(level=0)
        l=[]
        gr=[]
        
        #added this on; should wrap into function because does same as above
        and_or_creat = pandas.read_csv(path_to_and_or_creat)
        and_or_creat = and_or_creat.set_index('ADMIT_ID')
        my_dataset = my_dataset.set_index('ADMIT_ID')
        my_dataset = my_dataset.join(and_or_creat, how='left')
        my_dataset = my_dataset.reset_index(level=0)
        and_or_creat = []

    if use_meds==1:
        print "Loading meds..."
        m = pandas.read_csv(path_to_meds, nrows=nrows)
        m = m.set_index('ADMIT_ID')
        colNames = ['_UNIQUE_ID_'+c for c in m.columns]
        m.columns = colNames
        m = m.loc[:,m.sum()>200] #only take meds that have been administered more than 100 times (this is by admission, too). reduces from 8.7 to 3 gb
        my_dataset = my_dataset.set_index('ADMIT_ID')
        my_dataset = my_dataset.join(m,how='left')
        my_dataset.reset_index(level=0,inplace=True)
        m=[]
        
    
    return my_dataset

def set_types(allSamp):
    '''Currently unused.  Was much faster to just save and load, because read_csv default converts to float64.
        maybe 'pd.to_numeric() would be faster than astype(), obviating this save load step
    '''
    codes,other_str,meds,labs,other_cols = get_data_groups(allSamp)
    l=[]
    print "converting codes"
    allSamp[codes]=allSamp[codes].astype(str)
    print "converting other strings"
    allSamp[other_str]=allSamp[other_str].astype(str)
    print "converting meds"
    allSamp[meds]=allSamp[meds].astype(np.float64)
    print "converting labs"
    allSamp[labs]=allSamp[labs].astype(np.float64)
    print "# other cols",len(other_cols)
    for c in other_cols:
        if c in codes or c in other_str:
            allSamp[c]=allSamp[c].astype(str)
        else:
            try:
                allSamp[c]=allSamp[c].astype(np.float64)
            except:
                print "couldn't convert to float64",c
                l.append(c)
    return l

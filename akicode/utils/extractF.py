'''Does most of feature extraction.
Eg, adding all previous lab X, etc.
'''

import pandas
import pdb
import wrang
import numpy as np
import pickle


def load_obj(a_file ):
    with open(a_file, 'rb') as f:#http://stackoverflow.com/questions/19201290/how-to-save-a-dictionary-to-a-file-in-python
        return pickle.load(f)

def concatGroupVar(dfWgroupVar,dfToAddTo,groupVar):
    ''' dfWgroupVar is larger dataframe with index,dfToAddTo is the smaller dataframe,groupVar is 
    grouping variable
    '''
    return pandas.concat([dfWgroupVar[[groupVar]],dfToAddTo],axis=1,copy=False)

def extractFeatures(data,groupVar,drgPrec,dxPrec,cptPxPrec,
                    icdPxPrec,useLabs,useMeds,useAdmin,useAKI_dx,cat_too,
                    #hardcode just in case. will pass from calling script however.
                    dx_path = '/homedirec/user/CMS32_DESC_LONG_SHORT_DX.csv',
                    #locations
                    loc_path = '/homedirec/user/locCode ICU.csv',
                    #icd9 px
                    cms32Px_path = '/homedirec/user/CMS32_DESC_LONG_SHORT_SG.csv',
                    #cpt px
                    cptProd_path = '/homedirec/user/cpt product list.csv',
                    #drg
                    drgs_path = '/homedirec/user/drg2mdcxw2014.csv',
                    #meds
                    meds_path = '/homedirec/user/med_name_dictionary.pkl'
                   
                   ):
    '''data is dataframe, groupVar is eg 'sampleNo', precisions refer to top-level binary
    representation (how far to 'drill up')
    '''
    print "cptPxPrec",cptPxPrec
    data.sort_values([groupVar,'ADMIT_ID'],inplace=True,axis=0)
    cols = data.columns
    feat=[]
    
    if useLabs ==1:

        lab_names = [el for el in cols if 'UNIQUE_LAB_RESULT_VALUE' in el] + [el for el in cols if 'ABNORMAL_LAB_FLAG' in el]       
        labs = data[lab_names].convert_objects(convert_numeric=True)
        labs = labs.round(5)
        labs = labs.rename({c:'LAB_'+c for c in labs.columns})        
        
        
        labs = concatGroupVar(data,labs,groupVar)
        # print labs
        f = {c:['min','mean','max','sum','var'] for c in lab_names} #prod leads to inf in feature vector
        labs = labs.groupby(groupVar).agg(f)
        labs.columns = ['_'.join(col).strip() for col in labs.columns.values]#http://stackoverflow.com/questions/14507794/python-pandas-how-to-flatten-a-hierarchical-index-in-columns
        labs.columns = labs.columns.get_level_values(0)
        #labs = labs.groupby(groupVar).mean()
        
        feat.append(labs)
        print "lab info",labs.info()
        
        #add AKI Dx
    if useAKI_dx:     
        aki_dx_names = ['or', 'and', 'AKI_by_creat', 'AKI_by_admi']
        aki_dx = data[aki_dx_names]
        aki_dx = aki_dx.rename({c:'AKI_DX_' + c for c in aki_dx.columns})
        aki_dx = concatGroupVar(data, aki_dx, groupVar)
        aki_dx = aki_dx.groupby(groupVar).sum()
        feat.append(aki_dx)
        print "aki dx info", aki_dx.info()
        
    if useMeds==1:
        med_names = [el for el in data.columns if '_UNIQUE_ID_MED_' in el]
        medications = data[med_names]
        med_name_dictionary = load_obj(meds_path)#this was made with the actual med data
        names = []
        for el in med_names:
            if '_UNIQUE_ID_MED_DESCRIPTION' in el:
                names.append(el)
            else:
                names.append(str(el)+': ['+str(med_name_dictionary[el])+']')
        #[med_name_dictionary[el] for el in med_names]
        medications.columns=names
        medications = concatGroupVar(data,medications,groupVar)
        #print "med info",medications.info()
        #medications = medications.groupby(groupVar).sum().astype(np.int8)#don't: missing values
        #medications = medications.groupby(groupVar).sum()#medications can be summed too memory intensive
        groupsOfMeds=[]
        for g in np.array_split(medications,50):
            groupsOfMeds.append(g.groupby(groupVar).sum())
        medications=pandas.concat(groupsOfMeds)
        medications = medications.groupby(medications.index).sum() #get the ones that weren't summed before
        print "med info",medications.info()
        feat.append(medications)
    
    if useAdmin==1:
        #keys
        #diagnoses

        dx = pandas.read_csv(dx_path)
        #locations
        loc = pandas.read_csv(loc_path)
        #icd9 px
        cms32Px = pandas.read_csv(cms32Px_path)
        #cpt px
        cptProd = pandas.read_csv(cptProd_path)
        #drg
        drgs = pandas.read_csv(drgs_path)
            
        if cat_too==1:
            #groupVar='PATIENT_NUM'#CHANGE TO SAMPLENO
            #labs

            aDay =['ADMISSION_DAY']
            adDays=wrang.dummize(data[aDay],100,'ADDAY_')
            adDays = concatGroupVar(data,adDays,groupVar)
            adDays = adDays.groupby(groupVar).sum()#.astype(np.int8)
            #print "addays",adDays.info()
            feat.append(adDays)

            dDay  =['DISCHARGE_DAY']
            disDays=wrang.dummize(data[dDay],100,'DISDAY_')
            disDays = concatGroupVar(data,disDays,groupVar)
            disDays = disDays.groupby(groupVar).sum()#.astype(np.int8)
            feat.append(disDays)


            eth = ['ETHNIC_CD']
            ethnicit=wrang.dummize(data[eth],100,'ETHN_')
            ethnicit = concatGroupVar(data,ethnicit,groupVar)
            ethnicit = ethnicit.groupby(groupVar).first()
            feat.append(ethnicit)
            #print "eth",ethnicit.info()
            #print 

            mar = ['MARITAL_CD']
            maritStat=wrang.dummize(data[mar],100,'MARIT_')
            maritStat = concatGroupVar(data,maritStat,groupVar)
            maritStat = maritStat.groupby(groupVar).last()
            feat.append(maritStat)
            #print "marit",maritStat.info()
            #print 

            sex = ['SEX_CD']
            genders = wrang.dummize(data[sex],100,'GENDER_')
            genders = concatGroupVar(data,genders,groupVar)
            genders = genders.groupby(groupVar).last()#last gender
            feat.append(genders)


            #there is some weird bug with the DRGs where there is an entire dataframe (with two series) indexed by a single column of the dummified DRGS. Omitting. 
            # 'DRG1': dummize, Add?
            drg = ['DRG1']#what happened to drg2?
            diagnosisRelGroup = wrang.DummizeAndName(data[drg],drgPrec,drgs,'drg','drgdesc','DRG',1)
            diagnosisRelGroup = concatGroupVar(data,diagnosisRelGroup,groupVar)
            diagnosisRelGroup = diagnosisRelGroup.groupby(groupVar).sum()#.astype(np.int8)#I think that having a code assigned is an event. Therefore by adding we gain information whereas by averaging or just taking the last one, we would lose information
            feat.append(diagnosisRelGroup)


            ins = ['INSURANCE'] #use get insurance
            insurances=wrang.getIns(data[ins])
            insurances = concatGroupVar(data,insurances,groupVar)
            insurances = insurances.groupby(groupVar).last()#get most recent insurance plans
            feat.append(insurances)
            #print "in",insurances.info()
            #print 

            # 'DISCHARGE_DISPOSITION_1': dummize, Add (if they go to a location multiple times it should be weighted higher)
            dd = ['DISCHARGE_DISPOSITION_1']
            disDisp = wrang.dummize(data[dd],1000,'DISDIS_')
            disDisp = concatGroupVar(data,disDisp,groupVar)
            disDisp = disDisp.groupby(groupVar).sum()#.astype(np.int8)#maybe .last() would be better?
            feat.append(disDisp)

                    # DIAGNOS: dummize, add? Add b/c it's like saying: hospitalized and X was remarked at a hospital stay is different than simply having dx x. If it was only coded once, it may be less severe than if coded multiple time. The feature is therefore the number of assignments of Dx X, not necesarily the presence or absence of dx X. I think we lose information if we don't add. 
            #dxs = [el for el in cols if 'DIAGNOS' in el]
            dxs=[el for el in data.columns if 'DIAGNOSES_' in el]

            diagnoses = wrang.DummizeAndName(data[dxs],dxPrec,dx,'DIAGNOSIS CODE','LONG DESCRIPTION','DX',1)
            diagnoses = concatGroupVar(data,diagnoses,groupVar)
            diagnoses = diagnoses.groupby(groupVar).sum()#.astype(np.int8)#again, diagnosis codes are only assigned based on relevance to current stay. Therefore, need to get them all
            #print wrang.dummize(data[dx],4,'Dx_')
            print "# columns in dx", len(diagnoses.columns)
            print "# distinct colums in dx",len(set(diagnoses.columns))
            feat.append(diagnoses)
            print "dx", diagnoses.info()
            #print 

            # CPT4PX: dummize, add! Prodecures can occur multiple times
            cpt4px=[el for el in data.columns if 'CPT4_PROCEDURES_' in el]
 
            cptProcedures = wrang.DummizeAndName(data[cpt4px],cptPxPrec,cptProd,'CPT-4 Code','CPT-4 Procedure','CPT4PX',1)
            cptProcedures = concatGroupVar(data,cptProcedures,groupVar)
            cptProcedures = cptProcedures.groupby(groupVar).sum()#.astype(np.int8)#number of px X performed on patient, ever
            feat.append(cptProcedures)
            print "cpt",cptProcedures.info()
            #print 

            # ICD9PX: dummize, add! Prodecures can occur multiple times
            icd9px=[el for el in data.columns if 'ICD9_PROCEDURES_' in el]
   
            icdProcedures = wrang.DummizeAndName(data[icd9px],icdPxPrec,cms32Px,'PROCEDURE CODE','LONG DESCRIPTION','ICD9Px',1)
            icdProcedures = concatGroupVar(data,icdProcedures,groupVar)
            icdProcedures = icdProcedures.groupby(groupVar).sum()#.astype(np.int8)#same reasoning as above
            feat.append(icdProcedures)
            print "icd9",icdProcedures.info()

            # L1-L25: dummize, add (number of times gone to X)
            lx=['L1', 'L2', 'L3', 'L4', 'L5', 'L6', 'L7', 'L8', 'L9', 'L10', 'L11', 'L12', 'L13', 'L14', 'L15', 'L16', 'L17', 'L18', 'L19', 'L20', 'L21', 'L22', 'L23', 'L24', 'L25']
            data[lx] = data[lx].applymap(lambda x: str(x).replace('.0',''))#Some are floats with .0's.  Convert to string and strip trailing .0's. CHECK THIS IN COLUMNS!
            locations = wrang.DummizeAndName(data[lx],1000,loc,'Location','Name','LX',1)
            locations = concatGroupVar(data,locations,groupVar)
            locations = locations.groupby(groupVar).sum()#.astype(np.int8)#I think we definitely could add these. However, also might be better to get the last locations.
            feat.append(locations)
            #print "loc",locations.info()
        
        #print 
        # 'AGE_ON_ADMISSION' : don't dummize, can't add
        age = ['AGE_ON_ADMISSION']
        ages=data[age]
        ages = concatGroupVar(data,ages,groupVar)
        ages = ages.groupby(groupVar).max()#max age. Should also be .last()
        feat.append(ages)
        
        # 'TOT_LOS': don't dummize, Add (scale)
        los = ['TOT_LOS']
        lengthOfStay = data[los]
        lengthOfStay = concatGroupVar(data,lengthOfStay,groupVar)
        lengthOfStay = lengthOfStay.groupby(groupVar).sum()#also, last might be more informative
        feat.append(lengthOfStay)


        #print 
    ptNum=['PATIENT_NUM']
    ptNum=concatGroupVar(data,data[ptNum],groupVar)
    ptNum=ptNum.groupby(groupVar).last()
    feat.append(ptNum)
    
    #get if it's a target (positive sample) or not
    target=['target']
    target=concatGroupVar(data,data[target],groupVar)
    target=target.groupby(groupVar).mean()#this will throw error if targets are different
    feat.append(target)
    return feat

def joinFeatures(feat):
    print "in join"
    '''feat is a list of datframes from extractFeatures()'''
    firstFeat = feat.pop(0)
    def recursiveJoin(df):
        if feat != []:
            #newdf = df.join(feat.pop(0))
            #print df
            return recursiveJoin(df.join(feat.pop(0)))
        else:
            return df
    fullDf=recursiveJoin(firstFeat)
    return fullDf

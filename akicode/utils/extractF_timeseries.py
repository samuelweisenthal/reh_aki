'''Used for RNN input. very similar to extractF.py
but does not aggregate features over time - rather returns 
sequence.  This is a bit messier than extractF.py, so refer there'''

import pandas
import wrang
import numpy as np
import pickle


def load_obj(a_file ):
    with open(a_file, 'rb') as f:#http://stackoverflow.com/questions/19201290/how-to-save-a-dictionary-to-a-file-in-python
        return pickle.load(f)

def concatGroupVar(dfWgroupVar,dfToAddTo,groupVar, really=0):
    ''' dfWgroupVar is larger dataframe with index,dfToAddTo is the smaller dataframe,groupVar is 
    grouping variable
    '''
    if really:
        df = pandas.concat([dfWgroupVar[[groupVar]],dfToAddTo],axis=1,copy=False)
        return df
    else:
    #df.set_index(groupVar)
        return dfToAddTo

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
        #lab_names=['CO2', 'GFRC', 'CREAT', 'CL', 'PCREA', 'DB', 'CA', 'GFRB', 'ALB', 'TP', 'GAP', 'PO4', 'AST', 'UN', 'HA1C', 'K', 'ALT', 'HGB', 'GLU']
        
        lab_names = [el for el in cols if 'UNIQUE_LAB_RESULT_VALUE' in el] + ['ABNORMAL_LAB_FLAG: AB,GFRB', 'ABNORMAL_LAB_FLAG: AB,GFRC', 'ABNORMAL_LAB_FLAG: H,ALB', 'ABNORMAL_LAB_FLAG: H,ALT', 'ABNORMAL_LAB_FLAG: H,AST', 'ABNORMAL_LAB_FLAG: H,CA', 'ABNORMAL_LAB_FLAG: H,CL', 'ABNORMAL_LAB_FLAG: H,CO2', 'ABNORMAL_LAB_FLAG: H,CREAT', 'ABNORMAL_LAB_FLAG: H,DB', 'ABNORMAL_LAB_FLAG: H,GAP', 'ABNORMAL_LAB_FLAG: H,GLU', 'ABNORMAL_LAB_FLAG: H,HA1C', 'ABNORMAL_LAB_FLAG: H,HGB', 'ABNORMAL_LAB_FLAG: H,K', 'ABNORMAL_LAB_FLAG: H,PCREA', 'ABNORMAL_LAB_FLAG: H,PO4', 'ABNORMAL_LAB_FLAG: H,TP', 'ABNORMAL_LAB_FLAG: H,UN', 'ABNORMAL_LAB_FLAG: HH,CA', 'ABNORMAL_LAB_FLAG: HH,CL', 'ABNORMAL_LAB_FLAG: HH,CO2', 'ABNORMAL_LAB_FLAG: HH,GLU', 'ABNORMAL_LAB_FLAG: HH,HA1C', 'ABNORMAL_LAB_FLAG: HH,K', 'ABNORMAL_LAB_FLAG: HH,PO4', 'ABNORMAL_LAB_FLAG: L,ALB', 'ABNORMAL_LAB_FLAG: L,CA', 'ABNORMAL_LAB_FLAG: L,CL', 'ABNORMAL_LAB_FLAG: L,CO2', 'ABNORMAL_LAB_FLAG: L,CREAT', 'ABNORMAL_LAB_FLAG: L,GAP', 'ABNORMAL_LAB_FLAG: L,GLU', 'ABNORMAL_LAB_FLAG: L,HA1C', 'ABNORMAL_LAB_FLAG: L,HGB', 'ABNORMAL_LAB_FLAG: L,K', 'ABNORMAL_LAB_FLAG: L,PCREA', 'ABNORMAL_LAB_FLAG: L,PO4', 'ABNORMAL_LAB_FLAG: L,TP', 'ABNORMAL_LAB_FLAG: L,UN', 'ABNORMAL_LAB_FLAG: LL,CA', 'ABNORMAL_LAB_FLAG: LL,CL', 'ABNORMAL_LAB_FLAG: LL,CO2', 'ABNORMAL_LAB_FLAG: LL,GLU', 'ABNORMAL_LAB_FLAG: LL,K', 'ABNORMAL_LAB_FLAG: LL,PO4', 'ABNORMAL_LAB_FLAG: nan,ALB', 'ABNORMAL_LAB_FLAG: nan,ALT', 'ABNORMAL_LAB_FLAG: nan,AST', 'ABNORMAL_LAB_FLAG: nan,CA', 'ABNORMAL_LAB_FLAG: nan,CL', 'ABNORMAL_LAB_FLAG: nan,CO2', 'ABNORMAL_LAB_FLAG: nan,CREAT', 'ABNORMAL_LAB_FLAG: nan,DB', 'ABNORMAL_LAB_FLAG: nan,GAP', 'ABNORMAL_LAB_FLAG: nan,GFRB', 'ABNORMAL_LAB_FLAG: nan,GFRC', 'ABNORMAL_LAB_FLAG: nan,GLU', 'ABNORMAL_LAB_FLAG: nan,HA1C', 'ABNORMAL_LAB_FLAG: nan,HGB', 'ABNORMAL_LAB_FLAG: nan,K', 'ABNORMAL_LAB_FLAG: nan,PCREA', 'ABNORMAL_LAB_FLAG: nan,PO4', 'ABNORMAL_LAB_FLAG: nan,TP', 'ABNORMAL_LAB_FLAG: nan,UN']
        
        labs = data[lab_names].convert_objects(convert_numeric=True)
        labs = labs.round(5)
        labs = labs.rename({c:'LAB_'+c for c in labs.columns})        
        
        
        #labs = concatGroupVar(data,labs,groupVar)
        # print labs
        #f = {c:['min','mean','max','sum','var'] for c in lab_names} #prod leads to inf in feature vector
        #labs = labs.groupby(groupVar).agg(f)
        #labs.columns = ['_'.join(col).strip() for col in labs.columns.values]#http://stackoverflow.com/questions/14507794/python-pandas-how-to-flatten-a-hierarchical-index-in-columns
        #labs.columns = labs.columns.get_level_values(0)
        #labs = labs.groupby(groupVar).mean()
        
        feat.append(labs)
        print "lab info",labs.info()
        
        #add AKI Dx
    if useAKI_dx:     
        aki_dx_names = ['or', 'and', 'AKI_by_creat', 'AKI_by_admi']
        aki_dx = data[aki_dx_names]
        aki_dx = aki_dx.rename({c:'AKI_DX_' + c for c in aki_dx.columns})
        aki_dx = concatGroupVar(data, aki_dx, groupVar)
        #aki_dx = aki_dx.groupby(groupVar).sum()
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
        #groupsOfMeds=[]
        #for g in np.array_split(medications,50):
        #    groupsOfMeds.append(g.groupby(groupVar).sum())
        #medications=pandas.concat(groupsOfMeds)
        #medications = medications.groupby(medications.index).sum() #get the ones that weren't summed before
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
            #adDays = adDays.groupby(groupVar).sum()#.astype(np.int8)
            #print "addays",adDays.info()
            feat.append(adDays)

            dDay  =['DISCHARGE_DAY']
            disDays=wrang.dummize(data[dDay],100,'DISDAY_')
            disDays = concatGroupVar(data,disDays,groupVar)
            #disDays = disDays.groupby(groupVar).sum()#.astype(np.int8)
            feat.append(disDays)
            #print "diddays",disDays.info()
            #print 


            # 'MARITAL_CD': dummize, can't add
            # 'ETHNIC_CD' : dummize, can't add
            # 'SEX_CD': dummize, can't add

            eth = ['ETHNIC_CD']
            ethnicit=wrang.dummize(data[eth],100,'ETHN_')
            ethnicit = concatGroupVar(data,ethnicit,groupVar)
            #ethnicit = ethnicit.groupby(groupVar).first()
            feat.append(ethnicit)
            #print "eth",ethnicit.info()
            #print 

            mar = ['MARITAL_CD']
            maritStat=wrang.dummize(data[mar],100,'MARIT_')
            maritStat = concatGroupVar(data,maritStat,groupVar)
            #maritStat = maritStat.groupby(groupVar).last()
            feat.append(maritStat)
            #print "marit",maritStat.info()
            #print 

            sex = ['SEX_CD']
            genders = wrang.dummize(data[sex],100,'GENDER_')
            genders = concatGroupVar(data,genders,groupVar)
            #genders = genders.groupby(groupVar).last()#last gender
            feat.append(genders)


            #there is some weird bug with the DRGs where there is an entire dataframe (with two series) indexed by a single column of the dummified DRGS. Omitting. 
            # 'DRG1': dummize, Add?
            drg = ['DRG1']#what happened to drg2?
            diagnosisRelGroup = wrang.DummizeAndName(data[drg],drgPrec,drgs,'drg','drgdesc','DRG',1)
            diagnosisRelGroup = concatGroupVar(data,diagnosisRelGroup,groupVar)
            #diagnosisRelGroup = diagnosisRelGroup.groupby(groupVar).sum()#.astype(np.int8)#I think that having a code assigned is an event. Therefore by adding we gain information whereas by averaging or just taking the last one, we would lose information
            feat.append(diagnosisRelGroup)


#             # 'TOT_CHARGE1': don't dummize, Add!
#             # 'PRO_CHARGE1': don't dummize, Add!
#             # 'HSP_CHARGE1': don't dummize, Add!
#             # 'GRAND_TOTAL': don't dummize, Add!
#             # 'GRAND_TOTAL_PRO'
#             # 'GRAND_TOTAL_HSP'
        #     pay = ['TOT_CHARGE1', 'PRO_CHARGE1', 'HSP_CHARGE1', 'GRAND_TOTAL',
        #            'GRAND_TOTAL_PRO', 'GRAND_TOTAL_HSP'] #don't dummize 
        #     payments=data[pay].convert_objects(convert_numeric=True)
        #     payments = concatGroupVar(data,payments,groupVar)
        #     payments = payments.groupby(groupVar).sum()#it would perhaps be best to take the last payment? reflects the patient's state better?
        #     feat.append(payments) #often missing

            # 'INSURANCE': dummize. don't add
            ins = ['INSURANCE'] #use get insurance
            insurances=wrang.getIns(data[ins])
            insurances = concatGroupVar(data,insurances,groupVar)
            #insurances = insurances.groupby(groupVar).last()#get most recent insurance plans
            feat.append(insurances)
            #print "in",insurances.info()
            #print 

            # 'DISCHARGE_DISPOSITION_1': dummize, Add (if they go to a location multiple times it should be weighted higher)
            dd = ['DISCHARGE_DISPOSITION_1']
            disDisp = wrang.dummize(data[dd],1000,'DISDIS_')
            disDisp = concatGroupVar(data,disDisp,groupVar)
            #disDisp = disDisp.groupby(groupVar).sum()#.astype(np.int8)#maybe .last() would be better?
            feat.append(disDisp)

                    # DIAGNOS: dummize, add? Add b/c it's like saying: hospitalized and X was remarked at a hospital stay is different than simply having dx x. If it was only coded once, it may be less severe than if coded multiple time. The feature is therefore the number of assignments of Dx X, not necesarily the presence or absence of dx X. I think we lose information if we don't add. 
            #dxs = [el for el in cols if 'DIAGNOS' in el]
            #dxs=[el for el in data.columns if 'DIAGNOSES_' in el]
            dxs = ['DIAGNOSES_0', 'DIAGNOSES_1', 'DIAGNOSES_2', 'DIAGNOSES_3', 'DIAGNOSES_4', 'DIAGNOSES_5', 'DIAGNOSES_6', 'DIAGNOSES_7', 'DIAGNOSES_8', 'DIAGNOSES_9', 'DIAGNOSES_10', 'DIAGNOSES_11', 'DIAGNOSES_12', 'DIAGNOSES_13', 'DIAGNOSES_14', 'DIAGNOSES_15', 'DIAGNOSES_16', 'DIAGNOSES_17', 'DIAGNOSES_18', 'DIAGNOSES_19', 'DIAGNOSES_20', 'DIAGNOSES_21', 'DIAGNOSES_22', 'DIAGNOSES_23', 'DIAGNOSES_24', 'DIAGNOSES_25', 'DIAGNOSES_26', 'DIAGNOSES_27', 'DIAGNOSES_28', 'DIAGNOSES_29', 'DIAGNOSES_30', 'DIAGNOSES_31', 'DIAGNOSES_32', 'DIAGNOSES_33', 'DIAGNOSES_34', 'DIAGNOSES_35', 'DIAGNOSES_36', 'DIAGNOSES_37', 'DIAGNOSES_38', 'DIAGNOSES_39', 'DIAGNOSES_40', 'DIAGNOSES_41', 'DIAGNOSES_42', 'DIAGNOSES_43', 'DIAGNOSES_44', 'DIAGNOSES_45', 'DIAGNOSES_46', 'DIAGNOSES_47', 'DIAGNOSES_48', 'DIAGNOSES_49', 'DIAGNOSES_50', 'DIAGNOSES_51', 'DIAGNOSES_52', 'DIAGNOSES_53']
            diagnoses = wrang.DummizeAndName(data[dxs],dxPrec,dx,'DIAGNOSIS CODE','LONG DESCRIPTION','DX',1)
            diagnoses = concatGroupVar(data,diagnoses,groupVar)
            #diagnoses = diagnoses.groupby(groupVar).sum()#.astype(np.int8)#again, diagnosis codes are only assigned based on relevance to current stay. Therefore, need to get them all
            #print wrang.dummize(data[dx],4,'Dx_')
            print "# columns in dx", len(diagnoses.columns)
            print "# distinct colums in dx",len(set(diagnoses.columns))
            feat.append(diagnoses)
            print "dx", diagnoses.info()
            #print 

            # CPT4PX: dummize, add! Prodecures can occur multiple times
            #cpt4px=[el for el in data.columns if 'CPT4_PROCEDURES_' in el]
            cpt4px = ['CPT4_PROCEDURES_0', 'CPT4_PROCEDURES_1', 'CPT4_PROCEDURES_2', 'CPT4_PROCEDURES_3', 'CPT4_PROCEDURES_4', 'CPT4_PROCEDURES_5', 'CPT4_PROCEDURES_6', 'CPT4_PROCEDURES_7', 'CPT4_PROCEDURES_8', 'CPT4_PROCEDURES_9', 'CPT4_PROCEDURES_10', 'CPT4_PROCEDURES_11', 'CPT4_PROCEDURES_12', 'CPT4_PROCEDURES_13', 'CPT4_PROCEDURES_14', 'CPT4_PROCEDURES_15', 'CPT4_PROCEDURES_16', 'CPT4_PROCEDURES_17', 'CPT4_PROCEDURES_18', 'CPT4_PROCEDURES_19', 'CPT4_PROCEDURES_20', 'CPT4_PROCEDURES_21', 'CPT4_PROCEDURES_22', 'CPT4_PROCEDURES_23', 'CPT4_PROCEDURES_24', 'CPT4_PROCEDURES_25', 'CPT4_PROCEDURES_26', 'CPT4_PROCEDURES_27', 'CPT4_PROCEDURES_28', 'CPT4_PROCEDURES_29', 'CPT4_PROCEDURES_30', 'CPT4_PROCEDURES_31', 'CPT4_PROCEDURES_32', 'CPT4_PROCEDURES_33', 'CPT4_PROCEDURES_34', 'CPT4_PROCEDURES_35', 'CPT4_PROCEDURES_36', 'CPT4_PROCEDURES_37', 'CPT4_PROCEDURES_38', 'CPT4_PROCEDURES_39', 'CPT4_PROCEDURES_40', 'CPT4_PROCEDURES_41', 'CPT4_PROCEDURES_42', 'CPT4_PROCEDURES_43', 'CPT4_PROCEDURES_44', 'CPT4_PROCEDURES_45', 'CPT4_PROCEDURES_46', 'CPT4_PROCEDURES_47', 'CPT4_PROCEDURES_48', 'CPT4_PROCEDURES_49', 'CPT4_PROCEDURES_50', 'CPT4_PROCEDURES_51', 'CPT4_PROCEDURES_52', 'CPT4_PROCEDURES_53', 'CPT4_PROCEDURES_54', 'CPT4_PROCEDURES_55', 'CPT4_PROCEDURES_56', 'CPT4_PROCEDURES_57', 'CPT4_PROCEDURES_58', 'CPT4_PROCEDURES_59', 'CPT4_PROCEDURES_60', 'CPT4_PROCEDURES_61', 'CPT4_PROCEDURES_62', 'CPT4_PROCEDURES_63', 'CPT4_PROCEDURES_64', 'CPT4_PROCEDURES_65', 'CPT4_PROCEDURES_66', 'CPT4_PROCEDURES_67', 'CPT4_PROCEDURES_68', 'CPT4_PROCEDURES_69', 'CPT4_PROCEDURES_70', 'CPT4_PROCEDURES_71', 'CPT4_PROCEDURES_72', 'CPT4_PROCEDURES_73', 'CPT4_PROCEDURES_74', 'CPT4_PROCEDURES_75', 'CPT4_PROCEDURES_76', 'CPT4_PROCEDURES_77', 'CPT4_PROCEDURES_78', 'CPT4_PROCEDURES_79', 'CPT4_PROCEDURES_80', 'CPT4_PROCEDURES_81', 'CPT4_PROCEDURES_82', 'CPT4_PROCEDURES_83', 'CPT4_PROCEDURES_84', 'CPT4_PROCEDURES_85', 'CPT4_PROCEDURES_86', 'CPT4_PROCEDURES_87', 'CPT4_PROCEDURES_88', 'CPT4_PROCEDURES_89', 'CPT4_PROCEDURES_90', 'CPT4_PROCEDURES_91', 'CPT4_PROCEDURES_92', 'CPT4_PROCEDURES_93', 'CPT4_PROCEDURES_94', 'CPT4_PROCEDURES_95', 'CPT4_PROCEDURES_96', 'CPT4_PROCEDURES_97', 'CPT4_PROCEDURES_98', 'CPT4_PROCEDURES_99', 'CPT4_PROCEDURES_100', 'CPT4_PROCEDURES_101', 'CPT4_PROCEDURES_102', 'CPT4_PROCEDURES_103', 'CPT4_PROCEDURES_104', 'CPT4_PROCEDURES_105', 'CPT4_PROCEDURES_106', 'CPT4_PROCEDURES_107', 'CPT4_PROCEDURES_108', 'CPT4_PROCEDURES_109', 'CPT4_PROCEDURES_110', 'CPT4_PROCEDURES_111', 'CPT4_PROCEDURES_112', 'CPT4_PROCEDURES_113', 'CPT4_PROCEDURES_114', 'CPT4_PROCEDURES_115', 'CPT4_PROCEDURES_116', 'CPT4_PROCEDURES_117', 'CPT4_PROCEDURES_118', 'CPT4_PROCEDURES_119', 'CPT4_PROCEDURES_120', 'CPT4_PROCEDURES_121', 'CPT4_PROCEDURES_122', 'CPT4_PROCEDURES_123', 'CPT4_PROCEDURES_124', 'CPT4_PROCEDURES_125', 'CPT4_PROCEDURES_126', 'CPT4_PROCEDURES_127', 'CPT4_PROCEDURES_128', 'CPT4_PROCEDURES_129', 'CPT4_PROCEDURES_130', 'CPT4_PROCEDURES_131', 'CPT4_PROCEDURES_132', 'CPT4_PROCEDURES_133', 'CPT4_PROCEDURES_134', 'CPT4_PROCEDURES_135', 'CPT4_PROCEDURES_136', 'CPT4_PROCEDURES_137', 'CPT4_PROCEDURES_138', 'CPT4_PROCEDURES_139', 'CPT4_PROCEDURES_140', 'CPT4_PROCEDURES_141', 'CPT4_PROCEDURES_142', 'CPT4_PROCEDURES_143', 'CPT4_PROCEDURES_144', 'CPT4_PROCEDURES_145', 'CPT4_PROCEDURES_146', 'CPT4_PROCEDURES_147', 'CPT4_PROCEDURES_148', 'CPT4_PROCEDURES_149', 'CPT4_PROCEDURES_150', 'CPT4_PROCEDURES_151', 'CPT4_PROCEDURES_152', 'CPT4_PROCEDURES_153', 'CPT4_PROCEDURES_154', 'CPT4_PROCEDURES_155', 'CPT4_PROCEDURES_156', 'CPT4_PROCEDURES_157', 'CPT4_PROCEDURES_158', 'CPT4_PROCEDURES_159', 'CPT4_PROCEDURES_160', 'CPT4_PROCEDURES_161', 'CPT4_PROCEDURES_162', 'CPT4_PROCEDURES_163', 'CPT4_PROCEDURES_164', 'CPT4_PROCEDURES_165', 'CPT4_PROCEDURES_166', 'CPT4_PROCEDURES_167', 'CPT4_PROCEDURES_168', 'CPT4_PROCEDURES_169', 'CPT4_PROCEDURES_170', 'CPT4_PROCEDURES_171', 'CPT4_PROCEDURES_172', 'CPT4_PROCEDURES_173', 'CPT4_PROCEDURES_174', 'CPT4_PROCEDURES_175', 'CPT4_PROCEDURES_176', 'CPT4_PROCEDURES_177', 'CPT4_PROCEDURES_178', 'CPT4_PROCEDURES_179', 'CPT4_PROCEDURES_180', 'CPT4_PROCEDURES_181', 'CPT4_PROCEDURES_182', 'CPT4_PROCEDURES_183', 'CPT4_PROCEDURES_184', 'CPT4_PROCEDURES_185', 'CPT4_PROCEDURES_186', 'CPT4_PROCEDURES_187', 'CPT4_PROCEDURES_188', 'CPT4_PROCEDURES_189', 'CPT4_PROCEDURES_190', 'CPT4_PROCEDURES_191', 'CPT4_PROCEDURES_192', 'CPT4_PROCEDURES_193', 'CPT4_PROCEDURES_194', 'CPT4_PROCEDURES_195', 'CPT4_PROCEDURES_196', 'CPT4_PROCEDURES_197', 'CPT4_PROCEDURES_198', 'CPT4_PROCEDURES_199', 'CPT4_PROCEDURES_200', 'CPT4_PROCEDURES_201', 'CPT4_PROCEDURES_202', 'CPT4_PROCEDURES_203', 'CPT4_PROCEDURES_204', 'CPT4_PROCEDURES_205', 'CPT4_PROCEDURES_206', 'CPT4_PROCEDURES_207', 'CPT4_PROCEDURES_208', 'CPT4_PROCEDURES_209', 'CPT4_PROCEDURES_210', 'CPT4_PROCEDURES_211', 'CPT4_PROCEDURES_212', 'CPT4_PROCEDURES_213', 'CPT4_PROCEDURES_214', 'CPT4_PROCEDURES_215', 'CPT4_PROCEDURES_216', 'CPT4_PROCEDURES_217', 'CPT4_PROCEDURES_218', 'CPT4_PROCEDURES_219', 'CPT4_PROCEDURES_220', 'CPT4_PROCEDURES_221', 'CPT4_PROCEDURES_222', 'CPT4_PROCEDURES_223', 'CPT4_PROCEDURES_224', 'CPT4_PROCEDURES_225', 'CPT4_PROCEDURES_226', 'CPT4_PROCEDURES_227', 'CPT4_PROCEDURES_228', 'CPT4_PROCEDURES_229', 'CPT4_PROCEDURES_230', 'CPT4_PROCEDURES_231', 'CPT4_PROCEDURES_232', 'CPT4_PROCEDURES_233', 'CPT4_PROCEDURES_234', 'CPT4_PROCEDURES_235', 'CPT4_PROCEDURES_236', 'CPT4_PROCEDURES_237', 'CPT4_PROCEDURES_238', 'CPT4_PROCEDURES_239', 'CPT4_PROCEDURES_240', 'CPT4_PROCEDURES_241', 'CPT4_PROCEDURES_242', 'CPT4_PROCEDURES_243', 'CPT4_PROCEDURES_244', 'CPT4_PROCEDURES_245', 'CPT4_PROCEDURES_246', 'CPT4_PROCEDURES_247', 'CPT4_PROCEDURES_248', 'CPT4_PROCEDURES_249', 'CPT4_PROCEDURES_250', 'CPT4_PROCEDURES_251', 'CPT4_PROCEDURES_252', 'CPT4_PROCEDURES_253', 'CPT4_PROCEDURES_254', 'CPT4_PROCEDURES_255', 'CPT4_PROCEDURES_256', 'CPT4_PROCEDURES_257', 'CPT4_PROCEDURES_258', 'CPT4_PROCEDURES_259', 'CPT4_PROCEDURES_260', 'CPT4_PROCEDURES_261', 'CPT4_PROCEDURES_262', 'CPT4_PROCEDURES_263', 'CPT4_PROCEDURES_264', 'CPT4_PROCEDURES_265', 'CPT4_PROCEDURES_266', 'CPT4_PROCEDURES_267', 'CPT4_PROCEDURES_268', 'CPT4_PROCEDURES_269', 'CPT4_PROCEDURES_270', 'CPT4_PROCEDURES_271', 'CPT4_PROCEDURES_272', 'CPT4_PROCEDURES_273', 'CPT4_PROCEDURES_274', 'CPT4_PROCEDURES_275', 'CPT4_PROCEDURES_276', 'CPT4_PROCEDURES_277', 'CPT4_PROCEDURES_278', 'CPT4_PROCEDURES_279', 'CPT4_PROCEDURES_280', 'CPT4_PROCEDURES_281', 'CPT4_PROCEDURES_282', 'CPT4_PROCEDURES_283', 'CPT4_PROCEDURES_284', 'CPT4_PROCEDURES_285', 'CPT4_PROCEDURES_286', 'CPT4_PROCEDURES_287', 'CPT4_PROCEDURES_288', 'CPT4_PROCEDURES_289', 'CPT4_PROCEDURES_290', 'CPT4_PROCEDURES_291', 'CPT4_PROCEDURES_292', 'CPT4_PROCEDURES_293', 'CPT4_PROCEDURES_294', 'CPT4_PROCEDURES_295', 'CPT4_PROCEDURES_296', 'CPT4_PROCEDURES_297', 'CPT4_PROCEDURES_298', 'CPT4_PROCEDURES_299', 'CPT4_PROCEDURES_300', 'CPT4_PROCEDURES_301', 'CPT4_PROCEDURES_302', 'CPT4_PROCEDURES_303', 'CPT4_PROCEDURES_304', 'CPT4_PROCEDURES_305', 'CPT4_PROCEDURES_306', 'CPT4_PROCEDURES_307', 'CPT4_PROCEDURES_308', 'CPT4_PROCEDURES_309', 'CPT4_PROCEDURES_310', 'CPT4_PROCEDURES_311', 'CPT4_PROCEDURES_312', 'CPT4_PROCEDURES_313']
            cptProcedures = wrang.DummizeAndName(data[cpt4px],cptPxPrec,cptProd,'CPT-4 Code','CPT-4 Procedure','CPT4PX',1)
            cptProcedures = concatGroupVar(data,cptProcedures,groupVar)
            #cptProcedures = cptProcedures.groupby(groupVar).sum()#.astype(np.int8)#number of px X performed on patient, ever
            feat.append(cptProcedures)
            print "cpt",cptProcedures.info()
            #print 

            # ICD9PX: dummize, add! Prodecures can occur multiple times
            #icd9px=[el for el in data.columns if 'ICD9_PROCEDURES_' in el]
            icd9px = ['ICD9_PROCEDURES_0', 'ICD9_PROCEDURES_1', 'ICD9_PROCEDURES_2', 'ICD9_PROCEDURES_3', 'ICD9_PROCEDURES_4', 'ICD9_PROCEDURES_5', 'ICD9_PROCEDURES_6', 'ICD9_PROCEDURES_7', 'ICD9_PROCEDURES_8', 'ICD9_PROCEDURES_9', 'ICD9_PROCEDURES_10', 'ICD9_PROCEDURES_11', 'ICD9_PROCEDURES_12', 'ICD9_PROCEDURES_13', 'ICD9_PROCEDURES_14', 'ICD9_PROCEDURES_15', 'ICD9_PROCEDURES_16', 'ICD9_PROCEDURES_17', 'ICD9_PROCEDURES_18', 'ICD9_PROCEDURES_19', 'ICD9_PROCEDURES_20', 'ICD9_PROCEDURES_21', 'ICD9_PROCEDURES_22', 'ICD9_PROCEDURES_23', 'ICD9_PROCEDURES_24', 'ICD9_PROCEDURES_25', 'ICD9_PROCEDURES_26', 'ICD9_PROCEDURES_27', 'ICD9_PROCEDURES_28', 'ICD9_PROCEDURES_29', 'ICD9_PROCEDURES_30', 'ICD9_PROCEDURES_31', 'ICD9_PROCEDURES_32', 'ICD9_PROCEDURES_33', 'ICD9_PROCEDURES_34', 'ICD9_PROCEDURES_35', 'ICD9_PROCEDURES_36', 'ICD9_PROCEDURES_37']
            icdProcedures = wrang.DummizeAndName(data[icd9px],icdPxPrec,cms32Px,'PROCEDURE CODE','LONG DESCRIPTION','ICD9Px',1)
            icdProcedures = concatGroupVar(data,icdProcedures,groupVar)
            #icdProcedures = icdProcedures.groupby(groupVar).sum()#.astype(np.int8)#same reasoning as above
            feat.append(icdProcedures)
            print "icd9",icdProcedures.info()

            # L1-L25: dummize, add (number of times gone to X)
            lx=['L1', 'L2', 'L3', 'L4', 'L5', 'L6', 'L7', 'L8', 'L9', 'L10', 'L11', 'L12', 'L13', 'L14', 'L15', 'L16', 'L17', 'L18', 'L19', 'L20', 'L21', 'L22', 'L23', 'L24', 'L25']
            data[lx] = data[lx].applymap(lambda x: str(x).replace('.0',''))#Some are floats with .0's.  Convert to string and strip trailing .0's. CHECK THIS IN COLUMNS!
            locations = wrang.DummizeAndName(data[lx],1000,loc,'Location','Name','LX',1)
            locations = concatGroupVar(data,locations,groupVar)
            #locations = locations.groupby(groupVar).sum()#.astype(np.int8)#I think we definitely could add these. However, also might be better to get the last locations.
            feat.append(locations)
            #print "loc",locations.info()
        
        #print 
        # 'AGE_ON_ADMISSION' : don't dummize, can't add
        age = ['AGE_ON_ADMISSION']
        ages=data[age]
        ages = concatGroupVar(data,ages,groupVar)
        #ages = ages.groupby(groupVar).max()#max age. Should also be .last()
        feat.append(ages)
        
        # 'TOT_LOS': don't dummize, Add (scale)
        los = ['TOT_LOS']
        lengthOfStay = data[los]
        lengthOfStay = concatGroupVar(data,lengthOfStay,groupVar)
        #lengthOfStay = lengthOfStay.groupby(groupVar).sum()#also, last might be more informative
        feat.append(lengthOfStay)
        
        #suspect there are missing values here as with hospital charge. Need to 
        #also figure out here how to impute
        # LOS_1-L0S_25: ?#apparently only LOS_1
    #   In [36]: [el for el in samples.columns if 'LOS_' in el]
    #   Out[36]: ['LOS_1']
#         loss=[el for el in cols if 'LOS_' in el]#don't dummize
#         lengthsOfStay=data[loss]
#         lengthsOfStay = concatGroupVar(data,lengthsOfStay,groupVar)
#         lengthsOfStay = lengthsOfStay.groupby(groupVar).sum()#add
#         feat.append(lengthsOfStay)

        #print 
    ptNum=['PATIENT_NUM']
    ptNum=concatGroupVar(data,data[ptNum],groupVar)
    #ptNum=ptNum.groupby(groupVar).last()
    feat.append(ptNum)
    
    #get if it's a target (positive sample) or not
    target=['target']
    target=concatGroupVar(data,data[target],groupVar,really=1)
    #target=target.groupby(groupVar).mean()#this will throw error if targets are different
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

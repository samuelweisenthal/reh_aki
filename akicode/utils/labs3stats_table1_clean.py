'''Makes table 1.  It gathers some summary statistics 
about the cohort. This was originally
an ipython notebook, so it's a bit messy, but it runs.
'''

import pandas
from sklearn.decomposition import NMF
from sklearn.preprocessing import Imputer
import numpy as np
from defineOutcomes import any_in
from load_data import load_data
import dem
reload(dem)



pathToAdmit='/homedirec/user/ADMISSIONS.csv'


dxs = pandas.read_csv('/homedirec/user/or_and_byCreat_byAdm.csv', index_col=0)



all_samp = load_data('/homedirec/user/ADMISSIONS_split_px_dx.csv')



dial = pandas.read_csv('/homedirec/user/DialEx.csv')




n_p_h_df_path = '/homedirec/user/dxs_et_al.csv'



admits = dem.get_admits_for_dem(pathToAdmit)




c_px = dem.get_pr(admits,'CPT4_PROCEDURES',{'cpt_DialPx1':'909', #mostly 90999 unlisted dial
                                        'cpt_DialPx2': '050', 
                                        'cpt_DialPx3':'405',
                                        'anesthKidtrans':'00868'
                                       })




i_px = dem.get_pr(admits,'ICD9_PROCEDURES',{'OtherRenTrans':'5569', 
                                        'RenTrans':'V420',
                                        'RenRej':'5553', 
                                        'DialPx1':'3995', 
                                        'DialPx2':'5498',
                                         'RenalAuto':'5561'}) #don't see renal auto..




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




repr([el for el in admits.columns if 'ETHNIC' in el])




admits[['SEX_F']]
admits['EthAsian'] = admits[[el for el in admits.columns if 'ETHNIC_CD_AS' == el]]
admits['EthBlack'] = admits[[el for el in admits.columns if 'ETHNIC_CD_BL' == el]]
admits['EthWhite'] = admits[[el for el in admits.columns if 'ETHNIC_CD_WH' == el]]
admits['EthAmIn'] = admits[[el for el in admits.columns if 'ETHNIC_CD_AI' == el]]
admits['EthOther'] = admits[
    [el for el in admits.columns if (('ETHNIC' in el)&(el != 'ETHNIC_CD_AS')&(el != 'ETHNIC_CD_BL')&(el != 'ETHNIC_CD_WH'))]
].sum(axis=1)



ethn = admits[['EthAsian','EthBlack','EthWhite','EthAmIn','EthOther']]



df = pandas.concat([c_px, i_px, dx, dxs, ethn, admits[['SEX_F']]], axis=1)




d = {'dialysis':[el for el in df.columns if 'Dial' in el]}




df_c = pandas.DataFrame()
df_c['Dialysis(*)'] = (df[[el for el in df.columns if 'Dial' in el]].sum(axis=1) !=0)*1
df_c['OtherRenTrans:5569'] = df['OtherRenTrans:5569']
df_c['Rhab:72888'] = df['Rhab:72888']
df_c['CKDEndStage:5856'] = df['CKDEndStage:5856']
df_c['CKD:585'] = df['CKD:585']
df_c['Diab:250'] = df['Diab:250']
df_c['RenalFailUnspec:586'] = df['RenalFailUnspec:586']
df_c['CHF:428'] = df['CHF:428']
df_c['Shock:785'] = df['Shock:785']
df_c['Liver Failure(**)'] = (df[[el for el in df.columns if 'Liv' in el]].sum(axis=1) !=0)*1
df_c['SEX_F'] = df['SEX_F']
df_c[[el for el in df.columns if 'Eth' in el]] = df[[el for el in df.columns if 'Eth' in el]]
df_c['or'] = df['or']





df_cont = pandas.concat([dxs['or'], admits[['AGE_ON_ADMISSION', 'TOT_LOS']]], axis=1)
df_cont.pivot_table(index='or', aggfunc=[np.mean, np.median, np.std]).T.round(2)



table1 = pandas.concat([df_c.pivot_table(index='or', aggfunc='sum').T,df_c.pivot_table(index='or', aggfunc='sum').T/df_c.pivot_table(index='or', aggfunc='count').T], axis=1).round(2)




table1.columns = ['AKI-(count)','AKI+(count)','AKI-(%)','AKI-(%)']




df_cont = pandas.concat([dxs['or'], admits[['AGE_ON_ADMISSION', 'TOT_LOS']]], axis=1)


df['dial_or_es'] = df[['CKDEndStage:5856']].sum(axis=1)




df['dial_or_es_bin'] = 1*(df['dial_or_es']>0)


to_ex_dial = pandas.concat([admits['PATIENT_NUM'],df[['dial_or_es_bin', 'OtherRenTrans:5569']]], axis=1)



len(set(dial[
    dial['PATIENT_NUM'].isin(dial[dial['keep']==0]['PATIENT_NUM'])
    &
    dial['PATIENT_NUM'].isin(dial[dial['OtherRenTrans:5569']==1]['PATIENT_NUM'])
    
    ]['PATIENT_NUM']))





from load_data import load_data




len(set(all_samp['PATIENT_NUM']))




print len(all_samp)
all_samp = all_samp[all_samp['AGE_ON_ADMISSION']>=18]
print len(all_samp)




len(all_samp['PATIENT_NUM'].unique())




len(dial)


dial.tail()



all_samp[['ADMIT_ID', 'PATIENT_NUM']].sort_values(['PATIENT_NUM', 'ADMIT_ID']).tail()


dial = dial.set_index('ADMIT_ID')
dial= dial['keep']


all_samp = all_samp.set_index('ADMIT_ID')


dxs['creat'] = dxs['or'] - dxs['AKI_by_admi'] + dxs['and']



joined_with_ex = all_samp.join(dial)
joined_with_ex = joined_with_ex.join(dxs)



joined_with_ex = joined_with_ex[joined_with_ex['keep'] == 1] 




joined_with_ex = joined_with_ex.reset_index()



print len(joined_with_ex) 
print len(all_samp)
print len(all_samp) - len(joined_with_ex) 



print "aki+ in all adult with ex", joined_with_ex['or'].sum()
print "aki+ % in all adult with ex", joined_with_ex['or'].sum()/float(len(joined_with_ex))*100




len(joined_with_ex['PATIENT_NUM'].unique())




import nice_plots
reload(nice_plots)



admits_per_pt = joined_with_ex['PATIENT_NUM'].value_counts()




pts_w_mult_hosp = admits_per_pt[admits_per_pt>1].index



len(pts_w_mult_hosp)



h_from_rehosp = joined_with_ex[joined_with_ex['PATIENT_NUM'].isin(pts_w_mult_hosp)]



len(h_from_rehosp['PATIENT_NUM'].unique())



len(h_from_rehosp)


rehosp_coho = h_from_rehosp['ADMIT_ID']



len(h_from_rehosp) - len(h_from_rehosp['PATIENT_NUM'].unique())


h_from_rehosp['creat'].sum()


sorted_rehosp = h_from_rehosp.sort_values(['PATIENT_NUM', 'ADMIT_ID'])


not_primary_hosp = []
for ptid, admits in sorted_rehosp.groupby('PATIENT_NUM'):
    not_primary_hosp.append(admits[1:])



n_p_h_df = pandas.concat(not_primary_hosp, axis=0)


n_p_h_df[['ADMIT_ID', 'PATIENT_NUM', 'or', 'and', 'creat', 'AKI_by_admi']].to_csv(n_p_h_df_path)



import nice_plots
reload(nice_plots)


print len(n_p_h_df[n_p_h_df['or']==1])
print len(n_p_h_df[n_p_h_df['or']==1]['PATIENT_NUM'].unique())
print len(n_p_h_df[n_p_h_df['or']==0])
print len(n_p_h_df[n_p_h_df['or']==0]['PATIENT_NUM'].unique())


len(rehosp_coho)



rehosp_coho.head()


cat_of_i = df_c.loc[rehosp_coho]


print "N aki+",cat_of_i['or'].sum()
print "%",cat_of_i['or'].sum()/len(cat_of_i['or'])*100
print "N aki-",len(cat_of_i['or']) - cat_of_i['or'].sum() 


cont_of_i = df_cont.loc[rehosp_coho]


table1_cat = pandas.concat([cat_of_i.pivot_table(index='or', aggfunc='sum').T,df_c.pivot_table(index='or', aggfunc='sum').T/df_c.pivot_table(index='or', aggfunc='count').T], axis=1).round(2)
table1_cont = cont_of_i.pivot_table(index='or', aggfunc=[np.mean, np.median, np.std]).T.round(2)
table1_cat.head()



table1_cat.columns = ['AKI-(count)','AKI+(count)','AKI-(%)','AKI+(%)']
table1_cat


list(table1_cat.index)


my_map = {'CHF:428':'CHF',
 'CKD:585':'CKD',
 'CKDEndStage:5856':'End stage CKD',
 'Diab:250':'Diabetes',
 'Dialysis(*)':'Dialysis(*)',
 'EthAmIn':'Eth:American Indian',
 'EthAsian':'Eth:Asian',
 'EthBlack':'Eth:Black',
 'EthOther':'Eth:Other',
 'EthWhite':'Eth:White',
 'Liver Failure(**)':'Liver Failure(**)',
 'OtherRenTrans:5569':'Renal Transplant',
 'RenalFailUnspec:586':'Unspecified Renal Failure',
 'Rhab:72888':'Rhabdomyolysis',
 'SEX_F':'Female',
 'Shock:785':'Shock'}



table1_cat.index = map(lambda x: my_map[x], table1_cat.index)


table1_cat


list(table1_cat.index)



table1_cat = table1_cat.loc[ 
    ['Female',
     'Eth:American Indian',
     'Eth:Asian',
     'Eth:Black',
     'Eth:Other',
     'Eth:White',
     
     'CKD',
     'End stage CKD',
     'Dialysis(*)',
     'Renal Transplant',
     'Unspecified Renal Failure',
     
     'CHF',
     'Diabetes',
     'Shock',
     'Liver Failure(**)',
     'Rhabdomyolysis',
        
        
    ]
    
    
    
    
]


table1_cat



table1_cat = table1_cat[['AKI+(count)','AKI+(%)','AKI-(count)','AKI-(%)']]



table1_cat



table1_cat2 = pandas.DataFrame()



table1_cat2['AKI+ count (%)'] = table1_cat['AKI+(count)'].map(str)+' ('+(table1_cat['AKI+(%)']*100).map(str)+')'
table1_cat2['AKI- count (%)'] = table1_cat['AKI-(count)'].map(str)+' ('+(table1_cat['AKI-(%)']*100).map(str)+')'



table1_cat2



print table1_cat2.to_latex(bold_rows=True)


table1_cont



print table1_cont.to_latex(bold_rows=True)


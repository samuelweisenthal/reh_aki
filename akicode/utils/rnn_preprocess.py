import extractF_timeseries
import pdb
from collections import defaultdict
import drop_rare
from keras.preprocessing.sequence import pad_sequences

def get_data_for_rnn(allSamp, _ ,
                                 precDRG, precDX, precicdPx,preccptPx,
                                 use_labs,use_meds,use_admin,useAKI_dx,
                                 cat_too,
                                 #dx
                                 dx_path = None,
                                 #locations
                                 loc_path = None,
                                 #icd9 px
                                 cms32Px_path = None,
                                 #cpt px
                       		 cptProd_path = None,
                                 #drg
                                 drgs_path = None,
                                 #meds
                                 meds_path = None):


    allSamp = allSamp.sort_values(['sampleNo','ADMIT_ID'],axis=0)

    y = allSamp[['sampleNo','target']].groupby('sampleNo').mean()

    groups = allSamp.groupby('sampleNo')['PATIENT_NUM'].mean()


    data = extractF_timeseries.joinFeatures(extractF_timeseries.extractFeatures(allSamp,'sampleNo',
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

    #Drop rare features

    print data.info()
    print "removing rare"
    data = drop_rare.rem_rare_wrap(thresh=100, df=data)

    print data.info()

    X = data[[el for el in data.columns if el != 'target' and el !='PATIENT_NUM' and el !='ADMIT_ID']]


    f = defaultdict(list)
    for sampleno, etc in X.groupby('sampleNo'):
        for hosp in etc.values:
        #print sampleno, etc.values
            f[sampleno].append(hosp)
    print "padding"
    final_X = pad_sequences(f.values(), dtype=float, maxlen=10) #if no maxlen, uses all memory
    print "padded seq shape", final_X.shape
    return final_X, y, groups

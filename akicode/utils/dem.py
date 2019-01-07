'''Helper functions for demographics'''

import pandas

def get_admits_for_dem(pathToAdmit, nrows=None):
    
    cols_oi = ['ADMIT_ID', 'PATIENT_NUM', 'DIAGNOSES', 
               'CPT4_PROCEDURES', 'ICD9_PROCEDURES', 'GRAND_TOTAL', 
               'AGE_ON_ADMISSION', 'SEX_CD', 'ETHNIC_CD', 'TOT_LOS',]

    admits = pandas.read_csv(pathToAdmit, usecols=cols_oi, nrows=nrows)

    admits = admits.sort_values(['PATIENT_NUM', 'ADMIT_ID'])

    admits = admits.set_index('ADMIT_ID')

    print "#admits", len(admits)

    admits = pandas.get_dummies(admits, columns=['ETHNIC_CD'])

    admits['SEX_F'] = (admits['SEX_CD']=='F')*1

    return admits

def get_pr(df, col_oi, i_d):
    df = df.copy()
    df[col_oi] = df[col_oi].astype(str)
    df[col_oi] = df[col_oi].apply(lambda x: x.replace('.',''))
    df[col_oi] = df[col_oi].apply(lambda x: x.split('\r\n'))
    
    names, dois = zip(*sorted(i_d.iteritems(), key=lambda (k,v): -len(v)))
    print names, dois
    d = {}
    past_doi = ''
    for doi in dois:
        if len(doi) != len(past_doi):
            df[col_oi] = df[col_oi].apply(lambda x: [el[0:len(doi)] for el in x])
        d[doi] = df[col_oi].apply(lambda x: doi in x)*1

    mydf = pandas.DataFrame(d)[list(dois)] #reorder 
    nnames = [] 
    for n,i in zip(names, dois):
        nnames.append(n+':'+str(i))
        
    mydf.columns = list(nnames)
    return mydf


def collapse(d, df):
    df_collapsed = pandas.DataFrame(columns=[d.keys()])
    
    for k,v in d.iteritems():
        print k,v
        df_collapsed[k] = df[v].sum(axis=1)
        df_collapsed[k] = (df_collapsed[k]!=0)*1
    
    return df_collapsed

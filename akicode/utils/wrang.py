# -*- coding: utf-8 -*-
"""
Created on Fri Aug 19 08:23:48 2016
Converts categorical variabels to dummy variables that can be used as features.  Also names features according to some dictionary file
Is useful for seeing what exactly is going into a model
@author: user
"""


#diagnoses = patients[patients.columns.values[75:129]]

import clean
import pandas
import numpy as np
from functools import partial
import copy

def OnlyTakeFirstDig(x,prec):#don't drill down the missing values
    if str(x)!='nan': 
        return str(x)[0:prec]
    else:
        return 'nan'

def dummize(df,DigitsToTake,RenameLabel, numberSplits=20):
#need to record the rest of the commands below..problem is that the list of col names gets long.
    '''
    dummize is now fast AND memory-cheap.
    ds = wrang.dummize(patients,['DRG1', 'DRG2'],3,'DRG')
    --df must be a dataframe--it can't be a series. If it's a series, call pandas.DataFrame() on it
    Added new parameters threshold and numberSplits--the lowest freq of a code for it to be included and the number of chunks to iterate by (to save memory) respectively. Might eventually make these arguments, but for now will hardcode. 
    '''
    if '_' not in RenameLabel:
        print "Put an underscore in the rename label"
        return None
    truncator = partial(OnlyTakeFirstDig, prec=DigitsToTake)
    df = df.fillna('nan')#re (in remDot) chokes on floats
    df = df.applymap(clean.remDot)#do before truncating!
    df = df.applymap(truncator) 
    
        #On removing infrequent items: Do it with entire dataset, not now 
    
    #numberSplits=20 #for larger data (like px) need more splits. Splits is just how many chunks it makes rather than iterating over entire dataset. CPTpx needed about 20, so I am just splitting all by 20 (or 10, apparently..)
    #print "splits",numberSplits
    stacked=df.stack()#stack the dataframe
    stacked.index=stacked.index.droplevel(1)#drop the dx1,dx2, etc...now we lump all codes rather than trying to expand them as numbered, which is too expensive.  I am not certain that this step is actually necessary; maybe could do the split without..I think the downstream split is really what's important.
    blocks=[]
    for block in np.array_split(stacked,numberSplits):#split into blocks--otherwise large sparse matrices-> memory error
        ablock=pandas.get_dummies(block).groupby(block.index).sum()#when we have blocks, we can collapse them by index (correspondng to an admitid)
        #print ablock.info()
        blocks.append(ablock)#put all the blocks in a list
    fullFrame=pandas.concat(blocks).replace(np.NAN,0)#concat and replace nan with 0 (if didn't have procedure in block, will be nan--> no one in the block had it)
    #print fullFrame.info()

    fullFrame=fullFrame.groupby(fullFrame.index).sum()#sum again because array_split split some admissions
    #print fullFrame.info()
    #
 
    fullFrame.rename(columns={c:RenameLabel + str(c) for c in fullFrame.columns},inplace=True)#label
    #print "fullframe col",len(fullFrame.columns)
    #print "dist ff col", len(set(fullFrame.columns))
    return fullFrame   

def dummize2(df,DigitsToTake,RenameLabel):
#need to record the rest of the commands below..problem is that the list of col names gets long.
    '''
    dummize is much faster but much more memory-intensive than dummize2. Dummize considers dx1 != dx1, which leads to massive memory requirements
    ds = wrang.dummize(patients,['DRG1', 'DRG2'],3,'DRG')
    --df must be a dataframe--it can't be a series. If it's a series, call pandas.DataFrame() on it
    '''
    if '_' not in RenameLabel:
        print "Put an underscore in the rename label"
        return None    
    truncator = partial(OnlyTakeFirstDig, prec=DigitsToTake)
    df = df.applymap(truncator) 
    
    #Because the memory requirements are considerable for the cpt procedures, I wanted to remove rare ones. For example if they only occur in 3 patients, they may not be worth expanding
    #http://stackoverflow.com/questions/32511061/remove-low-frequency-values-from-pandas-dataframe   
    df = pandas.get_dummies(df,prefix_sep = '=')#convert to dummies; 
    #print "after get_dummies",df.info()
    df.rename(columns={c:c.split('=')[1] for c in df.columns},inplace=True)#rename columns
    df = df.groupby(lambda x:x,axis=1).sum().astype(np.int8)#equivalent to df_of_int = df_of_int.groupby(level=0,axis=1).sum()
    df.rename(columns={c:RenameLabel + clean.remDot(c) for c in df.columns},inplace=True)#label
    return df  

def dummize3(df,DigitsToTake,RenameLabel):
    '''
    This is a less memory intensive but extremely slow verison of get_dummies. dummize2 considers dx1 == dx2, avoiding the data accordion step of get_dummies.
    ds = wrang.dummize(patients[['DRG1', 'DRG2']],3,'DRG_') 
    -- put an underscore in the rename label
    -- the columns must be homogenous (eg, all diagnoses; no index columns added in)
    patients[['DRG1', 'DRG2']] NOT patients[['ADMIT_ID','DRG1', 'DRG2']]
    -- digitsToTake is precision.
    -- if precision is less than 3, remember that nan will be truncated to 'n'
    '''
    if '_' not in RenameLabel:
        print "Put an underscore in the rename label"
        return None
    truncator = partial(OnlyTakeFirstDig, prec=DigitsToTake)
    df = df.applymap(truncator) 
    bi = df.apply(lambda x: x.value_counts(),axis=1).fillna(0).astype(np.int8)# http://stackoverflow.com/questions/40636636/pandas-get-dummies-with-identical-same-column-names/40636879?noredirect=1#comment68510317_40636879
    bi.rename(columns={c:RenameLabel + clean.remDot(c) for c in bi.columns},inplace=True)#rename so that eg dx1,dx2 are treated equally. Need to remove period from codes
    return bi   

def getDictToConvertCodeToName(prec,dxDF,indexDesc,textDesc):
    
    '''
    (dxCodeToName,_) = wrang.getDictToConvertCodeToName(1000000,dxDF,'DIAGNOSIS CODE','LONG DESCRIPTION')
    '''
    #converts a file with a code and description into a dictionary that takes in some number ('prec') of the digits of that code and outputs (fullcode: desc)
    #Built this to rename codes such as ICD, DRG, CPT
    import collections
    #dict = collections.defaultdict(list)
    cut_codes=[str(code)[0:prec] for code in dxDF[indexDesc]]
    dxDF['cut_codes']=cut_codes
    listDict=collections.defaultdict(list)
    for cut_code,full_code,desc in zip(dxDF['cut_codes'],dxDF[indexDesc],dxDF[textDesc]):
            listDict[cut_code].append((full_code,desc))
    stringDict={}
    for k,v in listDict.iteritems():
        #print k,v
        stringDict[k]=str(v)
    return (stringDict,listDict)

def nameCodes(origDF,prec,nameDF,oldName,newName,ip_stringFlag,op_stringFlag):
    #these ip_stringFlag must match EXACTLY    
    #ICD9Px_   PRECISION 3 , DF cms32Px
    #tdf = wrang.nameCodes(patientsE,3,cms32Px,'PROCEDURE CODE','LONG DESCRIPTION','ICD9Px_','ICD9Px')
    #CPT4_    PRECISION 3 , DF cptProd
    #tdf = wrang.nameCodes(patientsE,3,cptProd,'CPT-4 Code','CPT-4 Procedure','CPT4_','CPT4PX')
    #DRG_     PRECISION 4 , DF drg
    #tdf = wrang.nameCodes(patientsE,3,drg,'drg','drgdesc','DRG_','DRG')
    #dx       PRECISION 3 , DF dx
    #tdf = wrang.nameCodes(patientsE,3,dx,'DIAGNOSIS CODE','LONG DESCRIPTION','dx_','dx')
    #prec must match precision used to extract the original features (Digits to take in dummize)
    #Names dummy variables using dictionary made from specified dataframes
    #IF THE COMMAND BELOW DOESN'T WORK, RERUN STARTERS.PY--there is a last col that is nan        
    
    (Dict,_) = getDictToConvertCodeToName(prec,nameDF,oldName,newName)  
    Cols =[el for el in origDF.columns if ip_stringFlag in el]
    colNames = {}
    counter = 0
    for c in Cols:
        try:
            colNames[c] = op_stringFlag+'_'+str(c[len(ip_stringFlag):])+'_ONE OF:'+Dict[c[len(ip_stringFlag):]]
        except:
            colNames[c] = op_stringFlag+'_'+str(c[len(ip_stringFlag):])+'_ONE OF:UNKNOWN by sjw:count'+str(counter)
        counter=counter+1
    origDF.rename(columns=colNames,inplace=True)
    return origDF

def DummizeAndName(df,precision,keyFile,keyCodeNumCol,keyCodeDescCol,newPrefix,oneOrTwo):
    '''
    wrang.DummizeAndName(data[dxs],20,dx,'DIAGNOSIS CODE','LONG DESCRIPTION','DX')
    --wraps dummize and name into one
    --df must be df of interest only
    --nameTag is really internal
    --keyFile: where the translation data is
    --keyCodeNumCol : where the keys are
    --keyCodeDescCol : where the values are
    --newPrefix will be like 'newPrefix__codeNumber__ONEOF:['description...
    '''
    nameTag='nameTag_'
    if oneOrTwo ==1:
        return nameCodes(dummize(df,precision,nameTag),precision,keyFile,keyCodeNumCol,keyCodeDescCol,nameTag,newPrefix)
    if oneOrTwo ==2:
        return nameCodes(dummize2(df,precision,nameTag),precision,keyFile,keyCodeNumCol,keyCodeDescCol,nameTag,newPrefix)

def getIns(df):
    '''Takes a dataframe df[['INSURANCE']]'''
    import re
    myin = []
    ins = list(df['INSURANCE'])#will be passed a dataframe, this gets a list of the series
    for el in ins:
       if str(el) != 'nan':
           #print el
           #print el.split('0')
           ls = el.split('0')
           ls = [re.sub(r'\d+','',o.translate(None,'\n')) for o in ls if o!='']
           myin.append(ls)
           #print ls
       else:
           #print el,"is",el
           myin.append([])
    import pandas
    mi= pandas.DataFrame(myin,index=df['INSURANCE'].index)
    
    #note that this is double-bracketed [[]] to get a subset, which is a DATAFRAME rather than [] which returns series!
    primary = dummize(mi[[mi.columns[0]]],1000,'INSPRIM_')
    if len(mi.columns)>1:
        secondary = dummize(mi[[mi.columns[1]]],1000,'INSSEC_')
    else:
        secondary = np.empty(len(mi))
        secondary[:]=np.NAN
        secondary=pandas.DataFrame({'INSSEC_':secondary},index=mi.index)
    if len(mi.columns)>2:
        other = dummize(mi[mi.columns[2:]],1000,'INSOTHER_')
    else:
        other = np.empty(len(mi))
        other[:]=np.NAN
        other=pandas.DataFrame({'INSOTHER_':other},index=mi.index) 
    inss=pandas.concat([primary,secondary,other],axis=1)
    inss = clean.remSpecCharFromColumns(inss)#has carriage returns which interfere with pandas to_csv()
    return inss

def pivot_codes(pathToAdmit='/homedirec/user/ADMISSIONS_split_px_dx.csv', tag_oi='DIAG', rename_label='DX_', code_piece_oi='719'):
    
    my_cols = pandas.read_csv(pathToAdmit, index_col=0, nrows=1).columns
    col_oi = [el for el in my_cols if tag_oi in el] 
    col_tg = col_oi + ['ADMIT_ID']
    data = pandas.read_csv(pathToAdmit, index_col=0, use_cols=col_tg, dtype='str')
    data = data.astype(str)

    data['ADMIT_ID'] = data['ADMIT_ID'].astype(np.int64)
    
    dum = dummize(data[col_oi], 1000, rename_label)
    dum_oi = dum[[el for el in dum.columns if code_piece_oi in el]].copy()
    dum_oi['ADMIT_ID'] = data['ADMIT_ID']
    dum_oi = dum_oi.set_index('ADMIT_ID')
    dxs = pandas.read_csv('/homedirec/user/or_and_byCreat_byAdm.csv', usecols=['ADMIT_ID', 'or'], index_col=0)
    joined = dum_oi.join(dxs, how='inner')

    return pandas.pivot_table(joined, index=['or'], aggfunc='sum')

#to dummize http://stackoverflow.com/questions/39006270/how-to-convert-a-set-of-features-to-a-count-matrix-in-pandas



def process_meds(m):
    m = copy.deepcopy(m)    
    ow = m['DESCRIPTION'].str.extract('^(.+?)[0-9]') #http://chrisalbon.com/python/pandas_regex_to_create_columns.html
    m['DESCRIPTION']= ow
    m['THERA_CLASS_C'] = m['THERA_CLASS_C'].astype(str)
    m['PHARM_CLASS_C'] = m['PHARM_CLASS_C'].astype(str)
    m['PHARM_SUBCLASS_C']=m['PHARM_SUBCLASS_C'].astype(str)
    
    return m

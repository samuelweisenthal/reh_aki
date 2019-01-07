'''Gives frequencies for sub-codes within aggregated codes.  Was used to rename features that consisted of many subcodes'''


pta = '/homedirec/user/ADMISSIONS_split_px_dx.csv'
ptd = '/homedirec/user/or_and_byCreat_byAdm.csv'



from wrang import dummize
import pandas
import numpy as np


def pivot_codes(
    pathToAdmit=pta,
    tag_oi='DIAG',
    rename_label='DX_',
    code_piece_oi='719'):
    
    my_cols = pandas.read_csv(pathToAdmit, index_col=0, nrows=1).columns
    col_oi = [el for el in my_cols if tag_oi in el] 
    col_tg = col_oi + ['ADMIT_ID']
    data = pandas.read_csv(pathToAdmit, index_col=0, usecols=col_tg, dtype='str')
    data = data.astype(str)

    #data['ADMIT_ID'] = data['ADMIT_ID'].astype(np.int64)

    dum = dummize(data[col_oi], 1000, rename_label, numberSplits=20)

    dum_oi = dum[[el for el in dum.columns if code_piece_oi in el]].copy()
    dum_oi['ADMIT_ID'] = data.index
    dum_oi = dum_oi.set_index('ADMIT_ID')
    dxs = pandas.read_csv(ptd, usecols=['ADMIT_ID', 'or'], index_col=0)
    joined = dum_oi.join(dxs, how='inner')

    check = rename_label + code_piece_oi

    joined = joined[[el for el in joined.columns if check in el]+['or']]

    return pandas.pivot_table(joined, index=['or'], aggfunc='sum').T



pivot_codes(pathToAdmit=pta,
    tag_oi='DIAG',
    rename_label='DX_',
    code_piece_oi='719')




pivot_codes(pathToAdmit=pta, 
            tag_oi='CPT4_PROCED', rename_label='PC_', code_piece_oi='837')




pivot_codes(pathToAdmit=pta, 
            tag_oi='CPT4_PROCED', rename_label='PC_', code_piece_oi='J19')




pivot_codes(pathToAdmit=pta, 
            tag_oi='CPT4_PROCED', rename_label='PC_', code_piece_oi='828')




pivot_codes(pathToAdmit=pta, 
            tag_oi='CPT4_PROCED', rename_label='PC_', code_piece_oi='J30')




pivot_codes(pathToAdmit=pta, 
            tag_oi='DIAG', rename_label='DX_', code_piece_oi='724')




pivot_codes(pathToAdmit=pta, 
            tag_oi='CPT4_PROCED', rename_label='PC_', code_piece_oi='364')




pivot_codes(pathToAdmit=pta, 
            tag_oi='CPT4_PROCED', rename_label='PC_', code_piece_oi='883')




pivot_codes(pathToAdmit=pta, 
            tag_oi='CPT4_PROCED', rename_label='PC_', code_piece_oi='843')


# <h2>full



pivot_codes(pathToAdmit=pta, 
            tag_oi='CPT4_PROCED', rename_label='PC_', code_piece_oi='J16')




pivot_codes(pathToAdmit=pta,
    tag_oi='DIAG',
    rename_label='DX_',
    code_piece_oi='V440')




pivot_codes(pathToAdmit=pta,
    tag_oi='DIAG',
    rename_label='DX_',
    code_piece_oi='276')




pivot_codes(pathToAdmit=pta,
    tag_oi='DIAG',
    rename_label='DX_',
    code_piece_oi='V42')




pivot_codes(pathToAdmit=pta,
    tag_oi='DIAG',
    rename_label='DX_',
    code_piece_oi='E00')




pivot_codes(pathToAdmit=pta,
    tag_oi='DIAG',
    rename_label='DX_',
    code_piece_oi='959')


# 5000_fit3Q



pivot_codes(pathToAdmit=pta, 
            tag_oi='CPT4_PROCED', rename_label='PC_',
            code_piece_oi='J75')




pivot_codes(pathToAdmit=pta, 
            tag_oi='CPT4_PROCED', rename_label='PC_',
            code_piece_oi='841')


# For stats



pivot_codes(pathToAdmit=pta,
    tag_oi='DIAG',
    rename_label='DX_',
    code_piece_oi='285')


# For dialysis



pivot_codes(pathToAdmit=pta,
    tag_oi='CPT4_PROC',
    rename_label='PX_',
    code_piece_oi='909')



pivot_codes(pathToAdmit=pta,
    tag_oi='DIAG',
    rename_label='DX_',
    code_piece_oi='V45')


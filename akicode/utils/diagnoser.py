'''Diagnose by sCr, using fold and absolute change'''

import pandas
import numpy as np
def dx_fold(fold_inc,time_window,t):

    admits = pandas.DataFrame({'AKI':0},index=t['ADMIT_ID'].unique())#keep track of positives by admit number
    for admit_id,admit in t.groupby('ADMIT_ID'):

        for measurement in admit.itertuples():

            my_meas_index = measurement[0]
            baseline = admit.loc[my_meas_index]
            if baseline['RESULT_VALUE']==0:#check if baseline creatinine is zero. 
                print "ZERO, not accurate!"
            in_timeframe=[]

            for fut_meas in admit.loc[my_meas_index:].iloc[1:].itertuples():
                if fut_meas[2]-measurement[2]<=pandas.Timedelta(time_window):
                    in_timeframe.append(fut_meas[3])

                else:

                    break

            if in_timeframe:

                if np.max(np.array(in_timeframe)/float(measurement[3]))>=fold_inc:

                    admits.loc[admit_id]['AKI']=1#y not admits.loc[admit_id,'AKI']

                    break

    return admits

def dx_abs(abs_inc,time_window,t):

    admits = pandas.DataFrame({'AKI':0},index=t['ADMIT_ID'].unique())#keep track of positives by admit number
    for admit_id,admit in t.groupby('ADMIT_ID'):

        for measurement in admit.itertuples():

            my_meas_index = measurement[0]
            baseline = admit.loc[my_meas_index]
            if baseline['RESULT_VALUE']==0:#check if baseline creatinine is zero. 
                print "ZERO, not accurate!"
            in_timeframe=[]

            for fut_meas in admit.loc[my_meas_index:].iloc[1:].itertuples():
                if fut_meas[2]-measurement[2]<=pandas.Timedelta(time_window):
                    in_timeframe.append(fut_meas[3])

                else:

                    break

            if in_timeframe:

                if np.max(np.array(in_timeframe)-float(measurement[3]))>=abs_inc:

                    admits.loc[admit_id]['AKI']=1#y not admits.loc[admit_id,'AKI']

                    break

    return admits

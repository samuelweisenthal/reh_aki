import pandas

def saveSampled(data,undata,runTag,precDRG,precDX,precicdPx,preccptPx):
    print "saving sampled data to",'/homedirec/user/'+runTag+'_DRG_'+str(precDRG)+'_Dx_'+str(precDX)+'_iPx_'+str(precicdPx)+'_cPx_'+str(preccptPx)+'_sampled.csv'
    data.to_csv('/homedirec/user/'+runTag+'_DRG_'+str(precDRG)+'_Dx_'+str(precDX)+'_iPx_'+str(precicdPx)+'_cPx_'+str(preccptPx)+'_sampled.csv')
    print "saving UNsampled data to",'/homedirec/user/'+runTag+'_DRG_'+str(precDRG)+'_Dx_'+str(precDX)+'_iPx_'+str(precicdPx)+'_cPx_'+str(preccptPx)+'_UNsampled.csv'
    undata.to_csv('/homedirec/user/'+runTag+'_DRG_'+str(precDRG)+'_Dx_'+str(precDX)+'_iPx_'+str(precicdPx)+'_cPx_'+str(preccptPx)+'_UNsampled.csv')

def saveDumm(dummData,runTag,precDRG,precDX,precicdPx,preccptPx):
    print "saving dummData to",'/homedirec/user/'+str(runTag)+'_DRG_'+str(precDRG)+'_Dx_'+str(precDX)+'_iPx_'+str(precicdPx)+'_cPx_'+str(preccptPx)+'_dummSamp'+'.csv'
    dummData.to_csv('/homedirec/user/'+str(runTag)+'_DRG_'+str(precDRG)+'_Dx_'+str(precDX)+'_iPx_'+str(precicdPx)+'_cPx_'+str(preccptPx)+'_dummSamp'+'.csv')
    
def saveRaw(rawData,runTag):
    print "saving rawData to",'/homedirec/user/'+runTag+'_rawSamp.csv'
    save_loc = '/homedirec/user/'+runTag+'_rawSamp.csv'
    rawData.to_csv('/homedirec/user/'+runTag+'_rawSamp.csv')
    return save_loc

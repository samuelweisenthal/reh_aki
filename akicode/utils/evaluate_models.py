'''Helper functions to evaluate models and plot results'''

from sklearn.metrics import roc_curve, roc_auc_score, recall_score, precision_score, precision_recall_curve, average_precision_score, brier_score_loss
from sklearn.calibration import calibration_curve
from sklearn.model_selection import cross_val_score
import matplotlib.pyplot as plt
import numpy as np
import numpy.ma as ma
import pandas

#evaluate_models.pl_cm(X_val,y_val,0.075,ForestCvDecisions,f,0,2,'x')
#evaluate_models.pl_fea(f,10,X_train_and_cv,y_train_and_cv,"x")
#evaluate_models.plot_rocs_prs(LogisticRegression1CvDecisions,LogisticRegression2CvDecisions,ForestCvDecisions,NeuralNetworkCvDecisions,y_val,"LR1","LR2","RF","NN","x")

def plot_roc(y_train,TrainDecisions,y_cv,CvDecisions,classifierName):
    TrainFpr,TrainTpr,_=roc_curve(y_train,TrainDecisions)
    train_auc = roc_auc_score(y_train,TrainDecisions)
    CvFpr,CvTpr,_=roc_curve(y_cv,CvDecisions)
    cv_auc = roc_auc_score(y_cv,CvDecisions)
    
    plt.plot(TrainFpr,TrainTpr,label='Training:{:.2}'.format(train_auc),color='black',lw=1,linestyle='-.')
    plt.plot(CvFpr,CvTpr,label='Validation:{:.2}'.format(cv_auc),color='black',lw=1,linestyle='solid')
    plt.plot([0, 1], [0, 1], color='gray', lw=0.25,)
    plt.xlim([-0.02,1.0])
    plt.ylim([0,1.03])
    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate')
    plt.title(classifierName)
    plt.legend(loc="lower right")
    plt.show()
    
def plot_cal(y_train, TrainDecisions, y_cv, CvDecisions, classifierName):
    NBINS = 10
    NORMALIZE = False #all classifiers use predict_proba
    train_frac_pos, train_mean_pred_value = calibration_curve(y_train, TrainDecisions, n_bins=NBINS, normalize=NORMALIZE)
    train_brier = brier_score_loss(y_train, TrainDecisions)
    cv_frac_pos, cv_mean_pred_value = calibration_curve(y_cv, CvDecisions, n_bins=NBINS, normalize=NORMALIZE)
    cv_brier = brier_score_loss(y_cv, CvDecisions)
    
    plt.plot(train_mean_pred_value, train_frac_pos, label='Training Brier:{:.2}'.format(train_brier), color='black', lw=1, linestyle='-.')
    plt.plot(cv_mean_pred_value, cv_frac_pos, label='Validation Brier:{:.2}'.format(cv_brier), color='black', lw=1, linestyle='solid')
    plt.plot([0, 1], [0, 1], color='gray', lw=0.25,)
    plt.xlim([-0.02,1.0])
    plt.ylim([0,1.03])
    plt.xlabel('Mean predicted value')
    plt.ylabel('Fraction of positives')
    plt.title(classifierName)
    plt.legend(loc="lower right")
    plt.show()    
    
def plot_pr(y_train,TrainDecisions,y_cv,CvDecisions,classifierName):
    TrainFpr,TrainTpr,_=precision_recall_curve(y_train,TrainDecisions)
    train_auc = average_precision_score(y_train,TrainDecisions)
    CvFpr,CvTpr,_=precision_recall_curve(y_cv,CvDecisions)
    cv_auc = average_precision_score(y_cv,CvDecisions)
    
    plt.plot(TrainFpr,TrainTpr,label='Training:{:.2}'.format(train_auc),color='black',lw=1,linestyle='-.')
    plt.plot(CvFpr,CvTpr,label='Validation:{:.2}'.format(cv_auc),color='black',lw=1,linestyle='solid')
    plt.plot([0, 1], [0, 1], color='gray', lw=0.25,)
    plt.xlim([-0.02,1.0])
    plt.ylim([0,1.03])
    plt.xlabel('Sensitivity')
    plt.ylabel('PPV')
    plt.title(classifierName)
    plt.legend(loc="upper right")
    plt.show()

def plot_test_roc(y_test,TestDecisions,classifierName):

    testFpr,testTpr,_= roc_curve(y_test,TestDecisions)
    test_auc = roc_auc_score(y_test,TestDecisions)
    

    plt.plot(testFpr,testTpr,label='Test:{:.2}'.format(test_auc),color='black',lw=1,linestyle='solid')
    
    plt.plot([0, 1], [0, 1], color='gray', lw=0.25,)
    plt.xlim([-0.02,1.0])
    plt.ylim([0,1.03])
    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate')
    plt.title(classifierName)
    plt.legend(loc="lower right")
    plt.show()
    
def score_clf_cv(clf,X_train,y_train,cv):
    scores = cross_val_score(clf,X_train,y_train,cv=cv,scoring='roc_auc')
    print "AUC",scores.mean(),"+/-",scores.std()
    return (scores.mean(),scores.std())
    
def pl_cm(X_val,y_val,thr,ForestCvDecisions,f,f1,f2,name):
    dec = map(lambda x: (x> thr)*1,ForestCvDecisions)
    val_c = X_val.copy()
    val_c = val_c[f.sort('imp',ascending=False).T.columns[0:20]]
    val_c['t']=y_val
    val_c['p']=dec
    val_c['err']=np.NAN
    val_c.loc[(val_c['t']==0)&(val_c['p']==1),'err'] = 3#'fp'
    val_c.loc[(val_c['t']==0)&(val_c['p']==0),'err'] = 2#'tn'
    val_c.loc[(val_c['t']==1)&(val_c['p']==1),'err'] = 1#'tp'
    val_c.loc[(val_c['t']==1)&(val_c['p']==0),'err'] = 4#'fn'
    
    n_fp = len(val_c.loc[(val_c['t']==0)&(val_c['p']==1),'err'])
    n_tn = len(val_c.loc[(val_c['t']==0)&(val_c['p']==0),'err'])
    n_tp = len(val_c.loc[(val_c['t']==1)&(val_c['p']==1),'err'])
    n_fn = len(val_c.loc[(val_c['t']==1)&(val_c['p']==0),'err'])
    
    sen = np.round(recall_score(y_val,dec),2)
    ppv = np.round(precision_score(y_val,dec),2)
    npv = np.round((n_tn/float(n_tn+n_fn)),2)
    spec = np.round((n_tn/float(n_tn+n_fp)),2)
    print "sen",sen,"spec",spec,"ppv",ppv,"npv",npv
    fig,ax = plt.subplots(2,2,figsize=(10,10))
    fig.suptitle('Threshold: '+str(thr)+"; Sensitivity: "+str(sen)+"; Specificity: "+str(spec)+"; PPV: "+str(ppv)+"; NPV: "+str(npv),fontsize=15)
    val_c['siz']=np.NAN
    val_c.loc[(val_c['t']==0)&(val_c['p']==1),'siz'] = 3#'fp'
    val_c.loc[(val_c['t']==0)&(val_c['p']==0),'siz'] = 3#'tn'
    val_c.loc[(val_c['t']==1)&(val_c['p']==1),'siz'] = 50#'tp'
    val_c.loc[(val_c['t']==1)&(val_c['p']==0),'siz'] = 3#'fn'
    ax[0,0].set_ylabel('log '+str(val_c.columns[f2]))
    ax[0,0].set_title('TP (n='+str(n_tp)+")")
    ax[0,0].scatter(np.log(val_c[val_c.columns[f1]]),np.log(val_c[val_c.columns[f2]]),c=val_c['err'],alpha=0.9,s=val_c['siz'],linewidths=0.1)
    val_c['siz']=np.NAN
    val_c.loc[(val_c['t']==0)&(val_c['p']==1),'siz'] = 50#'fp'
    val_c.loc[(val_c['t']==0)&(val_c['p']==0),'siz'] = 3#'tn'
    val_c.loc[(val_c['t']==1)&(val_c['p']==1),'siz'] = 3#'tp'
    val_c.loc[(val_c['t']==1)&(val_c['p']==0),'siz'] = 3#'fn'
    ax[0,1].scatter(np.log(val_c[val_c.columns[f1]]),np.log(val_c[val_c.columns[f2]]),c=val_c['err'],alpha=.9,s=val_c['siz'],linewidths=0.1)
    ax[0,1].set_title('FP (n='+str(n_fp)+")")
    val_c['siz']=np.NAN
    val_c.loc[(val_c['t']==0)&(val_c['p']==1),'siz'] = 3#'fp'
    val_c.loc[(val_c['t']==0)&(val_c['p']==0),'siz'] = 3#'tn'
    val_c.loc[(val_c['t']==1)&(val_c['p']==1),'siz'] = 3#'tp'
    val_c.loc[(val_c['t']==1)&(val_c['p']==0),'siz'] = 50#'fn'
    ax[1,0].scatter(np.log(val_c[val_c.columns[f1]]),np.log(val_c[val_c.columns[f2]]),c=val_c['err'],alpha=.9,s=val_c['siz'],linewidths=0.1)
    ax[1,0].set_xlabel('log '+str(val_c.columns[f1]))
    ax[1,0].set_ylabel('log '+str(val_c.columns[f2]))
    ax[1,0].set_title('FN (n='+str(n_fn)+")")
    val_c['siz']=np.NAN
    val_c.loc[(val_c['t']==0)&(val_c['p']==1),'siz'] = 3#'fp'
    val_c.loc[(val_c['t']==0)&(val_c['p']==0),'siz'] = 50#'tn'
    val_c.loc[(val_c['t']==1)&(val_c['p']==1),'siz'] = 3#'tp'
    val_c.loc[(val_c['t']==1)&(val_c['p']==0),'siz'] = 3#'fn'
    ax[1,1].scatter(np.log(val_c[val_c.columns[f1]]),np.log(val_c[val_c.columns[f2]]),c=val_c['err'],alpha=.9,s=val_c['siz'],linewidths=0.1)
    ax[1,1].set_xlabel('log '+str(val_c.columns[f1]))
    ax[1,1].set_title('TN (n='+str(n_tn)+")")
    fig.savefig('./plots/'+name+'_cm.pdf')
    
    fp = np.round(val_c[(val_c['t']==0)&(val_c['p']==1)].mean(),2)
    fps = np.round(val_c[(val_c['t']==0)&(val_c['p']==1)].std(),2)
    tn = np.round(val_c[(val_c['t']==0)&(val_c['p']==0)].mean(),2)
    tns =  np.round(val_c[(val_c['t']==0)&(val_c['p']==0)].std(),2)
    tp =  np.round(val_c[(val_c['t']==1)&(val_c['p']==1)].mean(),2)
    tps =  np.round(val_c[(val_c['t']==1)&(val_c['p']==1)].std(),2)
    fn =  np.round(val_c[(val_c['t']==1)&(val_c['p']==0)].mean(),2)
    fns =  np.round(val_c[(val_c['t']==1)&(val_c['p']==0)].std(),2)
    pm = "replaceMeWithPM "
    fps,tns,tps,fns=fps.apply(lambda x: pm+str(x)),tns.apply(lambda x: pm+str(x)),tps.apply(lambda x: pm+str(x)),fns.apply(lambda x: pm+str(x))
   
    c = pandas.concat([tp,tps,fp,fps,tn,tns,fn,fns],names=['tp','tps','fp','fps','tn','tns','fn','fns',],axis=1)
    pandas.set_option('display.max_colwidth',900)
    c.index = [str(el)[0:900] for el in c.index]
    
    c.columns = ['TP','','FP','','TN','','FN','']
    print c[0:20].to_latex()
    
    return (fig,c)
    
    

def pl_fea(f,nfeat,X_train_and_cv,y_train_and_cv,name):

    hfont = {'fontname':'Helvetica'}#http://stackoverflow.com/questions/21321670/how-to-change-fonts-in-matplotlib-python
    fs=30
    rf_fav=f.sort('imp',ascending=False).T.columns[0:nfeat]
    rf_fav=rf_fav[0:nfeat]
    #rf_fav=[el[0:40] for el in rf_fav]
    names = {el:el for el in rf_fav}
    #n =X_train_and_cv[rf_fav].columns.size
    
    naxes=nfeat*nfeat
    fig,axes=plt.subplots(nrows=nfeat,ncols=nfeat,figsize=(40,40),squeeze=False,sharex=False,sharey=False)
    i = 0
    for el_j in rf_fav:
        j = 0
        for el_i in rf_fav:
            
            print el_i
            axes[i,j].tick_params(labelsize=20)
            #axes[i,j].tick_params(labelsize=20)
            if el_i == el_j:
                X_train_and_cv[el_i].plot.kde(ax=axes[i,j],lw=1)
                axes[i,j].set_axis_bgcolor('white')
                if j == 0:
                    axes[i,j].set_ylabel(names[el_j],fontsize=fs,**hfont)
                    #if i == 0:
                    #    axes[i,j].tick_params(axis='y',colors='white')
                else:
                    axes[i,j].get_yaxis().set_visible(False)
                if i == nfeat-1:
                    print "i3"
                    axes[i,j].set_xlabel(names[el_i],fontsize=fs)
                    #axes[i,j].get_yaxis().set_visible(False)
                    #axes[i,j].tick_params(axis='x',colors='white')
                    #axes[i,j].set_xticks(axes[i-1,j].get_xticks())
                #else:
                    #axes[i,j].get_xaxis().set_visible(False)


                print "e"
            else:
                if i > j :

                    axes[i,j].scatter(ma.masked_array(X_train_and_cv[el_i],y_train_and_cv),ma.masked_array(X_train_and_cv[el_j],y_train_and_cv),marker='o',color='navy',alpha=0.25,s=15)
                    axes[i,j].scatter(ma.masked_array(X_train_and_cv[el_i],1-y_train_and_cv),ma.masked_array(X_train_and_cv[el_j],1-y_train_and_cv),marker='o',color='red',alpha=0.25,s=10)
                    axes[i,j].set_axis_bgcolor('white')
                    if j == 0:
                        axes[i,j].set_ylabel(names[el_j],fontsize=fs)

                        #if i != 4:
                            #axes[i,j].get_xaxis().set_visible(False)
                    #else:
                        #axes[i,j].get_yaxis().set_visible(False)
                        #if i != 4:                   
                            #axes[i,j].get_xaxis().set_visible(False)

                else:
                    axes[i,j].scatter(ma.masked_array(np.log(X_train_and_cv[el_i]),y_train_and_cv),ma.masked_array(np.log(X_train_and_cv[el_j]),y_train_and_cv),marker='o',color='navy',alpha=0.25,s=15)
                    axes[i,j].scatter(ma.masked_array(np.log(X_train_and_cv[el_i]),1-y_train_and_cv),ma.masked_array(np.log(X_train_and_cv[el_j]),1-y_train_and_cv),marker='o',color='red',alpha=0.25,s=10)
                    axes[i,j].set_axis_bgcolor('white')
                    #axes[i,j].get_xaxis().set_visible(False)
                    #axes[i,j].get_yaxis().set_visible(False)

                if i == nfeat-1:
                    print "i3"
                    axes[i,j].set_xlabel(names[el_i],fontsize=fs)




            j += 1
        i += 1
    
    fig.savefig('./plots/'+name+'_f.pdf')    
    return fig
    
def plot_rocs_prs(c1,c2,c3,c4,y_cv,c1_n,c2_n,c3_n,c4_n,name):
    f,axs=plt.subplots(1,2,figsize=(10,5))
    ax1,ax2=axs
    fp1,tp1,_=roc_curve(y_cv,c1)
    auc1 = roc_auc_score(y_cv,c1)
    
    fp2,tp2,_=roc_curve(y_cv,c2)
    auc2 = roc_auc_score(y_cv,c2)
    
    fp3,tp3,_=roc_curve(y_cv,c3)
    auc3 = roc_auc_score(y_cv,c3)
    
    fp4,tp4,_ = roc_curve(y_cv,c4)
    auc4 = roc_auc_score(y_cv,c4)
    ax1.tick_params(labelsize=20)
    ax1.plot(fp1,tp1,label=c1_n+':{:.2}'.format(auc1),color='cyan',lw=1,linestyle='solid')
    ax1.plot(fp2,tp2,label=c2_n+':{:.2}'.format(auc2),color='orangered',lw=1,linestyle='solid')
    ax1.plot(fp3,tp3,label=c3_n+':{:.2}'.format(auc3),color='purple',lw=1,linestyle='solid')
    ax1.plot(fp4,tp4,label=c4_n+':{:.2}'.format(auc4),color='steelblue',lw=1,linestyle='solid')
    
    ax1.plot([0, 1], [0, 1], color='gray', lw=0.25,)
    ax1.set_xlim([-0.02,1.0])
    ax1.set_ylim([0,1.03])
    ax1.set_xlabel('1-specificity',fontsize=20)
    ax1.set_ylabel('sensitivity',fontsize=20)
    ax1.set_title("Receiver Operating Characteristic",fontsize=20)
    ax1.legend(loc="lower right",fontsize=15)
    
    p1,r1,_=precision_recall_curve(y_cv,c1)
    auc1 = average_precision_score(y_cv,c1)
    
    p2,r2,_=precision_recall_curve(y_cv,c2)
    auc2 = average_precision_score(y_cv,c2)
    
    p3,r3,_=precision_recall_curve(y_cv,c3)
    auc3 = average_precision_score(y_cv,c3)
    
    p4,r4,_ = precision_recall_curve(y_cv,c4)
    auc4 = average_precision_score(y_cv,c4)
    ax2.tick_params(labelsize=20)
    ax2.plot(p1,r1,label=c1_n+':{:.2}'.format(auc1),color='cyan',lw=1,linestyle='solid')
    ax2.plot(p2,r2,label=c2_n+':{:.2}'.format(auc2),color='orangered',lw=1,linestyle='solid')
    ax2.plot(p3,r3,label=c3_n+':{:.2}'.format(auc3),color='purple',lw=1,linestyle='solid')
    ax2.plot(p4,r4,label=c4_n+':{:.2}'.format(auc4),color='steelblue',lw=1,linestyle='solid')
    
    # Plot Precision-Recall curve
    
    ax2.set_xlabel('sensitivity',fontsize=20)
    ax2.set_ylabel('PPV',fontsize=20)
    ax2.set_ylim([0.0, 1.05])
    ax2.set_xlim([0.0, 1.0])
    ax2.set_title('Precision-Recall',fontsize=20)
    ax2.legend(loc="upper right",fontsize=15)
    f.tight_layout()
    f.savefig('./plots/'+name+'_rocpr.pdf') 
    f.savefig('./plots/'+name+'_rocpr.png')
    
'''    
hfont = {'fontname':'Helvetica'}#http://stackoverflow.com/questions/21321670/how-to-change-fonts-in-matplotlib-python
fs=20
rf_fav=f.sort('imp',ascending=False).T.columns[0:5]
rf_fav=rf_fav[0:5]
names = {el:el for el in rf_fav}
n =toViz[rf_fav].columns.size
naxes=n*n
fig,axes=plt.subplots(nrows=n,ncols=n,figsize=(30,30),squeeze=False,sharex=False,sharey=False)
i = 0
for el_j in rf_fav:
    j = 0
    for el_i in rf_fav:
        print el_i
        if el_i == el_j:
            toViz[el_i].plot.kde(ax=axes[i,j],lw=1)
            axes[i,j].set_axis_bgcolor('white')
            if j == 0:
                axes[i,j].set_ylabel(names[el_j],fontsize=fs,**hfont)
                if i == 0:
                    axes[i,j].tick_params(axis='y',colors='white')
            else:
                axes[i,j].get_yaxis().set_visible(False)
            if i == 4:
                print "i3"
                axes[i,j].set_xlabel(names[el_i],fontsize=fs)
                axes[i,j].get_yaxis().set_visible(False)
                axes[i,j].tick_params(axis='x',colors='white')
                #axes[i,j].set_xticks(axes[i-1,j].get_xticks())
            else:
                axes[i,j].get_xaxis().set_visible(False)
                
            print "e"
        else:
            if i > j :
                
                axes[i,j].scatter(ma.masked_array(toViz[el_i],targ),ma.masked_array(toViz[el_j],targ),marker='o',color='b',alpha=0.25,s=3)
                axes[i,j].scatter(ma.masked_array(toViz[el_i],1-targ),ma.masked_array(toViz[el_j],1-targ),marker='o',color='r',alpha=0.25,s=3)
                axes[i,j].set_axis_bgcolor('white')
                if j == 0:
                    axes[i,j].set_ylabel(names[el_j],fontsize=fs)
                        
                    if i != 4:
                        axes[i,j].get_xaxis().set_visible(False)
                else:
                    axes[i,j].get_yaxis().set_visible(False)
                    if i != 4:                   
                        axes[i,j].get_xaxis().set_visible(False)
                
            else:
                axes[i,j].scatter(ma.masked_array(np.log(toViz[el_i]),targ),ma.masked_array(np.log(toViz[el_j]),targ),marker='o',color='b',alpha=0.25,s=3)
                axes[i,j].scatter(ma.masked_array(np.log(toViz[el_i]),1-targ),ma.masked_array(np.log(toViz[el_j]),1-targ),marker='o',color='r',alpha=0.25,s=3)
                axes[i,j].set_axis_bgcolor('white')
                axes[i,j].get_xaxis().set_visible(False)
                axes[i,j].get_yaxis().set_visible(False)
            
            if i == 4:
                print "i3"
                axes[i,j].set_xlabel(names[el_i],fontsize=fs)



            
        j += 1
    i += 1    
'''    

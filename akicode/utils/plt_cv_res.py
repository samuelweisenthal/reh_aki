'''Helper functions for plots'''

import pandas
import matplotlib.pyplot as plt
import numpy as np
from sklearn.calibration import calibration_curve
import pickle
import copy
import pdb
import nice_plots

def trunc_cols(my_cols, cutoff, n_coef):
    new_cols = []
    long_cols = []
    c = 1
    for l in my_cols:
        if len(l) > cutoff:
            l_c_n = '*' + str(c) + '* ' + l
            long_cols.append(l_c_n)
            new_l = l[0:cutoff] + ' *' + str(c) + '*'
            new_cols.append(new_l)
            c += 1
        else:
            new_cols.append(l)
    lists = [new_cols, long_cols]
    lists = ([el.replace('_UNIQUE_ID_','').replace('UNIQUE_','').replace('_',' ') for el in my_list] for my_list in lists)
    return lists

def plot_coef(coef, xlab, renamedP, renamedN, es_name, n_coef, n_char, _dir, plot_, figsize=(20,5)):
    '''Only truncates if renamed are None'''
    summary = coef.describe().T[['mean']]
    impPos = summary.sort_values(by='mean',na_position='first')[-n_coef:].index
    raw_col = impPos
    impNeg = summary.sort_values(by='mean',na_position='last')[:n_coef].index
    imp_dfP = coef[impPos]
    imp_dfN = coef[impNeg]

    Pcopy = copy.deepcopy(imp_dfP.columns)
    #imp_dfP.columns = [el[0:n_char] for el in imp_dfP.columns]
    if renamedP: 
        imp_dfP.columns, longP = renamedP, []
    else:
        imp_dfP.columns, longP = trunc_cols(imp_dfP.columns, n_char, n_coef)
    Ncopy = copy.deepcopy(imp_dfN.columns)
    #imp_dfN.columns = [el[0:n_char] for el in imp_dfN.columns]
    if renamedN: 
        imp_dfN.columns, longN = renamedN, []
    else:
        imp_dfN.columns, longN = trunc_cols(imp_dfN.columns, n_char, n_coef)
        
    plot_ = plot_
    if plot_:
        plt.rc('text', usetex=False)
        ax = imp_dfP.plot.box(rot=0, figsize=figsize, vert=False, return_type='axes')

        fn1 = _dir + es_name +'1.png'
        plt.xlabel(xlab)
        plt.xticks(rotation=90)
        ax.yaxis.tick_right()
        ax.yaxis.set_label_position("right")
        
        plt.tick_params(
        axis='both',          # changes apply to the x-axis
        which='both',      # both major and minor ticks are affected
        bottom='off',      # ticks along the bottom edge are off
        left = 'off',
        right = 'off',
        top='off')        # ticks along the top edge are off
        ax = plt.gca()
        ax.spines['left'].set_visible(False)
        ax.spines['top'].set_visible(False)
        
        plt.savefig(fn1, dpi=600, bbox_inches='tight')
        
        plt.show()
        if es_name =='lr1':
            ax = imp_dfN.plot.box(rot=0, figsize=figsize, vert=False, return_type='axes')
            fn2 = _dir + es_name + '2.png'
            #plt.yticks(rotation=90)
            plt.tick_params(
            axis='both',          # changes apply to the x-axis
            which='both',      # both major and minor ticks are affected
            bottom='off',      # ticks along the bottom edge are off
            left = 'off',
            right = 'off',
            top='off')        # ticks along the top edge are off
            ax = plt.gca()
            ax.spines['left'].set_visible(False)
            ax.spines['top'].set_visible(False)
            plt.xlabel(xlab)
            plt.xticks(rotation=90)
            ax.yaxis.tick_right()
            ax.yaxis.set_label_position("right")
            plt.savefig(fn2, dpi=600, bbox_inches='tight')
        
            plt.show()
            
    return imp_dfP, longP, imp_dfN, longN, raw_col
    #plt.tight_layout()
    
    
    
def plot_trials_coef(_dir, es_name, xlab, n_coef, n_char, plot_, figsize=(20,5), renamedP=None, renamedN=None, get_coef_only=False):

    num = 1
    fname = _dir + 'data_rs_' + str(num) + '.pickle'
    with open(fname, 'rb') as f: d = pickle.load(f)
    [best_clf, m_d_outers, param_res, all_param_res, all_param_full_res, y_trues_hats, searches, best_searches, full_searches, best_full_search, all_co, best_co, all_full_co, best_full_co, feat_names, _] = d
    #pdb.set_trace()
    if 'lr' in es_name:
        co = [el[0] for el in all_co[es_name]]
    else:
        co = all_co[es_name]
    coefs = pandas.DataFrame(co, columns=feat_names)
    coef = pandas.DataFrame(co, columns=feat_names)
    for i in range(2,51):
        num = i
        fname = _dir + 'data_rs_'+str(num) + '.pickle'
        with open(fname, 'rb') as f: d = pickle.load(f)
        [best_clf, m_d_outers, param_res, all_param_res, all_param_full_res, y_trues_hats, searches, best_searches, full_searches, best_full_search, all_co, best_co, all_full_co, best_full_co, feat_names, _] = d
        if 'lr' in es_name:
            co = [el[0] for el in all_co[es_name]]
        else:
            co = all_co[es_name]
        coef += pandas.DataFrame(co, columns=feat_names)
	coefs = coefs.append(pandas.DataFrame(co, columns=feat_names))
    if get_coef_only:
	return coefs
    else:
	 impP, lp, impN, ln, raw_col = plot_coef(coef, xlab, renamedP, renamedN, plot_=plot_, es_name=es_name, n_coef=n_coef, n_char=n_char, _dir=_dir, figsize=figsize)
   	 return impP, lp, impN, ln, raw_col



def plt_w(met, curve, xlab, ylab, qoi, ax, m_d_outers, alpha, identity, plt_first=False):
    #for i in reversed(range(4,5)):
    for i in reversed(range(len(m_d_outers[met]))):
        ((y, x), auc) = zip([(el[0],el[1]) for el in m_d_outers[curve]], m_d_outers[curve])[i]
        if met == 'roc':
            x, y = y, x
        plt_(x, y, xlab, ylab, i, ax=ax, alpha=alpha, plt_first=plt_first)
    if met != 'pr': 
        if identity:
            ax.plot([0, 1], [0, 1], color='k', lw=2)
        

def plt_(x, y, xlabel, ylabel, i, ax, alpha, plt_first=False):
    #plt.plot(tpr, fpr, label='AUC:{:.2}'.format(auc), color='black', lw=1)
    if i ==0 and plt_first:
        ax. plot(x, y, color='black', lw=2, linestyle='--') #marker='o'
    else:
        ax.plot(x, y, color='gray', lw=.5, alpha=alpha)
    #plt.plot(x, y, color='black', lw=1, alpha=1)
    ax.tick_params(
    axis='both',          # changes apply to the x-axis
    which='both',      # both major and minor ticks are affected
    bottom='off',      # ticks along the bottom edge are off
    left = 'off',
    right = 'off',
    top='off')        # ticks along the top edge are off
    #ax = plt.gca()
    ax.spines['right'].set_visible(False)
    ax.spines['top'].set_visible(False)
    ax.set_xlim([0,1])
    ax.set_ylim([0,1])
    ax.set_xlabel(xlabel)
    ax.set_ylabel(ylabel)
    ax.legend(loc='lower right')
    #ax.set_aspect('equal', adjustable='box')

def change_n_bins_cal_curves(m_d_outers, y_trues_hats, n_bins=10): #5 same as old  
    '''Modifies m_d_outers in place'''
    from sklearn.calibration import calibration_curve
    new_cal_curves = {}
    new_cal_curves['cal_curve'] = [calibration_curve(*y_trues_hats[i], n_bins=n_bins) for i in range(len(m_d_outers['bri']))]
    new_cal_curves['bri'] = m_d_outers['bri']
    return new_cal_curves

def plot_trials(_dir, met, curve, qoi, t, x_txt, y_txt, xlab, ylab, alpha=0.5, identity=1, ax=None, plt_first=False):
    
    macro_means = []
    macro_stds = []
    micros = [] 
    for i in reversed(range(1,51)):
        num = i
        fname = _dir + 'data_rs_'+str(num)+'.pickle'
        with open(fname, 'rb') as f: d = pickle.load(f)
        [best_clf, m_d_outers, param_res, all_param_res, all_param_full_res, y_trues_hats, searches, best_searches, full_searches, best_full_search, all_co, best_co, all_full_co, best_full_co, feat_names, _] = d

        if met == 'bri':
            m_d_outers = change_n_bins_cal_curves(m_d_outers, y_trues_hats, n_bins=10)
        if i ==1 and plt_first:
            plt_w(met, curve, xlab, ylab, qoi, ax, m_d_outers, alpha=alpha, identity=identity, plt_first=plt_first)
        else:
            plt_w(met, curve, xlab, ylab, qoi, ax, m_d_outers, alpha=alpha, identity=identity)
        s_mean = np.mean(m_d_outers[met])  
        s_std = np.std(m_d_outers[met])
        micros += m_d_outers[met]
        macro_means.append(s_mean)
        macro_stds.append(s_std)

    micro_mean = np.mean(micros)
    micro_std = np.std(micros)

    macro_mean_mean = np.mean(macro_means)
    macro_mean_std = np.std(macro_means)
    macro_std_mean = np.mean(macro_stds)
    macro_std_std = np.std(macro_stds)

    #t = qoi + ': ' + str(round(s_mean,4)) + ' $\pm\ $' + str(round(s_std,4))
    micro_text = qoi + ' Micro: '+str(round(micro_mean,2)) + ' $\pm\ $' + str(round(micro_std,2))
    macro_text = qoi + ' Macro: '+str(round(macro_mean_mean,2)) + ' $\pm\ $' + str(round(macro_mean_std,2))
    print qoi + ' Micro: '+str(round(micro_mean,5)) + ' $\pm\ $' + str(round(micro_std,5))
    print qoi + ' Macro: '+str(round(macro_mean_mean,5)) + ' $\pm\ $' + str(round(macro_mean_std,5))
    ax.text(x_txt, y_txt, micro_text, fontsize=10)
    ax.text(x_txt, y_txt-0.1, macro_text, fontsize=12)
    #plt.text(0.52, 0.2, 'matplotlib')
    #plt.text(0.52, 0.1, 'matplotlib')
    ax.set_title(t) 
    # fn = _dir + t + '.png'
    # plt.savefig(fn, dpi=600, bbox_inches='tight')



def get_all_pred(n_iterations, _dir):
    y_h_alls = {}
    y_t_alls = {}
    for i in range(1, n_iterations+1):
        fname = _dir + 'data_rs_' + str(i) + '.pickle'
        with open(fname, 'rb') as f: d = pickle.load(f)
        [best_clf, m_d_outers, param_res, all_param_res, all_param_full_res, y_trues_hats, searches, best_searches, full_searches, best_full_search, all_co, best_co, all_full_co, best_full_co, feat_names, splits] = d
        indices = np.concatenate([split[1] for split in splits])
        yhats = np.concatenate([el[1] for el in y_trues_hats])
        ytrues_by_res = np.concatenate([el[0] for el in y_trues_hats])
        df = pandas.DataFrame({'ind':indices, 'yhat':yhats, 'y_trues_by_res':ytrues_by_res}).sort_values('ind')
        #print df.head()
        y_h_alls[i] = df['yhat'].values
        y_t_alls[i] = df['y_trues_by_res'].values
    return y_h_alls, y_t_alls


def plt_error(data, let_y, thresh, linspace_start, linspace_end):

    oi = data[data['y']==let_y]
    #oi['error'][oi['error']>thresh].hist(bins=100, cumulative=False, color='gray', edgecolor='none')
    #ax = plt.gca()
    #ax.set_yscale('log')
    #oi[oi['error']<=thresh]['error'].hist(bins=100, cumulative=False, color='blue', edgecolor='none')
    #oi['error'].hist(bins=100, color='gray', cumulative=True, edgecolor='none')
    oi['error'].hist(bins=100, color='black')
    for i in np.linspace(linspace_start,linspace_end,2000*(abs(linspace_end - linspace_start))):
        plt.axvline(x=i, c='red', lw=5, alpha=0.002)
    plt.grid(False)

    plt.tick_params(
    axis='both',          # changes apply to the x-axis
    which='both',      # both major and minor ticks are affected
    bottom='off',      # ticks along the bottom edge are off
    left = 'off',
    right = 'off',
    top='off')        # ticks along the top edge are off

    plt.xlabel('Error')
    plt.ylabel('Log Frequency')
    plt.title('y='+str(let_y))
    ax = plt.gca()
    ax.set_yscale('log')
    ax.spines['right'].set_visible(False)
    ax.spines['top'].set_visible(False)

def plt_scatter(data, X_df, y):
    from pandas.tools.plotting import scatter_matrix
    '''must pass only columns to show in X_df'''
    for cmap, cvec in zip(['Reds', 'RdBu_r', 'winter_r', 'winter_r'],[data['abserror'], data['error'], data['mean_y_hat'], y]):
        f = scatter_matrix(X_df, c=cvec, cmap=cmap, edgecolor='face', diagonal='kde')
        
        
def plt_perf(_dir, identity=1, plt_first=True):
    fig, (ax1, ax2, ax3) = plt.subplots(nrows=1, ncols = 3, sharey=True, figsize=(15,4))
    plot_trials(_dir=_dir, met='roc', curve ='roc_curve', qoi='AUC', 
                t='ROC', 
                x_txt=0.26, y_txt=0.2, 
                xlab='False Positive Rate', ylab='True Positive Rate', ax=ax1, plt_first=plt_first)
    plot_trials(_dir=_dir, met='bri', curve ='cal_curve', qoi='Brier', 
            t='Calibration', 
            x_txt=0.01, y_txt=0.93,
            xlab='Mean Predicted Value', ylab='Fraction Positives', ax=ax2, identity=identity, plt_first=plt_first)
    ax2.spines['left'].set_visible(False)
    plot_trials(_dir=_dir, met='pr', curve ='pr_curve', qoi='AUC', 
                t='PPV', 
                x_txt=0.3, y_txt=0.8,
                xlab='Recall', ylab='Precision', ax=ax3, plt_first=plt_first)
    ax3.spines['left'].set_visible(False)
    fn = _dir + 'per.pdf'
    plt.savefig(fn, dpi=600, bbox_inches='tight')
    
    
def std_by_pp(agg_res, _dir):

    fig, (ax1, ax2, ax3) = plt.subplots(nrows=1, ncols = 3, sharey=True, figsize=(10,3))

    ax1.scatter(agg_res['mean_y_hat'], agg_res['std_y_hat'], alpha=0.01, color='gray')
    #ax = plt.gca()
    # ax.set_yscale('log')
    # ax.set_xscale('log')
    #ax1.set_xlabel('Mean Predicted Probability')
    ax1.set_ylabel('STD Predicted Probability')
    ax1.tick_params(
    axis='both',          # changes apply to the x-axis
    which='both',      # both major and minor ticks are affected
    bottom='off',      # ticks along the bottom edge are off
    left = 'off',
    right = 'off',
    top='off')        # ticks along the top edge are off

    ax1.spines['right'].set_visible(False)
    ax1.spines['top'].set_visible(False)
    ax1.set_title('All Samples')
    #plt.gca().set_aspect('equal', adjustable='box')


    ax2.scatter(agg_res[agg_res['y']==1]['mean_y_hat'], agg_res[agg_res['y']==1]['std_y_hat'], alpha=0.01, color='gray')
    print len(agg_res[agg_res['y']==1])
    #ax = plt.gca()
    # ax.set_yscale('log')
    # ax.set_xscale('log')
    ax2.set_title('AKI +')
    ax2.set_xlabel('Mean Predicted Probability')
    #ax2.set_ylabel('STD Predicted Probability')
    ax2.tick_params(
    axis='both',          # changes apply to the x-axis
    which='both',      # both major and minor ticks are affected
    bottom='off',      # ticks along the bottom edge are off
    left = 'off',
    right = 'off',
    top='off')        # ticks along the top edge are off

    ax2.spines['right'].set_visible(False)
    ax2.spines['top'].set_visible(False)
    ax2.spines['left'].set_visible(False)
    #plt.gca().set_aspect('equal', adjustable='box')

    ax3.scatter(agg_res[agg_res['y']==0]['mean_y_hat'], agg_res[agg_res['y']==0]['std_y_hat'], alpha=0.01, color='gray')
    ax3.set_title('AKI -')
    print len(agg_res[agg_res['y']==0])
    #ax = plt.gca()
    # ax.set_yscale('log')
    # ax.set_xscale('log')
    #ax3.set_xlabel('Mean Predicted Probability')
    #ax3.set_ylabel('STD Predicted Probability')
    ax3.tick_params(
    axis='both',          # changes apply to the x-axis
    which='both',      # both major and minor ticks are affected
    bottom='off',      # ticks along the bottom edge are off
    left = 'off',
    right = 'off',
    top='off')        # ticks along the top edge are off

    ax3.spines['right'].set_visible(False)
    ax3.spines['top'].set_visible(False)
    ax3.spines['left'].set_visible(False)
    #plt.gca().set_aspect('equal', adjustable='box')

    fn = _dir + 'mean_std.pdf'

    plt.savefig(fn, dpi=600, bbox_inches='tight')
    
    
def get_coef_table(df):

    #c = pandas.concat([df.T.mean(axis=1), df.T.std(axis=1), df.T.min(axis=1), df.T.max(axis=1)], axis=1)
    c = pandas.concat([df.T.mean(axis=1), df.T.std(axis=1)], axis=1)
    #c.columns = ['mean', 'std', 'min', 'max']
    c.columns = ['mean', 'std']
    c = c.sort_values('mean', ascending=False)
    return c

def count_nonzero_feat(es_name, _dir):

    imP, longP, imN, longN, raw_col = plot_trials_coef(_dir=_dir, xlab='', es_name=es_name, plot_=0, n_coef=1000000, n_char=20000, figsize=(2.5,20), renamedP=None, renamedN=None)
    _feat = get_coef_table(imP)
    print "num nonzero (if mean and std 0, then removed)",es_name,':', len(_feat['mean']) - len(_feat[(_feat['mean']==0) & (_feat['std']==0)])
 
    
def make_nice(ax):
    ax.tick_params(
    axis='both',          # changes apply to the x-axis
    which='both',      # both major and minor ticks are affected
    bottom='off',      # ticks along the bottom edge are off
    left = 'off',
    right = 'off',
    top='off')        # ticks along the top edge are off
    ax.spines['right'].set_visible(False)
    ax.spines['top'].set_visible(False)
    ax.spines['bottom'].set_visible(False)


def error_by_dx(df, _dir):
    fn = 'error_by_dx'    
    yneg = 2
    fig, (ax0, ax1, ax2, ax3, ax4, ax5) = plt.subplots(nrows=6, sharex=True, figsize=(10,8))

    #if bypt: ax1.set_title('Patient-specific', fontsize=25)
    #else: ax1.set_title('Hospitalization-specific', fontsize=25)
    my_c = 'gray'

    for ax, qoi, nqoi, lab in zip([ax0, ax1, ax2, ax3, ax4, ax5], 
                                  ['or', 'AKI_by_admi', 'creat', 'and', 'AKI_by_admi', 'creat'], 
                                  ['nor','nAKI_by_admi', 'ncreat', 'nand', 'creat', 'AKI_by_admi'], 
                                  ['code $\lor$ sCr', 'code', 'sCr', 'code $\land$ sCr', 'code - sCr', 'sCr - code']):

        cases_to_plot = df[(df[qoi]==1) & (df[nqoi]==0)]['error']

        h = ax.hist(cases_to_plot, bins=1000, color=my_c)
        ypos =  int(np.max(h[0])/float(2))
        make_nice(ax)
        offset = -0.3
        ax.text(offset, ypos, lab, fontsize=25)
        if (qoi == 'creat' and nqoi == 'AKI_by_admi'): ax.spines['bottom'].set_visible(True)
        mn = round(np.mean(cases_to_plot),2)
        std = round(np.std(cases_to_plot),2)
        N = '(N='+str(len(cases_to_plot))+')'
        mn = 'Mean: '+str(mn)
        std = 'STD: '+str(std)
        mytxt = mn+'; '+std
        ax.text(0.01, ypos*1.7, mytxt)
        ax.text(offset, ypos*0.2, N, fontsize=15)
        #ax1.set_title('AKI +')
        plt.plot([0,0],[1,1])
        plt.tight_layout()


    fn = _dir + fn + '.pdf'
    plt.savefig(fn, dpi=600, bbox_inches='tight')
    

def add_errors(data, agg_res, groups, X):
    
    data['y'] = agg_res['y']
    data['groups'] = groups

    data['mean_y_hat'] = agg_res['mean_y_hat']
    data['std_y_hat'] = agg_res['std_y_hat']

    data['error'] = (data['mean_y_hat']-data['y'])
    data['abserror'] = abs(data['mean_y_hat']-data['y'])

    #X.shape

    #errors = data.sort_values('error')

    all_data = pandas.DataFrame(X)
    all_data['error'] = (data['mean_y_hat']-data['y'])
    all_data['var_error'] = data['std_y_hat']
    all_data['y'] = data['y']
    all_data['groups'] = data['groups']
    
    return data, all_data

def make_reg(X, agg_res, groups, outcome,
             multiplier, data, tag_oi, _dir):

    data, all_data = add_errors(data, agg_res, groups, X)

    #

    abserrors = data.sort_values('abserror')

    data.sort_values('abserror').head()


    #fig, (ax1, ax2) = plt.subplots(nrows=1, ncols=2)

    #ax1.hist(data[data['y']==outcome]['error'])
    #ax1.set_title("raw error hist")
    my_y = data[data['y']==outcome]['error']*multiplier #use mult to make errors nonneg
    my_X = data[data['y']==outcome]
    my_g = my_X['groups']
    my_X = my_X[my_X.columns[0:-7]]
    my_X = my_X[[el for el in my_X.columns if tag_oi in el]]
    feat_names = my_X.columns

    #ax2.hist(my_y)
    #ax2.set_title('target hist')
    if outcome == 1: dirname = 'case_error'
    else: dirname = "contr_error"
    my_dir = _dir + dirname + '/' + tag_oi + '/'
    print "saving to", my_dir
    import os 
    if not os.path.exists(my_dir):
        os.makedirs(my_dir)
    
    y_path = my_dir + 'y_error.csv'
    X_path = my_dir + 'X_error.csv'
    g_path = my_dir + 'group_error.csv'

    my_y.to_csv(y_path)
    my_X.to_csv(X_path)
    my_g.to_csv(g_path)
    return my_y, my_X, my_g, feat_names


def regr(agg_res, X, y, groups, feat_names, alpha):
    from sklearn.model_selection import GroupKFold
    X, y, groups = np.array(X), np.ravel(np.array(y)), np.ravel(np.array(groups))

    gkf=GroupKFold(n_splits=5)#not sure if this has to be done again, but can't hurt

    fv = gkf.split(X, y, groups=groups)

    splits=[]
    for train,test in fv:

        tr = sum(y[train])/float(len(y[train]))

        te = sum(y[test])/float(len(y[test]))

        splits.append((train,test))

    train_val_ix,test_ix = splits[0][0],splits[0][1]

    X_train_and_cv = X[train_val_ix]
    y_train_and_cv = y[train_val_ix]
    groups_train_and_cv = groups[train_val_ix]
    X_test = X[test_ix]
    y_test = y[test_ix]
    groups_test = groups[test_ix]

    gkf=GroupKFold(n_splits=5)#not sure if this has to be done again, but can't hurt
    fv = gkf.split(X_train_and_cv, y_train_and_cv, groups=groups_train_and_cv)

    splits=[]
    for train,test in fv:

        tr = sum(y_train_and_cv[train])/float(len(y_train_and_cv[train]))

        te = sum(y_train_and_cv[test])/float(len(y_train_and_cv[test]))

        splits.append((train,test))

    train_ix,val_ix = splits[0][0],splits[0][1]

    X_train = X_train_and_cv[train_ix]
    y_train = y_train_and_cv[train_ix]
    groups_train = groups_train_and_cv[train_ix]
    X_val = X_train_and_cv[val_ix]
    groups_val = groups_train_and_cv[val_ix]
    y_val = y_train_and_cv[val_ix]

    gkf=GroupKFold(n_splits=5)#not sure if this has to be done again, but can't hurt
    fv = gkf.split(X_train_and_cv, y_train_and_cv, groups=groups_train_and_cv)
    
    from sklearn.linear_model import Lasso
    from sklearn.pipeline import make_pipeline
    from sklearn.preprocessing import Imputer, StandardScaler
    lr1     = make_pipeline(Imputer(missing_values=np.NAN,
                                          strategy="mean",
                                          axis=0,
                                          verbose=10),
                       StandardScaler(),
                        #ub 2e-3
                       #Lasso(alpha=0.01, random_state=1)) #train time inc w c!!
                       Lasso(alpha=0.015, random_state=1)) #train time inc w c!!
    lr1=lr1.fit(X_train,y_train)
    lr1.named_steps['lasso'].coef_
    c = pandas.DataFrame(lr1.named_steps['lasso'].coef_,
                     index=feat_names,columns=['imp l1'])
    print c[c['imp l1']!=0].sort_values('imp l1',ascending=False)
    from sklearn.metrics import mean_squared_error
    print "train", mean_squared_error(y_train, lr1.predict(X_train))
    print "val", mean_squared_error(y_val, lr1.predict(X_val))
    print "test", mean_squared_error(y_test, lr1.predict(X_test))
    
    
    
def get_nhosp_per_id(agg_res):
    print "#hosp", len(agg_res)
    ids = agg_res['pt_id']
    unique_ids = ids.unique()
    print "#pt", len(unique_ids)
    pt_to_nhosp = {el:len(agg_res[agg_res['pt_id']==el]) 
               for el in unique_ids}
    return pt_to_nhosp

def agg_by_nhosp(data):
    num_hosp = data.copy()
    num_hosp['error'] = num_hosp['mean_y_hat'] - num_hosp['y']
    num_hosp['abs_error'] = np.abs(num_hosp['error'])

    pt_to_nhosp = get_nhosp_per_id(num_hosp)

    agg_pt = num_hosp.groupby(
        'pt_id',as_index='False').agg(['mean', 'std'])


    agg_pt['n_hosp'] = [pt_to_nhosp[x] for x in agg_pt.index]

    all_errors = agg_pt.groupby('n_hosp').agg(['mean','std'])
    print "num num hosp", len(all_errors)
    all_errors = all_errors.reset_index()
    all_errors['log_n_hosp'] = np.log(all_errors['n_hosp'])
    
    return all_errors

def utiliz(agg_res, _dir):
    fig, ([[ax1, ax5,ax9], [ax2,ax6,ax10]]) = plt.subplots(nrows=2, ncols=3, figsize=(10,7))

    #x,y,fn,alpha,xlab,ylab
    my_scale = 'log_n_hosp'
    my_scale = 'n_hosp'

    d = agg_by_nhosp(agg_res) #all data

    nice_plots.plt_scatter_ax(d[my_scale],
                           d['abs_error']['mean']['mean'],
                           1, 'Mean Abs Error', ax1)
    ax1.spines['bottom'].set_visible(False)
    ax1.set_ylabel('Mean Abs Error', fontsize=20)
    ax1.set_title('All Samples')
    ax1.tick_params(
        axis='both',          # changes apply to the x-axis
        which='both',      # both major and minor ticks are affected
        bottom='off',      # ticks along the bottom edge are off
        )        # ticks along the top edge are off

    ax1.get_xaxis().set_ticks([])
    nice_plots.plt_scatter_ax(d[my_scale],
                           d['std_y_hat']['mean']['mean'],
                           1, 'Mean STD $\hat{P_P}$', ax2)
    #ax2.spines['bottom'].set_visible(False)
    ax2.set_ylabel('Mean STD $P_P$', fontsize=20)

    mn = min(d['n_hosp'])
    mx = max(d['n_hosp'])
    lab = 'Min='+str(mn)+', Max='+str(mx)
    ax2.set_xlabel(lab)

    d = agg_by_nhosp(agg_res[agg_res['y']==1])

    nice_plots.plt_scatter_ax(d[my_scale],
                           d['abs_error']['mean']['mean'],
                           1, 'Mean Abs Error', ax5)
    ax5.spines['bottom'].set_visible(False)
    ax5.get_xaxis().set_ticks([])
    ax5.set_title('AKI +')

    nice_plots.plt_scatter_ax(d[my_scale],
                           d['std_y_hat']['mean']['mean'],
                           1, 'Mean STD $\hat{P_P}$', ax6)
    #ax6.spines['bottom'].set_visible(False)

    mn = min(d['n_hosp'])
    mx = max(d['n_hosp'])
    lab = 'Min='+str(mn)+', Max='+str(mx)
    ax6.set_xlabel(lab)
    ax6.text(-1,-0.025,'No. Hospitalizations', fontsize=25)

    d = agg_by_nhosp(agg_res[agg_res['y']==0])

    nice_plots.plt_scatter_ax(d[my_scale],
                           d['abs_error']['mean']['mean'],
                           1, 'Mean Abs Error', ax9)
    ax9.set_title('AKI -')
    ax9.get_xaxis().set_ticks([])
    ax9.spines['bottom'].set_visible(False)

    nice_plots.plt_scatter_ax(d[my_scale],
                           d['std_y_hat']['mean']['std'],
                           1, 'Mean STD $\hat{P_P}$', ax10)
    #ax10.spines['bottom'].set_visible(False)

    mn = min(d['n_hosp'])
    mx = max(d['n_hosp'])
    lab = 'Min='+str(mn)+', Max='+str(mx)
    ax10.set_xlabel(lab)

    fn = _dir + 'Nhosp_error.pdf'
    plt.tight_layout()
    plt.savefig(fn, dpi=600, bbox_inches='tight')

    
def by_pt(agg_res, _dir):
    
    mean_pts = agg_res.groupby('pt_id').mean()
    one_or_zero = mean_pts[(mean_pts['mean_y_t']==0) | (mean_pts['mean_y_t']==1)]
    not_one_or_zero = mean_pts[(mean_pts['mean_y_t']!=0) & (mean_pts['mean_y_t']!=1)]

    def plt_pred(df,alpha,fn,ax):


        ax.scatter( df['mean_y_hat'], df['mean_y_t'], alpha=0.05, color='gray')
        if fn == 'ad':
            ax.set_ylabel('$P_O$', fontsize=25)
            ax.set_xlabel('$P_P$', fontsize=25)
        else:
            ax.set_ylabel('$\overline{P_O}$', fontsize=25)
            ax.set_xlabel('$\overline{P_P}$', fontsize=25)
        ax.tick_params(
        axis='both',          # changes apply to the x-axis
        which='both',      # both major and minor ticks are affected
        bottom='off',      # ticks along the bottom edge are off
        left = 'off',
        right = 'off',
        top='off')        # ticks along the top edge are off

        ax.spines['right'].set_visible(False)
        ax.spines['top'].set_visible(False)
        ax.plot([0,0],[1,1])
        #if fn == 'ad': plt.title('Observed vs. Predicted Hospitalization Risk', fontsize=25)
        #else: plt.title('Observed vs. Predicted Patient Risk', fontsize=25)
        #plt.gca().set_aspect('equal', adjustable='box')
        #fn = _dir + fn + '.pdf'
        #plt.savefig(fn, dpi=600, bbox_inches='tight')

    def plt_dis_by_pt(agg_res, mean_pts):

        fig, ([[ax1,ax2], [ax3,ax4]]) = plt.subplots(nrows=2, ncols=2, figsize=(10,10))

        plt_pred(agg_res,alpha=0.005, ax=ax1, fn=None)

        from sklearn.calibration import calibration_curve
        frac_pos, mean_pv = calibration_curve(agg_res['mean_y_t'], agg_res['mean_y_hat'], n_bins=10)

        import nice_plots
        reload(nice_plots)
        fn = _dir + 'pt_cal.pdf' 

        nice_plots.plt_scatter_ax2(mean_pv, frac_pos, 1, ax=ax1, xlab='Mean Predicted Value or $P_P$', ylab='Fraction Positives or $P_O$')
        ax1.plot(mean_pv,frac_pos, color='red')
        ax1.plot([0,1],[0,1], lw=2, color='black', linestyle='--')
        ax1.set_aspect('equal', adjustable='box')


        plt_pred(mean_pts,alpha=0.005,ax=ax3,fn=None)

        frac_pos, mean_pv  = calibration_curve(one_or_zero['mean_y_t'], one_or_zero['mean_y_hat'], n_bins=10)
        nice_plots.plt_scatter_ax2(mean_pv, frac_pos, 1, ax=ax3, xlab='Mean Predicted Value or $\overline{P_P}$',ylab='Fraction Positives or $\overline{P_O}$')
        ax3.plot(mean_pv, frac_pos, color='red')
        ax3.plot([0,1],[0,1], lw=2, color='black', linestyle='--')
        ax3.set_aspect('equal', adjustable='box')

        ax2.set_visible(False)
        ax4.set_visible(False)

        fn = _dir + 'cal_dist.pdf'
        plt.savefig(fn, dpi=600, bbox_inches='tight')
        
        
    def make_nice(ax):
        ax.tick_params(
        axis='both',          # changes apply to the x-axis
        which='both',      # both major and minor ticks are affected
        bottom='on',      # ticks along the bottom edge are off
        left = 'off',
        right = 'off',
        top='off')        # ticks along the top edge are off
        ax.spines['right'].set_visible(False)
        ax.spines['top'].set_visible(False)

    def plt_stk(df, ypos, yneg, fn, bypt):
        if bypt: fig, (ax1,ax2,ax3) = plt.subplots(nrows=3, sharex=True, figsize=(5,5))
        else: fig, (ax1,ax2,ax3) = plt.subplots(nrows=3, sharex=True, figsize=(5,5))
        #if bypt: ax1.set_title('Patient-specific', fontsize=25)
        #else: ax1.set_title('Hospitalization-specific', fontsize=25)
        my_c = 'gray'
        ax1.hist(df['mean_y_hat'][df['mean_y_t']==1], bins=1000, color=my_c)
        make_nice(ax1)
        ax1.tick_params(axis='both', which='both', bottom='off')
        ax1.spines['bottom'].set_visible(False)
        ax1.set_xlim([0,1])
        offset = -0.45
        if bypt: ax1.text(offset, ypos, '$\overline{P_O}$=1', fontsize=25)
        else: ax1.text(offset, ypos, 'AKI+', fontsize=25)
        #ax1.set_title('AKI +')
        plt.plot([0,0],[1,1])
        plt.tight_layout()

        if bypt:
            ax2.hist(df['mean_y_hat'][df['mean_y_t']==0.5], bins=1000, color=my_c)
            if bypt: ax2.text(offset, 2.5, '$\overline{P_O}$=0.5', fontsize=25)
            else: ax2.text(offset, 2.5, 'AKI-', fontsize=25) 
            make_nice(ax2)
            ax2.tick_params(axis='both', which='both', bottom='off')
            ax2.spines['bottom'].set_visible(False)
            #ax2.set_title('AKI -')
            #plt.xlabel('$P(AKI)$')
            plt.plot([0,0],[1,1])
            plt.tight_layout()
        else:
            ax2.hist(df['mean_y_hat'][df['mean_y_t']==0.5], bins=1000, color=my_c)
            if bypt: ax2.text(offset, 2.5, '$\overline{P_O}$=0.5', fontsize=25)
            else: ax2.text(offset, 2.5, 'AKI-', fontsize=25) 
            make_nice(ax2)
            ax2.tick_params(axis='both', which='both', bottom='off')
            ax2.spines['bottom'].set_visible(False)
            ax2.set_visible(False)
            #ax2.set_title('AKI -')
            #plt.xlabel('$P(AKI)$')
            plt.plot([0,0],[1,1])
            plt.tight_layout()

        ax3.hist(df['mean_y_hat'][df['mean_y_t']==0], bins=1000, color=my_c)
        if bypt: ax3.text(offset, yneg, '$\overline{P_O}$=0', fontsize=25)
        else: ax3.text(offset, yneg, 'AKI-', fontsize=25) 
        #ax2.set_title('AKI -')
        if bypt: plt.xlabel('$\overline{P_P}$')
        else: plt.xlabel('$P_P$')
        make_nice(ax3)
        plt.plot([0,0],[1,1])
        plt.tight_layout()


        # f=plt.gca()
        #ax2.set_yscale('log')
        fn = _dir + fn + '.pdf'
        plt.savefig(fn, dpi=600, bbox_inches='tight')

    plt_dis_by_pt(agg_res, mean_pts)
    plt_stk(agg_res, ypos=25, yneg=5000, fn='byhosp', bypt=0)
    plt_stk(mean_pts, ypos=5, yneg=2500, fn='bypt', bypt=1)

    
    
def make_agg_res(_dir, dxs_et_al, y, groups):

    y_h_alls, y_t_alls = get_all_pred(n_iterations=50, _dir=_dir)

    agg_res = pandas.DataFrame(
        {'mean_y_hat':pandas.DataFrame(y_h_alls).mean(axis=1),
         'std_y_hat':pandas.DataFrame(y_h_alls).std(axis=1),
         'mean_y_t':pandas.DataFrame(y_t_alls).mean(axis=1),
         'std_y_t':pandas.DataFrame(y_t_alls).std(axis=1)
        }
    )

    agg_res['or'] = list(dxs_et_al['or']) #use lists so doesn;t try to merge on index
    agg_res['and'] = list(dxs_et_al['and'])
    agg_res['AKI_by_admi'] = list(dxs_et_al['AKI_by_admi'])
    agg_res['creat'] = list(dxs_et_al['creat'])
    agg_res['y']=y
    agg_res['error'] = abs(agg_res['mean_y_hat']-agg_res['y'])
    agg_res['nor']= 1-agg_res['or']
    agg_res['nand'] = 1-agg_res['and']
    agg_res['nAKI_by_admi'] = 1-agg_res['AKI_by_admi']
    agg_res['ncreat'] = 1-agg_res['creat']
    agg_res['pt_id'] = groups

    return agg_res





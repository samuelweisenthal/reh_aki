'''Helper functions for plots'''

import pandas
import matplotlib.pyplot as plt
font = {'family': 'serif', 'weight':'bold', 'size':'20'}
plt.rc('font', **font)
plt.rc('text', usetex=True)

def nice_hist(my_df, fname='a_plot', xlab='xlab', ylab='ylab', yscal='linear'):
    f = my_df.hist(bins=100, color='gray', grid=False)
    plt.xlabel(xlab)
    plt.ylabel(ylab)
    plt.yscale(yscal)
    plt.tick_params(
    axis='both',          # changes apply to the x-axis
    which='both',      # both major and minor ticks are affected
    bottom='off',      # ticks along the bottom edge are off
    left = 'off',
    right = 'off',
    top='off')        # ticks along the top edge are off
    f.spines['right'].set_visible(False)
    f.spines['top'].set_visible(False)
    plt.savefig(fname, dpi=600, bbox_inches='tight')
    
    
    
def plt_scatter(x,y,fn,alpha,xlab,ylab):

    plt.scatter(x, y, alpha=alpha, color='gray')

    plt.xlabel(xlab, fontsize=20)
    plt.ylabel(ylab, fontsize=20)
    plt.tick_params(
    axis='both',          # changes apply to the x-axis
    which='both',      # both major and minor ticks are affected
    bottom='off',      # ticks along the bottom edge are off
    left = 'off',
    right = 'off',
    top='off')        # ticks along the top edge are off
    f = plt.gca()
    f.spines['right'].set_visible(False)
    f.spines['top'].set_visible(False)
    #plt.gca().set_aspect('equal', adjustable='box')
    plt.savefig(fn, dpi=600, bbox_inches='tight')
    
def plt_scatter_ax(x,y,alpha,ylab,ax,s=15):

    ax.scatter(x, y, alpha=alpha, color='gray', s=s)

    #ax.set_xlabel(xlab, fontsize=20)
    #ax.set_ylabel(ylab, fontsize=20)
    ax.tick_params(
    axis='both',          # changes apply to the x-axis
    which='both',      # both major and minor ticks are affected
    bottom='off',      # ticks along the bottom edge are off
    left = 'off',
    right = 'off',
    top='off')        # ticks along the top edge are off
    ax.spines['right'].set_visible(False)
    ax.spines['top'].set_visible(False)
    plt.tight_layout()
    #plt.gca().set_aspect('equal', adjustable='box')
    #plt.savefig(fn, dpi=600, bbox_inches='tight')
    

def plt_scatter_ax2(x,y,alpha,ax,ylab=None,xlab=None,s=15):

    ax.scatter(x, y, alpha=alpha, color='red', s=s)

    if xlab: ax.set_xlabel(xlab, fontsize=20)
    if ylab: ax.set_ylabel(ylab, fontsize=20)
    ax.tick_params(
    axis='both',          # changes apply to the x-axis
    which='both',      # both major and minor ticks are affected
    bottom='off',      # ticks along the bottom edge are off
    left = 'off',
    right = 'off',
    top='off')        # ticks along the top edge are off
    ax.spines['right'].set_visible(False)
    ax.spines['top'].set_visible(False)
    plt.tight_layout()
    #plt.gca().set_aspect('equal', adjustable='box')
    #plt.savefig(fn, dpi=600, bbox_inches='tight')
     

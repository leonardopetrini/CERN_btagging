# Implementation of all functions used for b-tagging
# including training with sklearn and xgboost and plotting
# ROCs and efficiencies
# Leonardo Petrini, CERN Summer Student Program 2018

import ROOT as root
from root_pandas import read_root

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

import sklearn.ensemble as skensamble
from sklearn import metrics

from scipy import interpolate

def load_root(inputFileName, columns, chunksize):
    '''Directly load a tree from .root file, returns a pandas dataframe'''
    chunksize = int(chunksize)
    
    for df in read_root(inputFileName, 'bTag_AntiKt4EMTopoJets',  \
                    columns=columns, flatten=True, chunksize=chunksize):
        tree = df
        break
    
    tree = tree.drop('__array_index', 1)
    
    return tree

def compute_log_likelihood_ratios(tree):
    '''Compute llr from probabilities'''
    tree['jet_ip2d_llr_bl'] = np.log((tree['jet_ip2d_pb']+1e-10)/(tree['jet_ip2d_pu']+1e-10))
    tree['jet_ip2d_llr_bc'] = np.log((tree['jet_ip2d_pb']+1e-10)/(tree['jet_ip2d_pc']+1e-10))
    tree['jet_ip2d_llr_cl'] = np.log((tree['jet_ip2d_pc']+1e-10)/(tree['jet_ip2d_pu']+1e-10))
    tree['jet_ip3d_llr_bl'] = np.log((tree['jet_ip3d_pb']+1e-10)/(tree['jet_ip3d_pu']+1e-10))
    tree['jet_ip3d_llr_bc'] = np.log((tree['jet_ip3d_pb']+1e-10)/(tree['jet_ip3d_pc']+1e-10))
    tree['jet_ip3d_llr_cl'] = np.log((tree['jet_ip3d_pc']+1e-10)/(tree['jet_ip3d_pu']+1e-10))

    tree = tree.drop(['jet_ip2d_pb','jet_ip2d_pc','jet_ip2d_pu',
            'jet_ip3d_pb','jet_ip3d_pc','jet_ip3d_pu'], 1)
    
    return tree

def jets_preselection(tree):
    '''Applies MV2 preselection rules for jet_pt, eta, JVT.'''
    
    initial_size = tree.size
    tree = tree.loc[(tree['jet_pt'] > 20e3) & (np.abs(tree['jet_eta']) < 2.5)]
    tree = tree.loc[(tree['jet_pt'] > 60e3) | (np.abs(tree['jet_eta']) > 2.4) | (tree['jet_JVT'] > 0.59)]
    tree = tree.loc[tree['jet_LabDr_HadF'] != 15]
    
    print('Final size after preselection = ',  tree.size / float(initial_size) * 100, '%')
    return tree

def train_test_splitting(tree, rate=.8):
    '''Test train splitting, rate = #train/#test'''
    tree['is_train'] = np.random.uniform(0, 1, len(tree)) <= rate
    train, test = tree[tree['is_train']==True], tree[tree['is_train']==False]
    
    return train, test

def select_features(tree, to_remove=[]):
    '''Takes list of features to remove and returns the features list'''
    features = tree.columns.tolist()
    features.remove('jet_LabDr_HadF')
    features.remove('jet_JVT')
    features.remove('jet_mv2c10')
    if 'jet_bH_pt' in features:
        features.remove('jet_bH_pt')
    if 'is_train' in features:
        features.remove('is_train')
    if 'weights' in features:
        features.remove('weights')
    for f in to_remove:
        features.remove(f)
        
    return features

def plot_features_importances(features, clf):
    feature_importance = clf.feature_importances_
    plt.figure("clf.feature_importances_", figsize=(10,4))
    plt.bar(range(len(feature_importance)), feature_importance*100, align='center')
    plt.xticks(range(len(feature_importance)), features, rotation='vertical')
    plt.title('Feature importance')
    plt.ylabel('Importance (%)')
    plt.xlabel('Features')
    plt.grid()
    
def compute_roc(labels, predicted_probabilities, rejection_jets='c', cut_pt=np.asarray([0,0])):
    '''Takes vector of labels, predicted probabilites and rejection jets: either c or l and returns fpr and 
    tpr to build roc curve. 
    In addition a cut in pt can be given as input.'''

    if cut_pt.any() == True:
        labels = labels[cut_pt]
        predicted_probabilities = predicted_probabilities[cut_pt]
    if rejection_jets == 'c':
        cut = (labels == 5) + (labels == 4)
    elif rejection_jets == 'l':
        cut = (labels == 5) + (labels == 0)
    else:
        raise ValueError('rejection jets can only be c or l!')
        
    if len(predicted_probabilities.shape) > 1:
        fpr, tpr, _ = metrics.roc_curve(labels[cut], predicted_probabilities[cut].T[2], pos_label=5)
    else:
        fpr, tpr, _ = metrics.roc_curve(labels[cut], predicted_probabilities[cut], pos_label=5)
    
    return fpr, tpr

def find_index(vector_of_values, bins):
    '''Assign each value in vector to a certain bin for reweighting.
    Returns bin indices.'''
    
    idxs = []
    for value in vector_of_values:
        idxs.append(np.argmin(np.abs(bins[:-1] - value)))
    return idxs
    
def sample_reweighting(tree):
    '''Reweights samples depending on eta and pt'''
    bins_pt = np.linspace(0,2e6,50)
    bins_eta = np.linspace(0,2.5,20)
    
    hist_b = np.histogram2d(tree['jet_pt'][tree['jet_LabDr_HadF']==5], np.abs(tree['jet_eta'][tree['jet_LabDr_HadF']==5]), bins=[bins_pt,bins_eta])[0]
    hist_c = np.histogram2d(tree['jet_pt'][tree['jet_LabDr_HadF']==4], np.abs(tree['jet_eta'][tree['jet_LabDr_HadF']==4]), bins=[bins_pt,bins_eta])[0]
    hist_l = np.histogram2d(tree['jet_pt'][tree['jet_LabDr_HadF']==0], np.abs(tree['jet_eta'][tree['jet_LabDr_HadF']==0]), bins=[bins_pt,bins_eta])[0]
    
    lc_rate = 1.0 * hist_c / hist_l
    bc_rate = 1.0 * hist_c / hist_b
    
    lc_rate[lc_rate == np.inf] = 0
    bc_rate[bc_rate == np.inf] = 0
    
    tree['weights'] = (tree['jet_LabDr_HadF'].values==5) * bc_rate[find_index(tree['jet_pt'], bins_pt),find_index(tree['jet_eta'].abs(), bins_eta)] + \
                  (tree['jet_LabDr_HadF'].values==0) * lc_rate[find_index(tree['jet_pt'], bins_pt),find_index(tree['jet_eta'].abs(), bins_eta)] + \
                  (tree['jet_LabDr_HadF'].values==4) * 1 
            
    tree['weights'].replace(np.nan, 0, inplace=True)
    
    return tree['weights']

def rejection_pt(test, probs1, probs2, num_cuts=15, b_eff=.77):
    '''Computes c and l rejection rate in function of pt for flat b_eff
    Takes all tree as input'''
    pt = test['jet_pt'].values
    cuts_pt = np.linspace(min(pt), 1.4e6, num_cuts)

    light_rejection_rf = []
    light_rejection_xgb = []

    for i, p in enumerate(cuts_pt):
        if i < num_cuts-1:
            cut_pt = (pt > cuts_pt[i]) * (pt < cuts_pt[i + 1])
            fpr, tpr = compute_roc(test['jet_LabDr_HadF'].values, probs1, rejection_jets='l', cut_pt=cut_pt)
            l_rej = interpolate.spline(tpr, 1/fpr, b_eff, order=1)
            light_rejection_rf.append(l_rej)

            fpr, tpr = compute_roc(test['jet_LabDr_HadF'].values, probs2, rejection_jets='l', cut_pt=cut_pt)
            l_rej = interpolate.spline(tpr, 1/fpr, b_eff, order=1)
            light_rejection_xgb.append(l_rej)

    c_rejection_rf = []
    c_rejection_xgb = []

    for i, p in enumerate(cuts_pt):
        if i < num_cuts-1:
            cut_pt = (pt > cuts_pt[i]) * (pt < cuts_pt[i + 1])
            fpr, tpr = compute_roc(test['jet_LabDr_HadF'].values, probs1, rejection_jets='c', cut_pt=cut_pt) 
            c_rej = interpolate.spline(tpr, 1/fpr, b_eff, order=1)
            c_rejection_rf.append(c_rej)
            fpr, tpr = compute_roc(test['jet_LabDr_HadF'].values, probs2, rejection_jets='c', cut_pt=cut_pt) 
            c_rej = interpolate.spline(tpr, 1/fpr, b_eff, order=1)
            c_rejection_xgb.append(c_rej)

    plt.figure(1, figsize=(8,5))
    plt.semilogy(cuts_pt[:-1], light_rejection_rf, '_', ms=15, lw=0, mew=3, label='light rej mv2', c='r')
    plt.semilogy(cuts_pt[:-1], light_rejection_xgb, '_', ms=15, lw=0, mew=3, label='light rej rf', c='b')
    plt.xlabel("$p_T$")
    plt.ylabel("Light-jet rejection for $\epsilon_b={}\%$".format(b_eff*100))
    plt.grid()
    plt.legend()

    plt.figure(2, figsize=(8,5))
    plt.semilogy(cuts_pt[:-1], c_rejection_rf, '_', ms=15, lw=0, mew=3, label='c rej mv2', c='r')
    plt.semilogy(cuts_pt[:-1], c_rejection_xgb, '_', ms=15, lw=0, mew=3, label='c rej rf', c='b')
    plt.grid()
    plt.xlabel("$p_T$")
    plt.ylabel("c-jet rejection for $\epsilon_b={}\%$".format(b_eff*100))
    plt.legend()
    
    return 0



def plot_roc_comparison(tpr1, fpr1, tpr2, fpr2):
    '''Plots rocs for comparison'''
    
    tpr = []
    
    fig=plt.figure(figsize=(10,7))

    gs=GridSpec(5,1)

    ax1=fig.add_subplot(gs[0:4,0])
    ax2=fig.add_subplot(gs[4,0])

    ax1.set_ylabel("light / c rejection")
    ax1.semilogy(tpr[0,1], 1/fpr[0,1], label='light w/ $p_t$')
    ax1.semilogy(tpr[0,0], 1/fpr[0,0], label='c w/ $p_t$')

    ax1.semilogy(tpr[1,1], 1/fpr[1,1], label='light w/o $p_t$')
    ax1.semilogy(tpr[1,0], 1/fpr[1,0], label='c w/o $p_t$')

    plt.setp(ax1.get_xticklabels(), visible=False)
    ax1.set_xlim([0.55, 1])
    ax1.set_ylim([1, 1e3])

    ax1.grid()
    ax1.legend()

    rate_light = 1 /fpr[0,1][1:] / interpolate.spline(tpr[1,1][1:], 1/fpr[1,1][1:], tpr[0,1][1:], order=1)
    ax2.plot(tpr[0,1][1:], 1/rate_light, c='r', lw=.8,  label='l-jets')

    rate_c = 1 /fpr[0,0][1:] / interpolate.spline(tpr[1,0][1:], 1/fpr[1,0][1:], tpr[0,0][1:], order=1)
    ax2.plot(tpr[0,0][1:], 1/rate_c, c='b', lw=.8,  label='c-jets')

    ax2.axhline(y=1, color='black', linestyle='-.', lw=.5)
    ax2.grid()
    ax2.set_xlabel("b-efficiency")
    ax2.set_ylabel("rate")
    ax2.set_xlim([0.55, 1])
    ax2.set_ylim([0.7, 1.5])
    ax2.legend(fontsize = 'x-small')

    #plt.savefig('figures/efficiency_rejection_rate.eps', format='eps')
    plt.show()
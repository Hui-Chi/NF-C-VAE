import numpy as np
import matplotlib.pyplot as plt
import glob
from matplotlib.cm import get_cmap
from cycler import cycler
import sklearn
from sklearn.utils import shuffle
#import energyflow as ef
#from energyflow.archs import DNN
from energyflow.utils import data_split, remap_pids, to_categorical
from sklearn.metrics import roc_auc_score, roc_curve
import math
import tqdm
import tensorflow as tf
import time


def define_SB_SR_SB(y_thing,sb_low, sb_high, sr_low, sr_high):
    y_bg_binary = np.vectorize(binary_side_band)(y_thing, sb_low, sb_high, sr_low, sr_high)
    print(np.unique(y_bg_binary,return_counts = True))
    return y_bg_binary

def binary_side_band(y_thing, sb_low, sb_high, sr_low, sr_high):
    if y_thing >= sr_low and y_thing < sr_high:
        return 1
    elif y_thing >= sb_low and y_thing < sb_high:
        return 0
    else:
        return -1

def defineXY(y_bg_binary,X_bg):
    side_band_indicator = (y_bg_binary == 0)
    # This is the background data in the SB
    X_sideband = X_bg[side_band_indicator]
    y_sideband = y_bg_binary[side_band_indicator]
    
    within_bounds_indicator = (y_bg_binary == 1)
    # This is the background data in the SR
    X_selected = X_bg[within_bounds_indicator]
    y_selected = y_bg_binary[within_bounds_indicator]
    return X_sideband, y_sideband, X_selected, y_selected

def prep_and_shufflesplit_data(X_sideband, X_selected, X_sig, anomaly_ratio, size_each = 76000, shuffle_seed = 62, val = 0.2, test_size_each = 5000): 

    #how much bg and signal data to take?
    anom_size = round(anomaly_ratio * size_each) 
    bgsig_size = size_each - anom_size

    # make sure we have enough data.
    assert (size_each <= X_sideband.shape[0]) 
    assert (anom_size + test_size_each <= X_sig.shape[0]) # need test_size_each for S and SR B only
    assert (bgsig_size + test_size_each <= X_selected.shape[0]) # need test_size_each for S and SR B only
    
    # select N=size_each from bkg in sidebands
    this_X_sb = X_sideband[:size_each] 
    this_y_sb = np.zeros(size_each) # 0 for bg in SB
    
    # select N=bgsig_size from bkg in signal region
    this_X_bgsig = X_selected[:bgsig_size]
    this_y_bgsig = np.ones(bgsig_size) # 1 for bg in SR
    
    # select N=anom_size of signal
    this_X_sig = X_sig[:anom_size]
    this_y_sig = np.ones(anom_size) # 1 for signal in SR
    
    """
    Shuffle + Train-Val-Test Split (not test set)
    """
    # Combine all 3 data sets
    this_X = np.concatenate([this_X_sb, this_X_bgsig, this_X_sig])
    this_y = np.concatenate([this_y_sb, this_y_bgsig, this_y_sig])
    
    # Shuffle before we split
    this_X, this_y = shuffle(this_X, this_y, random_state = shuffle_seed)
    
    
    (this_X_tr, this_X_v, _,
     this_y_tr, this_y_v, _) = data_split(this_X, this_y, val=val, test=0)
        
    print('-> Preparing SB and SR events:')
    print('Size of SB:',this_X_sb.shape)
    print('Size of bkg in SR:',this_X_bgsig.shape)
    print('Size of sig:',this_X_sig.shape)
    print('S/B: %0.2f'%np.divide(this_X_sig.shape[0],this_X_bgsig.shape[0],where=this_X_bgsig.shape[0]!=0))
    print('\n')    
    
    """
    Get the test set
    """
    
    # select the data
    this_X_test_P = X_sig[anom_size:anom_size+test_size_each] #take a slice of signal with size test_size_each for testing
    this_X_test_N = X_selected[bgsig_size:bgsig_size+test_size_each] #take a slide of bkg in SR with test_size_each for testing
    
    this_y_test_P = np.ones(test_size_each) # 1 for signal events
    this_y_test_N = np.zeros(test_size_each) # 0 for bkg events (in the SR)
        
    # Shuffle the combination    
    this_X_te = np.concatenate([this_X_test_P, this_X_test_N])
    this_y_te = np.concatenate([this_y_test_P, this_y_test_N])
    
    this_X_te, this_y_te = shuffle(this_X_te, this_y_te, random_state = shuffle_seed)
#     print('Size of test set:')
#     print(this_X_te.shape)
#     print('Test set distribution:')
#     print(np.unique(this_y_te,return_counts = True))
    
    
    X_train, X_val, X_test, y_train, y_val, y_test \
    = this_X_tr, this_X_v, this_X_te, this_y_tr, this_y_v, this_y_te
    
    """
    Data processing
    """
    #from sklearn import preprocessing
    #X_train = preprocessing.scale(X_train)
    #X_val = preprocessing.scale(X_val)
    #X_test = preprocessing.scale(X_test)
    
    from sklearn.preprocessing import StandardScaler
    scaler = StandardScaler()
    scaler.fit(X_train)
    #transform training dataset
    X_train = scaler.transform(X_train)
    # transform test dataset
    X_test = scaler.transform(X_test)
    # transform val dataset
    X_val = scaler.transform(X_val)
    
    # change Y to categorical Matrix
    Y_train = to_categorical(y_train, num_classes=2)
    Y_val = to_categorical(y_val, num_classes=2)
    Y_test = to_categorical(y_test, num_classes=2)
    
    print('-> Final setup:')
    print('Training set size, distribution:', X_train.shape)
    print(np.unique(y_train, return_counts = True))
    print('Validations set size, distribution:', X_val.shape)
    print(np.unique(y_val, return_counts = True))
    print('Test set size, distribution:', X_test.shape)
    print(np.unique(y_test, return_counts = True))
    
    return X_train, X_val, X_test, Y_train, Y_val, Y_test


def prep_benchmark_data(X_selected, X_sig, anom_size, shuffle_seed = 62): 

    #how much bg and signal data to take?
    #anom_size = round(anomaly_ratio * size_each)
    bg_size = round(X_selected.shape[0]/2)
    bgsig_size = bg_size - anom_size

    # make sure we have enough data #REVISIT
    assert (bg_size + bgsig_size <= X_selected.shape[0])
    assert (anom_size <= X_sig.shape[0])
    
    # select N=bg_size from bkg in SR
    this_X_bg = X_selected[:bg_size] 
    this_y_bg = np.zeros(bg_size) # 0 for bg in SR
    
    # select N=bg_size from bkg in SR (other half)
    this_X_bgsig = X_selected[bg_size:bg_size+bgsig_size] # never uses the last elements in the S+B mode
    this_y_bgsig = np.ones(bgsig_size) # 1 for bg in SR
    
    # select N=anom_size of signal
    # make this random, ie, always take # events randomly
    np.random.seed(seed=20)
    indices = np.random.choice(X_sig.shape[0], anom_size, replace=False)  
    this_X_sig = X_sig[indices]
    #this_X_sig = X_sig[:anom_size]
    this_y_sig = np.ones(anom_size) # 1 for signal in SR
    
    """
    Shuffle + Train-Val-Test Split (not test set)
    """
    # Combine all 3 data sets
    this_X = np.concatenate([this_X_bg, this_X_bgsig, this_X_sig])
    this_y = np.concatenate([this_y_bg, this_y_bgsig, this_y_sig])
    
    # Shuffle before we split
    this_X, this_y = shuffle(this_X, this_y, random_state = shuffle_seed)
    
    
    (this_X_tr, this_X_v, this_X_te,
     this_y_tr, this_y_v, this_y_te) = data_split(this_X, this_y, val=0.2, test=0.2)
        
    print('-> Preparing SB and SR events:')
    print('Size of B in SR:',this_X_bg.shape)
    print('Size of Bprime in SR:',this_X_bgsig.shape)
    print('Size of S+Bprime in SR:',this_X_bgsig.shape[0]+this_X_sig.shape[0])
    print('\n')    
        
    X_train, X_val, X_test, y_train, y_val, y_test \
    = this_X_tr, this_X_v, this_X_te, this_y_tr, this_y_v, this_y_te
    
    """
    Data processing
    """
    #from sklearn import preprocessing
    #X_train = preprocessing.scale(X_train)
    #X_val = preprocessing.scale(X_val)
    #X_test = preprocessing.scale(X_test)
    
    from sklearn.preprocessing import StandardScaler
    scaler = StandardScaler()
    scaler.fit(X_train)
    #transform training dataset
    X_train = scaler.transform(X_train)
    # transform test dataset
    X_test = scaler.transform(X_test)
    # transform val dataset
    X_val = scaler.transform(X_val)

    # change Y to categorical Matrix
    Y_train = to_categorical(y_train, num_classes=2)
    Y_val = to_categorical(y_val, num_classes=2)
    Y_test = to_categorical(y_test, num_classes=2)
    
    print('-> Final setup:')
    print('Training set size, distribution:', X_train.shape)
    print(np.unique(y_train, return_counts = True))
    print('Validations set size, distribution:', X_val.shape)
    print(np.unique(y_val, return_counts = True))
    print('Test set size, distribution:', X_test.shape)
    print(np.unique(y_test, return_counts = True))
    
    return X_train, X_val, X_test, Y_train, Y_val, Y_test


def pretty_plots(name):
    cmap = get_cmap(name) 
    colors = cmap.colors  
    default_cycler = (cycler(color=colors))
    plt.rc('lines', linewidth=3)
    plt.rc('axes', prop_cycle=default_cycler)
    #plt.rc.update({'font.size': 10})
    
    
def moving_average(x, w):
    return np.convolve(x, np.ones(w), 'valid') / w


def trainingNN(model, num_epoch, my_patience, batch_size, X_train, X_val, X_test, Y_train, Y_val, Y_test):
    history = tf.keras.callbacks.History()
    patience = my_patience
    earlystop = tf.keras.callbacks.EarlyStopping(monitor='val_loss', min_delta=0.001, patience=patience, verbose=1, mode='auto')

    #model.fit(X_train, Y_train[:,1], # dim 1
    model.fit(X_train, Y_train, 
    epochs=num_epoch,
    batch_size=batch_size,
    #validation_data=(X_val, Y_val[:,1]), # dim 1
    validation_data=(X_val, Y_val), 
    verbose=0,callbacks=[history,earlystop])

    Y_predict = model.predict(X_test)
    auc = roc_auc_score(Y_test[:,1], Y_predict[:,1])
    #auc = roc_auc_score(Y_test[:,1], Y_predict[:]) # dim 1
    roc = roc_curve(Y_test[:,1], Y_predict[:,1])
    #roc = roc_curve(Y_test[:,1], Y_predict[:]) # dim 1
    return auc, roc, history.history['loss'], history.history['acc'], history.history['val_loss'], history.history['val_acc']

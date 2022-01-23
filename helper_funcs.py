# -*- coding: utf-8 -*-
"""
Created on Tue Jan 18 3:29:24 2022

@author: bshreyas
"""
import numpy as np
import matplotlib.pyplot as plt
import os 
import argparse as ap
from modules.preprocessor import Preprocessor

from scipy.signal import butter, lfilter


def butter_bandpass(lowcut, highcut, fs, order=6):
    nyq = 0.5 * fs
    low = lowcut / nyq
    high = highcut / nyq
    b, a = butter(order, [low, high], btype='band')
    return b, a

def butter_bandpass_filter(data, lowcut, highcut, fs, order=6):
    b, a = butter_bandpass(lowcut, highcut, fs, order=order)
    y = lfilter(b, a, data)
    return y

def plotTrainingCurve(loss,val_loss, model_name):
    #Plot training curve
    plt.plot(loss)
    plt.plot(val_loss)
    plt.title('model loss')
    plt.ylabel('Mean Squared Error')
    plt.xlabel('epoch')
    plt.legend(['train', 'val'], loc='upper left')
    plt.savefig('Train_cruve_{}.jpg'.format(model_name))

def loadHistory(npfile):
    return np.load(npfile,allow_pickle=True)

def flatten(t):
    return [item for sublist in t for item in sublist]

## Funtion to obtain first deravative of the signal ##
## Return an array with deravative signal ##
def getIntegral(sig):
        derivative = []
        count = 0
        for i in range (len(sig)):
            if i == 0 :
                x = sig[i].copy()
                count+=1
                derivative.append(sig[i])
                continue
            elif count == i:
                x_next = sig[i].copy()
                derivative.append(x_next + x)
                x = sig[i].copy()
                count+=1
        return np.array (derivative)
        
if __name__ == '__main__':
    
    parser = ap.ArgumentParser()
    parser.add_argument("-mp","--metrics_path", required = True , help = "Path to metrics file")
    args = vars(parser.parse_args())
     
    ## Get args
    m_path = args["metrics_path"]

    ## Load File
    
    files = os.listdir(m_path)
    loss =[]
    val_loss =[]
    for file in files:
        if not file.startswith('.'):
            history = loadHistory(os.path.join(m_path, file))
            history = dict(enumerate(history.flatten(), 1))[1]
            loss.append(history['loss'])
            val_loss.append(history['val_loss'])
    
    loss = flatten(loss)
    val_loss = flatten(val_loss)
    plotTrainingCurve(loss, val_loss, 'DeepPhys')
    
    lowcut = 0.2
    highcut = 2
    fs = 50
    p = Preprocessor()
    sig = p.loadData('../test.dat')
    sig = butter_bandpass_filter(sig, lowcut, highcut, fs, order=6)
    sig = getIntegral(sig)
    m = p.plotHR(sig, 50)
    label = p.loadData('../s01_trial22.dat')
    label = getIntegral(label)
    m = p.plotHR(label, 50)
 
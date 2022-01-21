# -*- coding: utf-8 -*-
"""
Created on Tue Jan 18 3:29:24 2022

@author: bshreyas
"""
import numpy as np
import matplotlib.pyplot as plt
import os 
import argparse as ap
def plotTrainingCurve(loss,val_loss, model_name):
    #Plot training curve
    plt.plot(loss)
    plt.plot(val_loss)
    plt.title('model loss')
    plt.ylabel('accuracy')
    plt.xlabel('epoch')
    plt.legend(['train', 'val'], loc='upper left')
    plt.savefig('Train_cruve_{}.jpg'.format(model_name))

def loadHistory(npfile):
    return np.load(npfile,allow_pickle=True)

def flatten(t):
    return [item for sublist in t for item in sublist]

def plotHR(self,signal, sampling_rate):
        working_data, measures = hp.process(signal, sampling_rate)
        hp.plotter(working_data, measures)
    
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
        history = loadHistory(os.path.join(m_path, file))
        history = dict(enumerate(history.flatten(), 1))[1]
        loss.append(history['loss'])
        val_loss.append(history['val_loss'])
    
    loss = flatten(loss)
    val_loss = flatten(val_loss)
    plotTrainingCurve(loss, val_loss, 'DeepPhys')
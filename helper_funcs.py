# -*- coding: utf-8 -*-
"""
Created on Tue Jan 18 3:29:24 2022

@author: bshreyas
"""
import numpy as np
import matplotlib.pyplot as plt

def plotTrainingCurve(history, model_name):
    #Plot training curve
    plt.plot(history['loss'])
    plt.plot(history['val_loss'])
    plt.title('model loss')
    plt.ylabel('accuracy')
    plt.xlabel('epoch')
    plt.legend(['train', 'val'], loc='upper left')
    plt.savefig('Train_cruve_{}.jpg'.format(model_name))

def loadHistory(npfile):
    return np.load(npfile,allow_pickle=True)
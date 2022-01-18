# -*- coding: utf-8 -*-
"""
Created on Fri Nov 19 10:19:24 2021

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
    return np.load(npfile)
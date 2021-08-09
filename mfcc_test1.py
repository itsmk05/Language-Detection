#pip install python_speech_features
#pip install scipy

import numpy as np
#import matplotlib.pyplot as plt
#import pandas as pd
from python_speech_features import mfcc
from python_speech_features import logfbank
import scipy.io.wavfile as wav
from sklearn import svm, metrics
from sklearn.externals import joblib

#Train data
train_data = np.zeros((96,997,26))
train_label = np.zeros((96))
for i in range (1,96): 
    (rate,sig) = wav.read("E:/Gauru/Project2k18/L1/%d.wav"%(i))
    #small changes are not considered 
    mfcc_feat = mfcc(sig,rate,nfft=1104)
    #humans cant percieve loudness in linear scale but in log
    train_data[i,:,:] = np.mean(logfbank(sig,rate),axis=0)
    train_label[i] = np.round(i)
##    print(train_data[:,:])
##    print(train_data.shape)
    print(i)
nsamples, nx, ny = train_data.shape
train_dataset = train_data.reshape((nsamples,nx*ny))


#Test data
test_data = np.zeros((1,997, 26))
test_label = np.zeros((1))
(rate,sig) = wav.read("E:/Gauru/Project2k18/check.wav")
mfcc_feat = mfcc(sig,rate)
test_data[0,:,:] = np.mean(logfbank(sig,rate),axis=0)
test_label[0] = np.round(1)
print(test_data[:,:])
print(test_data.shape)
print()
nsamples, nx, ny = test_data.shape
test_dataset = test_data.reshape((nsamples,nx*ny))



#Creating Model
model = svm.SVC(kernel='linear')
#Training data
model = model.fit(train_dataset,train_label)
#Testing Data
output = model.predict(test_dataset)
print()
print()
print("Tested")

#Performance
acc = metrics.accuracy_score (test_label,output)
conf_matrix = metrics.confusion_matrix (test_label,output)
report = metrics.classification_report (test_label,output)
print('Accuracy : ')
print(acc)
print()
print('Confusion Matrix : ')
print(conf_matrix)
print()
print('Report : ')
print(report)
print()

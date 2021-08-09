#pip install python_speech_features
#pip install scipy

from scipy.io import wavfile
from matplotlib import pyplot as plt
import numpy as np
import matplotlib.pyplot as plt
#import pandas as pd
from python_speech_features import mfcc
from python_speech_features import logfbank
#from pynfft.nfft import NFFT
import scipy.io.wavfile as wav
from sklearn import svm, metrics
from sklearn.externals import joblib


train_data = np.zeros((135,997,26))
train_label = np.zeros((135))
count=0
for j in range (1,10):
    for i in range (1,16): 
        (rate,sig) = wav.read("E:/Prasad-Pc/Gauru/Project2k18/L%d/%d.wav"%(j,i))
        #small changes are not considered 
        mfcc_feat = mfcc(sig,rate,nfft=1104)
        #humans cant percieve loudness in linear scale but in log
        train_data[count,:,:] = np.mean(logfbank(sig,rate),axis=0)
        train_label[count] = np.round(count)
        count = count+1
    ##    print(train_data[:,:])
    ##    print(train_data.shape)
        print(count)

nsamples, nx, ny = train_data.shape
train_dataset = train_data.reshape((nsamples,nx*ny))



test_data = np.zeros((1,997, 26))
test_label = np.zeros((1))
(rate,sig) = wav.read("E:/Prasad-Pc/Gauru/Project2k18/check.wav")
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

#op = np.round(output[0])

### Load the data and calculate the time of each sample
##path1 = 'check.wav'
##samplerate, data = wavfile.read(path1)
##times = np.arange(len(data))/float(samplerate)
##
### Make the plot
### You can tweak the figsize (width, height) in inches
##plt.figure(1)
##plt.subplot(2,1,1)
##plt.fill_between(times, data[:,0], data[:,1], color='k') 
##plt.xlim(times[0], times[-1])
##plt.xlabel('time (s)')
##plt.ylabel('amplitude')
### You can set the format by changing the extension
### like .pdf, .svg, .eps
##plt.savefig('plot.png', dpi=100)
###plt.show()
##
##
### Load the data and calculate the time of each sample
##path2 = ("E:/Prasad-Pc/Gauru/Project2k18/L%d/2.wav"%(int(output[0])))
##samplerate1, data1 = wavfile.read(path2)
##times1 = np.arange(len(data1))/float(samplerate1)
##
### Make the plot
### You can tweak the figsize (width, height) in inches
##plt.subplot(2,1,2)
##plt.fill_between(times1, data1[:,0], data1[:,1], color='k') 
##plt.xlim(times1[0], times1[-1])
##plt.xlabel('time (s)')
##plt.ylabel('amplitude')
### You can set the format by changing the extension
### like .pdf, .svg, .eps
##plt.savefig('plot.png', dpi=100)
##plt.show()
##

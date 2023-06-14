#!/usr/bin/env python
# coding: utf-8

# In[35]:


import sys
import wave
import scipy.io as sio
import scipy.io.wavfile

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

import os
import librosa

import scipy

from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
from sklearn.preprocessing import MinMaxScaler

# 모델 학습 시 사용
from sklearn.svm import SVC
from sklearn.utils import shuffle

# 모델 저장 시 이용
import pickle
import joblib


# In[36]:


if __name__ == '__train__':
    print(sys.argv[0]) # train.py
    print(sys.argv[1])


# In[9]:


#file_path='fmcc_train.ctl'
file_path = sys.argv[1]


# In[10]:


# train_ctl 파일 읽어온 뒤 리스트에 담기
def raw_to_wav(file_path):
    with open(file_path) as f:
        lines = f.readlines()

    # 파일명을 담은 리스트를 기준으로 raw -> wav파일로 변환
    train_files_names = [i.strip("\n") for i in lines] # \n값 제거
    for i in train_files_names:
        with open("raw16k/train/{0}.raw".format(i), "rb") as inp_f:
            data = inp_f.read()
            with wave.open("raw16k/train/{0}.wav".format(i), "wb") as out_f:
                out_f.setnchannels(1) # 바이트 순서 Little Endian, 채널: 1 모노
                out_f.setsampwidth(2) # number of bytes (16bit = 2byte)
                out_f.setframerate(16000)
                out_f.writeframesraw(data)


# In[11]:


raw_to_wav(file_path)


# In[12]:


def train_dataset(file_path):
    dataset = []
    with open(file_path) as f:
        lines = f.readlines()
    train_files_names = [i.strip("\n") for i in lines] # \n값 제거
    
    for train_file in train_files_names:
        file_name='raw16k/train/' + train_file + ".wav"
        audio, sr = librosa.load(file_name, sr=16000)
        # 남/녀 별로 labeling
        # 0 : 남자 , 1: 여자
        if "M" in train_file[0]:
            dataset.append([file_name, audio, 0])
        elif "F" in train_file[0]:
            dataset.append([file_name, audio, 1])
    
    print("TrainDataset 생성 완료")
    return pd.DataFrame(dataset,columns=['fname', 'data','label'])


# In[15]:


# 음성의 길이 중 가장 긴 길이를 구합니다.

def get_max(data):

   max_data = -999
   for i in data:
       if len(i) > max_data:
           max_data = len(i)

   return max_data


def zero_pad(data, max_length):
   padded_data = []
   for d in data:
       if len(d) < max_length:
           pad_width = max_length - len(d)
           padded_d = np.pad(d, (0, pad_width), mode='constant')
       else:
           padded_d = d[:max_length]
       padded_data.append(padded_d)
   return np.array(padded_data)


# In[16]:


train_wav = train_dataset(file_path)


# In[17]:


# 데이터 길이 중 가장 큰 데이터로 길이 설정
train_x = np.array(train_wav.data)

train_max = get_max(train_x)

max_data = np.max(train_max)


# In[18]:


train_x = zero_pad(train_x, max_data)


# In[19]:


#mfcc 특징 추출
def preprocess_dataset(data):
    mfccs = []
    for i in data:
        mfcc = librosa.feature.mfcc(y=i,sr=16000,n_mfcc=40,   # n_mfcc:return 될 mfcc의 개수를 정해주는 파라미터, 더 다양한 데이터 특징을 추출하려면 값을 증가시키면 됨. 일반적으로 40개 추출
                                                  n_fft=400,  # n_fft:frame의 length를 결정하는 파라미터 
                                                  hop_length=160) # hop_length의 길이만큼 옆으로 가면서 데이터를 읽음(10ms기본)
        mfccs.append(mfcc.flatten())
    return pd.DataFrame(mfccs)


# In[20]:


train_mfccs = preprocess_dataset(train_x)


# In[21]:


# 데이터셋 재설정하기
train_set = pd.DataFrame()
train_set['fname'] = train_wav['fname']

train_set = pd.concat([train_set,train_mfccs],axis=1)
train_set['label'] = train_wav['label']


# In[22]:


# 데이터셋 (train, test) 셔플
shuffle_train=shuffle(train_set, random_state = 20)
shuffle_train


# In[23]:


X = shuffle_train.drop(['label', 'fname'], axis=1)
feature_names = list(X.columns)  # 특징 번호 리스트

X = X.values  # 특징벡터 값 전체

y=shuffle_train.label.values


# In[24]:


# scaling(정규화)
scaler = MinMaxScaler()
X_scaled = scaler.fit_transform(X)

X_scaled.shape


# In[25]:


# 기존 트레인셋을 분할하여 정확도 테스트
# Fit an SVM model
X_train, X_val, y_train, y_val = train_test_split(X_scaled, y, test_size = 0.2, random_state = 30, shuffle = True)

clf = SVC(kernel = 'rbf', C = 10, gamma = 0.01, probability=True)
clf.fit(X_train, y_train)

print(accuracy_score(clf.predict(X_val), y_val))


# In[28]:


# 실제 트레인 10000개 데이터 전체를 학습
clf = SVC(kernel = 'rbf', C = 10, gamma = 0.01, probability=True)
clf.fit(X_scaled, y)


# In[30]:


#학습 모델 저장
saved_model = joblib.dump(clf, 'svm.pkl')
print("svm.pkl 모델 저장 완료")


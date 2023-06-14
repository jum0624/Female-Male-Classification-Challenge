#!/usr/bin/env python
# coding: utf-8

# In[24]:


import sys
import wave
import scipy.io as sio
import scipy.io.wavfile


import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

import os

#mfcc 특징 추출 시 사용
import librosa

import scipy

from sklearn.metrics import accuracy_score
from sklearn.preprocessing import MinMaxScaler

# 모델 학습 시 사용
from sklearn.svm import SVC
from sklearn.utils import shuffle

# 모델 저장 시 이용
import pickle
import joblib


# In[25]:


if __name__ == '__test__':
    print(sys.argv[0]) # test.py
    print(sys.argv[1])
    file_path = sys.argv[1]


# In[26]:


train_file_path = 'fmcc_train.ctl'
file_path = sys.argv[1]


# In[27]:


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
                
def test_raw_to_wav(file_path):
    with open(file_path) as f:
        lines = f.readlines()

    # 파일명을 담은 리스트를 기준으로 raw -> wav파일로 변환
    test_files_names = [i.strip("\n") for i in lines] # \n값 제거
    for i in test_files_names:
        with open("raw16k/test/{0}.raw".format(i), "rb") as inp_f:
            data = inp_f.read()
            with wave.open("raw16k/test/{0}.wav".format(i), "wb") as out_f:
                out_f.setnchannels(1) # 바이트 순서 Little Endian, 채널: 1 모노
                out_f.setsampwidth(2) # number of bytes (16bit = 2byte)
                out_f.setframerate(16000)
                out_f.writeframesraw(data)


# In[28]:


raw_to_wav(train_file_path)
test_raw_to_wav(file_path)


# In[30]:


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

def test_dataset(test_file_path):
    dataset = []
    with open(file_path) as f:
        lines = f.readlines()
    test_files_names = [i.strip("\n") for i in lines] # \n값 제거
    for test_file in test_files_names:
        test_file = test_file.split(" ")
        fname = test_file[0]
        audio, sr = librosa.load('raw16k/test/' + fname + ".wav", sr=16000)
        dataset.append(['raw16k/test/'+fname+".raw", audio])
    
    print("TestDataset 생성 완료")
    return pd.DataFrame(dataset, columns=['fname','data'])


# In[31]:


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


# In[32]:


train_wav = train_dataset(train_file_path)
test_wav = test_dataset(file_path)


# In[33]:


train_x = np.array(train_wav.data)
test_x = np.array(test_wav.data)

train_max = get_max(train_x)
test_max = get_max(test_x)

max_data = np.max(train_max)
print('가장 긴 길이 :', max_data)


# In[34]:


train_x = zero_pad(train_x, max_data)
test_x = zero_pad(test_x, max_data)


# In[35]:


def preprocess_dataset(data):
    mfccs = []
    for i in data:
        mfcc = librosa.feature.mfcc(y=i,sr=16000,n_mfcc=40,   # n_mfcc:return 될 mfcc의 개수를 정해주는 파라미터, 더 다양한 데이터 특징을 추출하려면 값을 증가시키면 됨. 일반적으로 40개 추출
                                                  n_fft=400,  # n_fft:frame의 length를 결정하는 파라미터 
                                                  hop_length=160) # hop_length의 길이만큼 옆으로 가면서 데이터를 읽음(10ms기본)
        mfccs.append(mfcc.flatten())
    return pd.DataFrame(mfccs)


# In[36]:


train_mfccs = preprocess_dataset(train_x)

test_mfccs = preprocess_dataset(test_x)

train_mfccs.head()


# In[37]:


# 데이터셋 재설정하기
train_set = pd.DataFrame()
train_set['fname'] = train_wav['fname']
test_set = pd.DataFrame()
test_set['fname'] = test_wav['fname']

train_set = pd.concat([train_set,train_mfccs],axis=1)
train_set['label'] = train_wav['label']
test_set = pd.concat([test_set,test_mfccs],axis=1)


# In[38]:


# 데이터셋 (train, test) 셔플
shuffle_train=shuffle(train_set, random_state = 20)
shuffle_test=shuffle(test_set, random_state = 20)
shuffle_test_fname = shuffle_test.fname.values  # 이후 결과 txt 파일을 만들기 위해 fname 저장
shuffle_train


# In[39]:


X = shuffle_train.drop(['label', 'fname'], axis=1)
feature_names = list(X.columns)  # 특징 번호 리스트

X = X.values  # 특징벡터 값 전체

y=shuffle_train.label.values


# In[40]:


# test 데이터 label, fname 컬럼 삭제 (특징 데이터만 호출하기 위함)
X_test = shuffle_test.drop(['fname'], axis=1)
X_test = X_test.values


# In[41]:


# scaling(정규화)
scaler = MinMaxScaler()
X_scaled = scaler.fit_transform(X)
X_test_scaled = scaler.transform(X_test)

X_scaled.shape


# In[42]:


# 예측결과 txt 파일로 변환 및 저장
def to_txt_test(restored_df):
    y_pred = restored_df.label.values
    str_labels = pd.Series(y_pred).map({0: 'male', 1: 'feml'})
    
    test_predict_df = pd.DataFrame()
    test_predict_df['fname'] = restored_df.fname.values
    test_predict_df['y_pred'] = str_labels
    
    test_predict_df.to_csv('강력한컴공_test_results.txt', sep = " ", index=False, header=False, lineterminator='\n')
    print('강력한컴공_test_results.txt 생성완료')


# In[43]:


model = joblib.load('svm.pkl')
y_pred = model.predict(X_test_scaled) # 예측값 호출
shuffle_test['label'] = y_pred
restored_df = shuffle_test.sort_index()  # 셔플된 데이터 다시 인덱스를 기준으로 오름차순 정렬
predict_df = to_txt_test(restored_df) # 예측 값 호출 후 txt파일 저장


# In[ ]:





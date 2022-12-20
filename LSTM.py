# 드라이브에 마운틴
from google.colab import drive
drive.mount('/content/gdrive')

import warnings
warnings.filterwarnings(action='ignore') 

# Load Data
import numpy as np
import pandas as pd
from collections import Counter

# 시각화
import seaborn as sns
import matplotlib.pyplot as plt

# 자연어처리 
from konlpy.tag import *
from gensim.models import Word2Vec, FastText
from tensorflow.keras.preprocessing.text import Tokenizer
from tensorflow.keras.preprocessing.sequence import pad_sequences
from sklearn.feature_extraction.text import TfidfTransformer, TfidfVectorizer

# LSTM 
from sklearn.model_selection import train_test_split
from tensorflow.keras.layers import Embedding, Dense, LSTM, GRU, Bidirectional
from tensorflow.keras.models import Sequential
from tensorflow.keras.models import load_model
from tensorflow.keras.callbacks import EarlyStopping, ModelCheckpoint
from tensorflow.keras.utils import to_categorical

csv_lst = [] # 필요한 데이터 제목들
df_lst = []
for lst in csv_lst:
  df_lst.append(pd.read_csv(f'/content/gdrive/MyDrive/산공설/유튜브 크롤링/youtube_data3/{lst}.csv', encoding='UTF-8'))

df_lst
data = pd.concat(df_lst, ignore_index=True)


# 조회수 1K 단위로 구간 나누기
data_cat = data.copy()           
print(data_cat.info())
view_lst = []
for view_data in data_cat['조회수']:
  vd1 = view_data[0:-1].split(',')[0]
  vd2 = view_data[0:-1].split(',')[1]
  vd_dm = int(vd1+vd2)
  if len(view_data) > 8:
    vd3 = view_data[0:-1].split(',')[2]
    vd_om = int(vd1+vd2+vd3)
    view_lst.append(vd_om)
  else:
    view_lst.append(vd_dm)

data_cat.insert(4,'조회수_정수',view_lst)

data_cat['조회수_정수'] = (data_cat['조회수_정수'].floordiv(1000))    
# 1M 이후부터는 1,000,000 단위로 조회수 잘라야 할 것 같음 

data_cat['조회수_정수'] = data_cat['조회수_정수'].astype('int64') # 데이터 int 타입으로 바꿔주기

import math # 구간 나눠주기 -> multiclass classification

for index, 조회수_정수 in enumerate(data_cat['조회수_정수']):
  if 조회수_정수 >=10000:  
    data_cat['조회수_정수'][index] =  10000
  elif 조회수_정수 >=5000:  
    data_cat['조회수_정수'][index] =  5000
  elif 조회수_정수 >=1000:  
    data_cat['조회수_정수'][index] =  1000
  elif 조회수_정수 >=500:  
    data_cat['조회수_정수'][index] =  500
  elif 조회수_정수 >=100:  
    data_cat['조회수_정수'][index] = 100
  elif 조회수_정수 >=50: 
    data_cat['조회수_정수'][index] =  50
  elif 조회수_정수 >=10: 
    data_cat['조회수_정수'][index] = 10
  # elif view >=5: data_cat['views'][index] = 5
  # elif view >5: data_cat['views'][index] = math.floor(data_cat['views'][index]/5) * 5
  elif 조회수_정수 >0: 
    data_cat['조회수_정수'][index] = 1
   
  
#Processing
#Morphology Analysis
twitter = Okt()       # 트위터 형태소 분석기    
komoran = Komoran()   # Komoran 형태소 분석기
kkma = Kkma()         # kkma 형태소 분석

# 말뭉치 만들기
def make_corpus(text):
    corpus = []
    for s in text:
      corpus.append(s.split()) 
    return corpus
  
# 형태소 분석 결과 보여주기
def make_corpus_show_morph(text):
    corpus = []
    for s in text:
        corpus.append(['/'.join(p) for p in twitter.pos(s)]) 
    return corpus

# 형태소 분석 대호 tokenizing
def make_corpus_morph(text):
    corpus = []
    for s in text:
        corpus.append([p[0] for p in twitter.pos(s)])
    return corpus

corpus = make_corpus(data['제목'])
corpus = make_corpus_show_morph(data['제목'])
corpus = make_corpus_morph(data['제목'])
data_cat['tokenized'] = corpus

# Train-Test-Split
# 조회수 구간 라벨 값으로 대체
views = data_cat['조회수_정수'].map(cat_to_id)
data_cat['조회수_정수'] = views
data_cat.head()
# 구간별 샘플링
data_cat_0_random = data_cat[data_cat['조회수_정수']==0].sample(128,random_state=0).index.tolist()
data_cat_1_random = data_cat[data_cat['조회수_정수']==1].sample(128,random_state=0).index.tolist()
data_cat_2_random = data_cat[data_cat['조회수_정수']==2].sample(128,random_state=0).index.tolist()
data_cat_3_random = data_cat[data_cat['조회수_정수']==3].sample(128,random_state=0).index.tolist()
data_cat_4_random = data_cat[data_cat['조회수_정수']==4].sample(128,random_state=0).index.tolist()
data_cat_5_random = data_cat[data_cat['조회수_정수']==5].sample(128,random_state=0).index.tolist()
data_cat_6_random = data_cat[data_cat['조회수_정수']==6].sample(128,random_state=0).index.tolist()
# 랜덤 인덱싱
random_index = data_cat_0_random + data_cat_1_random + data_cat_2_random + data_cat_3_random + data_cat_4_random + data_cat_5_random + data_cat_6_random
y = data_cat['조회수_정수'][random_index] # 데이터 편향을 줄이기 위한 랜덤 샘플링
Sample_X = X[random_index]
X_train, X_test, y_train, y_test = train_test_split(Sample_X, y, test_size = 0.2, random_state = 42)

#Tokenize & Embed
tokenizer = Tokenizer()
tokenizer.fit_on_texts(X_train)

threshold = 2                         
total_cnt = len(tokenizer.word_index) # 단어의 수
rare_cnt = 0                          # 등장 빈도수가 threshold보다 작은 단어의 개수를 카운트
total_freq = 0                        # 훈련 데이터의 전체 단어 빈도수 총 합
rare_freq = 0                         # 등장 빈도수가 threshold보다 작은 단어의 등장 빈도수의 총 합

# 단어와 빈도수의 쌍(pair)을 key와 value로 받는다.
for key, value in tokenizer.word_counts.items():
    total_freq = total_freq + value

    # 단어의 등장 빈도수가 threshold보다 작으면
    if(value < threshold):
        rare_cnt = rare_cnt + 1
        rare_freq = rare_freq + value

print('단어 집합(vocabulary)의 크기 :',total_cnt)
print('등장 빈도가 %s번 이하인 희귀 단어의 수: %s'%(threshold - 1, rare_cnt))
print("단어 집합에서 희귀 단어의 비율:", (rare_cnt / total_cnt)*100)
print("전체 등장 빈도에서 희귀 단어 등장 빈도 비율:", (rare_freq / total_freq)*100)

# 전체 단어 개수 중 빈도수 1이하인 단어는 제거.
# 0번 패딩 토큰을 고려하여 + 1
vocab_size = total_cnt - rare_cnt + 1
print('단어 집합의 크기 :',vocab_size)

tokenizer = Tokenizer(vocab_size) 
tokenizer.fit_on_texts(X_train)

# keras tokenizer -> 텍스트 시퀀스 숫자 시퀀스로 변환
X_train = tokenizer.texts_to_sequences(X_train)
X_test = tokenizer.texts_to_sequences(X_test)

# Padding
def below_threshold_len(max_len, nested_list):
  cnt = 0
  for s in nested_list:
    if(len(s) <= max_len):
        cnt = cnt + 1
  print('전체 샘플 중 길이가 %s 이하인 샘플의 비율: %s'%(max_len, (cnt / len(nested_list))*100))

max_len = 45
below_threshold_len(max_len, X_train)
# 모든 샘플의 길이 45으로 맞추기
X_train = pad_sequences(X_train, maxlen = max_len)
X_test = pad_sequences(X_test, maxlen = max_len)

# LSTM model
# Bi-Directional LSTM
model2 = Sequential([
    Embedding(vocab_size, 64),
    Bidirectional(LSTM(128, dropout = 0.2)),
    Dense(64, activation='relu'),
    Dense(64, activation='relu'),
    Dense(64, activation='relu'),
    Dense(7,activation='softmax')
])

print(model2.summary())
es = EarlyStopping(monitor='val_loss', mode='min', verbose=1, patience=5)
mc = ModelCheckpoint('Bi-LSTM_class6.h5', monitor='val_acc', mode='max', verbose=1, save_best_only=True)

model2.compile(optimizer='adam', loss='CategoricalCrossentropy', metrics=['acc'])
history = model2.fit(X_train, y_train, epochs=20, callbacks=[es, mc], batch_size=64, validation_data=(X_test, y_test))

# 모델 저장하기
from keras.models import load_model

model2.save('./gdrive/MyDrive/--')

# Classification Report
from sklearn.metrics import classification_report
print(classification_report(true, pred))

# Confusion Matrix
from sklearn.metrics import confusion_matrix

conf_mat=confusion_matrix(true, pred)  # confusion matrix 계산

# Class 별 Accuracy
class_accuracy=100*conf_mat.diagonal()/conf_mat.sum(1)  # 각 클래스 별 accuracy 담은 list
class_accuracy = pd.DataFrame(class_accuracy, columns = ['Accuracy']) # dataframe 화하기

class_accuracy.insert(0, 'View Category', list(cat_to_id.keys())[:20])
class_accuracy

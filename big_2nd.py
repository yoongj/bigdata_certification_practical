# 출력을 원하실 경우 print() 함수 활용
# 예시) print(df.head())

# getcwd(), chdir() 등 작업 폴더 설정 불필요
# 파일 경로 상 내부 드라이브 경로(C: 등) 접근 불가

import pandas as pd
import numpy as np

train = pd.read_csv("train.csv", encoding= 'UTF8')
test = pd.read_csv("X_test.csv", encoding= 'cp949')
# print('original_train shape', train.shape) # 3500
# print('original_test shape', test.shape)   # 2482
# print('\n\n')

pd.set_option('display.max_columns',None)
pd.set_option('display.max_rows',None)
pd.options.display.float_format = '{:.5f}'.format

# 사용자 코딩
############EDA####################
# print(train.info())
# print(train['환불금액'].head(50))
# print('총구매액',sorted(train['총구매액'])[:20])
# print('최대구매액',sorted(train['최대구매액'])[:20])

# print(train.sort_values(by= '총구매액')[:20])

############preprocessing##################
# 총구매액, 최대구매액 이상치(음수)처리 (제거)
# print('train 총구매 음수\n', train[train['총구매액']<=0],'\n\n')
# print('test 총구매 음수\n', test[test['총구매액']<=0],'\n\n')
# print('train 최대구매 음수\n', train[train['최대구매액']<=0],'\n\n')
# print('test 최대구매 음수\n', test[test['최대구매액']<=0],'\n\n')

train= train[(train['총구매액']>0) & (train['최대구매액']>0)]
test= test[(test['총구매액']>0) & (test['최대구매액']>0)]
# print('음수 제거 train shape', train.shape) # 3490 (-10)
# print('음수 제거 test shape', test.shape)   # 2473 (-9)
# print('\n\n')


# 총구매액 < 최대구매액 제거
# print('train 총구매액<최대구매액\n', train[train['총구매액']<train['최대구매액']],'\n\n')
# print('test 총구매액<최대구매액\n', test[test['총구매액']<test['최대구매액']],'\n\n')

train= train[train['총구매액']>=train['최대구매액']]
test= test[test['총구매액']>=test['최대구매액']]
# print('총<최대 제거 train shape', train.shape)  # 3430 (-60)
# print('총<최대 제거 test shape', test.shape)    # 2438 (-35)
# print('\n\n')


##########보류############
# # 총구매액 < 환불금액 제거
# print('train 총구매액<환불금액\n', train[train['총구매액']<train['환불금액']],'\n',len(train[train['총구매액']>train['환불금액']]),'\n\n')
# print('test 총구매액<환불금액\n', test[test['총구매액']<test['환불금액']],'\n',len(test[test['총구매액']>test['환불금액']]),'\n\n')
# train= train[train['총구매액']>train['환불금액']]
# test= test[test['총구매액']>test['환불금액']]
# print('총<환불 제거 train shape', train.shape)  # 1121 ????
# print('총<환불 제거 test shape', test.shape)    # 813 ????
# print('\n\n')
###########################


# 환불금액 int로 변환, 0이 있는지 확인 후, 결측치 0으로 변환
# print(train[train['환불금액']<=0])
# print(test[test['환불금액']<=0])
train['환불금액']= train['환불금액'].fillna(0).astype(int)
test['환불금액']= test['환불금액'].fillna(0).astype(int)
# print(train[train['환불금액']>=0])
# print(test[test['환불금액']>=0])
# print(train.head())
# print(test.info())

# 주구매상품, 주구매지점 라벨 인코딩
import sklearn.preprocessing
from sklearn.preprocessing import LabelEncoder, OrdinalEncoder, OneHotEncoder
# print(help(LabelEncoder))
# print(train['주구매상품'].nunique())
# print(train['주구매지점'].nunique())
# print(test['주구매상품'].nunique())
# print(test['주구매지점'].nunique())

# print(type(train[['주구매상품']]))

# LabelEncoder > OrdinalEncoder
# encoder_mer= OrdinalEncoder()
# train['주구매상품']= encoder_mer.fit_transform(train[['주구매상품']])
# test['주구매상품']= encoder_mer.transform(test[['주구매상품']])
# encoder_store= OrdinalEncoder()
# train['주구매지점']= encoder_store.fit_transform(train[['주구매지점']])
# test['주구매지점']= encoder_store.transform(test[['주구매지점']])


encoder_mer= OneHotEncoder(sparse_output = False)
train_mer= pd.DataFrame(encoder_mer.fit_transform(train[['주구매상품']]), columns= train['주구매상품'].unique())
test_mer= pd.DataFrame(encoder_mer.transform(test[['주구매상품']]), columns= train['주구매상품'].unique())
# print('********train_mer*********',train_mer.head())
train= pd.concat([train, train_mer], axis= 1)
test= pd.concat([test, test_mer], axis= 1)
# print(train.head())

encoder_store= OneHotEncoder(sparse_output = False)
train_store= pd.DataFrame(encoder_store.fit_transform(train[['주구매지점']]), columns= train['주구매지점'].unique())
test_store= pd.DataFrame(encoder_store.transform(test[['주구매지점']]), columns= train['주구매지점'].unique())
train= pd.concat([train, train_store], axis= 1)
test= pd.concat([test, test_store], axis= 1)

train.drop(columns= ['주구매상품', '주구매지점'], inplace= True)
test.drop(columns= ['주구매상품', '주구매지점'], inplace= True)

print(train.head())
print(test.head())


# # OneHotEncoder > get_dummies
# print(train.info())
# # encoder_mer= OneHotEncoder(sparse_output= False)
# train= pd.get_dummies(train, columns= ['주구매상품'], dtype= int)
# # print(train.info())
# print(train.head())


# print(train.describe())
# print(test.describe())
# print(train['방문일수'].describe())



# # 총구매액, 최대구매액, 환불금액 스케일링
# from sklearn.preprocessing import MinMaxScaler

# # print('before',train[['총구매액','최대구매액','환불금액']].describe())

# scaler= MinMaxScaler()
# train['총구매액']= scaler.fit_transform(train[['총구매액']])
# test['총구매액']= scaler.transform(test[['총구매액']])

# train['최대구매액']= scaler.fit_transform(train[['최대구매액']])
# test['최대구매액']= scaler.transform(test[['최대구매액']])

# train['환불금액']= scaler.fit_transform(train[['환불금액']])
# test['환불금액']= scaler.transform(test[['환불금액']])

# # print('after', train[['총구매액','최대구매액','환불금액']].describe())



################train#########################
y= train['성별']
X= train.drop('성별', axis= 1)

from sklearn.model_selection import train_test_split
# print(help(train_test_split))
train_x, test_x, train_y, test_y = train_test_split(X, y, test_size=0.2, random_state= 42)
# print(train_x)
# print(X.corr())

import sklearn.svm
from sklearn.tree import DecisionTreeClassifier
from sklearn.tree import ExtraTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.svm import SVC
# print(help(sklearn.svm))
# model= DecisionTreeClassifier(random_state= 42) # 58% # 스케일링 58%
"""
> 0.5814285714285714
[0.1624447  0.11095861 0.11963861 0.08550953 0.10239238 0.08012055
 0.05723865 0.12263077 0.09835427 0.06071194]

"""
model= RandomForestClassifier(random_state= 42) # 61% # 스케일링 62% # ordinal 인코딩
# model= ExtraTreeClassifier(random_state= 42) # 54% # 스케일링 54%
# model= SVC(random_state= 42) # 61% # 스케일링 61%

print(model)
model.fit(train_x, train_y)
score= model.score(test_x, test_y)
print(score)

# print(model.feature_importances_.flatten())

result= model.predict(test)
# print(result)
test['성별']= result
# print(test.info())
test.to_csv("result.csv", index=False)



# 답안 제출 참고
# 아래 코드는 예시이며 변수명 등 개인별로 변경하여 활용
# pd.DataFrame변수.to_csv("result.csv", index=False)
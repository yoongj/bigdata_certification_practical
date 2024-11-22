"""
[ Travel Insurance Prediction Data ]  
여행객의 정보들을 기반으로 여행보험 상품 가입 여부 예측
- index(id) 컬럼 포함 총 10개의 컬럼으로 되어있으며, 
train 데이터로 1490건, test 데이터로 497건의 자료를 제공
- 훈련데이터의 여행여부(0,1)의 비율은 7:3 ~ 8:2 정도의 비율로 되어 있었음
"""

import pandas as pd
from sklearn.preprocessing import OrdinalEncoder
from sklearn.model_selection import train_test_split

df= pd.read_csv('TravelInsurancePrediction.csv', index_col= 0)
# print(df.info())
# print(df['TravelInsurance'].value_counts())

### 라벨인코딩 (전처리)
oe= OrdinalEncoder()
obj_cols= ['Employment Type', 'GraduateOrNot', 'FrequentFlyer', 'EverTravelledAbroad']
df[obj_cols]= oe.fit_transform(df[obj_cols])
df= df.astype('int64')
# print(oe.categories_)
# print(df.info())
# print(df.head())


### 데이터 분할
train, test= train_test_split(df, stratify= df['TravelInsurance'])
# print(help(train_test_split))

# print(train, test)
# print(train['TravelInsurance'].value_counts()) # train 1490개
# # TravelInsurance
# # 0    958
# # 1    532
# print(test['TravelInsurance'].value_counts()) # test 497개
# # TravelInsurance
# # 0    319
# # 1    178

train_y= train['TravelInsurance']
train_x= train.drop('TravelInsurance', axis= 1)
# print(train_x)
# print(train_y)

test_y= test[['TravelInsurance']]
test_x= test.drop('TravelInsurance', axis= 1)


from sklearn.ensemble import RandomForestClassifier

rf= RandomForestClassifier()

rf.fit(train_x, train_y)
# print(rf.score(test_x, test_y))
# 0.806841046277666

pred= rf.predict(test_x)
# print(pred)

test_y['TravelInsurance']= pred
# print(test_y)
# test_y.to_csv('pred_TravelInsurance.csv')

print(rf.predict_proba(test_x))

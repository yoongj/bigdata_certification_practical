"""
중고차 판매 가격 예측
"""
import pandas as pd
from sklearn.preprocessing import OrdinalEncoder, LabelEncoder, OneHotEncoder
x_train= pd.read_csv('504_x_train.csv', index_col= 0)
y_train= pd.read_csv('504_y_train.csv', index_col= 0)
x_test= pd.read_csv('504_x_test.csv', index_col= 0)
# print(x_train)
# print(y_train)
# print(x_test)

# print(x_train.info())
# print(x_test.info())\
# 결측치 x

# 라벨인코딩
# print(x_train['fuelType'].value_counts())
obj_cols= ['model', 'transmission', 'fuelType']

oe= OrdinalEncoder() # 2차원 배열, float로 변환
x_train[obj_cols]= oe.fit_transform(x_train[obj_cols]).astype(int)
x_test[obj_cols]= oe.transform(x_test[obj_cols]).astype(int)
# print(x_train.info())
# print(x_test.info())

le= LabelEncoder() # 1차원 배열, int로 변환
# x_train['model']= le.fit_transform(x_train['model'])
# x_test['model']= le.transform(x_test['model'])
# print(x_train.info())
# print(x_test.info())

ohe= OneHotEncoder() # 새로운 df 만들어서 concat 해줘야 함..
# ohe.fit_transform(x_train[['transmission', 'fuelType']])
# ohe.transform(x_test[['transmission', 'fuelType']])
# print(x_train.info())
# print(x_test.info())

# print(x_train.describe())
# print(x_test.describe())


from sklearn.model_selection import train_test_split

x_train, x_val, y_train, y_val= train_test_split(x_train, y_train, test_size= 0.2)
# print(x_train)
# print(y_train)
# print(x_val)
# print(y_val)

from sklearn.ensemble import RandomForestRegressor
from lightgbm import LGBMRegressor
from sklearn.linear_model import LinearRegression

m1= RandomForestRegressor()
m2= LinearRegression()
m3= LGBMRegressor()

m1.fit(x_train, y_train)
m2.fit(x_train, y_train)
m3.fit(x_train, y_train)

print('rf score',m1.score(x_val, y_val))
print('lr score',m2.score(x_val, y_val))
print('lgbm score',m3.score(x_val, y_val))

"""
rf score 0.9582126377910597
lr score 0.7976450406584286
lgbm score 0.959269873497515
"""
"""
고객 구매 데이터를 사용해서
고객이 주문한 물품이 제 시간에 도착여부(Reached.on.Time_Y.N) 예측
"""

import pandas as pd

# dataset_url= 'https://www.kaggle.com/prachi13/customer-analytics?select=Train.csv'
df= pd.read_csv('E-Commerce Shipping Data.csv')
# print(df)

# print(df.info())
# 결측치 x


# 라벨인코딩
# print(df[['Warehouse_block', 'Mode_of_Shipment', 'Product_importance', 'Gender']])
from sklearn.preprocessing import OrdinalEncoder

oe= OrdinalEncoder()
df[['Warehouse_block', 'Mode_of_Shipment', 'Product_importance', 'Gender']]= oe.fit_transform(df[['Warehouse_block', 'Mode_of_Shipment', 'Product_importance', 'Gender']])
# print(oe.categories_)
# print(oe.feature_names_in_)
# print(df[['Warehouse_block', 'Mode_of_Shipment', 'Product_importance', 'Gender']])
df= df.astype('int')
# print(df[['Warehouse_block', 'Mode_of_Shipment', 'Product_importance', 'Gender']])
# print(df.info())
# print(df.describe())

from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split

y= df['Reached.on.Time_Y.N']
X= df.drop('Reached.on.Time_Y.N', axis= 1)
# print(X)
# print(y)

train_x, test_x, train_y, test_y= train_test_split(X, y, test_size= 0.2, shuffle= True, stratify= y)
model= RandomForestClassifier()
model.fit(train_x, train_y)
print(model.score(test_x, test_y))
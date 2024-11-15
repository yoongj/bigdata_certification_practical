import pandas as pd
import numpy as np
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import roc_auc_score

train = pd.read_csv("train.csv", encoding= 'UTF8')
test = pd.read_csv("X_test.csv", encoding= 'cp949')

# print(train.info())
# print(test.info())

# print(train['환불금액'].info())
# print(train['환불금액'].describe())

train['환불금액']= train['환불금액'].fillna(0)
test['환불금액']= test['환불금액'].fillna(0)

# print(train.info())
# print(test.info())

# print(train['주구매상품'].nunique())
# print(test['주구매상품'].nunique())
# print(train['주구매지점'].nunique())
# print(test['주구매지점'].nunique())

le= LabelEncoder()
train['주구매상품']= le.fit_transform(train['주구매상품'])
test['주구매상품']= le.transform(test['주구매상품'])

train['주구매지점']= le.fit_transform(train['주구매지점'])
test['주구매지점']= le.transform(test['주구매지점'])

# print(train.info())
# print(test.info())

y= train['성별']
X= train.drop('성별', axis= 1)

train_x, valid_x, train_y, valid_y= train_test_split(X, y, test_size= 0.2, random_state= 42)

# print('x', X.head())
# print('y', y.head())

model= RandomForestClassifier(random_state= 42)
model.fit(train_x, train_y)
print(model.score(valid_x, valid_y))
print(roc_auc_score(valid_y, model.predict_proba(valid_x)[:, 1]))
pred= model.predict(test)

pred_df= pd.DataFrame(pred, columns= ['pred'])
# print(pred_df)
pred_df.to_csv('result.csv', index=False)

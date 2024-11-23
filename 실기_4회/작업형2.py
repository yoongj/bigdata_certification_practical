import pandas as pd
from sklearn.preprocessing import OrdinalEncoder
from sklearn.ensemble import RandomForestClassifier

x_train= pd.read_csv('404_x_train.csv')
x_test= pd.read_csv('404_x_test.csv')
y_train= pd.read_csv('404_y_train.csv')

# print(x_train)
# print(x_test)
# print(y_train)

# 결측치 X
# print(x_train.info())
# print(x_test.info())

# 라벨인코딩
obj_cols= ['Gender', 'Ever_Married', 'Graduated', 'Profession', 'Spending_Score']
oe= OrdinalEncoder()
x_train[obj_cols]= oe.fit_transform(x_train[obj_cols])
x_test[obj_cols]= oe.transform(x_test[obj_cols])

# print(x_train.info())
# print(x_test.info())
# float형으로 변환됨

# 확인
# print(x_train)
# print(x_test)

# 모두 int형으로 변환
# flt_cols= ['Work_Experience', 'Family_Size']
# print(x_test['Family_Size'].value_counts())
x_train= x_train.astype(int)
x_test= x_test.astype(int)

# print(x_train.info())
# print(x_test.info())


# y값 라벨인코딩

# print(y_train)
oe_y= OrdinalEncoder()
y_train['Segmentation']= oe_y.fit_transform(y_train[['Segmentation']]).astype(int)
# print(oe_y.categories_)
# print(y_train)


# 모델링
model= RandomForestClassifier()
x_train= x_train.drop('ID', axis= 1)
y_train= y_train['Segmentation']
# print(x_train)
# print(y_train)
model.fit(x_train, y_train)
print(model.score(x_train, y_train))
# 0.9285501637392081

test_id= x_test['ID']
x_test= x_test.drop('ID', axis= 1)
pred= model.predict(x_test)
print(pred)

y_test= pd.DataFrame({'ID': test_id, 'Segmentation': pred})
print(y_test)

y_test.to_csv('y_pred.csv', index= False)


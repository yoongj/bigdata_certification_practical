import pandas as pd


x= pd.read_csv('X_train.csv', encoding= 'cp949')
y= pd.read_csv('y_train.csv', encoding= 'cp949')
df= pd.merge(x,y, on= 'cust_id', how= 'inner')

print(df)
df.to_csv('train.csv', index= False)
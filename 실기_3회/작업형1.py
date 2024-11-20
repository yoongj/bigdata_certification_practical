"""
문제 1 : 
[보스턴 데이터] 20640건 정도의 데이터 중 
컬럼들의 결측값을 전부 제거 후 
데이터를 처음부터 순서대로 70%를 추출하여 
'housing_median_age' 1분위수를 산출
"""

import pandas as pd

df= pd.read_csv('california_housing.csv')
# print(df.info())

df.dropna(axis= 0, inplace= True)
# print(df.info())

df= df.head(int(len(df)*0.7))
df.reset_index(inplace= True, drop= True)
# print(df)
# print(df.info())

# print(df.describe())
# print(df.describe().loc['25%'])

# longitude               -120.7900
# latitude                  33.9100
# housing_median_age        19.0000
# total_rooms             1419.0000
# total_bedrooms           294.0000
# population               787.0000
# households               276.0000
# median_income              2.4896
# median_house_value    114800.0000




"""
문제 2 : 
[국가별 유병률? 데이터] 
연도별(1990 ~ 2007:18개년도, 행) 대략 200개(193개) 정도의 국가(컬럼)의 데이터 중
2000년도 전체 국가 유병률의 평균보다 큰 국가 수 산출
"""
import pandas as pd

df= pd.read_csv('302_worlddata.csv', index_col= 0)
# print(df)

y_2000= df.iloc[1]
# print(y_2000)

mean_2000= y_2000.mean()
# 81.01036269430051
over_mean= y_2000[y_2000.values>mean_2000]
# print(len(over_mean))
# 76




"""
문제 3 :
[타이타닉  데이터]
컬럼별로 빈값 또는 결측값들의 비율을 확인하여 
가장 결측율이 높은 변수명을 출력
"""

import pandas as pd

df= pd.read_csv('titanic_url.csv', index_col= 0)
# print(df.info())
# 결측치 있는 컬럼 : 'Age', 'Cabin'

# object 컬럼에서 빈값 찾기
# obj_cols= ['Name','Age','Ticket','Cabin','Embarked']
# obj_df= df[['Name','Age','Ticket','Cabin','Embarked']]
# for i in obj_cols:
#     print(df[df[i].values==''])
# 없음

null_sum= df.isna().sum()
null_max= max(df.isna().sum())
print(null_max)
print(null_sum[null_sum==null_max].index[0])
# 아래 4줄 다시
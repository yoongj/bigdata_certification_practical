"""
Question 1 
문제 1 :
보스턴 데이터 범죄율 컬럼 top10 중 10번째 범죄율 값으로 1~10위의 범죄율 값을 변경 후 AGE 변수 80이상의 범죄율 평균 산출
"""
import pandas as pd
# df= pd.read_csv('housing.csv', header= None, delimiter=r"\s+",
#                 names= ['CRIM', 'ZN', 'INDUS', 'CHAS', 'NOX', 'RM', 'AGE', 'DIS', 'RAD', 'TAX', 'PTRATIO', 'B', 'LSTAT', 'MEDV'])
# # print(df)
# # print(df['ocean_proximity'].value_counts())

# crime_sort= df.sort_values(by= 'CRIM', ascending= False)
# # print(crime_sort.head(11))

# crime_sort.iloc[:10, 0]= crime_sort.iloc[9,0]
# # print(crime_sort.head(11))

# over_80= crime_sort[crime_sort['AGE']>=80]
# print(over_80['CRIM'].mean())



"""
Question 2

1) 주어진 데이터 첫번째 행 부터 순서대로 80%까지의 데이터를 추출 후 
2) 'total_bedrooms' 변수의 결측값(NA)을 'total_bedrooms' 변수의 중앙값으로 대체하고
3)  대체 전의 'total_bedrooms' 변수 표준편차값과 
    대체 후의 'total_bedrooms' 변수 표준편차값의 차이 산출
"""
# df= pd.read_csv('housing_2.csv')
# # print(df)
# # print(df.columns)

# ### 1
# df= df.head(int(len(df)*0.8))
# # print(df.info())

# # print(df['total_bedrooms'].describe())
# # mean       544.243625

# ### 2 - 4
# # print(df['total_bedrooms'].info())
# # print(df['total_bedrooms'].describe())
# # 50%        436.000000
# # print(df['total_bedrooms'].median())

# print('Before\n', df['total_bedrooms'].describe()) # 전 : std        435.900577
# before= df['total_bedrooms'].describe()['std']
# print('\n\n')
# df['total_bedrooms']= df['total_bedrooms'].fillna(df['total_bedrooms'].median())
# # df.fillna({'total_bedrooms':df['total_bedrooms'].median()}, inplace= True)
# print('After\n', df['total_bedrooms'].describe()) # 후 : std        433.925430
# after= df['total_bedrooms'].describe()['std']

# print('차이 : ', abs(after-before))




"""
Question 3

'population' 항목의 이상값의 합계를 계산.
이상값은 평균에서 1.5 * 표준편차를 초과하거나 미만인 값의 범위로 정한다.
"""
df= pd.read_csv('housing_2.csv')

# print(df.describe())
# 'households'열로 대체

# print(df['households'].describe())
dead_1st= df['households'].mean() - df['households'].std()*1.5
dead_3rd= df['households'].mean() + df['households'].std()*1.5
# print(dead_1st)
# print(dead_3rd)

outlier= df[(df['households']<dead_1st) | (df['households']>dead_3rd)]
print(outlier['households'].sum())
# 2020169.0





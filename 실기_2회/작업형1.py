"""
Question 1 
문제 1 :
보스턴 데이터 범죄율 컬럼 top10 중 10번째 범죄율 값으로 1~10위의 범죄율 값을 변경 후 AGE 변수 80이상의 범죄율 평균 산출
"""
import pandas as pd
df= pd.read_csv('housing.csv', header= None, delimiter=r"\s+", names= ['CRIM', 'ZN', 'INDUS', 'CHAS', 'NOX', 'RM', 'AGE', 'DIS', 'RAD', 'TAX', 'PTRATIO', 'B', 'LSTAT', 'MEDV'])
# print(df)
# print(df['ocean_proximity'].value_counts())

crime_sort= df.sort_values(by= 'CRIM', ascending= False)
# print(crime_sort.head(11))
df.sort

crime_sort.iloc[:10, 0]= crime_sort.iloc[9,0]
# print(crime_sort.head(11))

over_80= crime_sort[crime_sort['AGE']>=80]
print(over_80['CRIM'].mean())



"""
Question 2

주어진 데이터 첫번째 행 부터 순서대로 80%까지의 데이터를 추출 후 
'total_bedrooms' 변수의 결측값(NA)을 'total_bedrooms' 변수의 중앙값으로 대체하고
대체 전의 'total_bedrooms' 변수 표준편차값과 
대체 후의 'total_bedrooms' 변수 표준편차 값 산출
"""


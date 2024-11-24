"""
문제 1:
2L인 쓰레기 봉투 가격 평균 구하기
0원인 가격은 제외, ...

가격 컬럼 중 종량제 봉투가 존재하면 값이 0
용도 : 음식물쓰레기, 사용대상 : 가정용, 2L가격의 평균을 소수점 버린 후 정수로 출력
정답: 119
"""
import pandas as pd
import numpy as np

df= pd.read_csv('501_trash_bag.csv', encoding= 'cp949')
# print(df.info())

df_= df[(df['용도']=='음식물쓰레기') & (df['사용대상']== '가정용')]
# print(df_['2L가격'].value_counts())
result= df_[df_['2L가격'] != 0]
# print(result['2L가격'].value_counts())

# print(result['2L가격'].mean().astype(int))
# 119




"""
문제 2:
# BMI지수 = 몸무게(kg) / 키(m)의 제곱
# 2. 비만도가 정상에 속하는 인원수와 과체중에 속하는 인원수의 차이를 정수로 출력
정답 : 28
"""

bmi= pd.read_csv('502_bmi.csv')
# print(bmi)

bmi['bmi']= bmi['Weight']/(bmi['Height']/100)**2
# print(bmi)
# print(bmi['bmi'].describe())

ordinary= bmi[(18.5< bmi['bmi']) & (bmi['bmi']<= 23)] # 47개
over= bmi[(23< bmi['bmi']) & (bmi['bmi']<= 25)] # 19개
# print(ordinary.info())
# print(over.info())

# print(abs(len(ordinary)-len(over)))
# 28



"""
문제 3:
# 순 전입학생수 : 총 전입학생 수 - 총 전출학생 수
# 3. 순 전입학생이 가장 많은 학교의 전체 학생 수를 구하시오
정답 : 566
"""

student= pd.read_csv('503_students.csv', encoding= 'cp949')

student['순 전입학생']= student['총 전입학생'] - student['총 전출학생']
# print(student)



# - 고려
# pure_sum= student.groupby('학교').sum()
# # print(pure_sum)
# school= pure_sum['순 전입학생'].idxmax()
# print(pure_sum.loc[school,'전체 학생 수'])
# # 566

# - 고려 x > 0으로 치환
student['순 전입학생']= [0 if i <0 else i for i in student['순 전입학생']]
# print(student)
pure_sum= student.groupby('학교').sum()
result= pure_sum[pure_sum['순 전입학생'] == pure_sum['순 전입학생'].max()]['전체 학생 수']
print(result.item())
# 566
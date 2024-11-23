"""
문제 1: IQR 사분 범위 값 정수로 구하기
사분위수 Q1 - Q3의 절대값을 구하시오.(정수형으로 출력)
답: 18
"""
list = [2, 3, 3.2, 5, 7.5, 10, 11.8, 12, 23, 25, 31.5, 34]
import pandas as pd
df = pd.DataFrame({
    'value' : list
})

# start
q1= df['value'].describe()['25%']
q3= df['value'].describe()['75%']
# print(q3)
# print(q1)
# print(int(q3- q1))
# # 18

# q1= df['value'].quantile(0.25)
# q3= df['value'].quantile(0.75)
# print(q3)
# print(q1)
# print(int(q3- q1))
# # 18






"""
문제 2: 유튜브 동영상
파생변수 만들고 활용하여 다른 조건과 함께 인덱싱
reactions 중 love와wow의 비율이 0.4보다 크고 0.5보다 작은 타입 중 비디오의 개수를 구하시오.

num_loves와 num_wows를 매우 긍정적인 반응으로 정의할 때,
전체 반응수 (num_reactions) 중 매우 긍정적인 반응 수가 차지하는 비율 계산
그 비율이 0.5보다 작고 0.4보다 크며, 유형이 비디오에 해당하는 건수를 정수로 출력
답: 90
"""
import pandas as pd
df= pd.read_csv('402_facebook.csv')
# print(df)

df['so_positive']= df['num_loves']+df['num_wows']
ratio= df[(df['so_positive']<df['num_reactions']*0.5) & (df['so_positive']>df['num_reactions']*0.4)]

result= ratio[ratio['status_type'] == 'video']
# print(len(result))
# # 90





"""
문제 3: 넷플릭스
날짜형 데이터 핸들링 활용해서 다른 조건과 함께 인덱싱
2018년 1월 중 넷플릭스에서 상영중인 작품 개수를 구하시오.

2018년 1월에 넷플릭스에 등록된 컨텐츠 중에서 'United Kingdom'이 단독 제작한 컨텐츠 수를 정수로 출력
답: 6
"""


import pandas as pd
import numpy as np

df= pd.read_csv('403_netflix.csv') # 5 나옴
df= pd.read_csv('nf.csv')          # 6 나옴
# print(df.info())

drop_df= df.dropna(subset= 'date_added')
# print(df.info())

drop_df['add_year']= [i.split(' ')[-1] for i in drop_df['date_added']]
drop_df['add_month']= [i.split(' ')[0] for i in drop_df['date_added']]
# print(df.head())
print(drop_df['add_year'].value_counts())
print(drop_df['add_month'].value_counts())

date_1801= drop_df[(drop_df['add_year']== '2018') & (drop_df['add_month']== 'January')]
result= date_1801[date_1801['country']== 'United Kingdom']


# year_18= drop_df[drop_df['add_year']== '2018']
# month_01= year_18[year_18['add_month']== 'January']
# result= month_01[month_01['country']== 'United Kingdom']


print(len(result))




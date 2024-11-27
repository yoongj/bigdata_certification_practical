"""
문제 1: 
각 구급 보고서 별 출동시각과 신고시각의 차이를 '소요시간' 컬럼을 만들고 초(sec)단위로 구하고
소방서명 별 소요시간의 평균을 오름차순으로 정렬 했을때 3번째로 작은 소요시간의 값과 소방서명을 출력하라
"""
import pandas as pd
df = pd.read_csv('https://raw.githubusercontent.com/Datamanim/datarepo/main/krdatacertificate/e6_p1_1.csv')
# print(df['출동시각'])

df['신고시각']= [str(i) if len(str(i)) == 6 else "0"*(6-len(str(i)))+str(i) for i in df['신고시각']]
df['출동시각']= [str(i) if len(str(i)) == 6 else "0"*(6-len(str(i)))+str(i) for i in df['출동시각']]

df['신고sec']= [int(i[:2])*3600 + int(i[2:4])*60 + int(i[4:]) for i in df['신고시각']]
df['출동sec']= [int(i[:2])*3600 + int(i[2:4])*60 + int(i[4:]) for i in df['출동시각']]

df['소요시간']= df['출동sec']-df['신고sec']
df['소요시간']= [i+(24*60*60) if i<0 else i for i in df['소요시간']]

# 소방서별 소요시간의 평균
df= df[['소방서명','소요시간']]
sorted_df= df.groupby('소방서명').mean().sort_values(by= '소요시간')
# 답
# df.groupby(['소방서명'])['소요시각'].mean().sort_values().reset_index().iloc[2].values
sorted_df.reset_index(inplace= True)

# print(sorted_df.iloc[2]['소방서명'])
# print(sorted_df.iloc[2]['소요시간'])
# # 종로소방서
# # 175.5



"""
문제 2:
학교 세부유형이 일반중학교인 학교들 중 
일반중학교 숫자가 2번째로 많은 시도의 일반중학교 데이터만 필터하여 
해당 시도의 교원 한명 당 맡은 학생수가 가장 많은 학교를 찾아서 해당 학교의 교원수를 출력하라
"""
df = pd.read_csv('https://raw.githubusercontent.com/Datamanim/datarepo/main/krdatacertificate/e6_p1_2.csv')

df= df[df['학교세부유형'] == '일반중학교'] # 4960 > 3260
second= df.groupby('시도')['학교명'].count().sort_values().reset_index()['시도'].iloc[-2] # 서울

df= df[df['시도'] == second] # 서울 일반중학교 387개

df['학생수/교원']= df['일반학급_학생수_계']/df['교원수_총계_계']
df.sort_values(by= '학생수/교원', inplace= True)

# print(df.iloc[-2]['교원수_총계_계'])
# # -1 : 가장 아래 학교는 학생수0, 교원수0 임
# # 그래서 -2를 설정하여 출력
# # 33




"""
문제 3: 
5대 범죄(절도, 사기, 배임, 방화, 폭행)의 월별 총 발생건수를 총범죄수라고 표현하자. 
18,19년의 각각 분기별 총범죄수의 월평균 값을 구했을때 최대값을 가지는 년도와 분기를 구하고 
해당 분기의 최댓값의 사기가 발생한 월의 사기 발생건수를 출력하라
(1분기:1,2,3월 / 2분기 : 4,5,6월 / 3분기 7,8,9월 / 4분기 10,11,12월 , 1분기 월평균 : 1,2,3월의 총범죄수 평균)
"""
df = pd.read_csv('https://raw.githubusercontent.com/Datamanim/datarepo/main/krdatacertificate/e6_p1_3.csv')

df['총범죄수']= df['절도'] + df['사기'] + df['배임'] + df['방화'] + df['폭행']
# print(df['총범죄수'][:3].sum())

season_mean= pd.DataFrame({'범죄분류': ['18_1', '18_2', '18_3', '18_4', '19_1', '19_2', '19_3', '19_4']})
season_mean['월 평균 범죄수']= [df['총범죄수'][3*i:3*i+3].mean() for i in range(8)]
# print(season_mean)

result1= season_mean.sort_values(by= '월 평균 범죄수', ascending= False).iloc[0]['범죄분류']
# print(result1)
# 19_2

# 19년 2분기 : 19년 4,5,6월
result1_cols= ['2019년_4월', '2019년_5월', '2019년_6월']
# for month in result1_cols:
#     df[df['범죄분류']== month]

df_사기= df[df['범죄분류'].isin(result1_cols)]['사기']
print(df_사기.sort_values().iloc[-1])
# 27766
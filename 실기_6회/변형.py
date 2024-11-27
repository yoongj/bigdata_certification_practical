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
# print(df)


## 1트
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
# print(df_사기.sort_values().iloc[-1])
# 27766



## 2트
df['년도']= [i[:4] for i in df['범죄분류']]
df['월']= [i.split('_')[-1][:-1] for i in df['범죄분류']]
df['분기']= [(int(i)-1)//3 + 1 for i in df['월']]


result1= df.groupby(['년도','분기'])['총범죄수'].mean().idxmax()
# print('최댓값: ',result1[0],'년도',result1[1],'분기')
# 최댓값:  2019 년도 2 분기

result2= df[(df['년도']==result1[0]) & (df['분기']==result1[1])].dropna()['사기'].max()
# print(result2)
# 27766





"""
제 2유형 : 
예측 변수 General_Health, test.csv에 대해 ID별로 General_Health 값을 예측하여 제출, 
제출 데이터 컬럼은 ID와 General_Health 두개만 존재해야함. 평가지표는 f1score
"""

train = pd.read_csv('https://raw.githubusercontent.com/Datamanim/datarepo/main/krdatacertificate/ep6_p2_train.csv')
test = pd.read_csv('https://raw.githubusercontent.com/Datamanim/datarepo/main/krdatacertificate/ep6_p2_test.csv')

test_id= test['ID']
test= test.drop('ID', axis= 1)
train_y= train['General_Health']
train_x= train.drop(['ID','General_Health'], axis= 1)

# print(train_x)
# print(train_y)
# print(test)

from sklearn.preprocessing import LabelEncoder

y_le= LabelEncoder()
train_y= y_le.fit_transform(train_y)
y_class= y_le.classes_
print(y_le.classes_)
# print(train_y)



# 결측치 > x
# print(train_x.info())



# obj 컬럼
obj_cols= [i for i in train_x.columns if train_x[i].dtype == 'object']
# print(train[obj_cols].nunique())
from sklearn.preprocessing import OrdinalEncoder

oe= OrdinalEncoder()

train_x[obj_cols]= oe.fit_transform(train_x[obj_cols]).astype(int)
test[obj_cols]= oe.transform(test[obj_cols]).astype(int)

# print(train_x)
# print(test)



# 스케일링? > x
# print(train_x.describe())
# print(test.describe())



# 모델링
from sklearn.ensemble import RandomForestClassifier

rf= RandomForestClassifier()
rf.fit(train_x, train_y)

pred= rf.predict(test)
# pred= [y_class[i] for i in pred]
pred= oe.inverse_transform(pred)
# print(len(pred))

pred_df= pd.DataFrame({'ID': test_id,
              'General_Health': pred})

pred_df.to_csv('pred.csv', index= False)

print(pd.read_csv('pred.csv'))
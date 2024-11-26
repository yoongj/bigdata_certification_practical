# bigdata_certification_practical
빅데이터 분석기사 실기



<details>
    <summary><h2>실기체험</h2></summary>

## 
### ♠ big_2nd.py
#### EDA
- 데이터 로드 (3500, 11), (2482, 10)
- pandas set option
#### Preprocessing
- '총구매액', '최대구매액'이 0이하인 값 제거 (3490, 11), (2473, 10)
- '총구매액' < '최대구매액'인 데이터 제거 (3430, 11), (2438, 10)
- ~~'총구매액' < '환불금액'인 데이터 제거~~
- '환불금액' 결측치 0으로 대체 & int형으로 변환
- ~~'주구매상품', '주구매지점' 라벨인코딩~~
- '주구매상품', '주구매지점' 원핫인코딩
- '총구매액', '최대구매액', '환불금액' 스케일링
#### Modeling
- RF
- DT
- ET
- SVM

### ♠ big_2nd_q2.py
: big_2nd.py 다시
- 데이터 로드
- 결측치 대체 (0)
- string 데이터 LabelEncoding
- valid 데이터 split
- RandomForest 모델링
- valid score : 0.6171428571428571
- roc-auc score : 0.6512769042043046
- 결과 저장


### ♠ q_3.py
- 제 3유형  
![image](https://github.com/user-attachments/assets/738523fa-1d86-4985-88e0-d8232d6aeccd)
- 1) 카이제곱 통계량
    - scipy.stats.chi2_contingency
        - **두 개 이상의 범주형 변수**간의 **독립성 검정**
        - 이차원 배열(행렬)을 입력으로 받음
        - 통계량, p-value, 자유도, 기대값 테이블 을 반환
        - 이차원 배열은 각 범주에 해당하는 데이터들의 개수표
        - pd.crosstab으로 교차표를 만들어 이를 chi2_contingency의 입력으로 사용

    - scipy.stats.chisquare
        - **한 개의 범주형 변수**분포의 **일치 여부** 검정
        - 한 변수에 대한 관측값, 기댓값 총 두개의 값을 입력으로 받음
        - 통계량, p-value 를 반환

- 2) 로지스틱 회귀모형 계수
    - ~~X['Sex']= X['Sex'].apply(lambda x:1 if x=='male' else 0)~~
    - X.loc[:,'Sex']= [1 if x=='male' else 0 for x in X['Sex']]
    - model= LogisticRegression
    - model.coef_[0] : 독립변수별 계수
    - penalty 인자 : None 처리

- 3) 오즈비
    - np.exp(model.coef_[0][1])
    - 특정 변수의 계수를 지수함수로 사용하면 이 값이 **특정 변수가 한단위 증가할 때의 오즈비**가 됨.
    - 만약 두단위가 증가할 때의 오즈비를 구한다면, 계수 * 단위 수
    - 오즈비는 이진 종속변수에서 주로 사용 > 로지스틱 회귀모형(이진분류)
</details>



<details>
    <summary><h2>실기_2회</h2></summary>

## 
### ♠ 작업형1.py
- 1) 데이터 프레임 정렬
    - df.sort_value(by= '열이름', ascending= 오름차순)
    - df.sort_index

- 2) 통계값
    - 평균값 : mean()
    - 중앙값 : median()
    - 표준편차 : std()

- 3) 연산자 사용
    - or 연산자, | 연산자
    - df[(df['households']<dead_1st) | (df['households']>dead_3rd)]
    - 에러에 따라 다르게 사용


### ♠ 작업형2.py
- 데이터 확인 (info)
- 결측치 x
- object 타입 변수 인코딩 (OrdinalEncoder)
- model_selection.train_test_split
- fit & score
- score: 0.6654545454545454
</details>




<details>
    <summary><h2>실기_3회</h2></summary>

## 
### ♠ 작업형1.py
- 문제 1
    - 결측값 제거 : dropna(axis= 0) # 행제거
    - 데이터프레임 인덱스 : reset_index(inplace= True, drop= True) # 기존 인덱스가 열 값으로 들어옴.
- 문제 3
    - 결측치 갯수 : df.isna().sum()
    - 가장 많은/적은 갯수를 가진 인덱스 : .idxmax() / .idxmin()


### ♠ 작업형2.py
- OrdinalEncoder 인자 외우기 
(참고) https://data-yun.tistory.com/entry/Python-LabelEncoder-VS-OrdinalEncoder 
- 형변환 : astype(int) == astype('int64')
- 클래스 별 예측 확률 : model.predict_proba(test_x)
</details>




<details>
    <summary><h2>실기_4회</h2></summary>
   
## 
### ♠ 작업형1.py
- 문제 1
    - describe()['25%'] == quantile(0.25)
    - describe()['75%'] == quantile(0.75)

- 문제 3
    - df['col'].dropna  (x)
    - df.dropna(subset= ['col'])  (o)
    : 'col'열에서 NaN인 값인 행을 제거

### ♠ 작업형2.py
- y= train['result']
- X= train.drop('result', axis= 1) # axis= 1 : 열 기준 제거
</details>




<details>
    <summary><h2>실기_5회</h2></summary>
   
## 
### ♠ 작업형1.py
- 문제 1
    - 특정 열에서 결측치를 가진 행 drop : df.dropna(subset=['col'])
    - 하지만, 결측치가 아닌 특정 값을 가진 행 drop은? : df[df['col']!= 'vaule']
    > 쉽게 생각하기..!

- 문제 2
    - 제곱 : a**2

- 문제 3
    - 그룹화 : df.groupby('col').sum()  (sum 외에도 mean, median, count, min/max, var, std 등 가능)
    - 그룹화 다중 통계량 : df.groupby('col').agg(['mean', 'var'])

### ♠ 작업형2.py
- object 컬럼
    - LabelEncoder
        - 타겟값(y)을 대상으로 함
        - 따라서 1차원만 입력으로 받음
        - int형으로 반환
    - OrdinalEncoder
        - 독립변수(x)를 대상으로 함
        - 따라서 2차원 배열을 입력으로 받음
        - float형으로 반환  
    => OrdinalEncoder().fit_transform(train[['col1','col2']]).**astype(int)**

- validation 셋 분리해서 성능확인
    - sklearn.model_selection.train_test_split() 사용

- 여러 모델 비교
    - 분류 (Classifier)
        - sklearn.ensemble.RandomForestClassifier
        - sklearn.linear_model.LogisticRegres**sion**
        - xgboost.XGBClassifier
    - 회귀 (Regressor)
        - sklearn.ensemble.RandomForestRegressor
        - sklearn.linear_model.LinearRegres**sion**
        - xgboost.XGBRegressor
</details>
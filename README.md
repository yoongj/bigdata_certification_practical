# bigdata_certification_practical
빅데이터 분석기사 실기

## 2회차
### big_2nd.py
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

### big_2nd_q2.py
: big_2nd.py 다시
- 데이터 로드
- 결측치 대체 (0)
- string 데이터 LabelEncoding
- valid 데이터 split
- RandomForest 모델링
- valid score : 0.6171428571428571
- roc-auc score : 0.6512769042043046
- 결과 저장





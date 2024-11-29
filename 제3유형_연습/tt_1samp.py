"""
모집단이 1개 일때 (집단의 평균 vs 특정 값)

- 정규성 O : 단일 표본 t-검정 
- 정규성 X : 윌콕슨 부호검정

"""

import pandas as pd

df = pd.read_csv("mtcars.csv")

# 사용자 코딩
##### 1번) mpg열이 평균 20 검정, 유의수준 5%
# H0(귀무): mpg 열의 평균이 20이다.
# H1(대립): mpg 열의 평균이 20이 아니다.

# 정규성 검정
from scipy.stats import shapiro
normal= shapiro(df['mpg'])
# print(normal)
# static= 0.947564... , pvalue= 0.1228813...
# 귀무가설 채택 > 정규성을 따른다

# 가설 검정
from scipy.stats import ttest_1samp
ttest_result= ttest_1samp(df['mpg'], popmean= 20)
# print(ttest_result)
# static= 0.0850600..., pvalue= 0.9327606...
# 귀무가설 채택 > 평균이 20이다.

# 만약 비정규성 이라면, 윌콕슨 검정
from scipy.stats import wilcoxon
wil_result= wilcoxon(df['mpg']-20)
# print(wil_result)
# static= 249.0, pvalue= 0.7891259...



##### 2번) mpg 열의 평균이 17보다 크다고 할수 있는가
# H0(귀무): mpg열이 17보다 크지 않다.
# H1(대립): mpg열이 17보다 크다.

# 정규성 검정
from scipy.stats import shapiro
normal= shapiro(df.mpg)
print(normal)
# stat= 0.947564... , pvalue= 0.12881359...
# 귀무가설 채택 > 정규성을 따른다.

# 정규성을 띌 경우, ttest_1samp
from scipy.stats import ttest_1samp
ttest_result= ttest_1samp(df.mpg, 17, alternative= 'greater')
print(ttest_result)
# stat= 2.900840... , pvalue= 0.003394155...
# 귀무가설 기각 > 평균이 17보다 크다.

# 비정규성을 띌 경우, wilcoxon
from scipy.stats import wilcoxon
wil_result= wilcoxon(df.mpg-17, alternative= 'greater')
print(wil_result)
# stat= 395.5, pvalue= 0.00661516..
# 귀무가설 기각 > 평균이 17보다 크다.

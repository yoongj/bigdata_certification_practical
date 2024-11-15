import pandas as pd

# titanic_url = 'https://raw.githubusercontent.com/datasciencedojo/datasets/master/titanic.csv'
# titanic = pd.read_csv(titanic_url)
# titanic.to_csv('titanic_url.csv')
titanic= pd.read_csv('titanic_url.csv')

##### 1) #####
from scipy.stats import chi2_contingency
# # print(titanic.columns)
# ct= pd.crosstab(titanic['Sex'],titanic['Survived'])
# # print(ct)

# chi= chi2_contingency(ct)
# print(f"{chi[0]:.4f}")
# print(round(chi[0],3))


##### 2) #####
# Gender, SibSp, Parch, Fare를 독립변수, 로지스틱 회귀모형, Parch 계수값
from sklearn.linear_model import LogisticRegression

X= titanic[['Sex', 'SibSp', 'Parch', 'Fare']]
y= titanic['Survived']
# print(X.info())


# X['Sex']= [1 if i == 'male' else 0 for i in X['Sex']]
# X['Sex'] = X['Sex'].apply(lambda x: 1 if x == 'male' else 0)
X.loc[:,'Sex']= [1 if i == 'male' else 0 for i in X['Sex']]
print(X)
print(y)

model= LogisticRegression()
model.fit(X,y)
print(model.coef_)
print(round(model.coef_[0][2], 3))
print(f"{model.coef_[0][2]:.3f}")
print(model.score(X,y))


##### 3) #####
import numpy as np
print(np.exp(model.coef_[0][1]))


import pandas as pd
from scipy.stats import chi2_contingency

titanic= pd.read_csv('titanic_test.csv')
# print(titanic.columns)
ct= pd.crosstab(titanic['Sex'],titanic['Survived'])
print(ct)

chi= chi2_contingency()
print(chi)
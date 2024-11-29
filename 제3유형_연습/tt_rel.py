import pandas as pd

df= pd.read_csv('blood_pressure.csv')

# H0(귀무): 치료 후 혈압 감소 x
# H1(대립): 치료 후 혈압 감소 O

from scipy.stats import ttest_rel

result1= ttest_rel(df['bp_after'], df['bp_before'], alternative= 'less')
result2= ttest_rel(df['bp_before'], df['bp_after'], alternative= 'greater')

print(result1)
print(result2)
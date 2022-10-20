import pandas as pd
import pingouin as pg
from sklearn.metrics import cohen_kappa_score

df = pd.read_csv('input/Patricc 2 - video coding reliability.csv')
df = df.fillna(0)
columns  = list(df)
columns = columns[4:]

df_omer = df[df['Coder name'] == 'Omer']
df_sharon = df[df['Coder name'] == 'Sharon']

for col in columns:
    print(col)
    icc = pg.intraclass_corr(data=df, targets='exam', raters='Coder name', ratings=col)
    icc.set_index('Type')
    print(icc)



for col in columns:
    rater1 = df_omer[col]
    rater2 = df_sharon[col]
    print(col)
    print(cohen_kappa_score(rater1, rater2))
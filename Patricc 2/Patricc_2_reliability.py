import pandas as pd
import pingouin as pg
from sklearn.metrics import cohen_kappa_score

df = pd.read_csv('input/Patricc 2 - video coding reliability_short_.csv')
df = df.fillna(0)
columns  = list(df)
columns = columns[4:]
df_icc = df
df_omer = df[df['Coder name'] == 'Omer']
#df_omer = df_omer.add_prefix('omer_')
#df_omer = df_omer.reset_index()
df_sharon = df[df['Coder name'] == 'Sharon']
#df_sharon = df_sharon.add_prefix('sharon_')
#df_sharon = df_sharon.reset_index()
df = pd.concat([df_omer, df_sharon], axis=1)

#columns  = list(df)
#columns = columns[4:]
#df_omer.to_csv('df_omer.csv')
#df_sharon.to_csv('df_sharon.csv')
df.to_csv('df_spss.csv')
for col in columns:
    try:
        print(col)
        icc = pg.intraclass_corr(data=df_icc, targets='exam', raters='Coder name', ratings=col)
        icc.set_index('Type')
        print(icc)
    except:
        print('didnt work')
        pass


print('here')
for col in columns:
    rater1 = df_omer[col]
    rater2 = df_sharon[col]
    print('here too')
    print(col)
    print(cohen_kappa_score(rater1, rater2))
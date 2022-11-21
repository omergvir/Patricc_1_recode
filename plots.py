import os
import pandas as pd
from scipy import stats
from variables import robot_vs_tablet, granger_condition_list
from matplotlib import pyplot as plt

df = pd.read_csv('output/df_by_session_lag10_2200_201122.csv')
df_pf_robot = pd.read_csv('output/df_pf_robot_lag10_2200_201122.csv')
df_pf_vs = pd.read_csv('output/df_pf_vs_lag10_2200_201122.csv')
df = pd.read_csv('output/df_by_session_lag20_0700_211122.csv')
df_pf_robot = pd.read_csv('output/df_pf_robot_lag20_0700_211122.csv')
df_pf_vs = pd.read_csv('output/df_pf_vs_lag20_0700_211122.csv')
filter_col = [col for col in df if col.startswith('granger')]
df2 = df[filter_col]
means = df2.mean()
print(means)
means.sort_values(ascending=False, inplace=True)
df2 = df2[means.index]
df2.plot.box(rot=90, figsize=(10,20), ylim=(0,130))




#pf_robot_plot = df_pf_robot.plot(x='f', y='p', style='o', title='p(F) for all robot Granger tests')
#pf_robot_plot.set_xlim(0, 40)
#pf_robot_plot.set_ylabel("p")
#pf_vs_plot = df_pf_vs.plot(x='f', y='p', style='o', title='p(F) for all Granger tests')
#pf_vs_plot.set_ylim(0,0.1)
#pf_vs_plot.set_ylabel("p")
#pf_vs_plot.axhline(y=0.05, color='r', linestyle='-')

df_l10_int05 = pd.read_csv('output/df_by_session_lag10_int05.csv')
df_pf_robot_l10_int05 = pd.read_csv('output/df_pf_robot_lag10_int05.csv')
df_pf_vs_l10_int05 = pd.read_csv('output/df_pf_vs_lag10_int05.csv')
df_l20_int02 = pd.read_csv('output/df_by_session_lag20_int02.csv')
df_pf_robot_l20_int02 = pd.read_csv('output/df_pf_robot_lag20_int02.csv')
df_pf_vs_l20_int02 = pd.read_csv('output/df_pf_vs_lag20_int02.csv')
filter_col = [col for col in df if col.startswith('granger')]

pf_robot_plot = df_pf_robot.plot(x='f', y='p', style='o', title='p(F) for all robot Granger tests')
pf_robot_plot.set_xlim(0, 40)
pf_robot_plot.set_ylabel("p")
pf_vs_l10_int05_plot = df_pf_vs_l10_int05.plot(x='f', y='p', style='o', title='p(F) for all Granger tests')
pf_vs_l20_int02_plot = df_pf_vs_l20_int02.plot(x='f', y='p', style='o', title='p(F) for all Granger tests')

pf_vs_l10_int05_plot.set_ylim(0,0.1)
pf_vs_l10_int05_plot.set_ylabel("p")
pf_vs_l10_int05_plot.axhline(y=0.05, color='r', linestyle='-')

df_pf_vs_l10_int05.plot(x='f', y='p', style='o', title='l10_int05')
df_pf_vs_l20_int02.plot(x='f', y='p', style='o', title='l20_int02')
ax = df_pf_vs_l10_int05.plot(x='f', y='p', style='o', color='green',label='l10_int05')
df_pf_vs_l20_int02.plot(ax=ax, x='f', y='p', style='o', color='blue',label='l20_int02', ylim=(0,0.1))


plt.show()

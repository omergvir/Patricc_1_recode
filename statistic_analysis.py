import os
import pandas as pd
from scipy import stats
from variables import robot_vs_tablet, granger_condition_list

variations = ['_count_', '_total time_', '_normalized count_', '_normalized total time_']
#variations = ['_normalized total time_']
conditions = ['r', 't']
df = pd.read_csv('output/df_count_all_int_05.csv')

column_names = ["feature", "robot mean", "tablet mean", "pvalue"]
df_result = pd.DataFrame(columns = column_names)

for feature in robot_vs_tablet:
    for variation in variations:
        #print(df[[feature+variation+conditions[0], feature+variation+conditions[1]]].describe())
        #print(feature+variation+conditions[0], feature+variation+conditions[1])
        result = stats.ttest_rel(df[feature+variation+conditions[0]], df[feature+variation+conditions[1]])
        pvalue = result[1]

        robot_mean = df[feature+variation+conditions[0]].mean()
        tablet_mean = df[feature+variation+conditions[1]].mean()
        df_result.loc[len(df_result.index)] = [feature+variation, robot_mean, tablet_mean, pvalue]

#for feature in granger_condition_list:
#    variable_1 = 'granger_'+feature[0]+feature[1]+'_r'
#    print(variable_1)
#    df[variable_1] = pd.to_numeric(df[variable_1], downcast="float")
#    df[variable_2] = pd.to_numeric(df[variable_2], downcast="float")
#    variable_2 = 'granger_' + feature[0]+feature[1] + '_t'
#    result = stats.ttest_rel(df[variable_1], df[variable_2])
#    pvalue = result[1]
#    robot_mean = df[variable_1].mean()
#    tablet_mean = df[variable_2].mean()
#    df_result.loc[len(df_result.index)] = [feature[0]+feature[1], robot_mean, tablet_mean, pvalue]

df_result.to_csv(os.path.join("output", "df_t_test_results.csv"))
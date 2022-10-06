import pandas
import numpy as np

d1 = {"Name": ["Pankaj", "Lisa"], "ID": [1, 2]}
d2 = {"Name": "David", "ID": 3}



df1 = pandas.DataFrame(d1, index={1, 2})
df2 = pandas.DataFrame(d2, index={3})
cols = df1.shape[1]
max_lag = 5
df3 = pandas.DataFrame(np.zeros((max_lag, cols)))
df3.columns = list(df1.columns.values)
df3.rename(index={'0':'ddd'})

columns = list(df1.columns.values)
row = 'feature'
list_of_zeros = [0] * cols
a = pandas.crosstab(index=row, columns=list_of_zeros)
print('a')
print(a)

print('********\n', df1)
print('********\n', df2)
print('********\n', df3)
df4 = pandas.concat([df1, df3, df2])

print('********\n', df4)

['Child affect','Child affect:positive 1','Child affect:positive 2','Child affect:positive 3',
 'Child affective touch:affective touch','Child gesture','Child gaze:parent','Child gaze:props',
 'Child gaze:robot','Child gesture:point at prop','Child prop manipulation:child','Child utterance:utterance',
 'Conversational turns','Conversational turns:CP','Conversational turns:CPC','Conversational turns:PC',
 'Conversational turns:PCP','joint attention','Joint attention:props','Joint attention:robot','Mutual gaze:MG',
 'Non-verbal scaffolding','Non-verbal scaffolding:cognitive','Non-verbal scaffolding:cognitive','Non-verbal scaffolding:technical',
 'Verbal scaffolding','Verbal scaffolding:affective','Verbal scaffolding:cognitive','Verbal scaffolding:technical',
 'Parent affect','Parent affect:positive 1','Parent affect:positive 2','Parent affect:positive 3',
 'Parent affective touch:affective touch','Parent gesture:point at prop','Parent prop manipulation:parent',
 'Parent gaze:child','Parent gaze:props','Parent gaze:robot','Parent utterance:utterance']


d = {('a','b'):0}
ser = pd.Series(data = d, index = [('a','b')])
df_count_row_1 = df_count_row_1.append(ser)
a = df_count_row.to_frame()
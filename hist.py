import os
import sys
import pandas as pd
import numpy as np
import seaborn as sns
import warnings
import time
import math
import statsmodels.api as sm
import seaborn as sns
import matplotlib as mpl
from matplotlib import pyplot as plt
from scipy.stats import chi2_contingency
from statsmodels.tsa.stattools import grangercausalitytests
from statsmodels.stats.anova import AnovaRM
from functools import reduce
from shutil import rmtree
import pingouin as pg
import pickle

warnings.simplefilter(action='ignore', category=FutureWarning)
dir_path = os.path.dirname(sys.argv[0])
os.chdir(dir_path)
plt.rc('ytick', labelsize=7)


def granger(df,col1,col2,maxlag=5):
    c = df.columns[df.nunique() <= 1]  # creates a list of cols with constant values
    # print('these are the constant cols = ', c, 'checking cols = ', col1, col2)
    print(["granger coloumns = ", col1, col2])
    try:
        x = grangercausalitytests(df[[col1, col2]].diff().dropna(),maxlag=maxlag,verbose=False)  # null hypoposis col2 does not granger cause col1
        #x = grangercausalitytests(df[[col1, col2]].dropna(),maxlag=maxlag,verbose=False)  # null hypoposis col2 does not granger cause col1

        result = sm.tsa.stattools.adfuller(df[[col1, col2]].diff().dropna())
        # print the test statistic and p-value
        print(result[0], result[1])

        lags = list(range(1,maxlag+1))
        lag_pv = np.array([x[lag][0]['ssr_chi2test'][1] for lag in lags])
        min_pv = min(lag_pv)
        lag_chi2v = np.array([x[lag][0]['ssr_chi2test'][0] for lag in lags])
        best_chi2v = max(lag_chi2v)
        max_index = lag_chi2v.argmax(axis=0)
        #best_index = lag_pv.argmin(axis=0)
        best_pv = lag_pv[max_index]
        #best_chi2v = lag_chi2v[best_index]
        best_lag = np.array(lags)[lag_pv == best_pv] if len(lag_pv == best_pv) == 1 else np.array(lags)[lag_pv == best_pv][0]
        if min_pv < 0.05:
            min_pv = 1
        else:
            min_pv = 0

    except Exception as e:
        best_lag = 100
        best_pv = math.nan
        best_chi2v = math.nan
        lag_chi2v = math.nan
        lag_pv = math.nan
        min_pv = math.nan
        print(e)
    print(best_chi2v)
    # return([best_lag,best_pv])
    return(best_chi2v, min_pv, lag_pv, lag_chi2v)



def granger_mat(df,maxlag = 5):
    '''
    col1 is row col2 is columns
    each cell represents the best result for the test that the column does not granger cause the row
    if rejected then: col --granger cause-- row
    '''
    cols = df.columns
    mat = [[granger(df,col1,col2,maxlag) for col2 in cols] for col1 in cols]
    out_df = pd.DataFrame(mat, columns=cols, index=cols)
    return out_df


def entropy(Y):
    """
    Also known as Shanon Entropy
    """
    unique, count = np.unique(Y, return_counts=True, axis=0)
    prob = count/len(Y)
    en = np.sum((-1)*prob*np.log2(prob))
    return en
def jEntropy(Y,X):
    """
    H(Y;X)
    """
    #
    YX = np.c_[Y,X]
    return entropy(YX)
def cEntropy(Y, X):
    """
    conditional entropy = Joint Entropy - Entropy of X
    H(Y|X) = H(Y;X) - H(X)
    """
    return jEntropy(Y, X) - entropy(X)
def gain(Y, X):
    """
    Information Gain, I(Y;X) = H(Y) - H(Y|X)
    """
    return entropy(Y) - cEntropy(Y,X)
def Theils_U(Y,X):
    '''
    :return: U(Y|X) the uncertenty of Y given X
    '''
    return(gain(Y,X)/entropy(Y))
def Theils_U_matrix(df):
    #calculate Theils U for each set of 2 columns in the df
    cols = df.columns
    mat = [[Theils_U(df[col1], df[col2]) for col2 in cols] for col1 in cols]
    out_df = pd.DataFrame(mat,columns=cols,index=cols)
    return(out_df)
def transfer_time_to_s(x):
    # transdorms time column to seconds
    split_time = [float(y) for y in x.split(':')]
    time_in_s = np.multiply(split_time,[3600,60,1]).sum()
    return(time_in_s)
def cramer_v(x,y):
    #calculated cramers V for 2 arrays
    df_c = pd.DataFrame({"x": x, "y": y})
    cont_table = pd.crosstab(df_c["x"],df_c["y"])
    chi = chi2_contingency(cont_table,correction=False)[0]
    n = cont_table.sum().sum()
    v = np.sqrt((chi/n))
    return(v)
def cramer_v_matrix(df):
    #calculate cramers V for each set of 2 columns in the df
    cols = df.columns
    mat = [[cramer_v(df[col1], df[col2]) for col2 in cols] for col1 in cols]
    out_df = pd.DataFrame(mat,columns=cols,index=cols)
    return(out_df)
def trim_by_time(df):
    #trims the data from s_time to e_time
    max_time = 300 #sessions are trimmed at 5 minutes (300 sec)
    s_time = df["s_time"][df["action"] == 'start-end'].values[0]+0.001
    end_time = df["e_time"][df["action"] == 'start-end'].values[1]-0.001
    # print the difference between the start and end time
    print('duration')
    print(end_time-s_time)
    if end_time - s_time >= max_time:
        end_time = s_time + max_time
    df = df[(df["s_time"] >= s_time) & (df["e_time"] <= end_time)]
    return df, s_time, end_time

def ct_helper(df1,df2,sec,order):
    #counts if utterance at df2 started 1 sec after df1 starts or {sec} sec after it ended
    s_times = []
    e_times = []
    t_times = []
    for index, row in df1.iterrows():
        interaction_df = df2[(df2["s_time"]>= row["s_time"] +1)&(df2["s_time"]<= row["e_time"] +sec)]
        if len(interaction_df) == 0:
            continue
        s_times.append(row["s_time"])
        e_times.append(interaction_df["e_time"].values[0])
        t_times.append(interaction_df["e_time"].values[0]-row["s_time"])
    df_out = pd.DataFrame({"action":["Conversational turns"]*len(s_times),
                            "who":["Parent,Child"]*len(s_times),
                            "s_time":s_times,
                            "e_time":e_times,
                            "t_time":t_times,
                           "sub_action":[order]*len(s_times)})
    return(df_out)
def Conversational_turns(df,sec = 5):
    # creates Conversational turns data
    parent_utterance_df = df[df["action"] == 'Parent utterance']
    child_utterance_df = df[df["action"] == 'Child utterance']
    PC_frame = ct_helper(parent_utterance_df,child_utterance_df,sec,"PC")
    CP_frame = ct_helper(child_utterance_df,parent_utterance_df, sec, "CP")
    PCP_frame = ct_helper(PC_frame,CP_frame,sec,"PCP")
    CPC_frame = ct_helper(CP_frame,PC_frame, sec, "CPC")
    return(PC_frame,CP_frame,PCP_frame,CPC_frame)
def joint_attention(df,rap=True):
    #creates joint attention data
    #rap specifies to treat robots and props as both robots and props,
    #meaning that if the child gazed at robot and parent at robots and props it count as joint attention
    #counts if gaze at df2 started while df1 was gazing

    if rap:
        #deletes robot or prop and adds them as two seperate rows
        rap_df = df[df["sub_action"] == 'robot or prop']
        r_df = rap_df.copy()
        r_df["sub_action"] = "robot"
        p_df = rap_df.copy()
        p_df["sub_action"] = "props"
        df = pd.concat([df[(df["sub_action"] != 'robot or prop')],r_df,p_df]).reset_index(drop = True)

    parent_gaze_df = df[df["action"] == 'Parent gaze']
    child_gaze_df = df[df["action"] == 'Child gaze']
    s_times = []
    e_times = []
    t_times = []
    sub_action_gaze=[]
    #parent starts gaze
    for index, row in parent_gaze_df.iterrows():
        v1 = (child_gaze_df["s_time"] >= row["s_time"]) & (child_gaze_df["s_time"] <= row["e_time"]) #child gazed after parent
        v2 = (child_gaze_df["s_time"] <= row["s_time"]) & (child_gaze_df["e_time"] >= row["s_time"]) #parent gazed after child
        interaction_df = child_gaze_df[v1 | v2]
        interaction_df = interaction_df[interaction_df["sub_action"] == row["sub_action"]]
        if len(interaction_df) == 0:
            continue
        for ind, inter_row in interaction_df.iterrows():
            s_time = max(inter_row["s_time"],row["s_time"])
            e_time = min(inter_row["e_time"],row["e_time"])
            s_times.append(s_time)
            e_times.append(e_time)
            t_times.append(e_time-s_time)
            sub_action_gaze.append(row["sub_action"])

    df_out = pd.DataFrame({"action":["Joint attention"]*len(s_times),
                            "who":["Parent,Child"]*len(s_times),
                            "s_time":s_times,
                            "e_time":e_times,
                            "t_time":t_times,
                           "sub_action":sub_action_gaze})
    return(df_out)
def mutual_gaze(df):
    parent_gaze_df = df[df["action"] == 'Parent gaze']
    parent_gaze_at_child = parent_gaze_df[parent_gaze_df["sub_action"]=="child"]
    child_gaze_df = df[df["action"] == 'Child gaze']
    child_gaze_at_parent = child_gaze_df[child_gaze_df["sub_action"]=="parent"]

    s_times = []
    e_times = []
    t_times = []
    for index, row in parent_gaze_at_child.iterrows():
        v1 = (child_gaze_at_parent["s_time"] >= row["s_time"]) & (child_gaze_at_parent["s_time"] <= row["e_time"]) #child gazed after parent
        v2 = (child_gaze_at_parent["s_time"] <= row["s_time"]) & (child_gaze_at_parent["e_time"] >= row["s_time"]) #parent gazed after child
        interaction_df = child_gaze_at_parent[v1 | v2]
        if len(interaction_df) == 0:
            continue
        for ind, inter_row in interaction_df.iterrows():
            s_time = max(inter_row["s_time"],row["s_time"])
            e_time = min(inter_row["e_time"],row["e_time"])
            s_times.append(s_time)
            e_times.append(e_time)
            t_times.append(e_time-s_time)

    df_out = pd.DataFrame({"action":["Mutual gaze"]*len(s_times),
                            "who":["Parent,Child"]*len(s_times),
                            "s_time":s_times,
                            "e_time":e_times,
                            "t_time":t_times,
                           "sub_action":"MG"})
    return(df_out)
def fillnas(df):
    action = df["action"]
    who = df["who"]
    action_first_word = [act.split(" ")[0] for act in action]
    action_first_word = ["Parent" if (act == "Verbal") else act for act in action_first_word]
    who_filled = [w if isinstance(w,str) else act for w,act in zip(who,action_first_word)]
    df["who"] = who_filled

    sub = df["sub_action"]
    action_lsat_words = [reduce(lambda x, y: x + " " + y, act.split(" ")[1:]) for act in action]
    sub_filled = [s if isinstance(s,str) else word for s,word in zip(sub,action_lsat_words)]
    df["sub_action"] = sub_filled
    return(df)

def df_preprocess(df, file_base):

    print('start preprocess', file_base, time.time())
    file_split = file_base.split("_")
    lesson_split = list(file_split[0])
    order = lesson_split[1]
    if order == '3':
        order = '2'
    condition = lesson_split[2]
    participant = file_split[1]
    coder = file_split[2]
    #print (order, condition, participant, coder)

    #parameters
    conversation_turn = 5 #max time (sec) between utterances to include as conversational turn
    #give names to columns
    df.columns =["action","who","s_time","e_time","t_time","sub_action"]
    df = df.convert_dtypes()  #converts to string
    #convert to time objects
    for time_col in ["s_time","e_time","t_time"]:
        df[time_col] = df[time_col].apply(transfer_time_to_s)

    df, s_time, e_time = trim_by_time(df) #trims data
    df = fillnas(df)# fill nas
    PC_frame, CP_frame, PCP_frame, CPC_frame = Conversational_turns(df,conversation_turn) #makes conversational turns
    ja_frame = joint_attention(df) #makes joint attentions
    mg_frame = mutual_gaze(df) # makes mutual gaze
    df = pd.concat([df,PC_frame, CP_frame, PCP_frame, CPC_frame,ja_frame,mg_frame]).reset_index(drop=True)
    df["action:sub_action"] = df["action"] + ":" + df["sub_action"].fillna("")

    # add session parameters
    #print(file_base, time.time())
    file_split = file_base.split("_")
    lesson_split = list(file_split[0])
    order = lesson_split[1]
    condition = lesson_split[2]
    participant = file_split[1]
    coder = file_split[2]
    #print (order, condition, participant, coder)
    df['order'] = order
    df['condition'] = condition
    df['participant'] = participant
    df['coder'] = coder
    df['start_time'] = s_time
    df['end_time'] = e_time
    return(df)

def add_session_parameters(df, file_base):
    #print(file_base)
    file_split = file_base.split("_")
    lesson_split = list(file_split[0])
    order = lesson_split[1]
    condition = lesson_split[2]
    participant = file_split[1]
    coder = file_split[2]
    #print (order, condition, participant, coder)
    df['order'] = order
    df['condition'] = condition
    df['participant'] = participant
    df['coder'] = coder

    return df


def make_crosstab(df,col):
    #makes a crosstab for spesific column
    time_sums = pd.crosstab(df[col],"sum",values = df["t_time"],aggfunc = 'sum') #sums time for each action
    time_counts = pd.crosstab(df[col], "count", values=df["t_time"],aggfunc='count')  # count number of events for each action
    end_of_vid = df['e_time'].max()
    start_of_vid = df['s_time'].min()
    interaction_length_1 = df["e_time"].values[-1]-df["s_time"].values[0]
    interaction_length = end_of_vid - start_of_vid
    time_sums_normalize = time_sums/interaction_length
    time_counts_normalize = time_counts / interaction_length
    out_df = pd.concat([time_sums,time_counts,time_sums_normalize,time_counts_normalize],axis= 1)
    out_df.columns = ["total time","count","normalized total time","normalized count"]

    return(out_df)

def time_window(df,col_base,act_base,col_count,act_count,s_w,e_w):
    '''
    df = dataframe not in time domain
    col_base = takes window for this column
    act_base = takes window for this action(or sub or action:sub)
    col_count = counts the # of events for this column
    act_count = counts the # of events for this action(or sub or action:sub)
    s_w = how long (in second) to start window after col_base
    e_w = how long to end the window after col_base
    return: dict of the actions and it's respective histogram
    '''
    action_e_times = df[df[col_base]==act_base]["e_time"]
    counter_s_times = df[df[col_count]==act_count]["s_time"]
    in_time_frame = lambda  s_t,e_t: e_t + s_w <= s_t <= e_t + e_w
    time_count = {e_time: sum([in_time_frame(s_time,e_time) for s_time in counter_s_times]) for e_time in action_e_times}
    hist = np.histogram(list(time_count.values()), bins=list(range(1,max(list(time_count.values()))+2)))
    return({f"{col_base}&{act_base}&{col_count}&{act_count}":hist})

def all_windows(df,s_w,e_w):
    cols = ["action","sub_action","action:sub_action"]
    col_uniques = np.array([f"{col}&{p_a}" for col in cols for p_a in np.unique(df[col])])
    dict = {}
    [dict.update(time_window(df,col_act_base.split("&")[0],col_act_base.split("&")[1],col_act_count.split("&")[0],col_act_count.split("&")[1],s_w,e_w))
     for col_act_base in col_uniques for col_act_count in col_uniques[col_uniques !=col_act_base]]
    out_df = pd.DataFrame({"col base":[],
                           "action base":[],
                           "col count":[],
                           "action count":[],
                           "hist":[],
                           "bins" :[]})
    split_loc = lambda x,i: x.split("&")[i]
    for key,item in dict.items():
        out_df = out_df.append({"col base":split_loc(key,0),
                           "action base":split_loc(key,1),
                           "col count":split_loc(key,2),
                           "action count":split_loc(key,3),
                           "hist":item[0],
                            "bins":item[1]},ignore_index=True)
    return(out_df)


def transform_to_time_representation(df,col = "action",time_stamp_jumps=1):
    #return the df in the time domain
    #end_of_vid = df["e_time"].values[-1]
    end_of_vid = df['e_time'].max()
    #start_of_vid = df["s_time"].values[0]
    start_of_vid = df['s_time'].min()
    time_indicates = np.arange(start_of_vid, end_of_vid, time_stamp_jumps)
    # for each observation an array of times it appears in
    # times= pd.Series([np.arange(row["s_time"], row["e_time"], time_stamp_jumps) for index,row in df.iterrows()],
    #                  name = "times")
    times = pd.Series([(row["s_time"], row["e_time"]) for index,row in df.iterrows()],
                     name = "times")
    features = df[col].unique()
    feaueres_times = {feature: times[df[col] == feature].reset_index(drop=True) for feature in features} # for each feature the times it appears in
    feaueres_time_binary = {feature:[1 if sum([1 if tup[0]-time_stamp_jumps <= t < tup[1] else 0 for tup in feaueres_times[feature]]) >0 else 0
                                     for t in time_indicates]
                            for feature in features} # for each feature does it appear in time i
    feaueres_time_binary["time"] = time_indicates
    out_df = pd.DataFrame(feaueres_time_binary)
    return(out_df)

def time_window_hist(df,col_base,action_base,col_count,action_count = 'all',save_path = ''):
    '''
    :param df: dataframe
    :param col_base: [action,sub_action,action:sub_action] for action base
    :param action_base: action to count window for
    :param col_count: [action,sub_action,action:sub_action] for action count
    :param action_count: action to count in window
    :return: dataframe containing histograms
    '''
    df = df[df["action base"] == action_base]
    df = df[df["col base"] == col_base]
    df = df[df["col count"] == col_count]
    if not isinstance(action_count,str):
        if not ("all" in action_count):
            l1 = df["action count"].apply(lambda x: x in action_count)
            df = df[l1]
    elif action_count != "all":
        l1 = df["action count"].apply(lambda x: x in action_count)
        df = df[l1]
    df.reset_index(inplace = True)
    hist = df["hist"]
    bins = df["bins"]
    action_count = df["action count"]
    if isinstance(hist[0],str):
        make_array = lambda x: [int(y)  for y in x.strip("]").strip("[").split(" ") if y != '']
        hist = hist.apply(make_array)
        bins = bins.apply(make_array)
    x_axis = np.unique([y for x in bins.values for y in x])
    f = {}
    f[col_count] = []
    f.update({x:[] for x in x_axis})
    out_df = pd.DataFrame(f)
    for h,bin,act in zip (hist,bins,action_count):
        d = {}
        d[col_count] = act
        d.update(dict(zip(bin[:-1],h)))
        out_df= out_df.append(d,ignore_index=True)
    out_df.fillna(0,inplace=True)
    out_df.index = out_df[col_count].apply(lambda x: x.replace(" ","\n"))
    out_df.drop(col_count, axis=1,inplace=True)

    #if save_path != '':
    #    fig = sns.heatmap(out_df, annot=True, linewidths=.5)
    #    fig.set_title(action_base)
    #    plt.savefig(f"{save_path} {col_base} window hist.png")
    #    plt.show()

    return(out_df)

def drop_features(df):
    drop_cols = ['Joint attention:child', 'Joint attention:parent',
                'Child gaze:child', 'Parent gaze:parent', 'Child utterance:other vocalization',
                 'robot text:other']
    df_out = df
    for col in drop_cols:
        if col in df.columns:
            #df_out = df.drop(columns=col)
            df.drop(columns=col, inplace=True)
    df_out = df
    return df_out


def transform_to_row(df):
    features = ['Child affect','Child affect:positive 1','Child affect:positive 2','Child affect:positive 3',
         'Child affective touch:affective touch','Child gesture','Child gaze:parent','Child gaze:props',
         'Child gaze:robot','Child gesture:point at prop','Child prop manipulation:child','Child utterance:utterance',
         'Conversational turns','Conversational turns:CP','Conversational turns:CPC','Conversational turns:PC',
         'Conversational turns:PCP','Joint attention','Joint attention:props','Joint attention:robot','Mutual gaze:MG',
         'Non-verbal scaffolding','Non-verbal scaffolding:cognitive','Non-verbal scaffolding:cognitive','Non-verbal scaffolding:technical',
         'Verbal scaffolding','Verbal scaffolding:affective','Verbal scaffolding:cognitive','Verbal scaffolding:technical',
         'Parent affect','Parent affect:positive 1','Parent affect:positive 2','Parent affect:positive 3',
         'Parent affective touch:affective touch','Parent gesture:point at prop','Parent prop manipulation:parent',
         'Parent gaze:child','Parent gaze:props','Parent gaze:robot','Parent utterance:utterance']
    index = list(df.index)
    cols = df.shape[1]
    #print(index)
    #print(features)
    #remove rows with irrelevant features
    for label in index:
        if label not in features:
            df = df.drop(label)
    #add missing features to session count matrix
    for feature in features:
        if feature not in index:
            df_temp = pd.DataFrame(np.zeros((1, cols)))
            df_temp.columns = list(df.columns.values)
            df_temp.rename(index={0:feature})
            df = pd.concat([df, df_temp])




def add_remove_features_granger(df, features):
    cols = list(df)
    for col in cols:
        if col not in features:
            df = df.drop([col], axis=1)
    for feature in features:
        if feature not in cols:
            df[feature] = 0
    return df

def add_remove_features(df_count_row, features, file_base):
    #removes unnecessary features
    #add features where if they are missing
    #adds session parameters (condition, order, participant, coder)
    index_list = (list(df_count_row.index.values))
    df_count_row_1 = df_count_row
    for i in range(len(index_list)):
        current = index_list[i][0]
        if index_list[i][0] not in features:
            df_count_row_1 = df_count_row_1.drop(labels=[index_list[i]])
    new_index_list = (list(df_count_row_1.index.values))
    for index in range(len(new_index_list)):
        new_index_list[index] = new_index_list[index][0]
    sub_features = ['total time', 'count', 'normalized count', 'normalized total time']
    for feature in features:
        if feature not in new_index_list:
            for sub_feature in sub_features:
                new_row = {(feature, sub_feature): 0}
                new_row_ser = pd.Series(data=new_row, index=[(feature, sub_feature)])
                df_count_row_1 = df_count_row_1.append(new_row_ser)
    file_split = file_base.split("_")
    lesson_split = list(file_split[0])
    order = lesson_split[1]
    condition = lesson_split[2]
    participant = file_split[1]
    coder = file_split[2]
    #print (order, condition, participant, coder)
    df_count_row_1['order'] = order
    df_count_row_1['condition'] = condition
    df_count_row_1['participant'] = participant
    df_count_row_1['coder'] = coder

    df_count_row_1 = df_count_row_1.to_frame()
    df_count_row_1 = df_count_row_1.transpose()
    return df_count_row_1


def session_to_participant(df):
    #converts row-session df to row-participant df
    df_robot = df[df['condition'] == 1]
    df_robot = df_robot.sort_values(by = 'participant')
    df_robot = df_robot.add_suffix('_r')
    df_robot = df_robot.reset_index(drop=True)
    df_tablet = df[df['condition'] == 2]
    df_tablet = df_tablet.sort_values(by = 'participant')
    df_tablet = df_tablet.add_suffix('_t')
    df_tablet = df_tablet.reset_index(drop=True)
    df_new = pd.concat([df_robot, df_tablet], axis=1)
    return df_new


def add_qualtrics_data(df_new):
    # read qualtrics to df
    qualtrics = pd.read_csv('Patricc 1 qualtrics data_minimal.csv')
    qualtrics = qualtrics.sort_values(by='participant number', ignore_index=True)
    qualtrics = qualtrics.drop(columns=['participant number'])
    # make sure participants are in the same order in both frames - sort
    df_new = df_new.sort_values(by='participants', ignore_index=True)
    df_new = pd.concat([qualtrics, df_new], axis=1)
    return df_new


def add_qualtrics_data_1(df_new):
    # read qualtrics to df
    qualtrics = pd.read_csv('Patricc 1 qualtrics data_minimal.csv')
    new_cols = ['child gender', 'child age', 'Nars_ss1', 'Nars_ss2', 'Nars_ss3']

    for col in new_cols:
        df_new[col] = 0
        for ind in df_new.index:
            print(ind)
            participant = df_new._get_value(ind, 'participant')
            ind_participant = int(participant) - 100 - 1
            val = qualtrics._get_value(ind_participant, col)
            df_new._set_value(ind, col, val)


    #df_new['child gender'] = 0
    #df_new['child age'] = 0
    #df_new['Nars_ss1'] = 0
    #for ind in df_new.index:
    #    print(ind)
    #    participant = df_new._get_value(ind, 'participant')
    #    ind_participant = int(participant) - 100 - 1
    #    gender = qualtrics._get_value(ind_participant, 'child gender')
    #    age = qualtrics._get_value(ind_participant, 'child age')
    #    df_new._set_value(ind, 'child gender', gender)
    #    df_new._set_value(ind, 'child age', gender)
        #df_new.at[ind, 'child gender'] = gender

    #df_new = pd.concat([qualtrics, df_new], axis=1)
    return df_new

def create_zero_df(features, row_num):
    df = pd.DataFrame(columns=features,
                      index=range(1, row_num+1)).fillna(0)
    return df


def granger_condition_tests(df, test_list):
    lag = 1
    f_all = pd.DataFrame()
    df_pf = pd.DataFrame(columns=['p', 'f'])
    df_lag_log = pd.DataFrame(columns=['col1', 'col2', 'lag'])
    df_chi_log = pd.DataFrame(columns=['col1', 'col2', 'lag'])
    df_granger_session = pd.DataFrame()
    i = 0
    for test in test_list:
        col1 = test[0]
        col2 = test[1]
        result, p, lag_log, chi_log = granger(df, col1, col2, maxlag=lag)
        df_pf.loc[len(df_pf.index)] = [p, result]
        #result_str = str(result[0])+','+str(result[1])
        df_lag_log.loc[len(df_lag_log)] = [col1, col2, lag_log]
        df_lag_log['lag'][i] = lag_log
        df_chi_log.loc[len(df_chi_log)] = [col1, col2, chi_log]
        df_chi_log['lag'][i] = chi_log

        df_granger_session[('granger_'+col1+'_'+col2)] = [result] #[result]

    return df_granger_session, df_pf, df_lag_log, df_chi_log


def add_derived_features(df):
    variations = ['_count_', '_total time_', '_normalized count_', '_normalized total time_']
    for variation in variations:
        df['Parent gaze:object'+variation+'t'] = df['Parent gaze:tablet'+variation+'t']
        df['Parent gaze:object'+variation+'r'] = df['Parent gaze:robot'+variation+'r']+df['Parent gaze:props'+variation+'r']
        df['Child gaze:object'+variation+'t'] = df['Child gaze:tablet'+variation+'t']
        df['Child gaze:object'+variation+'r'] = df['Child gaze:robot'+variation+'r']+df['Child gaze:props'+variation+'r']
    return df


def add_object_features_time(df):
    df['Child gaze:object'] = 0
    df['Parent gaze:object'] = 0

    df['Child gaze:object'] = df['Child gaze:tablet']+df['Child gaze:robot']+df['Child gaze:props']
    df.loc[df['Child gaze:object'] > 1, 'Child gaze:object'] = 1

    df['Parent gaze:object'] = df['Parent gaze:tablet']+df['Parent gaze:robot']+df['Parent gaze:props']
    df.loc[df['Parent gaze:object'] > 1, 'Parent gaze:object'] = 1
    return df


def convert_labels(df):
    cols = list(df)
    new_cols = []
    for col in cols:
        new_col = col[0] + '_' + col[1]
        new_cols.append(new_col)
    df.columns = new_cols
    return df


def add_object_features_row(df):
    df['Child gaze:object_normalized total time'] = 0
    df['Parent gaze:object_normalized total time'] = 0

    df['Child gaze:object_normalized total time'] = df['Child gaze:tablet_normalized total time']+\
                              df['Child gaze:robot_normalized total time']+df['Child gaze:props_normalized total time']
    df.loc[df['Child gaze:object_normalized total time'] > 1, 'Child gaze:object_normalized total time'] = 1

    df['Parent gaze:object_normalized total time'] = df['Parent gaze:tablet_normalized total time']+\
                                                     df['Parent gaze:robot_normalized total time']+\
                                                     df['Parent gaze:props_normalized total time']
    df.loc[df['Parent gaze:object_normalized total time'] > 1, 'Parent gaze:object_normalized total time'] = 1
    return df


def remove_count_features(df, features):
    condition = df._get_value(0, 'condition_')
    participant = df._get_value(0, 'participant_')
    order = df._get_value(0, 'order_')
    if order == '3':
        order = '2'
    cols = list(df)
    for col in cols:
        if col not in features:
            df = df.drop([col], axis=1)
    if condition == 'r':
        condition_temp = 1
    else:
        condition_temp = 2
    df['condition'] = condition_temp
    df['participant'] = participant
    df['order'] = order
    return df

def action_hist(df, df_action_occur, t_window, causes, effects):
    df_hist = pd.DataFrame(columns=['cause', 'effect', 's_time', 'condition','participant'])
    #iterate over all the elements in causes
    #reset the index of df to start from 0
    df = df.reset_index(drop=True)
    for cause in causes:
        #create dataframe that contains the rows of df where the values of 'action:sub_action' is cause
        df_cause = df.loc[df['action:sub_action'] == cause]
        print('cause: ', cause)
        #if the column 'cause' in the dataframe df_action_occur has the value cause
        if cause in df_action_occur['cause'].values:
            # create a variable that is the number of rows in df_cause
            cause_num = len(df_cause.index)
            #if the value of 'condition' in the first row of df is 'r'

            if df.at[0, 'condition'] == 'r':
            #if df._get_value(0, 'condition') == 'r':
                #add cause_num to the column 'robot count' at the row where the value of 'cause' is cause in df_action_occur
                df_action_occur.loc[df_action_occur['cause'] == cause, 'robot count'] += cause_num
            else:
                #add cause_num to the column 'child count' at the row where the value of 'cause' is cause in df_action_occur
                df_action_occur.loc[df_action_occur['cause'] == cause, 'tablet count'] += cause_num
        else:
            #if the value of 'condition' in the first row of df is 'r'
            if df._get_value(0, 'condition') == 'r':
                #add a row to df_action_occur where the value of 'cause' is cause and the value of 'robot count' is the number of rows in df_cause
                df_action_occur.loc[len(df_action_occur.index)] = [cause, len(df_cause.index), 0]

            else:
                #add a row to df_action_occur where the value of 'cause' is cause and the value of 'tablet count' is the number of rows in df_cause
                df_action_occur.loc[len(df_action_occur.index)] = [cause, 0, len(df_cause.index)]

        #iterate the rows of df_cause
        for index, row in df_cause.iterrows():
            #create variable which is that value of 's_time' of the row
            cause_time = row['s_time']
            time_window = [cause_time-t_window, cause_time+t_window]
            #iterate through the rows of df
            for index2, row2 in df.iterrows():
                #if the value of 'action:sub_action' of this row is not cause
                if row2['action:sub_action'] != cause:
                    #if the value of 's_time' of this row is within the time window
                    if row2['s_time'] >= time_window[0] and row2['s_time'] <= time_window[1]:
                        #create a variable that is the difference between the value of 's_time' of this row and the value of 's_time' of the cause row
                        time_diff = row2['s_time'] - cause_time
                        #add a new row to df_hist with the value of 'action:sub_action' of this row, the value of time_diff, the value of 'condition_' of this row, and the value of 'participant_' of this row
                        df_hist.loc[len(df_hist.index)] = [cause, row2['action:sub_action'], time_diff, row2['condition'], row2['participant']]

    return df_hist, df_action_occur

#write a function that takes in a dataframe, removes the rows where that value of 'action:sub_action' is not in the list of actions, and returns the dataframe
def remove_actions(df, actions):
    df = df[df['action:sub_action'].isin(actions)]
    return df


if __name__ == '__main__':
    from variables import count_features, granger_features, granger_condition_list, granger_robot_tests, \
        robot_vs_tablet, time_series_features, hist_features
    #parameters
    output_folder = "output"
    path_out = os.path.join(output_folder)
    run_calculations = 0
    t_window = 15
    Nbins = 6
    #cause = 'robot text:pick up'
    #cause = 'Child utterance:utterance'
    if run_calculations == 1:

        interval = 0.3 #time interval between time steps
        #which modules of analysis to run
        run_granger = 0
        run_count = 0
        stitch_buffer = 30 # number of rows to buffer between stitching time series from different sessions
        all_lag_log = pd.DataFrame(columns=['col1', 'col2', 'lag'])
        all_chi_log = pd.DataFrame(columns=['col1', 'col2', 'lag'])
        #load data
        files = os.listdir("files")
        #set window paramaters
        col_base, action_base, col_count, action_count = "action","Child gaze","action",'all'
        df_count_row_all = pd.DataFrame()
        df_granger_robot = pd.DataFrame()
        df_time_all = pd.DataFrame()
        df_time_tablet = pd.DataFrame()
        df_time_robot = pd.DataFrame()
        df_pf_robot = pd.DataFrame()
        df_pf_vs = pd.DataFrame()
        df_zeros = create_zero_df(granger_features, stitch_buffer)
        df_hist_all = pd.DataFrame()
        #create a dataframe with the columns 'cause', 'robot count', 'tablet count'
        df_action_occur = pd.DataFrame(columns=['cause', 'robot count', 'tablet count'])


        for file in files:

            output_folder = "output"
            output_folder_2 = "send_goren"
            file_time = time.time()
            print('new file = ', file_time)
            file_base = file[:-4]
            file_split = file_base.split("_")
            lesson_split = list(file_split[0])
            condition_temp = lesson_split[2]
            print('condition = ', condition_temp)
            if condition_temp == 'r':
                condition = 1
            else:
                condition = 2

            path = os.path.join(output_folder,file_base)
            path_out = os.path.join(output_folder)
            path_out_2 = os.path.join(output_folder_2)
            print(path_out)
            # re creates folder
            if os.path.exists(path):
                rmtree(path)
            os.makedirs(path)

            file_path = os.path.join("files",file)
            df = pd.read_csv(file_path, sep='\t', engine='python',header = None)
            df = df_preprocess(df, file_base) # make a df of the raw data
            # in the column of df called 'action:sub_action', replace values that start with 'Parent affect' to 'Parent affect'
            df.loc[df['action:sub_action'].str.startswith('Parent affect'), 'action:sub_action'] = 'Parent affect'
            # in the column of df called 'action:sub_action', replace the values that start with "Child affect" to "Child affect"
            df.loc[df['action:sub_action'].str.startswith('Child affect'), 'action:sub_action'] = 'Child affect'
            df = remove_actions(df, hist_features)


            #create a variable named effects that is a list of all the unique values of 'action:sub_action' in df
            effects = df['action:sub_action'].unique()
            causes = df['action:sub_action'].unique()


            df_hist, df_action_occur = action_hist(df, df_action_occur, t_window, causes, effects)
            df_hist_all = pd.concat([df_hist_all, df_hist], axis=0)


        #pickle df_hist_all to the main folder
        df_hist_all.to_pickle(os.path.join(path_out, 'df_hist_all_15.pkl'))
        #pickle df_action_occur to the main folder
        df_action_occur.to_pickle(os.path.join(path_out, 'df_action_occur_15.pkl'))




    #load the pickled df_hist_all
    df_hist_all = pd.read_pickle(os.path.join(path_out, 'df_hist_all_15.pkl'))
    #load the pickled df_action_occur
    df_action_occur = pd.read_pickle(os.path.join(path_out, 'df_action_occur_15.pkl'))

    # sort df_action_occur by values of 'robot count'
    df_action_occur = df_action_occur.sort_values(by=['robot count'], ascending=True)
    # create and display a bar plot of df_action_occur
    df_action_occur.plot.barh(x='cause', y=['robot count', 'tablet count'])
    # make the chart narrower so that the labels are not cut off
    plt.tight_layout()
    # add the title 'occurance of events' the the chart
    plt.title('occurance of events')
    plt.show()

    #create and disdplay a plot for each effect in df_hist_all, in the plat draw two overlaid histograms, one for the condition 'r' and one for the condition 't'. Set the title of the plot to be the effect

    #for cause in df_hist_all['cause'].unique():
    for cause in ['Parent utterance:utterance']:
        #create a variable named df_hist_cause that is a dataframe that contains only the rows of df_hist_all where the value of 'cause' is equal to the value of cause
        df_hist_cause = df_hist_all[df_hist_all['cause'] == cause]
        for effect in df_hist_cause['effect'].unique():
            #create a dataframe that contains the rows of df_hist_all where the value of 'effect' is effect and the value of 'condition' is 'r'
            df_hist_r = df_hist_cause[(df_hist_cause['effect'] == effect) & (df_hist_cause['condition'] == 'r')]
            #create a dataframe that contains the rows of df_hist_all where the value of 'effect' is effect and the value of 'condition' is 't'
            df_hist_t = df_hist_cause[(df_hist_cause['effect'] == effect) & (df_hist_cause['condition'] == 't')]
            #create a variable r_weights that is the value of the column 'robot count' in df_action_occur where the value of 'cause' is equal to cause
            r_weights = df_action_occur[df_action_occur['cause'] == cause]['robot count'].values[0]
            #create a list r_weights_list that is in the same length as df_hist_r['s_time'] and each element is the value of 1 divided by r_weights
            r_weights_list = [1/r_weights] * len(df_hist_r['s_time'])
            #multiple the values of r_weights_list by t_window*2
            r_weights_list = [x * t_window*4 for x in r_weights_list]
            #create a variable t_weights that is the value of the column 'tablet count' in df_action_occur where the value of 'cause' is equal to cause
            t_weights = df_action_occur[df_action_occur['cause'] == cause]['tablet count'].values[0]
            #create a list t_weights_list that is in the same length as df_hist_t['s_time'] and each element is the value of 1 divided by t_weights
            t_weights_list = [1/t_weights] * len(df_hist_t['s_time'])
            #multiple the values of t_weights_list by t_window*2
            t_weights_list = [x * t_window*4 for x in t_weights_list]

            #create a plot with two overlaid histograms, one for the values of 'time_diff' in df_hist_r and one for the values of 'time_diff' in df_hist_t. Set the title of the plot to be effect. Set the color of the histogram for 'r' to be red and the color of the histogram for 't' to be blue. Set the number of bins to Nbins.
            plt.hist(df_hist_r['s_time'], color='blue', alpha=0.5, label='robot', weights=r_weights_list)
            plt.hist(df_hist_t['s_time'], color='red', alpha=0.5, label='tablet', weights=t_weights_list)



            #set the title of the plot to be cause-effect
            plt.title(f"{cause}-{effect}")
            plt.legend(loc='upper right')
            #add a vertical line to the plot at x = 0
            plt.axvline(x=0, color='k', linestyle='--')




            #create a dataframe that has the columns 'count', participant'
            df_count_r = pd.DataFrame(columns=['count', 'participant'])
            #for each unique participant count the number of rows in df_hist_r where 's_time' is less than 0 and save the result in a new row of df_count_r with the value of 'count' being the count and the value of 'participant' being the participant
            for participant in df_hist_r['participant'].unique():
                df_count_r = df_count_r.append({'count':len(df_hist_r[(df_hist_r['s_time'] < 0) & (df_hist_r['participant'] == participant)]), 'participant':participant}, ignore_index=True)
            df_count_r_pre = df_count_r
            #calculate the mean and standard deviation of the values of 'count' in df_count_r
            mean_r = df_count_r['count'].mean()
            std_r = df_count_r['count'].std()
            #divide mean_r by r_weights
            mean_r = mean_r/r_weights
            #divide std_r by r_weights
            std_r = std_r/r_weights

            #overlay a horizontal line at y = mean_r and x = [-10,0], set the color of the line to be the same as the color of the histogram for 'r'
            plt.plot([-t_window,0], [mean_r, mean_r], color='blue')
            #plot two more lines on the same x range with y values of mean_r +/- std_r, set the color of the lines to be 'r' but make the lines dashed
            plt.plot([-t_window,0], [mean_r+std_r,mean_r+std_r], color='blue', linestyle='--', linewidth=2)
            plt.plot([-t_window,0], [mean_r-std_r,mean_r-std_r], color='blue', linestyle='--', linewidth=2)



            #create a dataframe that has the columns 'count', participant'
            df_count_r = pd.DataFrame(columns=['count', 'participant'])
            #for each unique participant count the number of rows in df_hist_r where 's_time' is less than 0 and save the result in a new row of df_count_r with the value of 'count' being the count and the value of 'participant' being the participant
            for participant in df_hist_r['participant'].unique():
                df_count_r = df_count_r.append({'count':len(df_hist_r[(df_hist_r['s_time'] > 0) & (df_hist_r['participant'] == participant)]), 'participant':participant}, ignore_index=True)
            df_count_r_post = df_count_r
            #calculate the mean and standard deviation of the values of 'count' in df_count_r
            mean_r = df_count_r['count'].mean()
            std_r = df_count_r['count'].std()
            #divide mean_r by r_weights
            mean_r = mean_r/r_weights
            #divide std_r by r_weights
            std_r = std_r/r_weights
            #overlay a horizontal line at y = mean_r and x = [-10,0], set the color of the line to be the same as the color of the histogram for 'r'
            plt.plot([0,t_window], [mean_r, mean_r], color='blue')
            #plot two more lines on the same x range with y values of mean_r +/- std_r, set the color of the lines to be 'r' but make the lines dashed
            plt.plot([0,t_window], [mean_r+std_r,mean_r+std_r], color='blue', linestyle='--', linewidth=2)
            plt.plot([0,t_window], [mean_r-std_r,mean_r-std_r], color='blue', linestyle='--', linewidth=2)

            #do the same thing for the condition 't'
            df_count_t = pd.DataFrame(columns=['count', 'participant'])
            for participant in df_hist_t['participant'].unique():
                df_count_t = df_count_t.append({'count':len(df_hist_t[(df_hist_t['s_time'] < 0) & (df_hist_t['participant'] == participant)]), 'participant':participant}, ignore_index=True)
            df_count_t_pre = df_count_t
            mean_t = df_count_t['count'].mean()
            std_t = df_count_t['count'].std()
            #divide mean_t by t_weights
            mean_t = mean_t/t_weights
            #divide std_t by t_weights
            std_t = std_t/t_weights




            plt.plot([-t_window,0], [mean_t, mean_t], color='red')
            plt.plot([-t_window,0], [mean_t+std_t,mean_t+std_t], color='red', linestyle='--', linewidth=2)
            plt.plot([-t_window,0], [mean_t-std_t,mean_t-std_t], color='red', linestyle='--', linewidth=2)
            #do the same thing for the condition 't'
            df_count_t = pd.DataFrame(columns=['count', 'participant'])
            for participant in df_hist_t['participant'].unique():
                df_count_t = df_count_t.append({'count':len(df_hist_t[(df_hist_t['s_time'] > 0) & (df_hist_t['participant'] == participant)]), 'participant':participant}, ignore_index=True)
            df_count_t_post = df_count_t
            mean_t = df_count_t['count'].mean()
            std_t = df_count_t['count'].std()
            #divide mean_t by t_weights
            mean_t = mean_t/t_weights
            #divide std_t by t_weights
            std_t = std_t/t_weights
            plt.plot([0,t_window], [mean_t, mean_t], color='red')
            plt.plot([0,t_window], [mean_t+std_t,mean_t+std_t], color='red', linestyle='--', linewidth=2)
            plt.plot([0,t_window], [mean_t-std_t,mean_t-std_t], color='red', linestyle='--', linewidth=2)
            plt.show()


            #create a dataframe that has the columns 'robot pre', 'robot post', 'tablet pre' and tablet post'
            df_count = pd.DataFrame(columns=['robot pre', 'robot post', 'tablet pre', 'tablet post'])
            #set the values of the column 'robot pre' to be the values of 'count' in df_count_r_pre
            df_count['robot pre'] = df_count_r_pre['count']
            #set the values of the column 'robot post' to be the values of 'count' in df_count_r_post
            df_count['robot post'] = df_count_r_post['count']
            #set the values of the column 'tablet pre' to be the values of 'count' in df_count_t_pre
            df_count['tablet pre'] = df_count_t_pre['count']
            #set the values of the column 'tablet post' to be the values of 'count' in df_count_t_post
            df_count['tablet post'] = df_count_t_post['count']
            #divide the values of the columns 'robot pre' and 'robot post' by r_weights
            df_count['robot pre'] = df_count['robot pre']/r_weights
            df_count['robot post'] = df_count['robot post']/r_weights
            #divide the values of the columns 'tablet pre' and 'tablet post' by t_weights
            df_count['tablet pre'] = df_count['tablet pre']/t_weights
            df_count['tablet post'] = df_count['tablet post']/t_weights
            #convert the values of df_count to numeric
            df_count = df_count.apply(pd.to_numeric)
            #create a plot of boxplots for the columns 'robot pre', 'robot post', 'tablet pre' and 'tablet post'
            df_count.boxplot(column=['robot pre', 'robot post', 'tablet pre', 'tablet post'])
            plt.title(f"{cause}-{effect}")





            #add a column named time to df_count_r_pre and set all its values to be 'pre'
            df_count_r_pre['time'] = 'pre'
            #add a column named time to df_count_r_post and set all its values to be 'post'
            df_count_r_post['time'] = 'post'
            #divide the values of the column 'count' in df_count_r_pre by r_weights
            df_count_r_pre['count'] = df_count_r_pre['count']/r_weights
            #divide the values of the column 'count' in df_count_r_post by r_weights
            df_count_r_post['count'] = df_count_r_post['count']/r_weights
            #concatenate df_count_r_pre and df_count_r_post into a new dataframe df_r_prepost
            df_r_prepost = pd.concat([df_count_r_pre, df_count_r_post])
            #conduct a repeated measures ANOVA with participant as subjects, time as within and count as the dependent variable using statmodels
            try:
                rm_anova_r = AnovaRM(data=df_r_prepost, depvar='count', subject='participant', within=['time']).fit()
                #print the results of the ANOVA
                print('robot pre post')
                print(rm_anova_r)
                #define ax
                ax = plt.gca()
                #print the anova results on the upper left corner of the plot using plt.text
                ax.text(0.05, 0.95, str(rm_anova_r), transform=ax.transAxes, fontsize=6, verticalalignment='top')
            except:
                pass

            #add a column named time to df_count_r_pre and set all its values to be 'pre'
            df_count_t_pre['time'] = 'pre'
            #add a column named time to df_count_r_post and set all its values to be 'post'
            df_count_t_post['time'] = 'post'
            #divide the value of the column 'count' in df_count_t_pre by t_weights
            df_count_t_pre['count'] = df_count_t_pre['count']/t_weights
            #divide the value of the column 'count' in df_count_t_post by t_weights
            df_count_t_post['count'] = df_count_t_post['count']/t_weights
            #concatenate df_count_r_pre and df_count_r_post into a new dataframe df_r_prepost
            df_t_prepost = pd.concat([df_count_t_pre, df_count_t_post])
            #conduct a repeated measures ANOVA with participant as subjects, time as within and count as the dependent variable using statmodels
            try:
                rm_anova_t = AnovaRM(data=df_t_prepost, depvar='count', subject='participant', within=['time']).fit()
                #print the results of the ANOVA
                print('tablet pre post')
                print(rm_anova_t)
                #define ax
                ax = plt.gca()
                #print the anova results on the upper right corner of the plot using plt.text
                ax.text(0.55, 0.95, str(rm_anova_t), transform=ax.transAxes, fontsize=6, verticalalignment='top')
            except:
                pass

            #add a column named time to df_count_r_pre and set all its values to be 'pre'
            df_count_r_post['condition'] = 'robot'
            #add a column named time to df_count_r_post and set all its values to be 'post'
            df_count_t_post['condition'] = 'tablet'
            #concatenate df_count_r_pre and df_count_r_post into a new dataframe df_r_prepost
            df_condition = pd.concat([df_count_t_post, df_count_r_post])
            #conduct a repeated measures ANOVA with participant as subjects, time as within and count as the dependent variable using statmodels
            try:
                rm_anova_t = AnovaRM(data=df_condition, depvar='count', subject='participant', within=['condition']).fit()
                #print the results of the ANOVA
                print('robot vs tablet post')
                print(rm_anova_t)
                #define ax
                ax = plt.gca()
                #print the anova results on the lower right corner of the plot using plt.text
                #ax.text(0.55, 0.05, str(rm_anova_t), transform=ax.transAxes, fontsize=6, verticalalignment='top')
            except:
                pass

            plt.show()








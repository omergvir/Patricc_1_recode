import os
import sys
import pandas as pd
import numpy as np
import seaborn as sns
import warnings
import time
import matplotlib as mpl
from matplotlib import pyplot as plt
from scipy.stats import chi2_contingency
from statsmodels.tsa.stattools import grangercausalitytests
from functools import reduce
from shutil import rmtree

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
        lags = list(range(1,maxlag+1))
        lag_pv = np.array([x[lag][0]['ssr_chi2test'][1] for lag in lags])
        best_pv = min(lag_pv)
        lag_chi2v = np.array([x[lag][0]['ssr_chi2test'][0] for lag in lags])
        best_chi2v = max(lag_chi2v)
        best_lag = np.array(lags)[lag_pv == best_pv] if len(lag_pv == best_pv) == 1 else np.array(lags)[lag_pv == best_pv][0]
        #if best_pv > 0.05:
        #    best_chi2v = 'ns'

    except Exception as e:
        best_lag = 100
        best_pv = 100
        best_chi2v = 0
        print(e)
    print(best_chi2v)
    # return([best_lag,best_pv])
    return(best_chi2v)



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
    if end_time - s_time >= max_time:
        end_time = s_time + max_time
    df = df[(df["s_time"] >= s_time) & (df["e_time"] <= end_time)]
    return df

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

    df = trim_by_time(df) #trims data
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
    participants = pd.unique(df_count_row_all[('participant', '')])
    new_columns = []
    for col in df.columns:
        new_column_r = col[0] + '_' + col[1] + '_' + 'r'
        new_column_t = col[0] + '_' + col[1] + '_' + 't'
        new_columns.append(new_column_r)
        new_columns.append(new_column_t)
        #print(new_column_r, new_column_t)
    df_new = pd.DataFrame(columns=new_columns)
    df_new['participants'] = participants
    for ind in df.index:
        participant = df[('participant', '')][ind]
        new_index = df_new.index[df_new['participants'] == participant].tolist()
        #print(new_index)
        if df[('condition','')][ind] == 'r':
            for col in df.columns:
                new_col = col[0] + '_' + col[1] + '_' +  'r'
                #print(new_col)
                #print(df_new.columns.get_loc(new_col))
                #df_new[new_col][new_index[0]] = df[col][ind]
                df_new.iat[new_index[0], df_new.columns.get_loc(new_col)] = df[col][ind]
        elif df[('condition','')][ind] == 't':
            for col in df.columns:
                new_col = col[0] + '_' + col[1] + '_' +  't'
                #df_new[new_col][new_index[0]] = df[col][ind]
                df_new.iat[new_index[0], df_new.columns.get_loc(new_col)] = df[col][ind]
    df_new.drop(['participant__r', 'condition__r', 'coder__r', 'participant__t', 'condition__t', 'coder__t'],
                axis=1, inplace=True)
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
    df_new['child gender'] = 0
    for ind in df_new.index:
        print(ind)
        participant = df_new._get_value(ind, 'participant')
        ind_participant = int(participant) - 100 - 1
        gender = qualtrics._get_value(ind_participant, 'child gender')
        df_new._set_value(ind, 'child gender', gender)
        #df_new.at[ind, 'child gender'] = gender

    #df_new = pd.concat([qualtrics, df_new], axis=1)
    return df_new

def create_zero_df(features, row_num):
    df = pd.DataFrame(columns=features,
                      index=range(1, row_num+1)).fillna(0)
    return df


def granger_condition_tests(df, test_list):
    lag = 10
    df_granger_session = pd.DataFrame()
    for test in test_list:
        col1 = test[0]
        col2 = test[1]
        result = granger(df, col1, col2, maxlag=lag)
        #result_str = str(result[0])+','+str(result[1])

        df_granger_session[('granger_'+col1+'_'+col2)] = [result] #[result]
    return df_granger_session


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
    cols = list(df)
    for col in cols:
        if col not in features:
            df = df.drop([col], axis=1)
    df['condition'] = condition
    df['participant'] = participant
    df['order'] = order
    return df


if __name__ == '__main__':
    from variables import count_features, granger_features, granger_condition_list, granger_robot_tests, robot_vs_tablet
    #parameters
    interval = 0.5 #time interval between time steps
    #which modules of analysis to run
    run_granger = 0
    run_count = 1
    stitch_buffer = 10 # number of rows to buffer between stitching time series from different sessions

    #load data
    files = os.listdir("files")
    #set window paramaters
    col_base, action_base, col_count, action_count = "action","Child gaze","action",'all'
    df_count_row_all = pd.DataFrame()
    df_granger_robot = pd.DataFrame()
    df_time_all = pd.DataFrame()
    df_time_tablet = pd.DataFrame()
    df_time_robot = pd.DataFrame()
    df_zeros = create_zero_df(granger_features, stitch_buffer)
    for file in files:
        output_folder = "output"
        file_time = time.time()
        print('new file = ', file_time)
        file_base = file[:-4]
        file_split = file_base.split("_")
        lesson_split = list(file_split[0])
        condition = lesson_split[2]
        path = os.path.join(output_folder,file_base)
        path_out = os.path.join(output_folder)
        print(path_out)
        # re creates folder
        if os.path.exists(path):
            rmtree(path)
        os.makedirs(path)

        file_path = os.path.join("files",file)
        df = pd.read_csv(file_path, sep='\t', engine='python',header = None)
        df = df_preprocess(df, file_base) # make a df of the raw data

        #if run_count == 1:
        pd.options.display.max_columns = 10
        print(f"made csv for {file_base}")
        pd.concat([make_crosstab(df,"action"),make_crosstab(df,"sub_action"),make_crosstab(df,"action:sub_action")]).to_csv(os.path.join(path,f"{file_base} sum_count.csv"))
        df_count = pd.concat([make_crosstab(df,"action"),make_crosstab(df,"sub_action"),make_crosstab(df,"action:sub_action")])
        #transform to row
        df_count_row = df_count.stack(level=0)
        df_count_row = add_remove_features(df_count_row, count_features, file_base)
        df_count_row = convert_labels(df_count_row)
        df_count_row = add_object_features_row(df_count_row)
        df_count_row = remove_count_features(df_count_row, robot_vs_tablet)

        df_time_action = transform_to_time_representation(df, "action", interval)
        df_time_action.drop(['time'], axis=1, inplace = True)
        df_time_sub_action_sub_action = transform_to_time_representation(df, "action:sub_action", interval)
        df_time_sub_action_sub_action.drop(['time'], axis=1, inplace = True)
        df_time = pd.concat([df_time_action, df_time_sub_action_sub_action], axis=1)
        df_time = add_remove_features_granger(df_time, granger_features)
        df_time = add_object_features_time(df_time)
        if condition == 'r':
            df_granger_session_robot = granger_condition_tests(df_time, granger_robot_tests)
            df_granger_robot = pd.concat([df_granger_robot, df_granger_session_robot], ignore_index=True)
        df_granger_session = granger_condition_tests(df_time, granger_condition_list)

        df_count_row = pd.concat([df_granger_session, df_count_row], axis=1)
        df_count_row_all = pd.concat([df_count_row_all, df_count_row], ignore_index=True)

        df_time = pd.concat([df_time, df_zeros], ignore_index=True)
        df_time_all = pd.concat([df_time_all, df_time], ignore_index=True)
        if condition == 'r':
            df_time_robot = pd.concat([df_time_robot, df_time], ignore_index=True)
        elif condition == 't':
            df_time_tablet = pd.concat([df_time_tablet, df_time], ignore_index=True)
        #df_time_sub_action_sub_action = drop_features(df_time_sub_action_sub_action)
        path_file_base = os.path.join(path,file_base)
        print(f"{path_file_base} action time rep.csv", time.time()-file_time)
        print(f"made time representation for {file_base}", time.time()-file_time)
        #all_windows_df = all_windows(df,1,5)
        #all_windows_df.to_csv(f"{path_file_base} windows.csv")
    #df_count_row_all = add_object_features_row(df_count_row_all)
    #df_count_row_all.to_csv(os.path.join(path_out, "df_by_session_1.csv"))
    #df_count_row_all = session_to_participant(df_count_row_all)
    #df_count_row_all = add_derived_features(df_count_row_all)
    df_count_row_all = add_qualtrics_data_1(df_count_row_all)
    df_count_row_all.to_csv(os.path.join(path_out, "df_by_session_lag10_.csv"))
    df_granger_robot.to_csv(os.path.join(path_out, "df_granger_robot_session_lag10_.csv"))
    #df_granger_robot = granger_condition_tests(df_time_robot, granger_robot_tests)
    #df_count_row_all.to_csv(os.path.join(path_out,"df_count_all_int_05.csv"))
    #df_granger_robot.to_csv(os.path.join(path_out, "df_granger_robot_int_05.csv"))



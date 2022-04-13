# -*- coding: utf-8 -*-
"""
Spyder Editor

This is a temporary script file.
"""

import pandas as pd
import os
import numpy as np
from hmmlearn import hmm


all_files = os.listdir('/home/data/stock/StockData/market_data/daily_data/')
all_files = [i for i in all_files if '.h5' in i]
daily_list = []
for i in all_files:
    daily_data = pd.read_hdf('/home/data/stock/StockData/market_data/daily_data/'+i)
    daily_list.append(daily_data[daily_data['ticker']=='002961'])
stock_data = pd.concat(daily_list)
stock_data = stock_data.sort_values('tradeDate')
stock_data = stock_data.set_index('tradeDate')
stock_data = stock_data[['turnoverVol','openPrice','closePrice']]
stock_data['last_closePrice'] = [np.nan] + list(stock_data['closePrice'][:-1])
stock_data['u_d_rate'] = stock_data['closePrice']/stock_data['last_closePrice']
stock_data = stock_data.dropna()
stock_data = stock_data[['turnoverVol','openPrice','closePrice','u_d_rate']]


'''
单次训练模型
三种状态 涨、跌、震荡，按照收益排序确定即可，而收益值是否符合实际暂时忽略
隐状态映射确认
'''
def train_and_get_states(data, n_iter = 20):
    max_model = None
    max_prob = -np.inf
    #训练三次取似然概率最大的为当前可信模型
    for i in range(3):
        model = hmm.GaussianHMM(n_components=3, covariance_type="diag",n_iter = n_iter)
        model.fit(data)
        #logp 很小 导致p接近0，因此直接拿logp进行比较  p = np.e**(model.score(data))
        logp = model.score(data)
        if logp > max_prob:
            max_prob = logp
            max_model = model
    logprob, states = max_model.decode(data, algorithm="viterbi")
    clear_state_list = []
    for state_id in range(len(max_model.transmat_)):
        reward = data[states==state_id][:,3].prod()
        clear_state_list.append((state_id,reward))
    clear_state_list.sort(key = lambda x:x[1],reverse = True)
    return max_model, clear_state_list, states

'''
策略回测
'''
total_data = stock_data.values
period = 365
total_length = len(total_data)

origin_money = 1
repositioy_state = 0 #0空仓 1满仓

for i in range(total_length - period):

    once_data = total_data[i:i + period]
    model, clear_state_list, states_list = train_and_get_states(once_data, 20)
    state_up = clear_state_list[0][0]
    state_stable = clear_state_list[1][0]
    state_down = clear_state_list[2][0]
    
    state_up_jump = np.argmax(model.transmat_[state_up])
    state_stable_jump = np.argmax(model.transmat_[state_stable])
    state_down_jump = np.argmax(model.transmat_[state_down])
    
    print("state_up is {} and next jump state is {}".format(state_up, state_up_jump))
    print("state_stable is {} and next jump state is {}".format(state_stable, state_stable_jump))
    print("state_down is {} and next jump state is {}".format(state_down, state_down_jump))
    
    today_state = states_list[-1]
    next_state = np.argmax(model.transmat_[today_state])
    if next_state == state_up:
        '''
        收盘前买入，收益根据第二天情况计算，空仓和满仓都跟着第二天收益计算
        '''
        if repositioy_state == 0:
            #扣除买入手续费
            pass
        origin_money = origin_money*total_data[i + period][3]
        repositioy_state = 1
    elif next_state == state_down:
        '''
        收盘前卖出，收益以当前为主，空仓和满仓收益都不考虑第二天收益
        '''
        if repositioy_state == 1:
            #扣除卖出手续费
            pass
        repositioy_state = 0
    else:
        '''
        横盘，此时不做操作，空仓无变化，满仓跟着第二天收益计算
        '''
        if repositioy_state == 1:
            origin_money = origin_money*total_data[i + period][3]
    
final_reward = '{:.2f}%'.format((origin_money-1)*100) 
print('回测收益为:{}'.format(final_reward))
    
    
        
    










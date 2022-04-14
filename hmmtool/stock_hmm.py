# -*- coding: utf-8 -*-
"""
Spyder Editor

This is a temporary script file.
"""

import pandas as pd
import os
import numpy as np
from hmmlearn import hmm


all_files = os.listdir('G:\data\daily\daily_data')
all_files = [i for i in all_files if '.h5' in i and i > '2019']
daily_list = []
for i in all_files:
    daily_data = pd.read_hdf('G:\data\daily\daily_data'+'\\'+ i)
    #daily_list.append(daily_data[daily_data['ticker']=='600764'])
    daily_list.append(daily_data[daily_data['ticker']=='002961'])
stock_data = pd.concat(daily_list)
stock_data = stock_data.sort_values('tradeDate')
stock_data = stock_data.set_index('tradeDate')
stock_data = stock_data[['turnoverVol','openPrice','closePrice']]
stock_data['last_closePrice'] = [np.nan] + list(stock_data['closePrice'][:-1])
stock_data['u_d_rate'] = stock_data['closePrice']/stock_data['last_closePrice']
stock_data = stock_data.dropna()
stock_data = stock_data[['turnoverVol','openPrice','closePrice','u_d_rate']]
stock_data.index = pd.to_datetime(stock_data.index)

'''
单次训练模型
三种状态 涨、跌、震荡，按照收益排序确定即可，而收益值是否符合实际暂时忽略
隐状态映射确认
'''
def train_and_get_states(data, n_iter = 20):
    max_model = None
    max_prob = -np.inf
    #训练三次取似然概率最大的为当前可信模型
    for i in range(4):
        model = hmm.GaussianHMM(n_components=3, covariance_type="full",n_iter = n_iter)
        #model = hmm.GMMHMM(n_components=3, n_mix=1, covariance_type="diag", n_iter = n_iter)
        model.fit(data)
        #logp 很小 导致p接近0，因此直接拿logp进行比较  p = np.e**(model.score(data))
        logp = model.score(data)
        if logp > max_prob:
            max_prob = logp
            max_model = model
    logprob, states = max_model.decode(data, algorithm="viterbi")
    clear_state_list = []
    for state_id in range(len(max_model.transmat_)):
        reward = data[states==state_id][:,2].prod()
        clear_state_list.append((state_id,reward))
    clear_state_list.sort(key = lambda x:x[1],reverse = True)
    return max_model, clear_state_list, states

'''
策略回测
'''
total_data = stock_data.values[:,[0,2,3]]
period = 100
total_length = len(total_data)

origin_money = 1
repositioy_state = 0 #0空仓 1满仓

buy_times = 0
sell_times = 0

x = []
y = []
c = []

for i in range(total_length - period):

    once_data = total_data[i:i + period]
    model, clear_state_list, states_list = train_and_get_states(once_data, 30)
    state_up = clear_state_list[0][0]
    state_stable = clear_state_list[1][0]
    state_down = clear_state_list[2][0]
    
    state_up_jump = np.argmax(model.transmat_[state_up])
    state_stable_jump = np.argmax(model.transmat_[state_stable])
    state_down_jump = np.argmax(model.transmat_[state_down])
    
    #print("state_up is {} and next jump state is {}".format(state_up, state_up_jump))
    #print("state_stable is {} and next jump state is {}".format(state_stable, state_stable_jump))
    #print("state_down is {} and next jump state is {}".format(state_down, state_down_jump))
    
    today_state = states_list[-1]
    next_state = np.argmax(model.transmat_[today_state])
    
    '''
    if i == 0:
        state_map = {state_up:0, state_stable:1, state_down:2}
        x.extend(stock_data.index[i:i + period])
        y.extend(stock_data['closePrice'].values[i:i + period])
        c.extend([state_map[s] for s in states_list])
        
        #logprob, states = model.decode(total_data[i + period:], algorithm="viterbi")
        #states_list = np.concatenate([states_list,states])
        #x.extend(stock_data.index)
        #y.extend(stock_data['closePrice'].values)
        #c.extend([state_map[s] for s in states_list])
    '''
    
    #预测状态是下一天的，执行的动作是在当天执行，因此看图的时候，下一天的状态颜色标记在当天上可以反映当天的收盘动作
    x.append(stock_data.index[i + period - 1])
    y.append(stock_data['closePrice'].values[i + period - 1])
    
    if next_state == state_up:
        '''
        收盘前买入，收益根据第二天情况计算，空仓和满仓都跟着第二天收益计算
        '''
        if repositioy_state == 0:
            #扣除买入手续费
            origin_money = origin_money * (1-0.0008)
            buy_times += 1
        origin_money = origin_money*total_data[i + period][2]
        repositioy_state = 1
        c.append(0)
    elif next_state == state_down:
        '''
        收盘前卖出，收益以当前为主，空仓和满仓收益都不考虑第二天收益
        '''
        if repositioy_state == 1:
            #扣除卖出手续费
            origin_money = origin_money * (1-0.0008)
            sell_times += 1
        repositioy_state = 0
        c.append(2)
    else:
        '''
        横盘，此时不做操作，空仓无变化，满仓跟着第二天收益计算
        '''
        if repositioy_state == 1:
            origin_money = origin_money*total_data[i + period][2]
        c.append(1)
    print("{} profit is {}".format(stock_data.index[i + period], origin_money))
    
final_reward = '{:.2f}%'.format((origin_money-1)*100) 
print("买入次数：{}".format(buy_times))
print("卖出次数：{}".format(sell_times))
print('回测收益为:{}'.format(final_reward))



'''
画图
'''
import matplotlib.pyplot as plt
    
plt.figure(figsize=(12,6),dpi = 100)
plt.plot(x,y,linewidth=0.5)
s = [5 for i in c]
plt.scatter(x, y, s = s, c = c)
plt.show
    










# -*- coding: utf-8 -*-
"""
@author: Jin Zitian
@time: 2022-2-18 16:55
"""

import numpy as np     


class HMM(object):
    
    def __init__(self, state_num, observe_num):
        self.s_n = state_num
        self.o_n = observe_num
        self.A = np.random.rand(self.s_n, self.s_n)
        self.A = self.A/self.A.sum(axis = 1).reshape([-1,1])
        self.B = np.random.rand(self.s_n, self.o_n)
        self.B = self.B/self.B.sum(axis = 1).reshape([-1,1])
        self.pi = np.random.rand(self.s_n)
        self.pi = self.pi/sum(self.pi)
        self.train_data = []
        
    #input_data 格式为 [[o1,o2,o3,...,ox], [o1,o2,o3,...,oy]]
    #支持多观测序列输入,观测序列的输入是index化后的值，需要提前根据实际的观测值字典去映射，解码出的隐状态值也是一样，都是index数据，需要再根据映射还原
    def add_data(self, input_data):
        self.train_data.extend(input_data)
    
    #计算所有的前向概率
    # [[o1,o2,o3,...,ot1], [o1,o2,o3,...,ot2]]
    # [t1 * s_n, t2 * s_n]
    def forward(self, o_seqs):
        self.alpha = [] 
        for seq in o_seqs:
            alpha = np.zeros((len(seq),self.s_n))
            for i in range(self.s_n):
                alpha[0,i] = self.pi[i] * self.B[i,seq[0]]
            for r in range(1,len(seq)):
                for i in range(self.s_n):
                    alpha[r,i] = sum([alpha[r-1,j]*self.A[j,i] for j in range(self.s_n)])*self.B[i,seq[r]]
            self.alpha.append(alpha)
            
    #计算所有的后向概率
    # [[o1,o2,o3,...,ot1], [o1,o2,o3,...,ot2]]
    # [t1 * s_n, t2 * s_n]
    def backward(self, o_seqs):
        self.beta = []
        for seq in o_seqs:
            beta = np.zeros((len(seq),self.s_n))
            for i in range(self.s_n):
                beta[len(seq)-1,i] = 1
            for r in range(len(seq)-2,-1,-1):
                for i in range(self.s_n):
                    beta[r,i] = sum([self.A[i,j]*self.B[j,seq[r+1]]*beta[r+1,j] for j in range(self.s_n)])
            self.beta.append(beta)
        
    #给定模型参数和观测序列，时刻t的状态为xx的概率
    # t * s_n
    # 多条观测序列输入 则为[t1 * s_n, t2 * s_n, ... , tk * s_n]
    def gamma_matrix(self):
        self.gamma = []
        for i in range(len(self.alpha)):
            alpha = self.alpha[i]
            beta = self.beta[i]
            self.gamma.append(alpha*beta/sum(alpha[len(alpha)-1]))
        
    #给定模型参数和观测序列，时刻t的状态为xx，且t+1的状态为yy的概率
    # t * s_n * s_n
    # 多条观测序列输入 则为[t1-1 * s_n * s_n, t2-1 * s_n * s_n, ... , tk-1 * s_n * s_n]
    def ksi_matrix(self):
        self.ksi = []
        for i in range(len(self.train_data)):
            seq = self.train_data[i]
            alpha = self.alpha[i]
            beta = self.beta[i]
            ksi = np.zeros((len(seq)-1, self.s_n, self.s_n))
            for t in range(len(seq)-1):
                for i in range(self.s_n):
                    for j in range(self.s_n):
                        ksi[t,i,j] = alpha[t,i]*self.A[i,j]*self.B[j,seq[t+1]]*beta[t+1,j]/sum(alpha[len(alpha)-1])
            self.ksi.append(ksi)
            
    #EM思想 Baum-Welch算法
    def train(self, maxStep = 10, delta = 0.01):
        step = 0
        while step < maxStep:
            print("=============== step {} ===============".format(step))
            #固定模型参数计算隐含数据
            self.forward(self.train_data)
            self.backward(self.train_data)
            self.gamma_matrix()
            self.ksi_matrix()
            #固定隐含数据计算模型参数
            new_pi = sum([gamma[0] for gamma in self.gamma])/len(self.gamma)
            new_A = sum([ksi.sum(axis = 0) for ksi in self.ksi])/np.reshape(sum([gamma[:-1].sum(axis = 0) for gamma in self.gamma]), [-1,1])
            sn_on_list = []
            for i in range(len(self.train_data)):
                seq = np.array(self.train_data[i])
                gamma = self.gamma[i]
                sn_on = []
                for o in range(self.o_n):
                    sn_o = (np.reshape(seq == o, [-1,1]) * gamma).sum(axis = 0).reshape([-1,1])
                    sn_on.append(sn_o)
                sn_on_list.append(np.concatenate(sn_on,axis = 1))
            new_B = sum(sn_on_list)/np.reshape(sum([gamma.sum(axis = 0) for gamma in self.gamma]), [-1,1])
            #误差小也停止
            pi_error = np.sum(np.square(new_pi - self.pi))
            A_error = np.sum(np.square(new_A - self.A))
            B_error = np.sum(np.square(new_B - self.B))
            print("pi_error is {}".format(pi_error))
            print("A_error is {}".format(A_error))
            print("B_error is {}".format(B_error))
            if pi_error < delta and A_error < delta and B_error < delta:
                self.pi = new_pi
                self.A = new_A
                self.B = new_B
                break
            self.pi = new_pi
            self.A = new_A
            self.B = new_B
            step += 1
    
    #viterbi算法
    #单条输入：[[o1,o2,o3,...,ot1]]
    #多条输入：[[o1,o2,o3,...,ot1],[o1,o2,o3,...,ot2]]
    #输出：[(prob1, [s1,s2,s3,...,st1]), (prob2, [s1,s2,s3,...,st2])]
    def decode(self, o_seq):
        result = []
        for i in range(len(o_seq)):
            seq = o_seq[i]
            last_max_state = [[-1]*self.s_n]
            max_state_prob_now = [self.pi[s]*self.B[s,seq[0]] for s in range(self.s_n)]
            for o in seq[1:]:
                current_last_max_state = [0]*self.s_n
                max_state_prob_new = [0]*self.s_n
                for ns in range(self.s_n):
                    candidates = [max_state_prob_now[bs]*self.A[bs,ns]*self.B[ns,o] for bs in range(self.s_n)]
                    max_index = np.argmax(candidates)
                    current_last_max_state[ns] = max_index
                    max_state_prob_new[ns] = candidates[max_index]
                last_max_state.append(current_last_max_state)
                max_state_prob_now = max_state_prob_new
            #状态回溯
            hidden_state = []
            current_state = np.argmax(max_state_prob_now)
            max_prob = max_state_prob_now[current_state]
            hidden_state.append(current_state)
            for current_t in range(len(seq)-1,0,-1):
                current_state = last_max_state[current_t][current_state]
                hidden_state.append(current_state)
            result.append((max_prob, hidden_state[::-1]))
        return result
            
    #计算概率 P(O|λ)
    #单条[[o1,o2,o3,...,ot1]]
    #多条[[o1,o2,o3,...,ot1],[o1,o2,o3,...,ot2]]
    #输出：[prob1, prob2]
    def estimate_prob(self, o_seq):
        self.forward(o_seq)
        result = []
        for alpha in self.alpha:
            result.append(sum(alpha[len(alpha)-1]))
        return result
        
    
if __name__ == '__main__':
    s1 = np.random.randint(6,size = 60)
    s2 = np.random.randint(6,size = 40)
    s = np.concatenate([s1,s2])
    sh = s.reshape([-1,1])
    myhmm = HMM(3,6)
    myhmm.add_data([s])
    myhmm.train(maxStep=50,delta=0.001)
    print(myhmm.pi)
    print(myhmm.A)
    print(myhmm.B)
    print(myhmm.estimate_prob([s]))
    
    
    from hmmlearn import hmm
    
    model = hmm.MultinomialHMM(n_components=3, n_iter=50, tol=0.01)
    model.fit(sh,lengths=[60,40])
    print(model.startprob_)
    print(model.transmat_)
    print(model.emissionprob_)
    print(np.e**model.score(sh))
    
    '''
    model = hmm.MultinomialHMM(n_components=3)
    model.startprob_=myhmm.pi
    model.transmat_=myhmm.A
    model.emissionprob_=myhmm.B
    '''
    
    ss = np.random.randint(6,size = 14)
    max_hidden_prob, hidden_state = myhmm.decode([ss])[0]
    print(max_hidden_prob, hidden_state)
    o_prob = myhmm.estimate_prob([ss])[0]
    print(o_prob)
    
    d = model.decode(ss.reshape([-1,1]), algorithm="viterbi")
    max_hidden_prob, hidden_state = np.e**d[0], list(d[1])
    print(max_hidden_prob, hidden_state)
    o_prob = np.e**(model.score(ss.reshape([-1,1])))
    print(o_prob)
        
        
        


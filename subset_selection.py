# -*- coding: utf-8 -*-
"""
Created on Sat Jan 14 15:21:39 2023

@author: jiay
"""

import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.datasets import load_boston
from itertools import combinations

class SubsetSelection(object):
    
    def __init__(self, k):
        '''
            k: a hyperparameter  
        '''
        self.k = k
        boston=load_boston()
        boston_df=pd.DataFrame(boston.data,columns=boston.feature_names)
        boston_df['Price']=boston.target
        xmat=boston_df.drop('Price',axis=1)
        y =boston_df['Price']
        self.X_list = xmat.columns 
        self.X_train,self.X_test,self.y_train,self.y_test= \
            train_test_split(xmat,y,test_size=0.2,random_state=3)

    def reg_model(self, variable_tuple):
        reg = LinearRegression()
        x = self.X_train[list(variable_tuple)]
        reg.fit(x, self.y_train)
        return {'model': variable_tuple, 'score':reg.score(x,self.y_train)}
    
    def optimal_subset(self):
        model_space = list(combinations(self.X_list, self.k)) 
        scores = list(map(self.reg_model, model_space))
        scores = sorted(scores, key=lambda x:x['score'], reverse=True) 
        return scores
    
    def forward_step(self):
        pass
    
    def backward_step(self):
        pass
    
    def forward_stage(self): 
        coeff = {k:0 for k in self.X_list}
        ytilt = self.y_train - self.y_train.mean()
        count = 0
        go = True
        while go:
            residual = ytilt
            if count > 0:
                for k,v in coeff.items():
                    residual = residual - v*self.X_train[k]
            out = [{'variable': col, 'corr': residual.corr(self.X_train[col])}
                   for col in self.X_list] 
            out = sorted(out, key=lambda x:abs(x['corr']), reverse=True) 
            v = out[0]['variable']
            model = LinearRegression()
            model.fit(self.X_train[[v]], residual)
            coeff[v] = coeff[v] + model.coef_[0]
            
            check = {k:v for k, v in coeff.items() if abs(v) > 0}
            if len(check) >= self.k:
                go = False
            count += 1
        return coeff

def validate_k(max_k):
    pass
    
    
if __name__ == '__main__':
    a = SubsetSelection(3)
    


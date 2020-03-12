#!/usr/bin/env python
# -*- coding: utf-8 -*-
import rospy
import numpy as np


class Calc(object):
    def __init__(self,):
        self.teta2 = 0.5
        self.teta3 = 0.15
        self.beta = 0.01
        self.xl = None
        self.cv = None
        self.p = None
        self.inv= None
        
    def conv(self, xi, xj):
        teta2 = self.teta2
        teta3 = self.teta3
        return teta2 + teta3 * xi * xj

    def learn(self, x, y):
        beta = self.beta
        cv = np.zeros([len(x),len(x)])
        for i in range(len(x)):
            for j in range(len(x)):
                cv[i,j] = self.conv(x[i],x[j])
                if i == j:
                    cv[i,j] += beta
        v = np.linalg.inv(cv)
        p = np.dot(v,np.array(y))    
        self.cv = cv
        self.inv = v
        self.p = p
        self.xl = x

    def predict(self, x):
        try:
            ns = len(self.xl)
        except:
            return None, None
        mu_l = []
        su_l = []
        for i in range(len(x)):
            v = np.zeros(ns)
            for j in range(ns):
                v[j] = self.conv(x[i],self.xl[j])
            c = self.conv(x[i],x[i]) + self.beta
            mu = np.dot(v,self.p)
            s = c - np.dot(v,np.dot(self.inv,v))
            mu_l.append(mu)
            su_l.append(s)
        return np.array(mu_l), np.array(su_l)
        
if __name__=="__main__":
    cl = Calc()
    
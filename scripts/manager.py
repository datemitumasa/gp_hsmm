#!/usr/bin/env python
# -*- coding: utf-8 -*-
import rospy
import numpy as np
import os
import sys

FILE = "class{0:03d}.npy"
POWH = 0.5
SIGH = 1.5
def mkdir(name):
    try:
        os.mkdir(name)
    except:
        pass
    return True

class Manger(object):
    def __init__(self,folda_name, category):
        """
        fold_name is folda directory : str
        category is object type , directory names : list
        """
        self.file = folda_name
        self.category = category
        self.data_set = []
        self.pow_dic = {}
        for c in self.category:
            name = self.file +"/"+ c + "/"
            i = 0
            d = []
            while not rospy.is_shutdown():
                try:
                    data = np.load(name+FILE.format(i),allow_pickle=True)
                    i += 1
                    d.append(data)
                except:
                    break
            self.data_set.append(d)

        for i in range(len(self.category)):
            ave_lens = self.check_avelen(self.data_set[i])
            print self.category[i]
            n = len(ave_lens)
            c_dic = {}
            for j in range(n):
#                print "class{}".format(j)
                dic = {}
                dic["avelen"] = ave_lens[j]
                c_dic[j] = dic
            self.pow_dic[self.category[i]] = c_dic

    def get_actioninfo(self,category, classnum):
        d = self.pow_dic[category]
        cd = d[classnum]
        length = cd["avelen"]
        return None, length

    def show_data(self,):
        rospy.loginfo("show power data")
        print "##########"
        for c in self.category:
            print c
            dic = self.pow_dic[c]
            n = len(dic)
            for i in range(n):
                print "class {}".format(i)
                cdic = dic[i]
                print "average length", cdic["avelen"]
            print "---"
        print "##########"
        
    def check_avelen(self,classes):
        avelen = []
        for c in classes:
            ls = []
            for d in c:
                ls.append(len(d))
            ave = np.average(ls)
            avelen.append(ave)
        return avelen

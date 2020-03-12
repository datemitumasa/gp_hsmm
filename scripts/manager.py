#!/usr/bin/env python
# -*- coding: utf-8 -*-
import rospy
import numpy as np
import os
import sys

FILE = "class{0:03d}.npy"
POWSET=["N","S","H"]
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
                    data = np.load(name+FILE.format(i))
                    i += 1
                    d.append(data)
                except:
                    break
            self.data_set.append(d)
            
        for i in range(len(self.category)):
            ave_lens = self.check_avelen(self.data_set[i])
            print self.category[i]
            avm_pows, sim_pows = self.check_power(self.data_set[i])
            n = len(ave_lens)
            c_dic = {}
            for j in range(n):
#                print "class{}".format(j)
                dic = {}
                dic["avelen"] = ave_lens[j]
                dic["pow"] = self.power_setter(avm_pows[j],sim_pows[j])
                dic["sig_param"] = sim_pows[j]
                dic["ave_param"] = avm_pows[j]
                c_dic[j] = dic
            self.pow_dic[self.category[i]] = c_dic
            
    def get_actioninfo(self,category, classnum):
        d = self.pow_dic[category]
        cd = d[classnum]
        length = cd["avelen"]
        pow_type = cd["pow"]
        if pow_type=="NNN":
            pow_type = None
        return pow_type, length

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
                print "power type(xyz): ", cdic["pow"]
                print "average_power [x,y,z]: ",cdic["ave_param"]
                print "sig_power [x,y,z]: ",cdic["sig_param"]
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
        
    def check_power(self,classes):
        avepow = []
        sigpow = []
        for c in classes:
            ml = []
            for d in c:
                ml.append(len(d))
            if len(ml)==0:
                m = 0
                avepow.append([np.nan,np.nan,np.nan])
                sigpow.append([np.nan,np.nan,np.nan])                
            else:
                m = np.max(ml)
                _avepow = []
                _sigpow = []
                print "###"
                for i in range(m):
                    x = []
                    y = []
                    z = []
                    for d in c:
                        try:
                            x.append(d[i][8]/10.0)
                            y.append(d[i][9]/10.0)
                            z.append(d[i][10]/10.0)
                        except:
                            continue
                    xn = np.average(x)
                    yn = np.average(y)
                    zn = np.average(z)
                    xs = np.var(x)
                    ys = np.var(y)
                    zs = np.var(z)
                    _avepow.append([xn,yn,zn])
                    _sigpow.append([xs,ys,zs])
                
                avmx = np.max(np.abs(np.array(_avepow)[:,0]))
                avmy = np.max(np.abs(np.array(_avepow)[:,1]))
                avmz = np.max(np.abs(np.array(_avepow)[:,2]))
                simx = np.max(np.array(_sigpow)[:,0])
                simy = np.max(np.array(_sigpow)[:,1])
                simz = np.max(np.array(_sigpow)[:,2])
                avepow.append([avmx,avmy,avmz])
                sigpow.append([simx,simy,simz])
        return avepow, sigpow
        
        
    def power_setter(self,ave_pow, sig_pow):
        type_set = POWSET
        name = ""
        for i in range(len(ave_pow)):
            print ave_pow[i],
            if ave_pow[i] > POWH:
                if sig_pow[i] > SIGH:
                    name += "H"
                else:
                    name += "N"
            else:
                name += "S"
        print "##"
        return name
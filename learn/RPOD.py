#!/usr/bin/env python
# encoding: utf8
#from __future__ import unicode_literals
import numpy as np
from RPGPHSMMs import GPSegmentation
from dataframe import Objects
import pandas as pd
import multiprocessing
import sys
sys.setrecursionlimit(10000)
OBJ = {"test_object":["recognized_object/exp_object/7"]}

CSVNAME=[

         ]

CSVCOLMN = ["x", "y", "z", "qx", "qy", "qz", "qw", "hand"]

TIMENAME = "time"
DATAFRAME = "./csv_data/marge_dataframe.csv"
TIMETHRED  = 15
DISTANCETHRED = 0.5
MAXDISTANCETHRED = 1.5
TIMESPARSE = 1.
SAVE = "save/{}/"

def dataframe_downsampling(frame):
    t = frame[TIMENAME]
    st = t[0]
    droplist = []
    for i in range(1, len(t)):
        if t[i] - st < TIMESPARSE:
            droplist.append(i)
        else:
            st = t[i]
    ff = frame.drop(droplist)
    return ff

class Manager(object):
    def __init__(self,):
        self.series = {}
        self.objects_data = {}
        objects = Objects(DATAFRAME)
        keys = OBJ.keys()
        for k in keys:
            self.objects_data[k] = list(objects.get_objects_data(OBJ[k]))
        self.objects_df = {}
        self.load_csv_data()
        self.i = 0
        
    def set_gp_data(self,):
        keys     = OBJ.keys()
        self.GPs = []
        for k in keys:
            ser  = self.series[k]
            df   = self.objects_df[k]
            gp   = GPSegmentation(ser, df, k)
            self.GPs.append(gp)
        
    def multi_learn(self,):
        for i in range(len(self.GPs)):
            self.GPs[i]._set_state()
        jobs     = []
        for i in range(len(self.GPs)):
            job  = multiprocessing.Process(target=self.GPs[i].learn_start(SAVE.format(self.i)), args=(i,))
            jobs.append(job)
            job.start()
        [job.join() for job in jobs]
        self.i += 1
        
        
    def load_csv_data(self):
        dfs = []
        for name in CSVNAME:
            _df =pd.read_csv(name)
            df = dataframe_downsampling(_df)
            dfs.append(df)
        keys = OBJ.keys()
        for k in keys:
            poses, time, _name, _ids = self.objects_data[k]
            of = pd.DataFrame()
            of["time"] = time
            of["pose"] = poses
            of["id"] = _ids
            of["name"] = _name
            datalist = []
            objectlist = []
            for df in dfs:
                t = df.time.values
                odf = of.loc[(of.time <= t[-1]) & (of.time >= t[0])]
                op = odf.pose.values
                ot = odf.time.values
                oi = odf.id.values
                first_time = ot[0]
                for i in range(len(op)):
                    p = op[i]
                    pdf = df.loc[(df.time >= ot[i]) & (df.time <= ot[i]+TIMETHRED)]
                    x = pdf.x.values
                    y = pdf.y.values
                    z = pdf.z.values
                    distances = np.power((x - p[0])**2 + (y - p[1])**2 + (z - p[2])**2, 0.5)
                    dis = np.min(distances)                    
                    if dis > DISTANCETHRED:
                        continue
                    else:
                        objectlist.append(oi[i])

            odf = of.loc[of.id.isin(objectlist)]
            op = odf.pose.values
            ot = odf.time.values
            oi = odf.id.values
            first_time = ot[0]
            for df in dfs:
                t = df.time.values
                for i in range(len(ot)):
                    p = op[i]
                    if ot[i] < t[0]:
                        continue
                    if ot[i] > t[-1]:
                        break
                    if i != len(ot)-1:
                        time_thred = ot[i+1] - ot[i]
                        if time_thred > TIMETHRED:
                            pdf = df.loc[(df.time >= first_time) & (df.time <= ot[i] + TIMETHRED)]
                            x = pdf.x.values
                            y = pdf.y.values
                            z = pdf.z.values
                            distances = np.power((x - p[0])**2 + (y - p[1])**2 + (z - p[2])**2, 0.5)
                            n = range(len(distances))
                            n.reverse()
                            for j in n:
                                if distances[j] <= MAXDISTANCETHRED:
                                    break
                            pdf = pdf[0:j+1]

                            pdf = pdf.reset_index()
                            del pdf["index"]
                            
                            first_time = ot[i+1]
                            datalist.append(pdf)
                    else:
                        pdf = df.loc[(df.time >= first_time) & (df.time <= ot[i] + TIMETHRED)]
                        x = pdf.x.values
                        y = pdf.y.values
                        z = pdf.z.values
                        distances = np.power((x - p[0])**2 + (y - p[1])**2 + (z - p[2])**2, 0.5)
                        n = range(len(distances))
                        n.reverse()
                        for j in n:
                            if distances[j] <= MAXDISTANCETHRED:
                                break
                        pdf = pdf[0:j+1]

                        
                        pdf = pdf.reset_index()
                        del pdf["index"]
                        datalist.append(pdf)
            odf = odf.reset_index()
            self.objects_df[k] = odf
            self.series[k] = datalist
            
                    


if __name__ == '__main__':
    test = Manager()
    test.set_gp_data()
    for i in range(20):
        test.multi_learn()
        
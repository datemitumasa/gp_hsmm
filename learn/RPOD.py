#!/usr/bin/env python
# encoding: utf8
#from __future__ import unicode_literals
import numpy as np
from RP import RPGPHSMM
from dataframe import Objects
import pandas as pd
import multiprocessing
import yaml
import sys
sys.setrecursionlimit(10000)


def dataframe_downsampling(frame, param):
    time_name = param["time_name"]
    t = frame[time_name]
    st = t[0]
    droplist = []
    for i in range(1, len(t)):
        if t[i] - st < param["data_time_sparse"]:
            droplist.append(i)
        else:
            st = t[i]
    ff = frame.drop(droplist)
    return ff

class RPOD(object):
    def __init__(self,):
        self.series = {}
        self.objects_df = {}
        self.GPs = []
        _path =  __file__.split("/")[:-1]
        self.path  = "/".join(_path) + "/"
        f = open(self.path + "../config/gp_hsmm_parameter.yaml", "r+")
        self.param = yaml.load(f)
        f.close()
        self.categorys = self.param.keys()
        try:
            self.categorys.remove("gp_hsmm_parameter")
        except:
            pass
        for cat in self.categorys:
            self.load_data(cat)
            self.set_learn_data(cat)

    def load_data(self, category):
        objects = Objects(self.path+ self.param[category]["object_csvdata"])
        _csv_data = self.param[category]["continuous_csvdata"]
        continuous_dataframe = []
        for csv in _csv_data:
            csv_data = pd.read_csv(self.path + csv)
            csv_data  = dataframe_downsampling(csv_data, self.param[category])
            continuous_dataframe.append(csv_data)
        poses, time, _name, _ids = objects.get_objects_data(self.param[category]["category"])
        of = pd.DataFrame()
        of["time"] = time
        of["pose"] = poses
        of["id"] = _ids
        of["name"] = _name
        datalist = []
        objectlist = []
        for jj in range(len(continuous_dataframe)):
        # for df in continuous_dataframe:
            df = continuous_dataframe[jj]
            t = df.time.values
            odf = of.loc[(of.time <= t[-1]) & (of.time >= t[0])]
            print len(odf)
            op = odf.pose.values
            ot = odf.time.values
            oi = odf.id.values
            try:
                first_time = ot[0]
                print "load {}".format(_csv_data[jj].encode("utf-8"))
            except:
                continue
            for i in range(len(op)):
                p = op[i]
                pdf = df.loc[(df.time >= ot[i]) & (df.time <= ot[i]+self.param[category]["time_thread"])]
                x = pdf.x.values
                y = pdf.y.values
                z = pdf.z.values
                distances = np.power((x - p[0])**2 + (y - p[1])**2 + (z - p[2])**2, 0.5)
                dis = np.min(distances)
                if dis > self.param[category]["distance_thread"]:
                    continue
                else:
                    objectlist.append(oi[i])

        odf = of.loc[of.id.isin(objectlist)]
        op = odf.pose.values
        ot = odf.time.values
        oi = odf.id.values
        first_time = ot[0]
        count = 0
        for jj in range(len(continuous_dataframe)):
        # for df in continuous_dataframe:
            df = continuous_dataframe[jj]
            count += 1
            t = df.time.values
            _odf = odf.loc[((odf.time >= np.min(t))&(odf.time <= np.max(t)))]
            op = _odf.pose.values
            ot = _odf.time.values
            oi = _odf.id.values
            try:
                first_time = ot[0]
            except:
                continue


            for i in range(len(ot)):
                p = op[i]
                if ot[i] < t[0]:
                    continue
                if ot[i] > t[-1]:
                    break
                if i != len(ot)-1:
                    time_thread = ot[i+1] - ot[i]
                    if time_thread > self.param[category]["time_thread"]:
                        pdf = df.loc[(df.time >= first_time) & (df.time <= ot[i] + self.param[category]["time_thread"])]
                        x = pdf.x.values
                        y = pdf.y.values
                        z = pdf.z.values
                        distances = np.power((x - p[0])**2 + (y - p[1])**2 + (z - p[2])**2, 0.5)
                        n = range(len(distances))
                        n.reverse()
                        for j in n:
                            if distances[j] <= self.param[category]["max_distance_thread"]:
                                break
                        pdf = pdf[0:j+1]

                        pdf = pdf.reset_index()
                        del pdf["index"]

                        first_time = ot[i+1]
                        datalist.append(pdf)
                else:
                    pdf = df.loc[(df.time >= first_time) & (df.time <= ot[i] + self.param[category]["time_thread"])]
                    x = pdf.x.values
                    y = pdf.y.values
                    z = pdf.z.values
                    distances = np.power((x - p[0])**2 + (y - p[1])**2 + (z - p[2])**2, 0.5)
                    n = range(len(distances))
                    n.reverse()
                    for j in n:
                        if distances[j] <= self.param[category]["max_distance_thread"]:
                            break
                    pdf = pdf[0:j+1]
                    pdf = pdf.reset_index()
                    del pdf["index"]
                    datalist.append(pdf)
        odf = odf.reset_index()
        self.objects_df[category] = odf
        self.series[category] = datalist

    def set_learn_data(self, category):
        ser  = self.series[category]
        df   = self.objects_df[category]
        gp   = RPGPHSMM(category, self.param[category])
        gp.load_data(ser, df)
        gp.set_gp_data()
        self.GPs.append(gp)

    def multi_learn(self,number):
        for i in range(len(self.GPs)):
            self.GPs[i].set_gp_data()
        jobs     = []
        for i in range(len(self.GPs)):
            job  = multiprocessing.Process(target=self.GPs[i].learn, args=(number,))
            job.start()
            jobs.append(job)
        [job.join() for job in jobs]



if __name__ == '__main__':
    rpod = RPOD()
    for i in range(10):
        rpod.multi_learn(i)
        print("{} learn finish".format(i +1))
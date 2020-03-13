#!/usr/bin/env python
# encoding: utf8
#from __future__ import unicode_literals
import numpy as np
from RPGPHSMMs import GPSegmentation
from dataframe import Objects
import pandas as pd
import sys
sys.setrecursionlimit(10000)


def dataframe_downsampling(frame, param):
    t = frame[param["time_name"]]
    st = t[0]
    droplist = []
    for i in range(1, len(t)):
        if t[i] - st < param["data_time_sparse"]:
            droplist.append(i)
        else:
            st = t[i]
    ff = frame.drop(droplist)
    return ff

class RPGPHSMM(object):
    def __init__(self,category="gp_hsmm_parametor", param=None):
        if param == None:
            _path =  __file__.split("/")[:-1]
            path  = "/".join(_path) + "/"
            f = open(path + "../config/parametor.yaml", "r+")
            param = yaml.load(f)[category]
            f.close()

        self.category            = category
        self.save                = "save/{}"
        self.parameter           = param
        self.series = []
        self.objects_df = pd.DataFrame()

    def load_data(self, continuous_dataframe, object_dataframe):
        self.series = continuous_csvdata
        self.objects_df = object_dataframe

    def set_gp_data(self,):
        ser  = self.series
        df   = self.objects_df
        gp   = GPSegmentation(ser, df, self.category, self.parameter)
        self.GP = gp

    def learn(self,number):
        self.GP._set_state()
        self.GP.learn_start(self.save.format(number))

if __name__ == '__main__':
    category = "gp_hsmm_parameter"
    _path =  __file__.split("/")[:-1]
    path  = "/".join(_path) + "/"
    f = open(path + "../config/parametor.yaml", "r+")
    param = yaml.load(f)[category]
    f.close()
    _csv_data = param["continuous_csvdata"]
    continuous_dataframe = []
    for csv in _csv_data:
        csv_data = pd.read_csv(path + csv)
        csv_data  = dataframe_downsampling(_csv_data, param)
        continuous_dataframe.append(csv_data)
    objects = Objects(path + param["object_csvdata"])

    poses, time, _name, _ids = objects.get_objects_data(param["category"])
    of = pd.DataFrame()
    of["time"] = time
    of["pose"] = poses
    of["id"] = _ids
    of["name"] = _name
    datalist = []
    objectlist = []
    for df in continuous_dataframe:
        t = df.time.values
        odf = of.loc[(of.time <= t[-1]) & (of.time >= t[0])]
        op = odf.pose.values
        ot = odf.time.values
        oi = odf.id.values
        first_time = ot[0]
        for i in range(len(op)):
            p = op[i]
            pdf = df.loc[(df.time >= ot[i]) & (df.time <= ot[i]+param["time_thred"])]
            x = pdf.x.values
            y = pdf.y.values
            z = pdf.z.values
            distances = np.power((x - p[0])**2 + (y - p[1])**2 + (z - p[2])**2, 0.5)
            dis = np.min(distances)                    
            if dis > param["distance_thread"]:
                continue
            else:
                objectlist.append(oi[i])

    object_dataframe = of.loc[of.id.isin(objectlist)]
    object_dataframe["id"] = range(len(object_dataframe))
    object_dataframe.reset_index()
    del object_dataframe["index"]


    for i in range(10):
        rpgphsmm = RPGPHSMM(category, param)
        rpgphsmm.load_data(continuous_dataframe, object_dataframe)
        rpgphsmm.set_gp_data()
        rpgphsmm.learn(i)
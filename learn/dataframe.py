#!/usr/bin/env python
# encoding: utf8
#from __future__ import unicode_literals
import numpy as np
import pandas as pd
import tf
DEFAULT = "./dataframe/marge_dataframes.csv"

class Objects(object):
    def __init__(self,name=None):
        
        if name == None:
            self.db = pd.read_csv(DEFAULT)
            print "use Default csv"
        else:
            
            self.db = pd.read_csv(name)
    def get_objects_data(self, querys):
        db = self.db.loc[self.db.name.isin(querys)]
        name = db.name.values
        print querys, len(db)
        if len(db) ==0:
        
            return [], [], [], []
        time = db.time.values
        ids = db.id.values
        x = db.x.values
        y = db.y.values
        z = db.z.values
        qx = db.qx.values
        qy = db.qy.values
        qz = db.qz.values
        qw = db.qw.values
        poses = []
        for i in range(len(x)):
            pose = [x[i], y[i], z[i], qx[i], qy[i], qz[i], qw[i]]
            poses.append(pose)
        return poses, time, name, ids


if __name__ == '__main__':
    pass

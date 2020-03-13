#!/usr/bin/env python
# encoding: utf8
#from __future__ import unicode_literals
import numpy as np
import pandas as pd
import tf
DEFAULT = "csvdata/object_data.csv"

class Objects(object):
    def __init__(self,name=None):
        
        if name == None:
            self.db = pd.read_csv(DEFAULT)
        else:
            
            self.db = pd.read_csv(name)
            
    def get_objects_data(self, querys):
        if type(querys) != type([]):
            print "dataframe query is wrong"
            return [], [], [], []
        check_type = querys[0]
        if type(check_type) == type("object_name"):
            db = self.db.loc[self.db.generic_name_0.isin(querys)]
            name = db.generic_name_0.values
        elif type(check_type) == type("object_name"):
            db = self.db.loc[self.db.specific_id_0.isin(querys)]
            name = db.specific_id_0.values
        else:
            return [], [], [], []
        
        time = db.ros_timestamp.values 
        ids = db.id.values
        x = db.position_x.values
        y = db.position_y.values
        z = db.position_z.values
        ex = db.szwht_x.values
        ey = db.szwht_y.values
        ez = db.szwht_z.values
        poses = []
        for i in range(len(x)):
            q = tf.transformations.quaternion_from_euler(ex[i],ey[i],ez[i])
            pose = [x[i], y[i], z[i], q[0], q[1], q[2], q[3]]
            poses.append(pose)
        return poses, time, name, ids

        
if __name__ == '__main__':
    pass

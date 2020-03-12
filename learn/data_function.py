#!/usr/bin/env python
# encoding: utf8
#from __future__ import unicode_literals

class RefObject(object):
    def __init__(self,):
        self.pose = []
        self.name = []
        self.time = None
        self.id = None
    def set_data(self,pose, name, time, Id):
        self.pose = pose
        self.name = name
        self.time = time        
        self.id = Id
        
class Datafunction(object):
    def __init__(self,):
        self.series = []
        self.ref_point = RefObject()
        self.clAss = None
        self.first_time = None
        self.last_time = None
        self.id = None
    
    def set_series(self,series):
        self.series = series
        
    def set_time(self, first, last):
        self.first_time = first
        self.last_time = last
        
    def set_id(self,Id):
        self.id = Id
    
    def set_class(self, clAss):
        self.clAss = clAss
                
    def set_ref_object(self,pose, name, time, Id):
        self.ref_point.set_data(pose, name, time, Id)
        
    def get_series(self,):
        return self.series
    
    def get_class(self,):
        return self.clAss
    
    def get_id(self,):
        return self.id
    
    def get_ref_point(self,):
        return self.ref_point

    def get_first_time(self,):
        return self.first_time

    def get_last_time(self):
        return self.last_time
        
if __name__ == '__main__':
    print("Hello")

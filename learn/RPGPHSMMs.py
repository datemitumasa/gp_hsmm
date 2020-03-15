#!/usr/bin/env python
# encoding: utf8
#from __future__ import unicode_literals
import GaussianProcessMiltiDim
import random
import math
import graphviz
import matplotlib.pyplot as plt

import time
import os
import tf
import pandas as pd
import numpy as np
from data_function import Datafunction

USE_JOINT = ["x", "y", "z", "qx", "qy", "qz", "qw"]
DIST = 0.9
########################
MAX = 700
MINIX = -32700
MIN = -700
"""
Cythonのコンパイルできないときは，

  E:\Python27_64\Lib\distutils\msvc9compiler.py

のget_build_version()の

  majorVersion = int(s[:-2]) - 6

を使いたいコンパラのバージョンに書き換える．
VC2012の場合は

 majorVersion = 11
"""
def dump():
    import sys
    sys.exit(1)
    
def rotM(p):
    # 回転行列を計算する
    px = p[0]
    py = p[1]
    pz = p[2]
     #物体座標系の 3->2->1 軸で回転させる
    Rx = np.array([[1, 0, 0],
                    [0, np.cos(px), np.sin(px)],
                    [0, -np.sin(px), np.cos(px)]])
    Ry = np.array([[np.cos(py), 0, -np.sin(py)],
                    [0, 1, 0],
                    [np.sin(py), 0, np.cos(py)]])
    Rz = np.array([[np.cos(pz), np.sin(pz), 0],
                    [-np.sin(pz), np.cos(pz), 0],
                    [0, 0, 1]])
    R = Rx.dot(Ry).dot(Rz)       
    return R

def calctransform(p,B,sP_B):
#    p = np.array([np.pi,np.pi/2, np.pi/3])
    R = rotM(p)
    A = R.T
    
    O = np.array([0, 0, 0])
    sP_O = np.dot(A, sP_B)
    rB_O = B - O
    rP_O = rB_O + sP_O
    return rP_O

def export_dataframe2series(frame):
    _series = []
    for name in USE_JOINT:
        _series.append(frame[name].values)
    series = np.array(_series).T
    return series

class GPSegmentation(object):
    SKIP_LEN = 1
    MIN_STATE = 3
    CORD_TRA = 0
    CORD_LAND1 = 1
    CORD_LAND2 = 2
    CORD_LAND3 = 3
    CORD_MOV = 4
    FIELD_CORD = 5

    def __init__(self, series, objects_data, category, parameter):
        self.parameter = parameter
        self.joint_states = []
        self.joint_state_stamp = []
        self.object_category = category
        self._set_state()
        print "set ",len(series), " data"
        for s in series:
            t = s.time.values
            self.joint_states.append(export_dataframe2series(s))
            self.joint_state_stamp.append(t)
        self.object_dataframe  = objects_data
        self.object_poses = self.object_dataframe.pose.values
        self.object_ids = self.object_dataframe.id.values
        self.object_stamps = self.object_dataframe.time.values
        

    def _set_state(self,):
        self.timethread = self.parameter["time_thread"]
        self.AVE_LEN = self.parameter["average_length"]
        self.MIN_LEN = self.parameter["min_length"]
        self.MAX_LEN = self.parameter["max_length"]
        self.dim = self.parameter["data_dimention"]
        self.iteration = self.parameter["learn_iteration"]
        self.t_n = self.parameter["landmark_setting"]["own_landmark_class"]
        self.h_n =  self.parameter["landmark_setting"]["z_and_y_axis_rotate_class"]
        self.h1_n = self.parameter["landmark_setting"]["z_axis_rotate_class"]
        self.h2_n = self.parameter["landmark_setting"]["no_rotate_class"]
        self.raw_n = self.parameter["landmark_setting"]["no_landmark_class"]

        cor = [0]*self.t_n + [1] * self.h1_n + [2] * self.h_n + [3] * self.h2_n + [4] *  self.raw_n 
        self.cordinates = cor
        self.numclass = len(cor)
        self.segmlen = 3
        self.gps = [GaussianProcessMiltiDim.GPMD(self.dim) for i in range(self.numclass)]

        self.segm_in_class = [[] for i in range(self.numclass)]
        self.segmclass = {}
        self.segmlandmark = {}
        self.segments = []
        self.segments_time = []
        self.landmarks = []
        self.landmark_lists = []
        self.trans_prob = np.ones((self.numclass, self.numclass))
        self.trans_prob_bos = np.ones(self.numclass)
        self.trans_prob_eos = np.ones(self.numclass)
        self.is_initialized = False
        self.land_choice = []
        self.data = []
        self.names = []
        self.time_list = []
        self.svdir = ""
        self.segmlandmark_df= {}
        self.segment_data = []
    def load_data(self,):
        self.segments = []
        self.segments_time = []
        self.is_initialized = False
        for oj in range(len(self.joint_states)):
            y = self.joint_states[oj]
            yt = self.joint_state_stamp[oj]
            segm = []
            """
            # ランダムに切る
            for i in range(len(y)/self.segmlen):
                segm.append( y[i*self.segmlen:i*self.segmlen+self.segmlen] )

            # 余りがあるか？
            remainder = len(y)%self.segmlen
            if remainder!=0:
                segm.append( y[len(y)-remainder:] )

            self.segments.append( segm )
            """

            i = 0
            stamp_list = []
            while i < len(y):
                length = random.randint(self.MIN_LEN, self.MAX_LEN)

                if i + length + 1 >= len(y):
                    length = len(y)-i
                segm.append(y[i: i+length + 1])
                stamp_list.append(yt[i: i+length + 1])
                i += length
            self.segments.append(segm)
            self.segments_time.append(stamp_list)

            for i, s in enumerate(segm):
                st = stamp_list[i]

                frames = self.object_dataframe.loc[(self.object_dataframe.time >= st[0]-self.timethread) & (self.object_dataframe.time <= st[-1])]
                lands = frames.pose.values
                ids = frames.id.values
                if len(frames) == 0 and self.t_n == 0:
                    print "no object, learn stop"
                    dump()
                while 1:
                    c = np.random.choice(range(self.numclass))
                    near_object = range(len(ids))
                    cor = self.cordinates[c]
                    if cor == self.CORD_TRA:
                        self.segmlandmark[id(s)] = np.zeros(self.dim)
                        break
                    else:
                        if len(near_object) == 0:
                            print "not mathch object data"
                            dump()
                        number = np.random.choice(near_object)
                        self.segmlandmark[id(s)] = lands[number]
                        Id = ids[number]
                        self.segmlandmark_df[id(s)] = frames.loc[(frames.id == Id)]
                        break
                self.segmclass[id(s)] = c

    def set_DataFunction(self,):
        i = 0
        self.segment_data = []
        for j in range(len(self.segments)):
            segm = self.segments[j]
            stamps = self.segments_time[j]
            for k in range(len(segm)):
                s = segm[k]
                st = stamps[k]
                num = id(s)
                cls = self.segmclass[num]
                
                d = Datafunction()
                d.set_time(st[0],st[-1])
                d.set_series(s)
                d.set_class(cls)
                try:
                    df = self.segmlandmark_df[num]
                    d.set_ref_object(df.pose.values[0], df.name.values[0], df.time.values[0], df.id.values[0])
                except:
                    df = None
                d.set_id(i)
                i+= 1
                self.segment_data.append(d)

    def check_quat(self,s, lands):
        n = len(lands)
        lands_list = []
        s = np.array(s)
        ss = s.T
        f_list = []
#        e_list = []
        for i in range(n):
            lx = np.power(ss[0] - lands[i][0],2)
            ly = np.power(ss[1] - lands[i][1],2)
            lz = np.power(ss[2] - lands[i][2],2)
            leng = lx+ly+lz
            f_lengths = np.sqrt(leng)
            f_length = f_lengths.min()
            if lands[i][4]==None:
                lands[i][4] = np.nan
            if math.isnan(lands[i][4]):
                continue
            lands_list.append(i)
            f_list.append(f_length)
        return lands_list
        
    def quat2quat(self, quat1, quat2):
        """
        quat1 --> quat2 = quat
        """
        key = False
        qua1 = -quat1[0:3]
        quv1 = quat1[3]
        qua2 = quat2[0:3]
        quv2 = quat2[3]
        if not key:
            qua = quv1 * qua2 + quv2 * qua1 + np.cross(qua1, qua2)
            quv = quv1 * quv2 - np.dot(qua1, qua2)
            if quv < 0.0:
                qua = qua * -1.
                quv = quv * -1.
        quat = np.r_[qua, np.array([quv])]
        return quat
        
# 遷移確率更新

    def normlize_time(self, num_step, max_time):
        step = float(max_time)/(num_step+1)
        time_stamp = []

        for n in range(num_step):
            time_stamp.append((n + 1) * step)
        return time_stamp

    def normalize_samples(self, d, nsamples):
        if len(d) == 1:
            return np.ones(nsamples) * d[0]
        else:
            return np.interp(range(nsamples),
                                np.linspace(0, nsamples - 1, len(d)), d)

    def load_model(self, basename):
        # GP読み込み
        print("now load model data")
        for c in range(self.numclass):
            filename = os.path.join(basename, "class%03d.npy" % c)
            self.segm_in_class[c] = [s for s in np.load(filename, allow_pickle=True)]

            landmarks = np.load(os.path.join(basename,
                                                "landmarks%03d.npy" % c), allow_pickle=True)

            for s, l in zip(self.segm_in_class[c], landmarks):
                self.segmlandmark[id(s)] = l

            self.update_gp(c)

        # 遷移確率更新
        self.trans_prob = np.load(os.path.join(basename, "trans.npy"), allow_pickle=True)
        self.trans_prob_bos = np.load(os.path.join(basename,
                                                      "trans_bos.npy"), allow_pickle=True)
        self.trans_prob_eos = np.load(os.path.join(basename,
                                                      "trans_eos.npy"), allow_pickle=True)
    def generate_class(self, basename):

        for i in range(self.numclass):
            gendata_lo,gendata_mi,gendata_hi, gendata_sigma=self.gps[i].generate2(np.arange(0, self.MAX_LEN, 1))
            gendata_hi = np.array(gendata_hi)
            gendata_lo = np.array(gendata_lo)

            gendata_mi = np.array(gendata_mi)
            gendata_sigma = np.array(gendata_sigma)
            np.savetxt(basename + "GP_m{0:d}.csv".format(i), gendata_mi,delimiter=",")
            np.savetxt(basename + "GP_sigma{0:d}.csv".format(i), gendata_sigma,delimiter=",")

        for i in range(self.numclass):
            gendata_lo,gendata_mi,gendata_hi, gendata_sigma=self.gps[i].generate2(np.arange(0, self.MAX_LEN, 0.1))
            gendata_hi = np.array(gendata_hi)
            gendata_lo = np.array(gendata_lo)

            gendata_mi = np.array(gendata_mi)
            gendata_sigma = np.array(gendata_sigma)
            np.savetxt(basename + "GP_high_m{0:d}.csv".format(i), gendata_mi,delimiter=",")
            np.savetxt(basename + "GP_high_sigma{0:d}.csv".format(i), gendata_sigma,delimiter=",")


    def gp_curve_old(self, basename):
        # GP読み込み
        gendata_lo_l = []
        gendata_mi_l = []
        gendata_hi_l = []
        for c in range(self.numclass):
            gendata_lo,gendata_mi,gendata_hi,_si=self.gps[c].generate2(np.arange(0, self.MAX_LEN,0.01))
            gendata_lo_l.append(gendata_lo)
            gendata_mi_l.append(gendata_mi)
            gendata_hi_l.append(gendata_hi)

        plt.figure(figsize=(20,20))
        for c in range(len(self.gps)):
            for d in range(self.dim):
                plt.subplot(self.dim, self.numclass, c+d*self.numclass+1)
                if self.dim == 1:
#                    plt.plot(np.linspace(0, self.MAX_LEN, len(gendata_lo_l[c])) , gendata_lo_l[c], "r-")
                    plt.plot(np.linspace(0, self.MAX_LEN, len(gendata_mi_l[c])) , gendata_mi_l[c], "r-")
                    plt.fill_between(np.linspace(0, self.MAX_LEN, len(gendata_hi_l[c])) , gendata_hi_l[c],gendata_lo[c], color="C0",alpha=.3)
                    plt.ylim(gendata_mi_l[c][8]-1.0, gendata_mi_l[c][8]+1.0)
                else:
#                    plt.plot(np.linspace(0, self.MAX_LEN, len(gendata_lo_l[c][:,d])) , gendata_lo_l[c][:,d], "r-")
                    plt.plot(np.linspace(0, self.MAX_LEN, len(gendata_mi_l[c][:,d])) , gendata_mi_l[c][:,d], "b-")
                    plt.fill_between(np.linspace(0, self.MAX_LEN, len(gendata_hi_l[c][:,d])) , gendata_hi_l[c][:,d], gendata_lo_l[c][:,d], color="r", alpha=.3)
        plt.tight_layout()
        plt.savefig( basename+"test.svg" )
        print("save")
        
    def update_gp(self, c):
        datay = []
        datax = []
        for s in self.segm_in_class[c]:
            try:
                s = self.cordinate_transform(s, self.segmlandmark[id(s)],
                                             self.cordinates[c])
            except:
                print self.segmlandmark.keys()

            datay += [y for y in s]
            datax += range(len(s))
        # 間引く,ひとところに固まることのないように
        self.gps[c].learn(np.array(datax), datay)

    def sample_class(self, landmark, segm):
        prob = []

        for c, gp in enumerate(self.gps):
            slen = len(segm)
            plen = 1.0
            if len(segm) > self.min_len:
                plen = (self.AVE_LEN**slen * math.exp(-slen) /
                        math.factorial(self.AVE_LEN))

                cord = self.cordinates[c]
                s = self.cordinate_transform(segm, landmark, cord)
                p = gp.calc_lik(range(len(s)), s)
                prob.append((math.exp(p) * plen))
            else:
                prob.append(0)

        accm_prob = [0]*self.numclass
        for i in range(len(prob)):
            accm_prob[i] = prob[i] + accm_prob[i-1]

        rnd = random.random() * accm_prob[-1]
        for i in range(len(prob)):
            if rnd <= accm_prob[i]:
                return i, prob[i]

        print "wrog prob, please check parametor"

    def calc_output_prob(self, c, segm, landmark):
        gp = self.gps[c]

        slen = len(segm)

        plen = 1.0
        if len(segm) >= self.MIN_LEN:
            plen = (self.AVE_LEN**slen * math.exp(-self.AVE_LEN) /
                    math.factorial(slen))

            cord = self.cordinates[c]
            s = self.cordinate_transform( segm, landmark, cord)
            p = gp.calc_lik(range(len(s)), s, self.MAX_LEN)
            return p + np.log(plen)
        else:
            return 0

    def save_model(self, basename):
        for n, segm in enumerate(self.segments):
            stamps = self.segments_time[n]
            classes = []
            clen = []
#            names = []
            stamp = []
            t=0
            for jk in range(len(segm)):
                s = segm[jk]
                st = stamps[jk]
                c = self.segmclass[id(s)]
                ts = st[0]
                classes += [c for i in range(len(s))]
                te = st[-1]
                t += len(s)
                dd = [c, len(s)]
                clen.append(dd)
                stamp.append([ts,te])
            np.savetxt(basename+"segm%03d.txt" % n, classes, fmt="%d")
            np.savetxt(basename+"slen%03d.txt" % n, np.array(clen,dtype=np.int))
            np.savetxt(basename+"stamps%03d.txt" % n, np.array(stamp))
            
        np.savetxt(basename+"cord.txt", np.array(self.cordinates), fmt="%d")
        plt.figure()
        for c in range(len(self.gps)):
            for d in range(self.dim):
                plt.subplot(self.dim, self.numclass, c + d * self.numclass + 1)
                for data in self.segm_in_class[c]:
                    trans_data = self.cordinate_transform( data,
                                                          self.segmlandmark[id(data)],
                                                          self.cordinates[c])

                    if self.dim == 1:
                        plt.plot(range(len(trans_data)), trans_data, "b-")
                    else:
                        plt.plot(range(len(trans_data)),
                                 trans_data[:, d], "b-")
                    plt.ylim(-1.1, 1.1)

        plt.savefig(basename+"class.png")
        np.savetxt(basename+"cordinate.txt", self.cordinates)
        # テキストでも保存
        np.save(basename + "trans.npy", self.trans_prob)
        np.save(basename + "trans_bos.npy", self.trans_prob_bos)
        np.save(basename + "trans_eos.npy", self.trans_prob_eos)
        names = ["class{}".format(i) for i in range(len(self.cordinates))]
        G = graphviz.Digraph(format="png")
        G.attr("node", shape="square", style="filled")
        chance = 1.0 / float(len(self.cordinates)) + 0.1
        for c in range(len(self.cordinates)):
            bos = self.trans_prob_bos[c]
            if bos > chance:
                G.edge("start",names[c],label=str(bos))
            for cc in range(len(self.cordinates)):
                trans = self.trans_prob[c][cc]
                if trans > chance:
                    G.edge(names[c],names[cc],label=str(trans))
            eos = self.trans_prob_eos[c]
            if eos > chance:
                G.edge(names[c],"end",label=str(eos))
        G.node("start",shape="circle",color="pink")
        G.render(basename+"trans")
        
        np.savetxt(basename + "lik.txt", [self.calc_lik()])

        for c in range(self.numclass):
            np.save(basename+"class%03d.npy" % c,
                       self.segm_in_class[c])
            np.save(basename+"landmarks%03d.npy" % c,[self.segmlandmark[id(s)] for s in self.segm_in_class[c]])
        np.save(basename + "cordinates.npy", self.cordinates)


    def cordinate_transform(self, s, land_pos, cord):
        land_pos = np.array(land_pos)
        if cord == self.CORD_TRA:
            ss = np.array(s).T
            ss_xyz = ss[:3]
            ss_qxyzw = ss[3:7]
            if self.dim > 7:
                ss_other = ss[7:]
                ss_other = ss_other.T
            ss_qxyzw = ss_qxyzw.T
            offset = np.zeros(len(ss_xyz))
            for i in range(len(ss_xyz)):
                offset[i] = ss_xyz[i][0]
            s_xyz = ss_xyz.T

            q = ss_qxyzw[0]
            R = tf.transformations.quaternion_matrix(q)

            v = []
            r_xyzw = ss_qxyzw[0]
            r_xyzw_inv = np.array([-r_xyzw[0],-r_xyzw[1],-r_xyzw[2],r_xyzw[3]])
            rot_inv = tf.transformations.euler_from_quaternion(r_xyzw_inv)
            pos = -s_xyz[0]
            pos = np.r_[pos,1.0]
            R_inv = tf.transformations.quaternion_matrix(r_xyzw_inv)
            pos_inv = np.dot(R_inv,pos)[0:3]


            for i, sss in enumerate(s_xyz):
                sr = calctransform(rot_inv, pos_inv, sss)
                x = sr[0]
                y = sr[1]
                z = sr[2]
                v.append([x, y, z])
            s_xyz = np.array(v)


            q = ss_qxyzw[0]
            ls = []
            for qq in ss_qxyzw:
                qt = self.quat2quat(q, qq)
                ls.append(qt)
            ls = np.array(ls)
            ss_o = s_xyz
            if self.dim > 3:
                ss = np.c_[ss_o, ls]
            if self.dim > 7:
                ss = np.c_[ss, ss_other]
        elif cord == self.CORD_LAND1:
            ss = np.array(s).T
            ss_xyz = ss[:3]
            ss_qxyzw = ss[3:7]
            if self.dim > 7:
                ss_other = ss[7:]
                ss_other = ss_other.T
            ss_qxyzw = ss_qxyzw.T
            s_xyz = ss_xyz.T

            v = []
            s_xyz = s_xyz - land_pos[0:3]
            t = -math.atan2(s_xyz[0][1], s_xyz[0][0])
            R = np.array([[np.cos(t), -np.sin(t)],
                             [np.sin(t), np.cos(t)]])
            zaw = t
            v = []
            for i, sss in enumerate(s_xyz):
                x = R[0][0] * sss[0] + R[0][1] * sss[1]
                y = R[1][0] * sss[0] + R[1][1] * sss[1]
                z = sss[2]
                v.append([x, y, z])

            s_xyz = np.array(v)
            if land_pos[4]==None:
                land_pos[4] = np.nan
            if math.isnan(land_pos[4]):
                if land_pos[2] > 0.25:
                    rxyzw = tf.transformations.quaternion_from_euler(0.0,0.0,zaw)
                else:
                    rxyzw = tf.transformations.quaternion_from_euler(0.0,-1.57,zaw)
            else:
                rxyzw = np.array([land_pos[3],land_pos[4],land_pos[5],land_pos[6]])

            q = np.array(rxyzw)
            ls = []
            for qq in ss_qxyzw:
                qt = self.quat2quat(q, qq)
                ls.append(qt)
            ls = np.array(ls)
            ss_o = s_xyz
            if self.dim > 3:
                ss = np.c_[ss_o, ls]

            if self.dim > 7:
                ss = np.c_[ss, ss_other]

        elif cord == self.CORD_LAND2:
            ss = np.array(s).T
            ss_xyz = ss[:3]
            ss_qxyzw = ss[3:7]
            if self.dim > 7:
                ss_other = ss[7:]
                ss_other = ss_other.T
            ss_qxyzw = ss_qxyzw.T
            s_xyz = ss_xyz.T

            v = []
            s_xyz = s_xyz - land_pos[0:3]





            t = -math.atan2(s_xyz[0][1], s_xyz[0][0])
            R = np.array([[np.cos(t), -np.sin(t)],
                             [np.sin(t), np.cos(t)]])
            zaw = t
            v = []
            for i, sss in enumerate(s_xyz):
                x = R[0][0] * sss[0] + R[0][1] * sss[1]
                y = R[1][0] * sss[0] + R[1][1] * sss[1]
                z = sss[2]
                v.append([x, y, z])

            t = -math.atan2(s_xyz[0][2], np.sqrt(s_xyz[0][0]**2 + s_xyz[0][1]**2))
            R = np.array([[np.cos(t), -np.sin(t)],
                             [np.sin(t), np.cos(t)]])
            vv = []
            for i, sss in enumerate(v):
                x = R[0][0] * sss[0] + R[0][1] * sss[2]
                z = R[1][0] * sss[0] + R[1][1] * sss[2]
                y = sss[1]
                vv.append([x, y, z])
            s_xyz = np.array(vv)
            if land_pos[4]==None:
                land_pos[4] = np.nan
            if math.isnan(land_pos[4]):
                if land_pos[2] > 0.25:
                    rxyzw = tf.transformations.quaternion_from_euler(0.0,0.0,zaw)
                else:
                    rxyzw = tf.transformations.quaternion_from_euler(0.0,-1.57,zaw)
            else:
                rxyzw = [land_pos[3],land_pos[4],land_pos[5],land_pos[6]]

            q = np.array(rxyzw)
            ls = []
            for qq in ss_qxyzw:
                qt = self.quat2quat(q, qq)
                ls.append(qt)
            ls = np.array(ls)
            ss_o = s_xyz
            if self.dim > 3:
                ss = np.c_[ss_o, ls]

            if self.dim > 7:
                ss = np.c_[ss, ss_other]

        elif cord == self.CORD_LAND3:
            ss = np.array(s).T
            ss_xyz = ss[:3]
            ss_h = ss[3]
            ss_qxyzw = ss[3:7]
            if self.dim > 7:
                ss_other = ss[7:]
                ss_other = ss_other.T
            ss_qxyzw = ss_qxyzw.T
            s_xyz = ss_xyz.T
            r_xyzw = np.array([land_pos[3],land_pos[4],land_pos[5],land_pos[6]])
            r_xyzw_inv = np.array([-r_xyzw[0],-r_xyzw[1],-r_xyzw[2],r_xyzw[3]])
            rot_inv = tf.transformations.euler_from_quaternion(r_xyzw_inv)
            pos = -land_pos[0:3]
            pos = np.r_[pos,1.0]
            R_inv = tf.transformations.quaternion_matrix(r_xyzw_inv)
            pos_inv = np.dot(R_inv,pos)[0:3]

            v = []

            for i, sss in enumerate(s_xyz):
                sr = calctransform(rot_inv, pos_inv, sss)
                x = sr[0]
                y = sr[1]
                z = sr[2]
                v.append([x, y, z])
            vv = v
            s_xyz = np.array(vv)
            q = ss_qxyzw[0]
            ls = []
            for qq in ss_qxyzw:
                qt = self.quat2quat(r_xyzw, qq)
                ls.append(qt)
            ls = np.array(ls)
            ss_o = s_xyz
            if self.dim > 3:
                ss = np.c_[ss_o, ls]

            if self.dim > 7:
                ss = np.c_[ss, ss_other]
        elif cord == self.CORD_MOV:
             ss = np.array(s)
        return ss

    def forward_filtering(self,  d, _joint_stamps, landdf):
        joint_stamps = np.array(_joint_stamps)
        T = len(d)
        a = np.ones((len(d), self.MAX_LEN, self.numclass)) # 前向き確率
        a[a==1.0] = None
        ll = np.ones((T, self.MAX_LEN, self.numclass),dtype='int')
        ll = ll * -2
        if len(landdf) !=0:
            poses  = landdf.pose.values
            stamps = landdf.time.values
            ids    = landdf.id.values
        else:
            poses  = pd.Series()
            stamps = pd.Series()
            ids    = pd.Series()

        for t in range(T):
            for k in range(self.MIN_LEN, self.MAX_LEN, self.SKIP_LEN):
                if t-k < 0:
                    break
                j_stamps = joint_stamps[t-k:t+1]
                lands    = poses[np.where((stamps >= j_stamps[0]-self.timethread) & (stamps <= j_stamps[-1]) )]
                land_ids = ids[np.where((stamps >= j_stamps[0]-self.timethread) & (stamps <= j_stamps[-1]) )]
                segm = d[t-k:t+1]
                if len(lands) == 0 and self.t_n == 0:
                    print "No object"
                for c in range(self.numclass):
                    out_prob = None
                    lm = None
                    cord = self.cordinates[c]
                    if cord in [ self.CORD_LAND1 , self.CORD_LAND2 , self.CORD_LAND3]:
                        near_lands = range(len(lands))
                        calc_probs = []
                        for iii in near_lands:
                            calc_prob = self.calc_output_prob(c, segm, lands[iii])
                            calc_probs.append(calc_prob)
                        if len(near_lands) != 0:
                            max_prob = np.argmax(calc_probs)
                            out_prob = np.max(calc_probs)
                            lm = land_ids[near_lands[max_prob]]
                            try:
                                ll[t,k,c] = lm
                            except:
                                ll[t,k,c] = lm


                        else:
                            # print "no obj"
                            lm = -2
                            ll[t,k,c] = lm
                            out_prob = MINIX
                    else:
                        out_prob = self.calc_output_prob( c, segm, [segm[0][0],segm[0][1],segm[0][2],
                                                                    segm[0][3],segm[0][4],segm[0][5],segm[0][6]])

                        ll[t,k,c] = -1

                    # 遷移確率
                    tt = t-k-1
                    log_array = []
                    if out_prob==None:
                        out_prob = MINIX
                    if tt >= 0:
                        for kk in range(self.MIN_LEN,self.MAX_LEN):
                            for cc in range(self.numclass):
                                if math.isnan(a[tt,kk,cc]):
                                    continue
                                log_array.append(a[tt, kk,cc] + np.log(self.trans_prob[cc, c]) + out_prob)
                        log_array = np.array(log_array)
                        if len(log_array)==0:
                            continue
                        min_log = np.min(log_array)
                        max_log = np.max(log_array)
                        if min_log >= MIN and max_log < MAX:
                            a[t, k, c]= np.log(np.exp(log_array).sum())
                        elif min_log < MIN and max_log < MAX:
                            min_T = MIN - min_log
                            if max_log +min_T > MAX:
                                min_T = MAX - max_log
                            log_array += min_T
                            a[t, k, c] = np.log(np.exp(log_array).sum())-min_T
                        elif max_log > MAX:
                            max_T = MAX - max_log
                            log_array += max_T
                            a[t, k, c] = np.log(np.exp(log_array).sum())-max_T

                    else:
                        # 最初の単語
                        a[t, k, c] = out_prob + np.log(self.trans_prob_bos[c])

                    if t == T - 1:
                        # 最後の単語
                        a[t, k, c] += np.log(self.trans_prob_eos[c])

        return a, ll

    def sample_idx(self, prob_lik):
        max_log_arg = np.nanargmax(prob_lik)
        max_log = prob_lik[max_log_arg]
        min_log_arg = np.nanargmin(prob_lik)
        min_log = prob_lik[min_log_arg]
        if min_log >= MIN and max_log < MAX:
            prob = np.exp(prob_lik)
        elif min_log < MIN and max_log <= MAX:
            min_T = MIN - min_log
            if max_log +min_T > MAX:
                min_T = MAX - max_log
            prob_lik += min_T
            prob = np.exp(prob_lik)
        elif max_log > MAX:
            max_T = MAX - max_log
            prob_lik += max_T
            prob = np.exp(prob_lik)
        prob = np.nan_to_num(prob)
        accm_prob = [0, ] * len(prob)
        for i in range(len(prob)):
            accm_prob[i] = prob[i] + accm_prob[i-1]
        accm_prob = np.array(accm_prob)
        accm_prob = accm_prob / accm_prob[-1] * 100.0
        r =  np.array(prob).reshape([-1,self.numclass])
        r = r / np.sum(r)
        _r = []
        for i in range(self.numclass):
            _r.append(np.sum(r[:,i]))
        if accm_prob[-1] ==0.0:
            print "non prob"
        try:
            rnd = np.random.uniform(0.0, accm_prob[-1])
        except:
            print("error")
            print(accm_prob)
            raw_input()
        for i in range(len(prob)):
            if rnd <= accm_prob[i]:
                return i


    def backward_sampling(self, a, d, stamps, ll):
        T = a.shape[0]
        t = T-1

        segm       = []
        segm_class = []
        land       = []
        stamp_list = []
        
        while True:
            try:
                idx = self.sample_idx( a[t].reshape(self.MAX_LEN * self.numclass))
            except:
                print a[t]
                print a
                dump()
            l = ll[t].reshape(self.MAX_LEN * self.numclass)[idx]
            test = np.zeros(self.numclass)
            for kk in range(self.MAX_LEN):
                ttt = a[t][:][kk]
                for ti in range(self.numclass):
                    test[ti] += ttt[ti]
            k = int(idx/(self.numclass))
            c = int(idx % self.numclass)
            if t - k < 0:
                print "warn"
                continue
            s = d[t-k:t+1]
            stamp = stamps[t-k:t+1]
            # パラメータ更新
            segm.insert(0, s)
            segm_class.insert(0, c)
            stamp_list.insert(0,stamp)
            land.insert(0, l)
            t = t-k-1
    
            if t <= 0:
                break
            
        return segm, segm_class, land, stamp_list

    def calc_start_prob(self,):
        self.trans_prob_bos = np.zeros(self.numclass)
        self.trans_prob_bos += 0.1

        # 数え上げる
        for n, segm in enumerate(self.segments):
            try:
                c = self.segmclass[id(segm[0])]
                self.trans_prob_bos[c] += 1.0
            except:
                pass
        self.trans_prob_bos = (self.trans_prob_bos /
                                   self.trans_prob_bos.sum())

    def calc_end_prob(self,):        
        self.trans_prob_eos = np.zeros(self.numclass)
        self.trans_prob_eos += 0.1

        # 数え上げる
        for n, segm in enumerate(self.segments):
            try:
                c = self.segmclass[id(segm[-1])]
                self.trans_prob_eos[c] += 1.0
            except:
                pass
        self.trans_prob_eos = (self.trans_prob_eos /
                                   self.trans_prob_eos.sum())


    def calc_trans_prob(self,):
        self.trans_prob = np.zeros((self.numclass, self.numclass))
        self.trans_prob += 0.1

        # 数え上げる
        for n, segm in enumerate(self.segments):
            for i in range(1, len(segm)):
                try:
                    cc = self.segmclass[id(segm[i-1])]
                    c = self.segmclass[id(segm[i])]
                except KeyError, e:
                    # gibss samplingで覗かれているものは無視
                    break
                self.trans_prob[cc, c] += 1.0

        # 正規化
        self.trans_prob = (self.trans_prob /
                                   self.trans_prob.sum(1).reshape(self.numclass, 1))

    def remove_ndarray(self, lst, elem):
        l = len(elem)
        for i, e in enumerate(lst):
            if len(e) != l:
                continue
            if id(e) == id(elem):
                lst.pop(i)
                return
        raise ValueError("ndarray is not found!!")


    def learn(self,):
        if not self.is_initialized:
            # GPの学習
            for i in range(len(self.segments)):
                for s in self.segments[i]:
                    c = self.segmclass[id(s)]
                    self.segm_in_class[c].append(s)
            # 各クラス毎に学習
            print "updateGP"
            for c in range(self.numclass):
                self.update_gp(c)

            self.is_initialized = True
        self.gp_curve_old(self.svdir)
        return self.update(True)

    def recog(self,):
        self.update(False)

    def update(self, learning_phase=True):
        cls = [0] * self.numclass
        for i in range(len(self.segments)):
            start_time = time.time()
            print i, "/",len(self.segments)
            d = self.joint_states[i]
            stamps = self.joint_state_stamp[i]
            #  そのファイルの生データ抽出
#           そのファイルの全分節抽出
            segm = self.segments[i]
#           そのファイルのランドマーク位置抽出
            for s in segm:
                c = self.segmclass[id(s)]
                try:
    #               対象の分節のクラスを抽出
    #               対象の分節の分類を全体から削除
                    self.segmclass.pop(id(s))
    #               対象の分節のランドマークを全体から削除
                    self.segmlandmark.pop(id(s))
                    self.segmlandmark_df.pop(id(s))
                except:
                    pass
                if learning_phase:
                    # パラメータ更新
                    # 対象の分節を全体から削除
                    self.remove_ndarray(self.segm_in_class[c], s)
            _st = time.time()
            if learning_phase:
                # GP更新
                for c in range(self.numclass):
                    self.update_gp(c)

                # 遷移確率更新
#                print (time.time()-start).to_sec()
                self.calc_trans_prob()
                self.calc_start_prob()
                self.calc_end_prob()
            print "update prametor time :", time.time()- _st


#            print "forward...",
            landdf = self.object_dataframe.loc[(self.object_dataframe.time >= np.min(stamps)-self.timethread) & (self.object_dataframe.time <= np.max(stamps))]
            _st = time.time()
            a, ll = self.forward_filtering(d, stamps , landdf)
            print "forward time :", time.time() - _st
#            print "backward...",
            pt = "{0:d}".format(self.number)

            segm, segm_class, landdf_list, stamp_list = self.backward_sampling(a, d, stamps, ll)
            self.segments[i] = segm
            self.segments_time[i] = stamp_list
            ccc = []
            for s, c, ll , ss in zip(segm, segm_class, landdf_list, stamp_list):

                self.segmclass[id(s)] = c
                ccc.append(c)
                if ll == -1:
                    _df = pd.DataFrame()
                    print s[0]
                    _df["pose"] = s[0]
                    _df["id"] = id(s[0])
                    _df["name"] = "my_hand"
                    _df["time"] = ss[0]
                else:
                    _df = self.object_dataframe.loc[(self.object_dataframe.id == ll)]
                pose = _df.pose.values[0]
                self.segmlandmark[id(s)] = pose
                self.segmlandmark_df[id(s)] = _df
                cls[c] += 1
                # パラメータ更新
                if learning_phase:
                    self.segm_in_class[c].append(s)
            if learning_phase:
                # GP更新
                for c in range(self.numclass):
                    self.update_gp(c)

                # 遷移確率更新
                self.calc_trans_prob()
                self.calc_start_prob()
                self.calc_end_prob()
                end_time = time.time()
                print self.object_category, " 1 term time:",end_time - start_time
        self.set_DataFunction()
        return True

    def calc_lik(self,):
        lik = 0
        for segm in self.segments:
            for s in segm:
                c = self.segmclass[id(s)]
                lik += self.gps[c].calc_lik(range(len(s)), s, self.MAX_LEN)
        return lik

    def learn_start(self, svdir):
        print svdir
        print("load data")
        st = time.time()
        self.load_data()
        
        print("learn start ",self.object_category)
        self.base_dir = svdir
        make_dir(self.base_dir)
        self.svdir = self.base_dir + "{}/".format(self.object_category)
        make_dir(self.svdir)
        
        liks = []
        self.number = 0
        for it in range(self.iteration):
            self.number = it
            print "-----", it, " ", self.category,"-----"
            flag = self.learn()
            try:
                lik = self.calc_lik()
            except:
                lik = 0.0
            self.gp_curve_old(self.svdir)
            if len(liks) > 3:
                if lik == liks[-1]:
                    break
            liks.append(lik)
            np.savetxt(self.svdir+"liks.txt",liks)
            self.save_model( self.svdir)
#            if not flag:
#                print("out")
#                return False
        print liks
        print("now saving")
        self.save_model( self.svdir)
        print self.calc_lik()
        self.generate_class(self.svdir)
        print"learn finish ",self.object_category, " time : ", time.time()-st
     
        return None

    def recog_start(self, svdir):
        print svdir
        print("load data")
        st = time.time()
        self.load_data()
        
        print("learn start ",self.object_category)
        self.base_dir = svdir
        make_dir(self.base_dir)
        self.svdir = self.base_dir + "{}/".format(self.object_category)
        make_dir(self.svdir)
        
        liks = []
        self.number = 0
        for it in range(1):
            self.number = it
            print "-----", it, "-----"
            flag = self.recog()
            try:
                lik = self.calc_lik()
            except:
                lik = 0.0
            self.gp_curve_old(self.svdir)
            np.savetxt(self.svdir+"liks.txt",liks)
            self.save_model( self.svdir)
            if len(liks) > 3:
                if lik == liks[-1]:
                    break
            liks.append(lik)
#            if not flag:
#                print("out")
#                return False
        print liks
        print("now saving")
        self.save_model( self.svdir)
        print self.calc_lik()
        self.generate_class(self.svdir)
        print"recog finish ",self.object_category, " time : ", time.time()-st
     
        return None



def make_dir(name):
    try:
        os.mkdir(name)
    except:
        pass

def main():
    jb = GPSegmentation()
    return True

if __name__ == '__main__':
    main()
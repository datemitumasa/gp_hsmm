#!/usr/bin/env python
# -*- coding: utf-8 -*-
import rospy
import numpy as np
import tf2_ros
import manager
import os
import math
from std_msgs.msg import Empty
from geometry_msgs.msg import TransformStamped
from sensor_msgs.msg import JointState
from gp_calc import Calc
from scipy.stats import norm, multivariate_normal
import itertools
SELF=False
FILE="GP_m{0:d}.csv"
SIGMA_FILE="GP_sigma{0:d}.csv"
COR ="cord.txt"
SLENS= "slen{0:03d}.txt"
LEN=25
MIN=5
E = 0.05
VAR = 0.01
VAR10 = 100.
import math
import tf

def smooth(xyz_l):
    data = []
    data.append(xyz_l[0])
    n = len(xyz_l)
    nn = len(xyz_l[0])
    for i in range(1,n-1):
        st = []
        for j in range(nn):
            p = (xyz_l[i-1][j]+4*xyz_l[i][j]+xyz_l[i+1][j])/6.0
            st.append(p)
        data.append(st)
    data.append(xyz_l[-1])
    return data

def _smooth(xyz_l):
    data = []
    data.append(xyz_l[0])
    n = len(xyz_l)
    nn = len(xyz_l[0])
    for i in range(1,n-1):
        st = []
        for j in range(nn):
            p = (xyz_l[i-1][j]+xyz_l[i][j]+xyz_l[i+1][j])/3.0
            st.append(p)
        data.append(st)
    data.append(xyz_l[-1])
    return data


def check_first(xyz_l, sigma, landmark_l):
    lik = []
    for i in range(len(xyz_l)-1):
        g_x = gaussian(xyz_l[i][0],sigma[i][0],landmark_l[0])
        g_y = gaussian(xyz_l[i][1],sigma[i][1],landmark_l[1])
        g_z = gaussian(xyz_l[i][2],sigma[i][2],landmark_l[2])
        lik.append(g_x*g_y*g_z)
    n = np.argmax(lik)
    return n

def marge_curve(xyz_l, sigma, trajector_mu, trajector_sig):
    xyz = []
    sig = []
    for i in range(3):
        m_mu, m_sig = marge_gp(xyz_l[i], sigma[i], trajector_mu[i],trajector_sig[i])
        xyz.append(m_mu)
        sig.append(m_sig)
    xyz = np.array(xyz)
    sigma = np.array(sig)
    xyz = xyz.T,sigma.T

    return xyz

def marge_gp(n1,s1,n2,s2):
    nlen= len(n1)
    n_m = []
    s_m = []
    for i in range(nlen):
        n = (s1[i]**2 *n2[i] + s2[i]**2 * n1[i])/(s1[i]**2 + s2[i]**2)
        s = np.sqrt((s1[i]**2 * s2[i]**2)/(s1[i]**2 + s2[i]**2))
        n_m.append(n)
        s_m.append(s)
    return n_m, s_m

def quat2quat(quat1, quat2):
    """
    quat1 --> quat2 = quat
    """
    key = False
    qua1 = -quat1[0:3]
    quv1 = quat1[3]
    if len(quat2.shape) ==1:
        qua2 = quat2[0:3]
        quv2 = quat2[3]
    else:
        qua2 = quat2[:,0:3]
        quv2 = quat2[:,3]
        key = True
    if not key:
        qua = quv1 * qua2 + quv2 * qua1 + np.cross(qua1, qua2)
        quv = quv1 * quv2 - np.dot(qua1, qua2)
        if quv < 0:
            quv = quv * -1
            qua = qua * -1
    else:
        qua = quv1 * qua2 + np.c_[qua1[0]*quv2,qua1[1]*quv2,qua1[2]*quv2] + np.c_[qua1[1]*qua2[:,2]+qua1[2]*qua2[:,1], qua1[0]*qua2[:,2]+qua1[2]*qua2[:,0], qua1[1]*qua2[:,0]+qua1[0]*qua2[:,1]]
        quv = quv1 * quv2 - (qua1[0] * qua2[:,0] + qua1[1] * qua2[:,1] + qua1[2] * qua2[:,2])
    nq = np.linalg.norm(qua)
    if nq != 0.0 and quv < 1.0:
        p = np.sqrt(1.0 - np.power(quv,2)) / nq
        qua_p = np.array(qua)*p
        quat = np.r_[qua_p, np.array([quv])]
    else:
        quat = np.array([0.,0.,0.,1.])
    return quat

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

def gaussian(nu, sigma, x):
#    p =     (1.0 / (np.sqrt(2 * np.pi)* sigma)) * np.exp(-(np.power((x-nu),2)/(2 * np.power(sigma,2))))
    _xn = x * VAR10
    xn = np.round(_xn) / VAR10
    p1 = norm.cdf(x =xn+VAR,loc=nu,scale=sigma)
    p2 = norm.cdf(x =xn-VAR,loc=nu,scale=sigma)
    p = p1 - p2
    return p



class Planner(object):
    def __init__(self,folda_name, category):
        """
        fold_name is folda directory : str
        category is object type , directory names : list
        """
        self.st = []
        self.buf = tf2_ros.Buffer()
        self.lis = tf2_ros.TransformListener(self.buf)
        self._broadcaster = tf2_ros.TransformBroadcaster()
        self.file = folda_name
        self.category = category
        self.powers = manager.Manger(self.file, self.category)
        self.actions = {}
        self.trans = {}
        self.bos = {}
        self.eos = {}
        for c in self.category:
            name = self.file + "/" + c + "/" +FILE
            sname = self.file + "/" + c + "/" +SIGMA_FILE
            cor = np.loadtxt(self.file + "/" + c + "/" +COR)
            n = len(cor)
            cdic = {}
            for i in range(n):
                dic = {}
                data = np.loadtxt(name.format(i),delimiter=",")
                sdata = np.loadtxt(sname.format(i),delimiter=",")
                dic["gp"] = data
                dic["sig"] = sdata
                dic["cor"] = cor[i]
                cdic[i] = dic
            self.actions[c]=cdic
            trans = np.load(self.file + "/" + c + "/" +"trans.npy")
            bos = np.load(self.file + "/" + c + "/" +"trans_bos.npy")
            sl = 0
            sts = []
            while not rospy.is_shutdown():
                try:
                    st = np.loadtxt(self.file + "/" + c + "/" + SLENS.format(sl))
                    sts.append(st)
                    sl += 1
                except:
                    break
            alst = np.zeros(len(cor))
            enst = np.zeros(len(cor))
            trst = np.zeros([len(cor),len(cor)])
            print c
            for i in range(len(sts)):
                st = sts[i]
                if len(st.shape)==2:
                    for sp in range(len(st)):
                        s = st[sp]
                        j = int(s[0])
                        if sp >0:
                            if int(st[sp-1][0]) == j:
                                continue
                        alst[j]+=1
                        if sp == 0:
                            continue
                        trst[int(st[sp-1][0])][j]+=1
                else:
                    j = int(st[0])
                    alst[j]+= 1
                enst[j]+=1
            eos = np.zeros(len(cor))
            tos = np.zeros(trst.shape)
            for i in range(len(cor)):
                if alst[i] ==0.:
                    eos[i] = E
                else:
                    eos[i] = enst[i]/alst[i] * (1.0-2*E) + E
                if trst[i].sum() == 0.0:
                    continue
                else:
                    tos[i] = trst[i] / trst[i].sum()
#            self.trans[c] = trans
            self.trans[c] = tos
            self.eos[c] = eos
            self.bos[c] = bos
        self.cord = {}
        self.back_action = None
        self.gps = [Calc() for i in range(3)]
        rospy.Subscriber("/hsrb/joint_states", JointState, self.broadcast,queue_size=10)

    def set_stamp(self, xyz, qxyzw,name="target_object"):
        ts = TransformStamped()
        ts.header.frame_id = "map"
        ts.child_frame_id = name
        ts.transform.translation.x = xyz[0]
        ts.transform.translation.y = xyz[1]
        ts.transform.translation.z = xyz[2]
        ts.transform.rotation.x = qxyzw[0]
        ts.transform.rotation.y = qxyzw[1]
        ts.transform.rotation.z = qxyzw[2]
        ts.transform.rotation.w = qxyzw[3]
        ts.header.stamp = rospy.Time.now()
        self.st = []
        self.st.append(ts)
        return None

    def set_stamp_sim(self, xyz, qxyzw,name="target_object"):
        ts = TransformStamped()
        ts.header.frame_id = "odom"
        ts.child_frame_id = name
        ts.transform.translation.x = xyz[0]
        ts.transform.translation.y = xyz[1]
        ts.transform.translation.z = xyz[2]
        ts.transform.rotation.x = qxyzw[0]
        ts.transform.rotation.y = qxyzw[1]
        ts.transform.rotation.z = qxyzw[2]
        ts.transform.rotation.w = qxyzw[3]
        ts.header.stamp = rospy.Time.now()
        self.st = []
        self.st.append(ts)
        return None


    def broadcast(self,data):
        """
        tf_stampを発行する関数
        param TransformStamped tf_stamp: tfのRosMessage
        """

        if len(self.st) == 0:
            return None
        for tf_stamp in self.st:
            tf_stamp.header.stamp = rospy.Time.now()
            self._broadcaster.sendTransform(tf_stamp)

    def sample_idx(self, probs):
        p = np.array(probs)
        norm = p.sum()
        shape = p.shape
        ps = p.reshape(shape[0] * shape[1])
        pp = ps / norm
        n = 0
        c = np.random.random()
        num = 0
        for i in range(len(pp)):
            n += pp[i]
            if c <= n:
                num = i
                break
            if i == (len(pp)-1):
                num = i
                break
        ln = shape[1]
        cl = num / ln
        length = num % ln

        numax = np.argmax(pp)
        cl = numax / ln
        length = numax % ln

        return cl ,length

    def calc_first_state(self,end_effector_pose, now_hand, object_pose, object_pose_inv):
        xyz = object_pose[:3]
        qxyzw = object_pose[3:]
        h_xyz = end_effector_pose[:3]
        x = xyz[0] - h_xyz[0]
        y = xyz[1] - h_xyz[1]
        z = xyz[2] - h_xyz[2]
        xaw = 0.0
        zaw = math.atan2(y, x)  + math.pi
        yaw = math.atan2(z, math.sqrt(y **2 + x **2))
        landmark_rol = {}
        dic = {}
        dic["e"] = [xaw,yaw,zaw]
        dic["q"] = tf.transformations.quaternion_from_euler(xaw,yaw,zaw)
        landmark_rol[2]=dic
        dic = {}
        dic["e"] = [xaw,0.0,zaw]
        dic["q"] = tf.transformations.quaternion_from_euler(xaw,0.0,zaw)
        landmark_rol[1]=dic
        dic = {}
        exyz = tf.transformations.euler_from_quaternion(qxyzw)
        xaw = exyz[0]
        yaw = exyz[1]
        zaw = exyz[2]
        dic["e"] = [xaw,yaw,zaw]
        dic["q"] = qxyzw
        landmark_rol[3]=dic
        landmark_rol[0]=dic
#---

        first_state={}
        x = np.sqrt(np.power((h_xyz[0]-xyz[0]),2)+np.power((h_xyz[1]-xyz[1]),2)+np.power((h_xyz[2]-xyz[2]),2))
        y = 0.0
        z = 0.0
        first_state[2] = [x,y,z]

        x = np.sqrt(np.power((h_xyz[0]-xyz[0]),2)+np.power((h_xyz[1]-xyz[1]),2))
        y = 0.0
        z = np.sqrt(np.power((h_xyz[2]-xyz[2]),2))
        first_state[1] = [x,y,z]
###
        rot = tf.transformations.euler_from_quaternion(object_pose_inv[3:])
        rot = np.array(rot)
        pos = np.array(object_pose_inv[:3])
        obj_pos_from_base_frame = np.array([h_xyz[0],h_xyz[1],h_xyz[2]])
        c3_pos = calctransform(rot,pos,obj_pos_from_base_frame)
        x = c3_pos[0]
        y = c3_pos[1]
        z = c3_pos[2]
        first_state[3] = [x,y,z]
        first_state[0] = [x,y,z]
        return first_state, landmark_rol

    def calc_next_prob(self, end_effector_pose, now_hand, object_pose, object_category,object_pose_inv,back_action):
        qxyzw = object_pose[3:]
        h_qxyzw = end_effector_pose[3:]
        o_q = quat2quat(np.array(qxyzw),np.array(h_qxyzw))
        first_state, landmark_rol = self.calc_first_state(end_effector_pose, now_hand, object_pose, object_pose_inv)

        acts = self.actions[object_category]
        liks = []
        nhand = now_hand
        n = len(acts)
        trans = self.trans[object_category]
        bos = self.bos[object_category]
        for i in range(n):
            lik = 1.0
            ac = acts[i]
            cor = int(ac["cor"])
            gp = ac["gp"]
            sig = ac["sig"]
            o = first_state[cor]
            o.append(nhand)
            o.extend(o_q)
            pl = 1.0
            for j in range(3):
                if cor==1:
                    if j >=1:
                        continue
                elif cor==2:
                    if j==2:
                        continue
                p = gaussian(gp[0][j],sig[0][j],o[j])
                pl *= p
            if cor==2:
                pl *= gaussian(gp[0][0],sig[0][0],o[0])**2
            elif cor==1:
                pl *=gaussian(gp[0][2],sig[0][2],o[2])
            lik *= pl
            p = gaussian(gp[0][3],sig[0][3],o[3])
            lik *= p
            if back_action == None:
                pp = bos[i]
            else:
                pp = trans[back_action][i]
            lik *= pp
            liks.append(lik)
        return liks, first_state, landmark_rol


    def choose_action(self, end_effector_pose, now_hand, object_pose, object_category, calc_type=0, back_action=None, top_command_act_cls_len=[None,None]):
        xyz = object_pose[:3]
        qxyzw = object_pose[3:]
        use_qxyzw = [-qxyzw[0],-qxyzw[1],-qxyzw[2],qxyzw[3]]
        self.set_stamp(xyz,qxyzw,"object")
        rospy.sleep(2.0)
        _object_inv = self.buf.lookup_transform("object","map",rospy.Time.now(),rospy.Duration(3.0))
        _object = _object_inv.transform
        object_inv = [_object.translation.x,_object.translation.y,_object.translation.z,
                      _object.rotation.x,_object.rotation.y,_object.rotation.z,_object.rotation.w]
#---
        liks, first_state,landmark_rol = self.calc_next_prob( end_effector_pose, now_hand, object_pose, object_category, object_inv,back_action)
        acts = self.actions[object_category]
        gp_params = {}
        for j in range(1,4):
            cdic = {}
            o = first_state[j]
            for i in range(3):
                self.gps[i].learn([0],[o[i]])
                dic = {}
                mu , sig = self.gps[i].predict(range(LEN))
                dic["mu"]=mu
                dic["sig"]=sig
                cdic[i] = dic
            gp_params[j] = cdic
        gp_params[0] =gp_params[3]

        acts = self.actions[object_category]
        n = len(acts)
        nl = len(acts[0]["gp"])
        action_liks = np.ones([n*2,nl])
        action_liks_ano = np.ones([n*2,nl])
        eos = self.eos[object_category]
        for i in range(n):
            action_liks[i] *= liks[i] * (1.0-eos[i])
            action_liks[i+n] *= liks[i] * eos[i]
            action_liks_ano[i] *= liks[i] * (1.0-eos[i])
            action_liks_ano[i+n] *= liks[i] * eos[i]
        for i in range(n):
            _pow, _length=self.powers.get_actioninfo(object_category,i)
            try:
                length = int(np.round(_length))
            except:
                length = None
            print length
            for j in range(nl):
                if j < MIN or length == None:
                    action_liks[i][j] = 0.0
                    action_liks[i+n][j] = 0.0
                    action_liks_ano[i][j] = 0.0
                    action_liks_ano[i+n][j] = 0.0
                else:
                    action_liks[i][j]*= (length**j * math.exp(-length) /
                                            math.factorial(j))
                    action_liks[i+n][j]*=(length**j * math.exp(-length) /
                                            math.factorial(j))
                    action_liks_ano[i+n][j]*= (length**j * math.exp(-length) /
                                            math.factorial(j))
                    action_liks_ano[i][j]*= (length**j * math.exp(-length) /
                                            math.factorial(j))

        trans =self.trans[object_category]
        tra_xyz = []
        tra_qxyzw = []
        tra_hand = []
        for i in range(n):
            ac = acts[i]
            cor = ac["cor"]
            gp = ac["gp"]
            sig = ac["sig"]
            dic=gp_params[cor]
            xyz_hand_mu =[]
            xyz_hand_sig = []
            for j in range(3):
                d = dic[j]
                xyz_hand_sig.append(d["sig"])
                xyz_hand_mu.append(d["mu"])
            xyz_t,_sig = marge_curve(gp.T , sig.T, xyz_hand_mu, xyz_hand_sig)
            qxyzw_t = gp[:,4:]
            hand = gp[:,3]
            tra_xyz.append(xyz_t)
            tra_qxyzw.append(qxyzw_t)
            tra_hand.append(hand)
        trajector_xyz = []
        trajector_qxyzw = []
        trajector_hand = []

        for i in range(n):
            ac = acts[i]
            cor = ac["cor"]
            gp = ac["gp"]
            sig = ac["sig"]
            dic = landmark_rol[cor]
            e = dic["e"]
            q = dic["q"]
            use_qxyzw = [-qxyzw[0],-qxyzw[1],-qxyzw[2],qxyzw[3]]

            xyz_list = []
            qxyzw_list = []
            hand_list = []
    #        print xyz
            rot = e
            rot = np.array(rot)
            pos = np.array(xyz)
            txyz = tra_xyz[i]
            tqxyzw = tra_qxyzw[i]
            thand = tra_hand[i]
            _pow, _length=self.powers.get_actioninfo(object_category,i)
            try:
                length = int(np.round(_length))
            except:
                length = None
            for j in range(len(txyz)):
                obj_pos_from_base_frame = np.array([txyz[j][0],txyz[j][1],txyz[j][2]])
                base_pos = calctransform(rot,pos,obj_pos_from_base_frame)
                q = quat2quat(np.array(use_qxyzw), np.array(tqxyzw[j]))
                xyz_list.append(base_pos)
                qxyzw_list.append(q)
                hand_list.append(thand[j])
#                liks, _first_state,_landmark_rol = self.calc_next_prob( [base_pos[0],base_pos[1],base_pos[2],q[0],q[1],q[2],q[3]], thand[j], object_pose, object_category, object_inv,i)
#                lik = np.sum(liks)
#                if j < MIN or length == None:
#                    action_liks[i][j] = 0.0
#                    action_liks[i+n][j] = 0.0
#                    action_liks_ano[i][j] = 0.0
#                    action_liks_ano[i+n][j] = 0.0
#                else:
#                    action_liks[i][j]*= (length**j * math.exp(-length) /
#                                            math.factorial(j)) * lik
#                    action_liks[i+n][j]*=(length**j * math.exp(-length) /
#                                            math.factorial(j)) * lik
#                    action_liks_ano[i+n][j]*= (length**j * math.exp(-length) /
#                                            math.factorial(j)) * lik
#                    action_liks_ano[i][j]*= (length**j * math.exp(-length) /
#                                            math.factorial(j)) * lik
            if cor in [1,2]:
                trajector_xyz.append(np.array(xyz_list))
            else:
                trajector_xyz.append(np.array(xyz_list))
            trajector_qxyzw.append(np.array(qxyzw_list))
            trajector_hand.append(np.array(hand_list))
        trajector = [trajector_xyz,trajector_qxyzw,trajector_hand]
        if not SELF:
            if back_action != None:
                if acts[back_action]["cor"]!=4:
                    action_liks[back_action]=0.0
                    action_liks[back_action+n]=0.0
        if calc_type==0:
            cl, length = self.sample_idx(action_liks)
            ac = action_liks
        elif calc_type==1:
            cl, length = self.sample_idx(action_liks_ano)
            ac = action_liks_ano
        if top_command_act_cls_len[0]!=None:
            cl = top_command_act_cls_len[0]
        if top_command_act_cls_len[1]!=None:
            length = top_command_act_cls_len[1]
        end = False
        if cl >= n:
            cl = cl - n
            end = True
        ap = acts[cl]
        cor = ap["cor"]
        tra_xyz = smooth(trajector[0][cl][:length,:])
#        if cor in [1,2]:
#            tra_xyz = smooth(trajector[0][cl][:length,:])
#        else:
#            tra_xyz = trajector[0][cl][:length,:]
        tra_qxyzw = trajector[1][cl][:length,:]
        tra = []
        for i in range(len(tra_xyz)):
            tra.append([tra_xyz[i],tra_qxyzw[i]])
        hand = trajector[2][cl][:length]
        power, _length=self.powers.get_actioninfo(object_category,cl)
        psum = ac.sum()
        if end:
            num = cl + n
        else:
            num = cl
        prob = ac[num,length] / psum
        return cl, length, tra, hand, power, end, ac, trajector, prob


    def make_trajector(self, cls, hand_pose, now_hand,object_pose, object_category,back_action=None):
        xyz = object_pose[:3]
        qxyzw = object_pose[3:]
        use_qxyzw = [-qxyzw[0],-qxyzw[1],-qxyzw[2],qxyzw[3]]
        self.set_stamp(xyz,qxyzw,"object")
        rospy.sleep(1.0)
        _object_inv = self.buf.lookup_transform("object","map",rospy.Time.now(),rospy.Duration(3.0))
        _object = _object_inv.transform
        object_inv = [_object.translation.x,_object.translation.y,_object.translation.z,
                      _object.rotation.x,_object.rotation.y,_object.rotation.z,_object.rotation.w]
#---
        liks, first_state,landmark_rol = self.calc_next_prob( hand_pose, now_hand, object_pose, object_category, object_inv,back_action)
        acts = self.actions[object_category]
        gp_params = {}
        for j in range(1,4):
            cdic = {}
            o = first_state[j]
            for i in range(3):
                self.gps[i].learn([0],[o[i]])
                dic = {}
                mu , sig = self.gps[i].predict(range(LEN))
                dic["mu"]=mu
                dic["sig"]=sig
                cdic[i] = dic
            gp_params[j] = cdic

        acts = self.actions[object_category]
        n = len(acts)
        nl = len(acts[0]["gp"])
        action_liks = np.ones([n*2,nl])
        action_liks_ano = np.ones([n*2,nl])
        eos = self.eos[object_category]
        for i in range(n):
            action_liks[i] *= liks[i] * (1.0-eos[i])
            action_liks[i+n] *= liks[i] * eos[i]
            action_liks_ano[i] *= liks[i] * (1.0-eos[i])
            action_liks_ano[i+n] *= liks[i] * eos[i]
        for i in range(n):
            _pow, _length=self.powers.get_actioninfo(object_category,i)
            try:
                length = int(np.round(_length))
            except:
                length = None
#            print length
            for j in range(nl):
                if j < MIN or length == None:
                    action_liks[i][j] = 0.0
                    action_liks[i+n][j] = 0.0
                    action_liks_ano[i][j] = 0.0
                    action_liks_ano[i+n][j] = 0.0
                else:
                    action_liks[i][j]*= (length**j * math.exp(-length) /
                                            math.factorial(j))
                    action_liks[i+n][j]*=(length**j * math.exp(-length) /
                                            math.factorial(j))
                    action_liks_ano[i+n][j]*= (length**j * math.exp(-length) /
                                            math.factorial(j))
                    action_liks_ano[i][j]*= (length**j * math.exp(-length) /
                                            math.factorial(j))


        tra_xyz = []
        tra_qxyzw = []
        tra_hand = []
        for i in range(n):
            ac = acts[i]
            cor = ac["cor"]
            gp = ac["gp"]
            sig = ac["sig"]
            dic=gp_params[cor]
            xyz_hand_mu =[]
            xyz_hand_sig = []
            for j in range(3):
                d = dic[j]
                xyz_hand_sig.append(d["sig"])
                xyz_hand_mu.append(d["mu"])
            if cor >= -1:
                xyz_t,_sig = marge_curve(gp.T , sig.T, xyz_hand_mu, xyz_hand_sig)
            else:
                xyz_t = xyz_hand_mu
            qxyzw_t = gp[:,4:]
            hand = gp[:,3]
            tra_xyz.append(xyz_t)
            tra_qxyzw.append(qxyzw_t)
            tra_hand.append(hand)
        trajector_xyz = []
        trajector_qxyzw = []
        trajector_hand = []

        for i in range(n):
            ac = acts[i]
            cor = ac["cor"]
            gp = ac["gp"]
            sig = ac["sig"]
            dic = landmark_rol[cor]
            e = dic["e"]
            q = dic["q"]
            use_qxyzw = [-qxyzw[0],-qxyzw[1],-qxyzw[2],qxyzw[3]]

            xyz_list = []
            qxyzw_list = []
            hand_list = []
    #        print xyz
            rot = e
            rot = np.array(rot)
            pos = np.array(xyz)
            txyz = tra_xyz[i]
            tqxyzw = tra_qxyzw[i]
            thand = tra_hand[i]
            for j in range(len(txyz)):
                obj_pos_from_base_frame = np.array([txyz[j][0],txyz[j][1],txyz[j][2]])
                base_pos = calctransform(rot,pos,obj_pos_from_base_frame)
                q = quat2quat(np.array(use_qxyzw), np.array(tqxyzw[j]))
                xyz_list.append(base_pos)
                qxyzw_list.append(q)
                hand_list.append(thand[j])
            trajector_xyz.append(np.array(xyz_list))
            trajector_qxyzw.append(np.array(qxyzw_list))
            trajector_hand.append(np.array(hand_list))
        trajector = [trajector_xyz,trajector_qxyzw,trajector_hand]
        _pow, _length=self.powers.get_actioninfo(object_category,cls)
        length = int(np.round(_length))
        ac = acts[cls]
        cor = ac["cor"]
        if cor != -1:
            tra_xyz = smooth(trajector[0][cls][:length,:])
        else:
            tra_xyz = trajector[0][cls][:length,:]
        tra_qxyzw = trajector[1][cls][:length,:]
        tra = []
        for i in range(len(tra_xyz)):
            tra.append([tra_xyz[i],tra_qxyzw[i]])
        hand = trajector[2][cls][:length]
        return tra, hand, _pow

    def make_trajector_sim(self, cls, hand_pose, now_hand,object_pose, object_category,back_action=None):
        xyz = object_pose[:3]
        qxyzw = object_pose[3:]
        use_qxyzw = [-qxyzw[0],-qxyzw[1],-qxyzw[2],qxyzw[3]]
        self.set_stamp_sim(xyz,qxyzw,"object")
        rospy.sleep(1.0)
        _object_inv = self.buf.lookup_transform("object","odom",rospy.Time.now(),rospy.Duration(3.0))
        _object = _object_inv.transform
        object_inv = [_object.translation.x,_object.translation.y,_object.translation.z,
                      _object.rotation.x,_object.rotation.y,_object.rotation.z,_object.rotation.w]
#---
        liks, first_state,landmark_rol = self.calc_next_prob( hand_pose, now_hand, object_pose, object_category, object_inv,back_action)
        acts = self.actions[object_category]
        gp_params = {}
        for j in range(1,4):
            cdic = {}
            o = first_state[j]
            for i in range(3):
                self.gps[i].learn([0],[o[i]])
                dic = {}
                mu , sig = self.gps[i].predict(range(LEN))
                dic["mu"]=mu
                dic["sig"]=sig
                cdic[i] = dic
            gp_params[j] = cdic

        acts = self.actions[object_category]
        n = len(acts)
        nl = len(acts[0]["gp"])
        action_liks = np.ones([n*2,nl])
        action_liks_ano = np.ones([n*2,nl])
        eos = self.eos[object_category]
        for i in range(n):
            action_liks[i] *= liks[i] * (1.0-eos[i])
            action_liks[i+n] *= liks[i] * eos[i]
            action_liks_ano[i] *= liks[i] * (1.0-eos[i])
            action_liks_ano[i+n] *= liks[i] * eos[i]
        for i in range(n):
            _pow, _length=self.powers.get_actioninfo(object_category,i)
            try:
                length = int(np.round(_length))
            except:
                length = None
#            print length
            for j in range(nl):
                if j < MIN or length == None:
                    action_liks[i][j] = 0.0
                    action_liks[i+n][j] = 0.0
                    action_liks_ano[i][j] = 0.0
                    action_liks_ano[i+n][j] = 0.0
                else:
                    action_liks[i][j]*= (length**j * math.exp(-length) /
                                            math.factorial(j))
                    action_liks[i+n][j]*=(length**j * math.exp(-length) /
                                            math.factorial(j))
                    action_liks_ano[i+n][j]*= (length**j * math.exp(-length) /
                                            math.factorial(j))
                    action_liks_ano[i][j]*= (length**j * math.exp(-length) /
                                            math.factorial(j))


        tra_xyz = []
        tra_qxyzw = []
        tra_hand = []
        for i in range(n):
            ac = acts[i]
            cor = ac["cor"]
            gp = ac["gp"]
            sig = ac["sig"]
            dic=gp_params[cor]
            xyz_hand_mu =[]
            xyz_hand_sig = []
            for j in range(3):
                d = dic[j]
                xyz_hand_sig.append(d["sig"])
                xyz_hand_mu.append(d["mu"])
            if cor >= 1:
                xyz_t,_sig = marge_curve(gp.T , sig.T, xyz_hand_mu, xyz_hand_sig)
            else:
                xyz_t = xyz_hand_mu
            qxyzw_t = gp[:,4:]
            hand = gp[:,3]
            tra_xyz.append(xyz_t)
            tra_qxyzw.append(qxyzw_t)
            tra_hand.append(hand)
        trajector_xyz = []
        trajector_qxyzw = []
        trajector_hand = []

        for i in range(n):
            ac = acts[i]
            cor = ac["cor"]
            gp = ac["gp"]
            sig = ac["sig"]
            dic = landmark_rol[cor]
            e = dic["e"]
            q = dic["q"]
            use_qxyzw = [-qxyzw[0],-qxyzw[1],-qxyzw[2],qxyzw[3]]

            xyz_list = []
            qxyzw_list = []
            hand_list = []
    #        print xyz
            rot = e
            rot = np.array(rot)
            pos = np.array(xyz)
            txyz = tra_xyz[i]
            tqxyzw = tra_qxyzw[i]
            thand = tra_hand[i]
            for j in range(len(txyz)):
                obj_pos_from_base_frame = np.array([txyz[j][0],txyz[j][1],txyz[j][2]])
                base_pos = calctransform(rot,pos,obj_pos_from_base_frame)
                q = quat2quat(np.array(use_qxyzw), np.array(tqxyzw[j]))
                xyz_list.append(base_pos)
                qxyzw_list.append(q)
                hand_list.append(thand[j])
            trajector_xyz.append(np.array(xyz_list))
            trajector_qxyzw.append(np.array(qxyzw_list))
            trajector_hand.append(np.array(hand_list))
        trajector = [trajector_xyz,trajector_qxyzw,trajector_hand]
        _pow, _length=self.powers.get_actioninfo(object_category,cls)
        length = int(np.round(_length))
        tra_xyz = smooth(trajector[0][cls][:length,:])
        tra_qxyzw = trajector[1][cls][:length,:]
        tra = []
        for i in range(len(tra_xyz)):
            tra.append([tra_xyz[i],tra_qxyzw[i]])
        hand = trajector[2][cls][:length]
        return tra, hand, _pow


    def set_motion(self,bow,object_category,length=None):
        c = []
        cat = object_category
        bos = self.bos[cat]
#        eos = np.load(self.file + "/" + cat + "/" +"trans_eos.npy")
        eos = self.eos[cat]
        trans = self.trans[cat]
        C = len(bos)
        c = bow
#        for i in range(C):
#            b = bow[i]
#            if b > 0:
#                c.append(i)
#            for j in range(b):
#                c.append(i)
        if length==None:
            l = len(c)
        else:
            l = length
        while not rospy.is_shutdown():
            permutation = list(itertools.permutations(c,l))
            liks = []
            for case in permutation:
                lik = 1.0
                for i in range(len(case)):
                    if i == 0:
                        lik *= bos[case[i]]
                    elif i > 0:
                        lik *= trans[case[i-1]][case[i]]
                    if i == len(case)-1:
                        lik *= eos[case[i]]
                liks.append(lik)
            j = np.argmax(liks)
            if liks[j] == 0.0:
                l += -1
            else:
                break
            if l ==0:
                return []
        per = permutation[j]
        print liks[j], np.power(liks[j],1./float(l))
        return per

    def resampling(self,hand_pose ,object_pose, cls, slen, number ,object_category, back_action=None):
        now_hand = 0.0
        xyz = object_pose[:3]
        qxyzw = object_pose[3:]
        use_qxyzw = [-qxyzw[0],-qxyzw[1],-qxyzw[2],qxyzw[3]]
        self.set_stamp(xyz,qxyzw,"object")
        rospy.sleep(1.0)
        _object_inv = self.buf.lookup_transform("object","map",rospy.Time.now(),rospy.Duration(3.0))
        _object = _object_inv.transform
        object_inv = [_object.translation.x,_object.translation.y,_object.translation.z,
                      _object.rotation.x,_object.rotation.y,_object.rotation.z,_object.rotation.w]
#---
        liks, first_state,landmark_rol = self.calc_next_prob( hand_pose, now_hand, object_pose, object_category, object_inv,back_action)
        acts = self.actions[object_category]
        gp_params = {}
        for j in range(1,4):
            cdic = {}
            o = first_state[j]
            for i in range(3):
                self.gps[i].learn([0],[o[i]])
                dic = {}
                mu , sig = self.gps[i].predict(range(LEN))
                dic["mu"]=mu
                dic["sig"]=sig
                cdic[i] = dic
            gp_params[j] = cdic

        acts = self.actions[object_category]
        n = len(acts)
        nl = len(acts[0]["gp"])
        action_liks = np.ones([n*2,nl])
        action_liks_ano = np.ones([n*2,nl])
        eos = self.eos[object_category]
        for i in range(n):
            action_liks[i] *= liks[i] * (1.0-eos[i])
            action_liks[i+n] *= liks[i] * eos[i]
            action_liks_ano[i] *= liks[i] * (1.0-eos[i])
            action_liks_ano[i+n] *= liks[i] * eos[i]
        for i in range(n):
            _pow, _length=self.powers.get_actioninfo(object_category,i)
            try:
                length = int(np.round(_length))
            except:
                length = None
#            print length
            for j in range(nl):
                if j < MIN or length == None:
                    action_liks[i][j] = 0.0
                    action_liks[i+n][j] = 0.0
                    action_liks_ano[i][j] = 0.0
                    action_liks_ano[i+n][j] = 0.0
                else:
                    action_liks[i][j]*= (length**j * math.exp(-length) /
                                            math.factorial(j))
                    action_liks[i+n][j]*=(length**j * math.exp(-length) /
                                            math.factorial(j))
                    action_liks_ano[i+n][j]*= (length**j * math.exp(-length) /
                                            math.factorial(j))
                    action_liks_ano[i][j]*= (length**j * math.exp(-length) /
                                            math.factorial(j))
        tra_xyz = []
        tra_qxyzw = []
        tra_hand = []
        for i in range(n):
            ac = acts[i]
            cor = ac["cor"]
            gp = ac["gp"]
            sig = ac["sig"]
            dic=gp_params[cor]
            xyz_hand_mu =[]
            xyz_hand_sig = []
            for j in range(3):
                d = dic[j]
                xyz_hand_sig.append(d["sig"])
                xyz_hand_mu.append(d["mu"])
            if cor >= 1:
                _xyz_t, _sig = marge_curve(gp.T , sig.T, xyz_hand_mu, xyz_hand_sig)
            else:
                _xyz_t = xyz_hand_mu
                _sig = xyz_hand_sig
            xyz_t = []
            for j in range(len(_xyz_t)):
                xyz_t.append(multivariate_normal.rvs(mean=_xyz_t[j],cov=_sig[j]))
            xyz_t = np.array(xyz_t)
            qxyzw_t = gp[:,4:]
            hand = gp[:,3]
            tra_xyz.append(xyz_t)
            tra_qxyzw.append(qxyzw_t)
            tra_hand.append(hand)
        trajector_xyz = []
        trajector_qxyzw = []
        trajector_hand = []

        for i in range(n):
            ac = acts[i]
            cor = ac["cor"]
            gp = ac["gp"]
            sig = ac["sig"]
            dic = landmark_rol[cor]
            e = dic["e"]
            q = dic["q"]
            use_qxyzw = [-qxyzw[0],-qxyzw[1],-qxyzw[2],qxyzw[3]]

            xyz_list = []
            qxyzw_list = []
            hand_list = []
    #        print xyz
            rot = e
            rot = np.array(rot)
            pos = np.array(xyz)
            txyz = tra_xyz[i]
            tqxyzw = tra_qxyzw[i]
            thand = tra_hand[i]
            for j in range(len(txyz)):
                obj_pos_from_base_frame = np.array([txyz[j][0],txyz[j][1],txyz[j][2]])
                base_pos = calctransform(rot,pos,obj_pos_from_base_frame)
                q = quat2quat(np.array(use_qxyzw), np.array(tqxyzw[j]))
                xyz_list.append(base_pos)
                qxyzw_list.append(q)
                hand_list.append(thand[j])
            trajector_xyz.append(np.array(xyz_list))
            trajector_qxyzw.append(np.array(qxyzw_list))
            trajector_hand.append(np.array(hand_list))
        trajector = [trajector_xyz,trajector_qxyzw,trajector_hand]
        _pow, _length=self.powers.get_actioninfo(object_category,cls)
        length = int(np.round(_length))
        ac = acts[cls]
        cor = ac["cor"]
        if cor != 3:
            tra_xyz = smooth(trajector[0][cls][:slen,:])
        else:
            tra_xyz = trajector[0][cls][:slen,:]
        tra_qxyzw = trajector[1][cls][:slen,:]
        tra = []
        for i in range(len(tra_xyz)):
            tra.append([tra_xyz[i],tra_qxyzw[i]])
        hand = trajector[2][cls][:length]
        return tra[number]
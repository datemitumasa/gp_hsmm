#!/usr/bin/env python
# -*- coding: utf-8 -*-
import rospy
import rosparam
from geometry_msgs.msg import Transform
from gp_hsmm.srv import TrajectoryOrder, TrajectoryOrderResponse, TrajectoryOrderRequest 
from gp_hsmm.msg import Motion
import planner
import tf2_ros
from std_srvs.srv import Empty, EmptyRequest, EmptyResponse
from sensor_msgs.msg import JointState
import tf
import math
import yaml
import numpy as np
import glob

FOLDA_PARAM = "learned_folda"
OBJECT_PARAM = "object_category"
HAND_PARAM =  "hand_state"
ENV="environment"


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


class ObjectGetter(object):
    def __init__(self):
        self.set_param(None)
        self.planner = planner.Planner(self._learned_folda,
                                       self._learned_category)
        rospy.Service("/gp_hsmm/trajectory/make_trajectory",TrajectoryOrder, self.make_trajectory)
        self.buf = tf2_ros.Buffer()
        self._lis = tf2_ros.TransformListener(self.buf)
        print "#########"
        rospy.loginfo("set up")
        print "#########"


    def set_param(self,data):
        self._learned_folda = rosparam.get_param(FOLDA_PARAM)
        self._learned_category = rosparam.get_param(OBJECT_PARAM).split("/")
        self._hand_frame = rosparam.get_param(HAND_PARAM)


    def set_stamp(self, xyz, qxyzw,name="target_object"):
        ts = Transform()
        ts.translation.x = xyz[0]
        ts.translation.y = xyz[1]
        ts.translation.z = xyz[2]
        ts.rotation.x = qxyzw[0]
        ts.rotation.y = qxyzw[1]
        ts.rotation.z = qxyzw[2]
        ts.rotation.w = qxyzw[3]
        return ts

    def set_transform(self, xyz_qxyzw,name="target_object"):
        ts = Transform()
        xyz = xyz_qxyzw[0]
        qxyzw = xyz_qxyzw[1]
        ts.translation.x =  xyz[0]
        ts.translation.y =  xyz[1]
        ts.translation.z =  xyz[2]
        ts.rotation.x =     qxyzw[0]
        ts.rotation.y =     qxyzw[1]
        ts.rotation.z =     qxyzw[2]
        ts.rotation.w =     qxyzw[3]
        return ts
        

    def make_trajectory(self, data):
        _obj_pose = data.object_pose
        obj_pose = [_obj_pose.position.x,_obj_pose.position.y,_obj_pose.position.z,
                    _obj_pose.orientation.x, _obj_pose.orientation.y, _obj_pose.orientation.z, _obj_pose.orientation.w]
        act_list = data.action_classes
        object_category = data.object_name
        s = rospy.Time.now()
        trajectory_list = []

        _hand = self.buf.lookup_transform("map",self._hand_frame ,rospy.Time.now(),rospy.Duration(3.0))
        h = _hand.transform
        end_effect = [h.translation.x, h.translation.y, h.translation.z,
                      h.rotation.x, h.rotation.y, h.rotation.z,h.rotation.w]
        actor = self.planner.set_motion(act_list,object_category)
        _back = None
        req = TrajectoryOrderResponse()
        for i in range(len(actor)):
            trajectory = self.planner.make_trajector(actor[i], end_effect, obj_pose, object_category, _back)
            motion = Motion()
            for k in range(len(trajectory)):
                xyzqxyzw = trajectory[k]
                ts = self.set_transform(xyzqxyzw)
                motion.trajectory.append(ts)


            trajectory_list.append(trajectory)
            end_effect = [trajectory[-1][0][0], trajectory[-1][0][1], trajectory[-1][0][2], trajectory[-1][1][0], trajectory[-1][1][1], trajectory[-1][1][2], trajectory[-1][1][3]]
            _back = actor[i]
            req.action.append(motion)
        rospy.loginfo(rospy.Time.now().to_sec()-s.to_sec())
        req.success = True
        return req
    def run(self,):
        rate = rospy.Rate(10)
        while not rospy.is_shutdown():
            self.planner.broadcast()
            rate.sleep()


if __name__=="__main__":
    rospy.init_node("test")
    _path =  __file__.split("/")[:-1]
    path  = "/".join(_path) + "/"
    f = open(path + "../config/trajectory_generator.yaml", "r+")
    param = yaml.load(f)
    f.close()

    rospy.set_param(FOLDA_PARAM, path + "action")
    _cat = glob.glob(path + "action/*/")
    cat = [s.split("/")[-2] for s in _cat]
    cat_str = "/".join(cat)
    rospy.set_param(OBJECT_PARAM,cat_str)
    rospy.set_param(HAND_PARAM, param["end_effector_tf"])
    
    obj = ObjectGetter()
    obj.run()
    rospy.spin()

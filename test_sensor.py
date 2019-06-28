# roslaunch realsense2_camera rs_camera.launch
#%%
%reload_ext autoreload
%autoreload 2
%matplotlib inline
import sys
import numpy as np
import matplotlib.pyplot as plt
try:
    sys.path.remove('/opt/ros/kinetic/lib/python2.7/dist-packages')
except ValueError:
    pass  # do nothing!
import cv2
sys.path.append('../')
import logging
import IPython
import os
import rosgraph.roslogging as rl
import rospy
import time
from cv_bridge import CvBridge, CvBridgeError
from sensor_msgs.msg import Image, CameraInfo
from autolab_core import Point, RigidTransform, YamlConfig
from perception import BinaryImage, CameraIntrinsics, ColorImage, DepthImage, RgbdImage
from visualization import Visualizer2D as vis
from gqcnn.grasping import RobustGraspingPolicy, CrossEntropyRobustGraspingPolicy, RgbdImageState, Grasp2D, SuctionPoint2D, GraspAction
#from gqcnn.utils import GripperMode, NoValidGraspsException
# from gqcnn.msg import GQCNNGrasp, BoundingBox
# from gqcnn.srv import GQCNNGraspPlanner, GQCNNGraspPlannerBoundingBox, GQCNNGraspPlannerSegmask
from rosComm import RosComm
#from practical import utils
import sensor_msgs
#from practical.webserver import sampleClient
#from practical.raiRobot import RaiRobot
#from practical.objectives import moveToPosition, gazeAt, scalarProductXZ, scalarProductZZ, distance
#from practical.vision import findBallPosition, calcDepth,findBallInImage, virtCamIntrinsics as intr
#import libry as ry
#%%
rosco = RosComm()
rospy.init_node('z')
#%%
intr = rosco.get_camera_intrinsics('/camera/color/camera_info')
#%%
rosco.subscribe_synced_rgbd('/camera/color/image_raw/', '/camera/depth/image_rect_raw/')
#%%
#rosco.threaded_synced_rgbd_cb('/camera/color/image_raw/', '/camera/depth/image_rect_raw/')
#%%
camera_int = CameraIntrinsics(frame='pcl', fx=intr['fx'], fy=intr['fy'], cx=intr['cx'], cy=intr['cy'], height=intr['height'], width=intr['width'])
#%%
cfg = YamlConfig('cfg/gqcnn_pj_dbg.yaml')
#%%
grasp_policy = CrossEntropyRobustGraspingPolicy(cfg['policy'])
# grasp_policy = RobustGraspingPolicy(cfg['policy'])
#%%
img = rosco.rgb
d = rosco.depth
#%%
color_im = ColorImage(img.astype(np.uint8), encoding="bgr8", frame='pcl')
#%%
depth_im = DepthImage(d.astype(np.float32), frame='pcl')
color_im = color_im.inpaint(rescale_factor=cfg['inpaint_rescale_factor'])
depth_im = depth_im.inpaint(rescale_factor=cfg['inpaint_rescale_factor'])
rgbd_im = RgbdImage.from_color_and_depth(color_im, depth_im)
rgbd_state = RgbdImageState(rgbd_im, cam_intr)

#%%
from cv_bridge import CvBridge, CvBridgeError
latest_rgb = CvBridge.imgmsg_to_cv2('/camera/color/image_raw/', "bgr8")

#%%

# roslaunch realsense2_camera rs_camera.launch filters:=pointcloud

#%%
%reload_ext autoreload
%autoreload 2
%matplotlib inline
#%%
import sys
import numpy as np
import matplotlib.pyplot as plt
import rospy
from autolab_core import Point, RigidTransform, YamlConfig
from perception import BinaryImage, CameraIntrinsics, ColorImage, DepthImage, RgbdImage
from visualization import Visualizer2D as vis
from gqcnn.grasping import RobustGraspingPolicy, CrossEntropyRobustGraspingPolicy, RgbdImageState, Grasp2D, SuctionPoint2D, GraspAction
from rosComm import RosComm
try:
    sys.path.remove('/opt/ros/melodic/lib/python2.7/dist-packages')
except ValueError:
    pass  # do nothing!
import cv2
sys.path.append('../')
#%%
# import logging
# import IPython
# import os
#import rosgraph.roslogging as rl
#import time
#from cv_bridge import CvBridge, CvBridgeError
#from sensor_msgs.msg import Image, CameraInfo
#from gqcnn.utils import GripperMode, NoValidGraspsException
# from gqcnn.msg import GQCNNGrasp, BoundingBox
# from gqcnn.srv import GQCNNGraspPlanner, GQCNNGraspPlannerBoundingBox, GQCNNGraspPlannerSegmask
#from practical import utils
#import sensor_msgs
#from practical.webserver import sampleClient
#from practical.raiRobot import RaiRobot
#from practical.objectives import moveToPosition, gazeAt, scalarProductXZ, scalarProductZZ, distance
#from practical.vision import findBallPosition, calcDepth,findBallInImage, virtCamIntrinsics as intr
#import libry as ry
#%%
rosco = RosComm()
rospy.init_node('z')
#%%
rosco.subscribe_synced_rgbd('/camera/color/image_raw/', '/camera/depth/image_rect_raw/')
#rosco.subscribe_synced_rgbd('/camera/color/image_raw/', '/camera/depth/color/points')
#rosco.subscribe_synced_rgbd('/camera/color/image_raw/', '/camera/aligned_depth_to_color/image_raw/')
#%%
intr = rosco.get_camera_intrinsics('/camera/color/camera_info')
#%%
intr
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
plt.imshow(img)
#%%
plt.imshow(d)
#print(img)
#print(d)
#%%
color_im = ColorImage(img.astype(np.uint8), encoding="bgr8", frame='pcl')
depth_im = DepthImage(d.astype(np.float32), frame='pcl')
color_im = color_im.inpaint(rescale_factor=cfg['inpaint_rescale_factor'])
depth_im = depth_im.inpaint(rescale_factor=cfg['inpaint_rescale_factor'])
rgbd_im = RgbdImage.from_color_and_depth(color_im, depth_im)
rgbd_state = RgbdImageState(rgbd_im, camera_int)
#%%
grasp = grasp_policy(rgbd_state)
#%%
img2 = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
img2 = cv2.circle(img2,(int(grasp.grasp.center.x),int(grasp.grasp.center.y)),2,(255,0,0),3)
plt.imshow(img2)

#%%
img3 = cv2.cvtColor(d, cv2.COLOR_BGR2RGB)
img3 = cv2.circle(img3,(int(grasp.grasp.center.x),int(grasp.grasp.center.y)),2,(255,0,0),3)
plt.imshow(img3)
#%%
vis.figure(size=(16,16))
vis.imshow(rgbd_im.color, vmin=0.5, vmax=2.5)
vis.grasp(grasp.grasp, scale=2.0,jaw_width=2.0, show_center=True, show_axis=True, color=plt.cm.RdYlBu(.1))
vis.title('Elite grasp with score: ' + str(grasp.q_value))
vis.show()
#%%
vis.figure(size=(16,16))
vis.imshow(rgbd_im.depth, vmin=0.5, vmax=2.5)
vis.grasp(grasp.grasp, scale=2.0,jaw_width=2.0, show_center=True, show_axis=True, color=plt.cm.RdYlBu(.1))
vis.title('Elite grasp with score: ' + str(grasp.q_value))
vis.show()
#%%
print(grasp.grasp.angle)
print(grasp.grasp.depth)
print(grasp.grasp.width)
print(grasp.grasp.axis)
#%%
grasp.grasp.approach_angle
#%%
grasp.grasp.pose() #returns the transformation from the grasp to the camera frame of reference
#%%

# Training 
# python3 tools/train.py data/training/Dexnet-2.0_testTraining --config_filename cfg/train_dex-net_2.0_test.yaml --name v3.0_trainedFromScratch

# Analysis
# python3 tools/analyze_gqcnn_performance.py v3.0_trainedFromScratch

# Grasping an Example
# python3 examples/policy.py v3.0_trainedFromScratch --depth_image data/examples/single_object/primesense/depth_4.npy --config_filename cfg/examples/replication/dex-net_2.0.yaml

# Finetuning
# python3 tools/finetune.py data/training/Dexnet-2.0_testFinetuning v3.0_trainedFromScratch --config_filename cfg/finetune_dex-net_2.0_test_trainedFromScratch.yaml --name v3.0_finetuned_trainedFromScratch

# Anaysis
# python3 tools/analyze_gqcnn_performance.py v3.0_finetuned_trainedFromScratch

# Grasing
# python3 examples/policy.py v3.0_finetuned_trainedFromScratch --depth_image data/examples/single_object/primesense/depth_4.npy --config_filename cfg/examples/replication/dex-net_2.0.yaml


# python3 tools/finetune.py data/gen/ABC-1k5-GQ/ GQ-Object-Wise --config_filename cfg/finetune_dex-net_2.0_test_org.yaml --split_name object_wise --name GQ-ABC-1k5-Obj-Wise 
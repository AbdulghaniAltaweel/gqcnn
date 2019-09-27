# -*- coding: utf-8 -*-
#%%
#%reload_ext autoreload
#%autoreload 2
#%matplotlib inline
#%%
from perception import RealSenseSensor
import numpy as np
import matplotlib.pyplot as plt
import rospy
from autolab_core import Point, RigidTransform, YamlConfig
from perception import BinaryImage, CameraIntrinsics, ColorImage, DepthImage, RgbdImage, SegmentationImage
from visualization import Visualizer2D as vis
from gqcnn.grasping import RobustGraspingPolicy, CrossEntropyRobustGraspingPolicy, RgbdImageState, Grasp2D, SuctionPoint2D, GraspAction


#%%

cam_id = '817612071012'
cam = RealSenseSensor(cam_id)

#%%
cam.start()
#%%
color_im, depth_im , nicht = cam.frames()

#myBin = SegmentationImage(depth_im.to_binary(0.01).raw_data)
#segMask = myBin.segment_mask(10)
#segMask.save('my_mask.png')
#%%
#color_im = ColorImage.open('color_im.png')
#depth_im = DepthImage.open('depth_im.png')
#%%
print(color_im.shape)
print(depth_im.shape)
#%%
intr = cam.color_intrinsics
camera_int = CameraIntrinsics(frame=intr.frame, fx=intr.fx, fy=intr.fy, cx=intr.cx, cy=intr.cy, height=intr.height, width=intr.width)
#camera_int.save('mySensor.intr')
#%%
cfg = YamlConfig('cfg/examples/replication/dex-net_2.0.yaml')
grasp_policy = CrossEntropyRobustGraspingPolicy(cfg['policy'])

#%%
#depth_im.threshold(front_thresh=0.00, rear_thresh=75.0)
color_im = color_im.inpaint(rescale_factor=cfg['inpaint_rescale_factor'])
depth_im = depth_im.inpaint(rescale_factor=cfg['inpaint_rescale_factor'])
depth_im.save('my_im_heoo9.npy')
depth_im.save('my_im_heoo9.png')
rgbd_im = RgbdImage.from_color_and_depth(color_im, depth_im)
rgbd_state = RgbdImageState(rgbd_im, cam.color_intrinsics)

#%%
cam.stop()
grasp = grasp_policy(rgbd_state)

#%%
#print(grasp.grasp.center.x)
#print(grasp.grasp.center.y)

#%%
vis.figure(size=(16,16))
vis.imshow(rgbd_im.color, vmin=0.5, vmax=2.5)
vis.grasp(grasp.grasp, scale=2.0,jaw_width=2.0, show_center=True, show_axis=True, color=plt.cm.RdYlBu(.1))
vis.title('Depth: ' + str(grasp.grasp.depth) + 'm ; Elite grasp with score: ' + str(grasp.q_value))
vis.show()
#%%
vis.figure(size=(16,16))
vis.imshow(rgbd_im.depth, vmin=0.0, vmax=2.5)
vis.grasp(grasp.grasp, scale=2.0,jaw_width=2.0, show_center=True, show_axis=True, color=plt.cm.RdYlBu(.1))
vis.title('Depth: ' + str(grasp.grasp.depth) + 'm ; Elite grasp with score: ' + str(grasp.q_value))
vis.show()

#%%
#grasp.grasp.approach_angle
#%%
#grasp.grasp.pose()

#%%


#%%


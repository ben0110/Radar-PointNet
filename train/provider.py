''' Provider class and helper functions for Frustum PointNets.

Author: Charles R. Qi
Date: September 2017
'''
from __future__ import print_function

#import cPickle as pickle
#import pcl
import sys
import os
import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
ROOT_DIR = os.path.dirname(BASE_DIR)
sys.path.append(BASE_DIR)
sys.path.append(os.path.join(ROOT_DIR, 'models'))
#import mayavi
#import mayavi.mlab as mlab

sys.path.append(os.path.join(ROOT_DIR, 'mayavi'))
#from viz_util import draw_lidar, draw_gt_boxes3d

from box_util import box3d_iou
from model_util import g_type2class, g_class2type, g_type2onehotclass
from model_util import g_type_mean_size
from model_util import NUM_HEADING_BIN, NUM_SIZE_CLUSTER

from dataset import KittiDataset
from collections import Counter
import kitti_utils
#import  pickle
import csv
import pandas
from pypcd import pypcd
import math
try:
    raw_input  # Python 2
except NameError:
    raw_input = input  # Python 3


def rotate_pc_along_y(pc, rot_angle):
    '''
    Input:
        pc: numpy array (N,C), first 3 channels are XYZ
            z is facing forward, x is left ward, y is downward
        rot_angle: rad scalar
    Output:
        pc: updated pc with XYZ rotated
    '''
    cosval = np.cos(rot_angle)
    sinval = np.sin(rot_angle)
    rotmat = np.array([[cosval, -sinval], [sinval, cosval]])
    pc[:, [0, 2]] = np.dot(pc[:, [0, 2]], np.transpose(rotmat))
    return pc

def inverse_rotate_pc_along_y(pc, rot_angle):
    '''
    Input:
        pc: numpy array (N,C), first 3 channels are XYZ
            z is facing forward, x is left ward, y is downward
        rot_angle: rad scalar
    Output:
        pc: updated pc with XYZ rotated
    '''
    cosval = np.cos(rot_angle)
    sinval = np.sin(rot_angle)
    rotmat = np.array([[cosval, sinval], [-sinval, cosval]])
    pc[:, [0, 2]] = np.dot(pc[:, [0, 2]], np.transpose(rotmat))
    return pc

def angle2class(angle, num_class):
    ''' Convert continuous angle to discrete class and residual.

    Input:
        angle: rad scalar, from 0-2pi (or -pi~pi), class center at
            0, 1*(2pi/N), 2*(2pi/N) ...  (N-1)*(2pi/N)
        num_class: int scalar, number of classes N
    Output:
        class_id, int, among 0,1,...,N-1
        residual_angle: float, a number such that
            class*(2pi/N) + residual_angle = angle
    '''
    angle = angle % (2 * np.pi)
    assert (angle >= 0 and angle <= 2 * np.pi)
    angle_per_class = 2 * np.pi / float(num_class)
    shifted_angle = (angle + angle_per_class / 2) % (2 * np.pi)
    class_id = int(shifted_angle / angle_per_class)
    residual_angle = shifted_angle - \
                     (class_id * angle_per_class + angle_per_class / 2)
    return class_id, residual_angle


def class2angle(pred_cls, residual, num_class, to_label_format=True):
    ''' Inverse function to angle2class.
    If to_label_format, adjust angle to the range as in labels.
    '''
    angle_per_class = 2 * np.pi / float(num_class)
    angle_center = pred_cls * angle_per_class
    angle = angle_center + residual
    if to_label_format and angle > np.pi:
        angle = angle - 2 * np.pi
    return angle


def size2class(size, type_name):
    ''' Convert 3D bounding box size to template class and residuals.
    todo (rqi): support multiple size clusters per type.

    Input:
        size: numpy array of shape (3,) for (l,w,h)
        type_name: string
    Output:
        size_class: int scalar
        size_residual: numpy array of shape (3,)
    '''
    size_class = g_type2class[type_name]
    size_residual = size - g_type_mean_size[type_name]
    return size_class, size_residual


def class2size(pred_cls, residual):
    ''' Inverse function to size2class. '''
    mean_size = g_type_mean_size[g_class2type[pred_cls]]
    return mean_size + residual
"""
def get_closest_radar_point(center,input_radar):
    cloud = pcl.PointCloud()
    cloud.from_array(input_radar[:,0:3])
    center_pc = pcl.PoinCloud()
    center_pc.from_array(center)
    kdtree = cloud
    [ind,sqdist] = kdtree.nearst_k_search_for_cloud(center_pc,0)
    closest_radar_point=np.array([cloud[ind[0][0]][0],cloud[ind[0][0]][1],cloud[ind[0][0]][2]])
"""


def get_box3d_center(box3d_list):
    ''' Get the center (XYZ) of 3D bounding box. '''
    box3d_center = (box3d_list[0, :] + \
                    box3d_list[6, :]) / 2.0
    return box3d_center
def get_radar_mask(input,input_radar):
    radar_mask = np.zeros((input.shape[0]), dtype=np.float32)
    gt_boxes3d = np.zeros((len(input_radar), 7), dtype=np.float32)
    for k in range(len(input_radar)):
        gt_boxes3d[k, 0] = input_radar[k, 0]
        gt_boxes3d[k, 1] = input_radar[k, 1]
        gt_boxes3d[k, 2] = input_radar[k, 2]+1.0
        gt_boxes3d[k, 3] = 5.0
        gt_boxes3d[k, 4] = (np.tan((7.5)*np.pi/180)*2)* math.sqrt(math.pow(input_radar[k,2],2)+math.pow(input_radar[k,0],2)) + 2.0
        gt_boxes3d[k, 5] = 4.0
        gt_boxes3d[k, 6] = np.arctan2(input_radar[k,0],input_radar[k,2]+1.0)
        print(gt_boxes3d)
    gt_corners = kitti_utils.boxes3d_to_corners3d(gt_boxes3d, transform=False)
    for k in range(len(gt_corners)):
        box_corners = gt_corners[k]
        fg_pt_flag = kitti_utils.in_hull(input[:,0:3], box_corners)
        radar_mask[fg_pt_flag] = 1.0
    radar_masks = []
    radar_masks.append(radar_mask)

    radar_mask_center=[]
    radar_mask_center.append(get_box3d_center(gt_corners[0]))
    """ seg_idx = np.argwhere(radar_mask == 1.0)
    pc_test = input[seg_idx.reshape(-1)]
    fig = mlab.figure(figure=None, bgcolor=(0.4, 0.4, 0.4), fgcolor=None, engine=None,
                      size=(1000, 500))
    mlab.points3d(input[:, 0], input[:, 1], input[:, 2], radar_mask, mode='point', colormap='gnuplot',
                  scale_factor=1, figure=fig)
    mlab.points3d(0, 0, 0, color=(1, 1, 1), mode='sphere', scale_factor=0.2, figure=fig)
    draw_gt_boxes3d([gt_corners[0]], fig, color=(1, 0, 0))
    mlab.orientation_axes()
    fig = plt.figure()
    ax = fig.add_subplot(111, projection="3d")
    ax.scatter(pc_test[:, 0], pc_test[:, 1], pc_test[:, 2], c=pc_test[:, 3:6], s=1)
    plt.show()"""
    return radar_masks,radar_mask_center



def get_radar_masks(input,input_radar):
    center_y=[]
    center_y.append(input_radar[0, 0]- ((np.tan((15)*np.pi/180)*2)* math.sqrt(math.pow(input_radar[0,2],2)+math.pow(input_radar[0,0],2)) + 2.0)/4 )
    center_y.append(input_radar[0, 0])
    center_y.append(input_radar[0, 0]+ ((np.tan((15)*np.pi/180)*2)* math.sqrt(math.pow(input_radar[0,2],2)+math.pow(input_radar[0,0],2)) + 2.0)/4 )
    radar_masks=[]
    radar_center_masks=[]
    corners3d=[]
    for i in range(3):
        gt_boxes3d = np.zeros(( len(input_radar), 7), dtype=np.float32)
        for k in range(len(input_radar)):
            gt_boxes3d[k, 0] = center_y[i]
            gt_boxes3d[k, 1] = input_radar[k, 1]
            gt_boxes3d[k, 2] = input_radar[k, 2]+1.0

            gt_boxes3d[k, 3] = 5.0
            gt_boxes3d[k, 4] = ((np.tan((15.0) * np.pi / 180.0) * 2) * math.sqrt(
                math.pow(input_radar[k, 2], 2) + math.pow(input_radar[k, 0], 2)) + 2.0)/2.0
            gt_boxes3d[k, 5] = 4.0

            gt_boxes3d[k, 6] = np.arctan2(input_radar[k, 0], input_radar[k, 2]+1.0)
        gt_corners = kitti_utils.boxes3d_to_corners3d(gt_boxes3d, transform=False)
        radar_center_masks.append(get_box3d_center(gt_corners[0]))
        radar_mask = np.zeros((input.shape[0]), dtype=np.float32)
        for j in range(len(gt_corners)):
            box_corners = gt_corners[j]
            corners3d.append(box_corners)
            fg_pt_flag = kitti_utils.in_hull(input[:,0:3], box_corners)
            radar_mask[fg_pt_flag] = 1.0
        radar_masks.append(radar_mask)
    """fig = mlab.figure(figure=None, bgcolor=(0.4, 0.4, 0.4), fgcolor=None, engine=None,
                      size=(1000, 500))
    mlab.points3d(input[:, 0], input[:, 1], input[:, 2], radar_masks[0], mode='point', colormap='gnuplot',
                  scale_factor=1, figure=fig)
    mlab.points3d(0, 0, 0, color=(1, 1, 1), mode='sphere', scale_factor=0.2, figure=fig)
    draw_gt_boxes3d(corners3d, fig, color=(1, 0, 0))

    mlab.orientation_axes()
    fig = plt.figure()
    ax = fig.add_subplot(111, projection="3d")
    ax.scatter(input[:, 0], input[:, 1], input[:, 2], c=input[:, 3:6], s=1)
    plt.show()"""


    return radar_masks,radar_center_masks

def load_GT_eval(indice,database,split):
    data_val=KittiDataset('radar_2', root_dir='/root/frustum-pointnets_RSC/dataset/',dataset=database, mode='TRAIN', split=split)
    id_list = data_val.sample_id_list
    obj_frame=[]
    corners_frame=[]
    size_class_frame=[]
    size_residual_frame=[]
    angle_class_frame=[]
    angle_residual_frame=[]
    center_frame=[]
    id_list_new=[]
    for i in range(len(id_list)):
        if(id_list[i]<indice+1):
            gt_obj_list = data_val.filtrate_objects(
                data_val.get_label(id_list[i]))
            #print("GT objs per frame", id_list[i],len(gt_obj_list))
            gt_boxes3d = kitti_utils.objs_to_boxes3d(gt_obj_list)
            gt_corners = kitti_utils.boxes3d_to_corners3d(gt_boxes3d, transform=False)
            obj_frame.append(gt_obj_list)
            corners_frame.append(gt_corners)
            angle_class_list=[]
            angle_residual_list=[]
            size_class_list=[]
            size_residual_list=[]
            center_list=[]
            for j in range(len(gt_obj_list)):

                angle_class, angle_residual = angle2class(gt_boxes3d[j][6],
                                                      NUM_HEADING_BIN)
                angle_class_list.append(angle_class)
                angle_residual_list.append(angle_residual)

                size_class, size_residual = size2class(np.array([gt_boxes3d[j][3], gt_boxes3d[j][4], gt_boxes3d[j][5]]),
                                                   "Pedestrian")
                size_class_list.append(size_class)
                size_residual_list.append(size_residual)

                center_list.append( (gt_corners[j][0, :] + gt_corners[j][6, :]) / 2.0)
            size_class_frame.append(size_class_list)
            size_residual_frame.append(size_residual_list)
            angle_class_frame.append(angle_class_list)
            angle_residual_frame.append(angle_residual_list)
            center_frame.append(center_list)
            id_list_new.append(id_list[i])

    return corners_frame,id_list_new



class FrustumDataset(object):
    ''' Dataset class for Frustum PointNets training/evaluation.
    Load prepared KITTI data from pickled files, return individual data element
    [optional] along with its annotations.
    '''

    def __init__(self,radar_file,database, npoints, split,
                 random_flip=False, random_shift=False, rotate_to_center=False,
                 overwritten_data_path=None, from_rgb_detection=False, one_hot=False,all_batches=False ,translate_radar_center=False,
                 store_data = False,proposals_3=False,no_color=False):#,generate_database=False):
        '''
        Input:
            npoints: int scalar, number of points for frustum point cloud.
            split: string, train or val
            random_flip: bool, in 50% randomly flip the point cloud
                in left and right (after the frustum rotation if any)
            random_shift: bool, if True randomly shift the point cloud
                back and forth by a random distance
            rotate_to_center: bool, whether to do frustum rotation
            overwritten_data_path: string, specify pickled file path.
                if None, use default path (with the split)
            from_rgb_detection: bool, if True we assume we do not have
                groundtruth, just return data elements.
            one_hot: bool, if True, return one hot vector
        '''
        self.no_color = no_color
        self.translate_to_radar_center= translate_radar_center
        self.dataset_kitti = KittiDataset(radar_file,root_dir='/root/frustum-pointnets_RSC/dataset/',dataset=database, mode='TRAIN', split=split)
        self.all_batches = all_batches
        self.npoints = npoints
        self.random_flip = random_flip
        self.random_shift = random_shift
        self.rotate_to_center = rotate_to_center
        self.one_hot = one_hot
        self.pc_lidar_list = []
        if(proposals_3):
            box_number='threeboxes'
        else:
            box_number= 'one_boxes'
        if overwritten_data_path is None:
            overwritten_data_path = os.path.join('/root/frustum-pointnets_RSC_RADAR_fil_PC_batch/',
                                                 'dataset/RSC/radar_'+ box_number +('_%s.pickle' % (split)))
        output_filename = overwritten_data_path
        self.from_rgb_detection = from_rgb_detection


        #list = os.listdir("/root/3D_BoundingBox_Annotation_Tool_3D_BAT/input/NuScenes/ONE/pointclouds_Radar")
        self.id_list = self.dataset_kitti.sample_id_list
        self.idx_batch = self.id_list
        batch_list = []
        self.radar_OI=[]
        self.batch_size = []
        self.batch_train =[]
        batch_list = []
        self.frustum_angle_list = []
        self.input_list = []
        self.label_list = []
        self.box3d_list = []
        self.type_list = []
        self.heading_list = []
        self.size_list = []
        self.radar_point_list =[]
        self.indice_box = []
        nbr_ped=0
        if store_data:
            for i in range(len(self.id_list)):
                pc_radar = self.dataset_kitti.get_radar(self.id_list[i])
                pc_lidar = self.dataset_kitti.get_lidar(self.id_list[i])
                m=0
                for j in range(len(pc_radar)):
                    #print(pc_radar[j].reshape(-1, 3).shape[0])
                    if (pc_radar[j,2]> 2.0):
                        if (proposals_3):
                            radar_masks,radar_center_masks = get_radar_masks(pc_lidar, pc_radar[j].reshape(-1, 3))
                        else:
                            radar_masks,radar_center_masks = get_radar_mask(pc_lidar, pc_radar[j].reshape(-1, 3))
                        for v in range(len(radar_masks)):
                            radar_mask = radar_masks[v]
                            if(np.count_nonzero(radar_mask==1)>300):
                                radar_idx = np.argwhere(radar_mask == 1)
                                pc_fil = pc_lidar[radar_idx.reshape(-1)]
                                """
                                fig = plt.figure()
                                ax = fig.add_subplot(111, projection="3d")
                                ax.scatter(pc_fil[:, 0], pc_fil[:, 1], pc_fil[:, 2], c=pc_fil[:,3:6], s=1)
                                plt.show()
                                """
                                self.radar_OI.append(j)
                                m=m+1
                                #radar_angle = 1.0 * np.arctan2(pc_radar[j,1],pc_radar[j,0]+1.0)
                                radar_angle = - np.arctan2(pc_radar[j,0],pc_radar[j,2]+1.0)

                                cls_label = np.zeros((pc_fil.shape[0]), dtype=np.int32)
                                gt_obj_list = self.dataset_kitti.filtrate_objects(
                                    self.dataset_kitti.get_label(self.id_list[i]))
                                gt_boxes3d = kitti_utils.objs_to_boxes3d(gt_obj_list)
                                gt_corners = kitti_utils.boxes3d_to_corners3d(gt_boxes3d, transform=False)
                                print("id",self.id_list[i])
                                print("corsners,",gt_corners)
                                for k in range(gt_boxes3d.shape[0]):
                                    box_corners = gt_corners[k]
                                    fg_pt_flag = kitti_utils.in_hull(pc_fil[:, 0:3], box_corners)
                                    cls_label[fg_pt_flag] = k + 1

                                print("all labels",np.count_nonzero(cls_label > 0))
                                if (np.count_nonzero(cls_label > 0) < 100):
                                    if(all_batches):

                                        center = np.ones((3)) * (-1.0)
                                        heading = 0.0
                                        size = np.ones((3))
                                        cls_label[cls_label > 0] = 0
                                        seg = cls_label
                                        rot_angle = 0.0
                                        box3d_center = np.random.rand(3) * (-10.0)
                                        print(box3d_center)
                                        box3d = np.array([[box3d_center[0], box3d_center[1], box3d_center[2], size[0], size[1],
                                                       size[2], rot_angle]])
                                        corners_empty = kitti_utils.boxes3d_to_corners3d(box3d, transform=False)
                                        bb_corners = corners_empty[0]
                                        batch = 0
                                        self.indice_box.append(0)
                                    else:
                                        continue

                                else:
                                    max = 0
                                    corners_max = 0

                                    for k in range(gt_boxes3d.shape[0]):
                                        count = np.count_nonzero(cls_label == k + 1)
                                        if count > max:
                                            max = count
                                            corners_max = k
                                    seg = np.where(cls_label == corners_max + 1, 1.0, 0.0)
                                    self.indice_box.append(corners_max + 1)

                                    """seg_idx = np.argwhere(seg == 1.0)
                                    pc_test = pc_fil[seg_idx.reshape(-1)]

                                    fig = mlab.figure(figure=None, bgcolor=(0.4, 0.4, 0.4), fgcolor=None, engine=None,
                                                      size=(1000, 500))
                                    mlab.points3d(pc_fil[:, 0], pc_fil[:, 1], pc_fil[:, 2], seg, mode='point', colormap='gnuplot',
                                                  scale_factor=1, figure=fig)
                                    mlab.points3d(0, 0, 0, color=(1, 1, 1), mode='sphere', scale_factor=0.2, figure=fig)
                                    draw_gt_boxes3d([gt_corners[corners_max]], fig, color=(1, 0, 0))
                                    mlab.orientation_axes()
                                    

                                    fig = plt.figure()
                                    ax = fig.add_subplot(111, projection="3d")
                                    ax.scatter(pc_test[:, 0], pc_test[:, 1], pc_test[:, 2], c=pc_test[:, 3:6], s=1)
                                    plt.show()"""

                                    bb_corners = gt_corners[corners_max]
                                    obj = gt_boxes3d[corners_max]
                                    size = np.array([obj[3], obj[4], obj[5]])
                                    print("size 3asba:", size)
                                    rot_angle = obj[6]
                                    batch = 1
                                print(np.count_nonzero(seg==1))
                                print("radar angle",radar_angle)
                                nbr_ped+=1
                                self.input_list.append(pc_fil)
                                self.frustum_angle_list.append(radar_angle)
                                self.label_list.append(seg)
                                self.box3d_list.append(bb_corners)
                                self.type_list.append("Pedestrian")
                                self.heading_list.append(rot_angle)
                                self.size_list.append(size)
                                self.batch_train.append(batch)
                                #proposal_center = kitti_utils.trans_RSC_to_Kitti(np.array([pc_radar[j]]))
                                self.radar_point_list.append(radar_center_masks[v])
                                batch_list.append(self.id_list[i])
                                #print(len(batch_list))
                                #print(len(self.input_list))



                self.batch_size.append(m)
            self.id_list= batch_list
            print("nbr_ped: ",nbr_ped)
            print("id_list",len(self.id_list))
            print("self.input_list",len(self.input_list))




    def __len__(self):
        return len(self.id_list)

    def __getitem__(self, index):
        ''' Get index-th element from the picked file dataset. '''
        # ------------------------------ INPUTS ----------------------------

        label_mask = self.batch_train[index]
        rot_angle = self.get_center_view_rot_angle(index)

        # Compute one hot vector
        if self.one_hot:
            cls_type = self.type_list[index]
            assert (cls_type in ['Car', 'Pedestrian', 'Cyclist'])
            one_hot_vec = np.zeros((3))
            one_hot_vec[g_type2onehotclass[cls_type]] = 1

        # Get point cloud
        if self.rotate_to_center:
            point_set = self.get_center_view_point_set(index)
        else:
            point_set = self.input_list[index]
        # Resample
        choice = np.random.choice(point_set.shape[0], self.npoints, replace=True)
        point_set = point_set[choice, :]

        # ------------------------------ LABELS ----------------------------
        seg = self.label_list[index]
        seg = seg[choice]

        if(self.no_color):
            ret_pts_features = np.ones((len(point_set), 1))
            point_set = np.concatenate((point_set[:,0:3], ret_pts_features), axis=1)

        # Get center point of 3D box
        if self.rotate_to_center:
            box3d_center = self.get_center_view_box3d_center(index)
            proposal_center = self.get_center_view_proposal(index)

        else:
            box3d_center = self.get_box3d_center(index)
            proposal_center = self.radar_point_list[index]

        if self.translate_to_radar_center:
            box3d_center = box3d_center - proposal_center
            point_set[:,0] = point_set[:,0] - proposal_center[0]
            point_set[:, 1] = point_set[:, 1] - proposal_center[1]
            point_set[:, 2] = point_set[:, 2] - proposal_center[2]
        # Heading
        if self.rotate_to_center:
            heading_angle = self.heading_list[index] - rot_angle
        else:
            heading_angle = self.heading_list[index]

        # Size
        size_class, size_residual = size2class(self.size_list[index],
                                               self.type_list[index])

        # Data Augmentation
        if self.random_flip:
            # note: rot_angle won't be correct if we have random_flip
            # so do not use it in case of random flipping.
            if np.random.random() > 0.5:  # 50% chance flipping
                point_set[:, 0] *= -1
                box3d_center[0] *= -1
                heading_angle = np.pi - heading_angle
        if self.random_shift:
            dist = np.sqrt(np.sum(box3d_center[0] ** 2 + box3d_center[1] ** 2))
            shift = np.clip(np.random.randn() * dist * 0.05, dist * 0.8, dist * 1.2)
            point_set[:, 2] += shift
            box3d_center[2] += shift

        angle_class, angle_residual = angle2class(heading_angle,
                                                  NUM_HEADING_BIN)
        #print("10 points",point_set[0:10])
        if self.one_hot:
            return point_set, seg, box3d_center, angle_class, angle_residual, \
                   size_class, size_residual, rot_angle, one_hot_vec,label_mask
        else:
            return point_set, seg, box3d_center, angle_class, angle_residual, \
                   size_class, size_residual, rot_angle,label_mask
    def get_center_view_rot_angle(self, index):
        ''' Get the frustum rotation angle, it isshifted by pi/2 so that it
        can be directly used to adjust GT heading angle '''
        return np.pi / 2.0 + self.frustum_angle_list[index]
        #return self.frustum_angle_list[index]

    def get_box3d_center(self, index):
        ''' Get the center (XYZ) of 3D bounding box. '''
        box3d_center = (self.box3d_list[index][0, :] + \
                        self.box3d_list[index][6, :]) / 2.0
        return box3d_center

    def get_center_view_box3d_center(self, index):
        ''' Frustum rotation of 3D bounding box center. '''
        box3d_center = (self.box3d_list[index][0, :] + \
                        self.box3d_list[index][6, :]) / 2.0
        return rotate_pc_along_y(np.expand_dims(box3d_center, 0), \
                                 self.get_center_view_rot_angle(index)).squeeze()
    def get_center_view_proposal(self,index):
        return rotate_pc_along_y(np.expand_dims(self.radar_point_list[index], 0), \
                                 self.get_center_view_rot_angle(index)).squeeze()

    def get_center_view_box3d(self, index):
        ''' Frustum rotation of 3D bounding box corners. '''
        box3d = self.box3d_list[index]
        box3d_center_view = np.copy(box3d)
        return rotate_pc_along_y(box3d_center_view, \
                                 self.get_center_view_rot_angle(index))

    def get_center_view_point_set(self, index):
        ''' Frustum rotation of point clouds.
        NxC points with first 3 channels as XYZ
        z is facing forward, x is left ward, y is downward
        '''
        # Use np.copy to avoid corrupting original data
        point_set = np.copy(self.input_list[index])
        return rotate_pc_along_y(point_set, \
                                 self.get_center_view_rot_angle(index))
def get_radar_pc_mask(input, input_radar):
    print("radar pc mask")
    radar_mask = np.zeros((input.shape[0]), dtype=np.float32)
    RoI_boxes3d = np.zeros((len(input_radar), 7), dtype=np.float32)
    for k in range(len(input_radar)):
        # print(pc_radar[j].reshape(-1, 3).shape[0])
        RoI_boxes3d[k, 0] = input_radar[k, 0]
        RoI_boxes3d[k, 1] = input_radar[k, 1]
        RoI_boxes3d[k, 2] = input_radar[k, 2] + 1.0
        RoI_boxes3d[k, 3] = 5.0
        RoI_boxes3d[k, 4] = (np.tan(15.0 * np.pi / 180.0) * 2) * math.sqrt(
            math.pow(input_radar[k, 2], 2) + math.pow(input_radar[k, 0], 2))
        RoI_boxes3d[k, 5] = 4.0
        RoI_boxes3d[k, 6] = np.arctan2(input_radar[k, 0], input_radar[k, 2] + 1.0)
        gt_corners = kitti_utils.boxes3d_to_corners3d(RoI_boxes3d, transform=False)
    radar_mask_list = []
    for k in range(len(gt_corners)):
        box_corners = gt_corners[k]
        fg_pt_flag = kitti_utils.in_hull(input[:, 0:3], box_corners)
        radar_mask_local = np.zeros((input.shape[0]), dtype=np.float32)
        radar_mask_local[fg_pt_flag] = 1.0
        radar_mask_list.append(radar_mask_local)
        radar_mask[fg_pt_flag] = 1.0

    return radar_mask, radar_mask_list, RoI_boxes3d
def get_bins_in_RRoI(point_set, Radar_roi):
    centers = []
    bin_pc = []
    trans = np.array([Radar_roi[0], Radar_roi[1], Radar_roi[2]])
    pc = point_set - trans
    pc = rotate_pc_along_y(pc, Radar_roi[6])
    min = np.array([np.min(pc[:, 0]), np.min(pc[:, 1]), np.min(pc[:, 2])])
    max = np.array([np.max(pc[:, 0]), np.max(pc[:, 1]), np.max(pc[:, 2])])
    corners = corneers_from_minmax(min, max)
    center = (min + max) / 2.0
    l = abs(max[2] - min[2])
    h = abs(max[1] - min[1])
    center_1 = center
    center_2 = center
    w = 1.0 / 8.0
    ds = 0
    boxes_1 = get_3d_box((h, w, l), 0.0, center)
    fg_pt_flag_1 = kitti_utils.in_hull(pc[:, 0:3], boxes_1)
    if (np.count_nonzero(fg_pt_flag_1 == 1) > 0):
        pc_1 = pc[fg_pt_flag_1, :]
        bin_pc.append(pc_1)
        centers.append(center)
    else:
        bin_pc.append(np.array([]))
        centers.append(center)
    size = [h, w, l]
    while center_2[0] < max[0]:
        center_1 = [center_1[0] - 1.0 / 8.0, center_1[1], center_1[2]]
        center_2 = [center_2[0] + 1.0 / 8.0, center_2[1], center_2[2]]
        boxes_1 = get_3d_box((h, w, l), 0.0, center_1)
        boxes_2 = get_3d_box((h, w, l), 0.0, center_2)
        fg_pt_flag_1 = kitti_utils.in_hull(pc[:, 0:3], boxes_1)
        fg_pt_flag_2 = kitti_utils.in_hull(pc[:, 0:3], boxes_2)
        if np.count_nonzero(fg_pt_flag_1 == 1) > 0:
            pc_1 = pc[fg_pt_flag_1, :]
            bin_pc.append(pc_1)
            centers.append(center_1)
        else:
            bin_pc.append(np.array([]))
            centers.append(center_1)

        if np.count_nonzero(fg_pt_flag_2 == 1) > 0:
            pc_2 = pc[fg_pt_flag_2, :]
            bin_pc.insert(0, pc_2)
            centers.insert(0, center_2)
        else:
            bin_pc.insert(0, np.array([]))
            centers.insert(0, center_2)
        fg_pt_flag = np.logical_or(fg_pt_flag_1, fg_pt_flag_2)
        pc = pc[~fg_pt_flag, :]
    return bin_pc, centers, size, trans

def corneers_from_minmax(min, max):
    corners = np.zeros((8, 3))
    corners[0,] = [min[0], max[1], min[2]]
    corners[1,] = [min[0], max[1], max[2]]
    corners[2,] = [max[0], max[1], max[2]]
    corners[3,] = [max[0], max[1], min[2]]
    corners[4,] = [min[0], min[1], min[2]]
    corners[5,] = [min[0], min[1], max[2]]
    corners[6,] = [max[0], min[1], max[2]]
    corners[7,] = [max[0], min[1], min[2]]
    return corners



def local_min_method(bin_pc, centers, size, radar_angle, trans):
    print(len(bin_pc), len(centers))
    bin_y_max = []
    for i in range(len(bin_pc)):
        if (bin_pc[i].size == 0):
            bin_y_max.append(centers[i][1] + size[0] / 2)
        else:
            bin_y_max.append(np.min(bin_pc[i][:, 1]))

    minimum = []
    if (bin_y_max[0] < bin_y_max[1]):
        minimum.append(1)
    else:
        minimum.append(-1)
    for m in range(1, len(bin_y_max) - 1):
        if (bin_y_max[m] < bin_y_max[m - 1] and bin_y_max[m] < bin_y_max[m + 1]):
            minimum.append(1)
        elif (bin_y_max[m] > bin_y_max[m - 1] and bin_y_max[m] > bin_y_max[m + 1]):
            minimum.append(-1)
        else:
            minimum.append(0)
    if (bin_y_max[len(bin_y_max) - 1] < bin_y_max[len(bin_y_max) - 1]):
        minimum.append(1)
    else:
        minimum.append(-1)
    print(minimum)
    local_min_indices = np.argwhere(np.array(minimum) == -1)
    pc_AB_list = []
    corners_AB = []
    for n in range(len(local_min_indices)):
        pc_AB = np.empty([0, 3])
        for m in range(n + 1, len(local_min_indices)):
            for o in range(local_min_indices[n][0], local_min_indices[m][0]):
                if (bin_pc[o].size != 0):
                    pc_AB = np.concatenate((pc_AB, bin_pc[o]))
            print("pc_AB_list:", len(pc_AB_list))
            if (len(pc_AB) > 0):
                min = np.array([np.min(pc_AB[:, 0]), np.min(pc_AB[:, 1]), np.min(pc_AB[:, 2])])
                max = np.array([np.max(pc_AB[:, 0]), np.max(pc_AB[:, 1]), np.max(pc_AB[:, 2])])
                corners = corneers_from_minmax(min, max)
                center = (min + max) / 2.0

                corners = inverse_rotate_pc_along_y(corners, radar_angle)
                corners = corners + trans
                pc = inverse_rotate_pc_along_y(pc_AB, radar_angle)
                pc = pc + trans
                pc_AB_list.append(pc)
                corners_AB.append(corners)
    return pc_AB_list, corners_AB

def divide_in_n_AB(bin_pc, n):
    pc_AB_list = []
    for i in range(0, len(bin_pc) - n, 1):
        pc_AB = np.empty([0, 3])
        # print(len(bin_pc))
        # print(i,i+n)
        for j in range(i, i + n):
            # print(j)
            # print(bin_pc[j].size)
            if bin_pc[j].size != 0:
                pc_AB = np.concatenate((pc_AB, bin_pc[j]))
        pc_AB_list.append(pc_AB)
    return pc_AB_list


def iterative_method(bin_pc, centers, size, radar_angle, trans):
    pc_AB_list = []
    corners_AB = []
    for i in range(3, 8):
        pc_AB_ = divide_in_n_AB(bin_pc, i)
        for pc_ in pc_AB_:
            if (len(pc_) > 0):
                min = np.array([np.min(pc_[:, 0]), np.min(pc_[:, 1]), np.min(pc_[:, 2])])
                max = np.array([np.max(pc_[:, 0]), np.max(pc_[:, 1]), np.max(pc_[:, 2])])
                corners = corneers_from_minmax(min, max)
                center = (min + max) / 2.0

                corners = inverse_rotate_pc_along_y(corners, radar_angle)
                corners = corners + trans
                corners_AB.append(corners)
                pc = inverse_rotate_pc_along_y(pc_, radar_angle)
                pc = pc + trans

                pc_AB_list.append(pc)
    return pc_AB_list, corners_AB

class RadarDataset(object):
    ''' Dataset class for Frustum PointNets training/evaluation.
    Load prepared KITTI data from pickled files, return individual data element
    [optional] along with its annotations.
    '''

    def __init__(self, radar_file, database,npoints,split,
                 random_flip=False, random_shift=False, rotate_to_center=False,
                 overwritten_data_path=None, from_rgb_detection=False, one_hot=False, all_batches=False,
                 translate_radar_center=False,
                 store_data=False, proposals_3=False, no_color=False):  # ,generate_database=False):
        '''
        Input:
            npoints: int scalar, number of points for frustum point cloud.
            split: string, train or val
            random_flip: bool, in 50% randomly flip the point cloud
                in left and right (after the frustum rotation if any)
            random_shift: bool, if True randomly shift the point cloud
                back and forth by a random distance
            rotate_to_center: bool, whether to do frustum rotation
            overwritten_data_path: string, specify pickled file path.
                if None, use default path (with the split)
            from_rgb_detection: bool, if True we assume we do not have
                groundtruth, just return data elements.
            one_hot: bool, if True, return one hot vector
        '''
        self.no_color = no_color
        self.translate_to_radar_center = translate_radar_center
        self.dataset_kitti = KittiDataset(radar_file, root_dir='/root/frustum-pointnets_RSC/dataset/',dataset=database, mode='TRAIN',
                                          split=split)
        self.all_batches = all_batches
        self.npoints = npoints
        self.random_flip = random_flip
        self.random_shift = random_shift
        self.rotate_to_center = rotate_to_center
        self.one_hot = one_hot
        self.pc_lidar_list = []
        if (proposals_3):
            box_number = 'threeboxes'
        else:
            box_number = 'one_boxes'
        if overwritten_data_path is None:
            overwritten_data_path = os.path.join('/root/frustum-pointnets_RSC_RADAR_fil_PC_batch/',
                                                 'dataset/RSC/radar_' + box_number + ('_%s.pickle' % (split)))
        output_filename = overwritten_data_path
        self.from_rgb_detection = from_rgb_detection

        # list = os.listdir("/root/3D_BoundingBox_Annotation_Tool_3D_BAT/input/NuScenes/ONE/pointclouds_Radar")
        self.id_list = self.dataset_kitti.sample_id_list
        self.idx_batch = self.id_list
        batch_list = []
        self.radar_OI = []
        self.batch_size = []
        self.batch_train = []
        batch_list = []
        self.frustum_angle_list = []
        self.input_list = []
        self.label_list = []
        self.box3d_list = []
        self.type_list = []
        self.heading_list = []
        self.size_list = []
        self.radar_point_list = []
        self.indice_box = []
        nbr_ped = 0
        if store_data:
            for i in range(len(self.id_list)):
                pc_radar = self.dataset_kitti.get_radar(self.id_list[i])
                pc_lidar = self.dataset_kitti.get_lidar(self.id_list[i])
                m = 0
                for j in range(len(pc_radar)):
                    # print(pc_radar[j].reshape(-1, 3).shape[0])
                    if (pc_radar[j, 2] > 2.0):
                        """if (proposals_3):
                            radar_masks, radar_center_masks = get_radar_masks(pc_lidar, pc_radar[j].reshape(-1, 3))
                        else:
                            radar_masks, radar_center_masks = get_radar_mask(pc_lidar, pc_radar[j].reshape(-1, 3))"""
                        radar_mask, radar_mask_list, RoI_boxes_3d = get_radar_pc_mask(pc_lidar, pc_radar)
                        for v in range(len(radar_mask_list)):
                            radar_mask = radar_mask_list[v]
                            print("radar_mask:", np.count_nonzero(radar_mask == 1))
                            if (np.count_nonzero(radar_mask == 1) > 300):
                                radar_idx = np.argwhere(radar_mask == 1)
                                pc_fil = pc_lidar[radar_idx.reshape(-1)]
                                gt_obj_list = self.dataset_kitti.filtrate_objects(
                                    self.dataset_kitti.get_label(self.id_list[i]))
                                gt_boxes3d = kitti_utils.objs_to_boxes3d(gt_obj_list)
                                gt_corners = kitti_utils.boxes3d_to_corners3d(gt_boxes3d, transform=False)
                                cls_label = np.zeros((pc_fil.shape[0]), dtype=np.int32)
                                for k in range(gt_boxes3d.shape[0]):
                                    box_corners = gt_corners[k]
                                    fg_pt_flag = kitti_utils.in_hull(pc_fil[:, 0:3], box_corners)
                                    cls_label[fg_pt_flag] = 1
                                if (float(np.count_nonzero(
                                        cls_label == 1)) > 100 and split == "train") or split == "val" or split == "test":
                                    bin_pc, centers, size, trans = get_bins_in_RRoI(pc_fil[:,0:3], RoI_boxes_3d[j])
                                    AB_pc, AB_corners = local_min_method(bin_pc, centers, size, RoI_boxes_3d[j][6],
                                                                         trans)
                                    #AB_pc, AB_corners = iterative_method(bin_pc, centers, size, RoI_boxes_3d[j][6], trans)
                                    for k in range(len(AB_corners)):
                                        for m in range(len(gt_corners)):
                                            # print("corners AB", AB_corners[k])
                                            # print("gt_corners[m]", gt_corners[m])
                                            if len(np.unique(AB_corners[k][:, 0])) == 1:
                                                continue
                                            iou_3d, iou_2d = box3d_iou(AB_corners[k], gt_corners[m])
                                            print(iou_3d)
                                            seg=[]
                                            bb_corners=[]
                                            rot_angle=0
                                            batch=0

                                            if iou_3d >= 0.3:
                                                """fig = mlab.figure(figure=None, bgcolor=(0.4, 0.4, 0.4), fgcolor=None, engine=None,
                                                                  size=(1000, 500))
                                                mlab.points3d(AB_pc[k][:, 0], AB_pc[k][:, 1], AB_pc[k][:, 2], mode='point',
                                                              colormap='gnuplot', scale_factor=1,
                                                              figure=fig)
                                                mlab.points3d(0, 0, 0, color=(1, 1, 1), mode='sphere', scale_factor=0.2, figure=fig)
                                                draw_gt_boxes3d([gt_corners[m]], fig, color=(1, 0, 0))
                                                draw_gt_boxes3d([AB_corners[k]], fig, color=(0, 0, 1))
                                                mlab.orientation_axes()
                                                raw_input()"""
                                                cls_label = np.zeros((AB_pc[k].shape[0]), dtype=np.int32)
                                                fg_pt_flag = kitti_utils.in_hull(AB_pc[k][:, 0:3], gt_corners[m])
                                                cls_label[fg_pt_flag] = 1
                                                seg = cls_label
                                                bb_corners = gt_corners[m]
                                                obj = gt_boxes3d[m]
                                                size = np.array([obj[3], obj[4], obj[5]])
                                                print("size 3asba:", size)
                                                rot_angle = obj[6]
                                                batch = 1

                                            elif iou_3d < 0.3  and (split == 'val' or split == 'test'):
                                                center = np.ones((3)) * (-1.0)
                                                heading = 0.0
                                                size = np.ones((3))
                                                cls_label = np.zeros((AB_pc[k].shape[0]), dtype=np.int32)
                                                seg = cls_label
                                                rot_angle = 0.0
                                                box3d_center = np.random.rand(3) * (-10.0)
                                                print(box3d_center)
                                                box3d = np.array(
                                                    [[box3d_center[0], box3d_center[1], box3d_center[2], size[0], size[1],
                                                      size[2], rot_angle]])
                                                corners_empty = kitti_utils.boxes3d_to_corners3d(box3d, transform=False)
                                                bb_corners = corners_empty[0]
                                                batch = 0
                                                self.indice_box.append(10)

                                                """self.AB.append(AB_pc[k])
                                                self.type_list.append("Pedestrian")
                                                self.box3d_list.append(gt_corners[m])
                                                self.AB_list.append(AB_corners[k])
                                                self.size_list.append([gt_boxes3d[m][3], gt_boxes3d[m][4], gt_boxes3d[m][5]])
                                                self.heading_list.append(gt_boxes3d[m][6])
                                                self.batch_list.append(self.ids[i])
                                                self.indice_box.append(m)"""
                                            #print(np.count_nonzero(seg == 1))
                                            #print("radar angle", radar_angle)
                                            nbr_ped += 1
                                            self.input_list.append(pc_fil)
                                            #self.frustum_angle_list.append(radar_angle)
                                            self.label_list.append(seg)
                                            self.box3d_list.append(bb_corners)
                                            self.type_list.append("Pedestrian")
                                            self.heading_list.append(rot_angle)
                                            self.size_list.append(size)
                                            self.batch_train.append(batch)
                                            # proposal_center = kitti_utils.trans_RSC_to_Kitti(np.array([pc_radar[j]]))
                                            #self.radar_point_list.append(radar_center_masks[v])
                                            batch_list.append(self.id_list[i])
                                """
                                fig = plt.figure()
                                ax = fig.add_subplot(111, projection="3d")
                                ax.scatter(pc_fil[:, 0], pc_fil[:, 1], pc_fil[:, 2], c=pc_fil[:,3:6], s=1)
                                plt.show()
                                """
                                """self.radar_OI.append(j)
                                m = m + 1
                                # radar_angle = 1.0 * np.arctan2(pc_radar[j,1],pc_radar[j,0]+1.0)
                                radar_angle = - np.arctan2(pc_radar[j, 0], pc_radar[j, 2] + 1.0)

                                cls_label = np.zeros((pc_fil.shape[0]), dtype=np.int32)
                                gt_obj_list = self.dataset_kitti.filtrate_objects(
                                    self.dataset_kitti.get_label(self.id_list[i]))
                                gt_boxes3d = kitti_utils.objs_to_boxes3d(gt_obj_list)
                                gt_corners = kitti_utils.boxes3d_to_corners3d(gt_boxes3d, transform=False)
                                print("id", self.id_list[i])
                                print("corsners,", gt_corners)
                                for k in range(gt_boxes3d.shape[0]):
                                    box_corners = gt_corners[k]
                                    fg_pt_flag = kitti_utils.in_hull(pc_fil[:, 0:3], box_corners)
                                    cls_label[fg_pt_flag] = k + 1

                                print("all labels", np.count_nonzero(cls_label > 0))
                                if (np.count_nonzero(cls_label > 0) < 100):
                                    if (all_batches):

                                        center = np.ones((3)) * (-1.0)
                                        heading = 0.0
                                        size = np.ones((3))
                                        cls_label[cls_label > 0] = 0
                                        seg = cls_label
                                        rot_angle = 0.0
                                        box3d_center = np.random.rand(3) * (-10.0)
                                        print(box3d_center)
                                        box3d = np.array(
                                            [[box3d_center[0], box3d_center[1], box3d_center[2], size[0], size[1],
                                              size[2], rot_angle]])
                                        corners_empty = kitti_utils.boxes3d_to_corners3d(box3d, transform=False)
                                        bb_corners = corners_empty[0]
                                        batch = 0
                                        self.indice_box.append(0)
                                    else:
                                        continue

                                else:
                                    max = 0
                                    corners_max = 0

                                    for k in range(gt_boxes3d.shape[0]):
                                        count = np.count_nonzero(cls_label == k + 1)
                                        if count > max:
                                            max = count
                                            corners_max = k
                                    seg = np.where(cls_label == corners_max + 1, 1.0, 0.0)
                                    self.indice_box.append(corners_max + 1)

                                    seg_idx = np.argwhere(seg == 1.0)
                                    pc_test = pc_fil[seg_idx.reshape(-1)]

                                    fig = mlab.figure(figure=None, bgcolor=(0.4, 0.4, 0.4), fgcolor=None, engine=None,
                                                      size=(1000, 500))
                                    mlab.points3d(pc_fil[:, 0], pc_fil[:, 1], pc_fil[:, 2], seg, mode='point', colormap='gnuplot',
                                                  scale_factor=1, figure=fig)
                                    mlab.points3d(0, 0, 0, color=(1, 1, 1), mode='sphere', scale_factor=0.2, figure=fig)
                                    draw_gt_boxes3d([gt_corners[corners_max]], fig, color=(1, 0, 0))
                                    mlab.orientation_axes()


                                    fig = plt.figure()
                                    ax = fig.add_subplot(111, projection="3d")
                                    ax.scatter(pc_test[:, 0], pc_test[:, 1], pc_test[:, 2], c=pc_test[:, 3:6], s=1)
                                    plt.show()

                                    bb_corners = gt_corners[corners_max]
                                    obj = gt_boxes3d[corners_max]
                                    size = np.array([obj[3], obj[4], obj[5]])
                                    print("size 3asba:", size)
                                    rot_angle = obj[6]
                                    batch = 1"""

                                # print(len(batch_list))
                                # print(len(self.input_list))

                self.batch_size.append(m)
            self.id_list = batch_list
            print("nbr_ped: ", nbr_ped)
            print("id_list", len(self.id_list))
            print("self.input_list", len(self.input_list))

    def __len__(self):
        return len(self.id_list)

    def __getitem__(self, index):
        ''' Get index-th element from the picked file dataset. '''
        # ------------------------------ INPUTS ----------------------------

        label_mask = self.batch_train[index]
        rot_angle = self.get_center_view_rot_angle(index)

        # Compute one hot vector
        if self.one_hot:
            cls_type = self.type_list[index]
            assert (cls_type in ['Car', 'Pedestrian', 'Cyclist'])
            one_hot_vec = np.zeros((3))
            one_hot_vec[g_type2onehotclass[cls_type]] = 1

        # Get point cloud
        if self.rotate_to_center:
            point_set = self.get_center_view_point_set(index)
        else:
            point_set = self.input_list[index]
        # Resample
        choice = np.random.choice(point_set.shape[0], self.npoints, replace=True)
        point_set = point_set[choice, :]

        # ------------------------------ LABELS ----------------------------
        seg = self.label_list[index]
        seg = seg[choice]

        if (self.no_color):
            ret_pts_features = np.ones((len(point_set), 1))
            point_set = np.concatenate((point_set[:, 0:3], ret_pts_features), axis=1)

        # Get center point of 3D box
        if self.rotate_to_center:
            box3d_center = self.get_center_view_box3d_center(index)
            proposal_center = self.get_center_view_proposal(index)

        else:
            box3d_center = self.get_box3d_center(index)
            proposal_center = self.radar_point_list[index]

        if self.translate_to_radar_center:
            box3d_center = box3d_center - proposal_center
            point_set[:, 0] = point_set[:, 0] - proposal_center[0]
            point_set[:, 1] = point_set[:, 1] - proposal_center[1]
            point_set[:, 2] = point_set[:, 2] - proposal_center[2]
        # Heading
        if self.rotate_to_center:
            heading_angle = self.heading_list[index] - rot_angle
        else:
            heading_angle = self.heading_list[index]

        # Size
        size_class, size_residual = size2class(self.size_list[index],
                                               self.type_list[index])

        # Data Augmentation
        if self.random_flip:
            # note: rot_angle won't be correct if we have random_flip
            # so do not use it in case of random flipping.
            if np.random.random() > 0.5:  # 50% chance flipping
                point_set[:, 0] *= -1
                box3d_center[0] *= -1
                heading_angle = np.pi - heading_angle
        if self.random_shift:
            dist = np.sqrt(np.sum(box3d_center[0] ** 2 + box3d_center[1] ** 2))
            shift = np.clip(np.random.randn() * dist * 0.05, dist * 0.8, dist * 1.2)
            point_set[:, 2] += shift
            box3d_center[2] += shift

        angle_class, angle_residual = angle2class(heading_angle,
                                                  NUM_HEADING_BIN)
        # print("10 points",point_set[0:10])
        if self.one_hot:
            return point_set, seg, box3d_center, angle_class, angle_residual, \
                   size_class, size_residual, rot_angle, one_hot_vec, label_mask
        else:
            return point_set, seg, box3d_center, angle_class, angle_residual, \
                   size_class, size_residual, rot_angle, label_mask

    def get_center_view_rot_angle(self, index):
        ''' Get the frustum rotation angle, it isshifted by pi/2 so that it
        can be directly used to adjust GT heading angle '''
        return np.pi / 2.0 + self.frustum_angle_list[index]
        #return self.frustum_angle_list[index]

    def get_box3d_center(self, index):
        ''' Get the center (XYZ) of 3D bounding box. '''
        box3d_center = (self.box3d_list[index][0, :] + \
                        self.box3d_list[index][6, :]) / 2.0
        return box3d_center

    def get_center_view_box3d_center(self, index):
        ''' Frustum rotation of 3D bounding box center. '''
        box3d_center = (self.box3d_list[index][0, :] + \
                        self.box3d_list[index][6, :]) / 2.0
        return rotate_pc_along_y(np.expand_dims(box3d_center, 0), \
                                 self.get_center_view_rot_angle(index)).squeeze()
    def get_center_view_proposal(self,index):
        return rotate_pc_along_y(np.expand_dims(self.radar_point_list[index], 0), \
                                 self.get_center_view_rot_angle(index)).squeeze()

    def get_center_view_box3d(self, index):
        ''' Frustum rotation of 3D bounding box corners. '''
        box3d = self.box3d_list[index]
        box3d_center_view = np.copy(box3d)
        return rotate_pc_along_y(box3d_center_view, \
                                 self.get_center_view_rot_angle(index))

    def get_center_view_point_set(self, index):
        ''' Frustum rotation of point clouds.
        NxC points with first 3 channels as XYZ
        z is facing forward, x is left ward, y is downward
        '''
        # Use np.copy to avoid corrupting original data
        point_set = np.copy(self.input_list[index])
        return rotate_pc_along_y(point_set, \
                                 self.get_center_view_rot_angle(index))


# ----------------------------------
# Helper functions for evaluation
# ----------------------------------

def get_3d_box(box_size, heading_angle, center):
    ''' Calculate 3D bounding box corners from its parameterization.

    Input:
        box_size: tuple of (l,w,h)
        heading_angle: rad scalar, clockwise from pos x axis
        center: tuple of (x,y,z)
    Output:
        corners_3d: numpy array of shape (8,3) for 3D box cornders
    '''

    def roty(t):
        c = np.cos(t)
        s = np.sin(t)
        return np.array([[c, 0, s],
                         [0, 1, 0],
                         [-s, 0, c]])

    R = roty(heading_angle)

    h, w, l = box_size
    x_corners = [w / 2, w / 2, -w / 2, -w / 2, w / 2, w / 2, -w / 2, -w / 2];
    y_corners = [h / 2, h / 2, h / 2, h / 2, -h / 2, -h / 2, -h / 2, -h / 2];
    z_corners = [l / 2, -l / 2, -l / 2, l / 2, l / 2, -l / 2, -l / 2, l / 2];
    corners_3d = np.dot(R, np.vstack([x_corners, y_corners, z_corners]))
    corners_3d[0, :] = corners_3d[0, :] + center[0];
    corners_3d[1, :] = corners_3d[1, :] + center[1];
    corners_3d[2, :] = corners_3d[2, :] + center[2];
    corners_3d = np.transpose(corners_3d)
    return corners_3d

def get_3d_box_batch(center, angle_class, angle_res, \
                                    size_class, size_res, rot_angle):
    ''' Calculate 3D bounding box corners from its parameterization.

    Input:
        box_size: tuple of (l,w,h)
        heading_angle: rad scalar, clockwise from pos x axis
        center: tuple of (x,y,z)
    Output:
        corners_3d: numpy array of shape (8,3) for 3D box cornders
    '''

    def roty(t):
        c = np.cos(t)
        s = np.sin(t)
        return np.array([[c, 0, s],
                         [0, 1, 0],
                         [-s, 0, c]])

    R = roty(class2angle(angle_class, angle_res, NUM_HEADING_BIN) + rot_angle)
    h, w, l = class2size(size_class, size_res)
    x_corners = [w / 2, w / 2, -w / 2, -w / 2, w / 2, w / 2, -w / 2, -w / 2];
    y_corners = [h / 2, h / 2, h / 2, h / 2, -h / 2, -h / 2, -h / 2, -h / 2];
    z_corners = [l / 2, -l / 2, -l / 2, l / 2, l / 2, -l / 2, -l / 2, l / 2];
    corners_3d = np.dot(R, np.vstack([x_corners, y_corners, z_corners]))
    corners_3d[0, :] = corners_3d[0, :] + center[0];
    corners_3d[1, :] = corners_3d[1, :] + center[1];
    corners_3d[2, :] = corners_3d[2, :] + center[2];
    corners_3d = np.transpose(corners_3d)
    return corners_3d

def compute_box3d_iou_batch_test(center_pred,
                      heading_class, heading_residual,
                      size_class, size_residual,
                      corners_3d_label):
    ''' Compute 3D bounding box IoU from network output and labels.
    All inputs are numpy arrays.

    Inputs:
        center_pred: (B,3)
        heading_logits: (B,NUM_HEADING_BIN)
        heading_residuals: (B,NUM_HEADING_BIN)
        size_logits: (B,NUM_SIZE_CLUSTER)
        size_residuals: (B,NUM_SIZE_CLUSTER,3)
        center_label: (B,3)
        heading_class_label: (B,)
        heading_residual_label: (B,)
        size_class_label: (B,)
        size_residual_label: (B,3)
    Output:
        iou2ds: (B,) birdeye view oriented 2d box ious
        iou3ds: (B,) 3d box ious
    '''
    batch_size = heading_class.shape[0]
    #heading_class = np.argmax(heading_logits, 1)  # B
    #heading_residual = np.array([heading_residuals[i, heading_class[i]] \
    #                             for i in range(batch_size)])  # B,
    #size_class = np.argmax(size_logits, 1)  # B
    #size_residual = np.vstack([size_residuals[i, size_class[i], :] \
    #                           for i in range(batch_size)])
    #print(heading_class_label, heading_residual_label)
    iou2d_list = []
    iou3d_list = []

    for i in range(batch_size):
        heading_angle = class2angle(heading_class[i],
                                    heading_residual[i], NUM_HEADING_BIN)
        box_size = class2size(size_class[i], size_residual[i])
        corners_3d = get_3d_box(box_size, heading_angle, center_pred[i])

        #heading_angle_label = class2angle(heading_class_label[i],
        #                                  heading_residual_label[i], NUM_HEADING_BIN)
        #box_size_label = class2size(size_class_label[i], size_residual_label[i])
        #corners_3d_label = get_3d_box(box_size_label,
        #                              heading_angle_label, center_label[i])
        for j in range(len(corners_3d_label)):
            iou_3d, iou_2d = box3d_iou(corners_3d, corners_3d_label[j])
            print(corners_3d, corners_3d_label[j])
            print("iou_3d:", iou_3d,"iou_2d" , iou_2d)
            iou3d_list.append(iou_3d)
            iou2d_list.append(iou_2d)
    return np.array(iou2d_list, dtype=np.float32), \
           np.array(iou3d_list, dtype=np.float32)

def compute_box3d_iou_batch(logits,center_pred,
                      heading_logits, heading_residuals,
                      size_logits, size_residuals,
                      center_label,
                      heading_class_label, heading_residual_label,
                      size_class_label, size_residual_label):
    ''' Compute 3D bounding box IoU from network output and labels.
    All inputs are numpy arrays.

    Inputs:
        center_pred: (B,3)
        heading_logits: (B,NUM_HEADING_BIN)
        heading_residuals: (B,NUM_HEADING_BIN)
        size_logits: (B,NUM_SIZE_CLUSTER)
        size_residuals: (B,NUM_SIZE_CLUSTER,3)
        center_label: (B,3)
        heading_class_label: (B,)
        heading_residual_label: (B,)
        size_class_label: (B,)
        size_residual_label: (B,3)
    Output:
        iou2ds: (B,) birdeye view oriented 2d box ious
        iou3ds: (B,) 3d box ious
    '''
    pred_val = np.argmax(logits, 2)
    batch_size = heading_logits.shape[0]
    heading_class = np.argmax(heading_logits, 1)  # B
    heading_residual = np.array([heading_residuals[i, heading_class[i]] \
                                 for i in range(batch_size)])  # B,
    size_class = np.argmax(size_logits, 1)  # B
    size_residual = np.vstack([size_residuals[i, size_class[i], :] \
                               for i in range(batch_size)])

    iou2d_list = []
    iou3d_list = []
    box_pred_nbr= 0.0
    for i in range(batch_size):
        # if object has low seg mask break
        if(np.sum(pred_val[i])<50):
            continue
        else:
            heading_angle = class2angle(heading_class[i],
                                        heading_residual[i], NUM_HEADING_BIN)
            box_size = class2size(size_class[i], size_residual[i])
            corners_3d = get_3d_box(box_size, heading_angle, center_pred[i])

            heading_angle_label = class2angle(heading_class_label[i],
                                              heading_residual_label[i], NUM_HEADING_BIN)
            box_size_label = class2size(size_class_label[i], size_residual_label[i])
            if (center_label[i][2] < 0.0):
                iou3d_list.append(0.0)
                iou2d_list.append(0.0)
            else:

                corners_3d_label = get_3d_box(box_size_label,
                                          heading_angle_label, center_label[i])

                iou_3d, iou_2d = box3d_iou(corners_3d, corners_3d_label)
                iou3d_list.append(iou_3d)
                iou2d_list.append(iou_2d)
            box_pred_nbr = box_pred_nbr+1.0

    return np.array(iou2d_list, dtype=np.float32), \
           np.array(iou3d_list, dtype=np.float32), np.array(box_pred_nbr,dtype=np.float32)
def compute_box3d_iou_batch_test1(output,center_pred,
                      heading_logits, heading_residuals,
                      size_logits, size_residuals,
                      center_label,
                      heading_class_label, heading_residual_label,
                      size_class_label, size_residual_label):
    ''' Compute 3D bounding box IoU from network output and labels.
    All inputs are numpy arrays.

    Inputs:
        center_pred: (B,3)
        heading_logits: (B,NUM_HEADING_BIN)
        heading_residuals: (B,NUM_HEADING_BIN)
        size_logits: (B,NUM_SIZE_CLUSTER)
        size_residuals: (B,NUM_SIZE_CLUSTER,3)
        center_label: (B,3)
        heading_class_label: (B,)
        heading_residual_label: (B,)
        size_class_label: (B,)
        size_residual_label: (B,3)
    Output:
        iou2ds: (B,) birdeye view oriented 2d box ious
        iou3ds: (B,) 3d box ious
    '''
    #pred_val = np.argmax(logits, 2)
    batch_size = heading_logits.shape[0]
    heading_class = heading_logits #np.argmax(heading_logits, 1)  # B
    heading_residual = heading_residuals#np.array([heading_residuals[i, heading_class[i]] \
                        #         for i in range(batch_size)])  # B,
    size_class =size_logits #np.argmax(size_logits, 1)  # B
    size_residual =size_residuals# np.vstack([size_residuals[i, size_class[i], :] \
                               #for i in range(batch_size)])

    iou2d_list = []
    iou3d_list = []
    box_pred_nbr= 0.0
    for i in range(batch_size):
        # if object has low seg mask continue
        if(np.sum(output[i])<50):
            continue
        else:
            heading_angle = class2angle(heading_class[i],
                                        heading_residual[i], NUM_HEADING_BIN)
            box_size = class2size(size_class[i], size_residual[i])
            corners_3d = get_3d_box(box_size, heading_angle, center_pred[i])

            heading_angle_label = class2angle(heading_class_label[i],
                                              heading_residual_label[i], NUM_HEADING_BIN)
            box_size_label = class2size(size_class_label[i], size_residual_label[i])
            if (center_label[i][2]<0.0):
                iou3d_list.append(0.0)
                iou2d_list.append(0.0)
            else:

                corners_3d_label = get_3d_box(box_size_label,
                                          heading_angle_label, center_label[i])

                iou_3d, iou_2d = box3d_iou(corners_3d, corners_3d_label)
                iou3d_list.append(iou_3d)
                iou2d_list.append(iou_2d)
            box_pred_nbr = box_pred_nbr+1.0

    return np.array(iou2d_list, dtype=np.float32), \
           np.array(iou3d_list, dtype=np.float32), np.array(box_pred_nbr,dtype=np.float32)


def compute_box3d_iou(center_pred,
                      heading_logits, heading_residuals,
                      size_logits, size_residuals,
                      center_label,
                      heading_class_label, heading_residual_label,
                      size_class_label, size_residual_label):
    ''' Compute 3D bounding box IoU from network output and labels.
    All inputs are numpy arrays.

    Inputs:
        center_pred: (B,3)
        heading_logits: (B,NUM_HEADING_BIN)
        heading_residuals: (B,NUM_HEADING_BIN)
        size_logits: (B,NUM_SIZE_CLUSTER)
        size_residuals: (B,NUM_SIZE_CLUSTER,3)
        center_label: (B,3)
        heading_class_label: (B,)
        heading_residual_label: (B,)
        size_class_label: (B,)
        size_residual_label: (B,3)
    Output:
        iou2ds: (B,) birdeye view oriented 2d box ious
        iou3ds: (B,) 3d box ious
    '''
    batch_size = heading_logits.shape[0]
    heading_class = np.argmax(heading_logits, 1)  # B
    heading_residual = np.array([heading_residuals[i, heading_class[i]] \
                                 for i in range(batch_size)])  # B,
    size_class = np.argmax(size_logits, 1)  # B
    size_residual = np.vstack([size_residuals[i, size_class[i], :] \
                               for i in range(batch_size)])

    iou2d_list = []
    iou3d_list = []
    for i in range(batch_size):
        # if object has low seg mask break
        heading_angle = class2angle(heading_class[i],
                                    heading_residual[i], NUM_HEADING_BIN)
        box_size = class2size(size_class[i], size_residual[i])
        corners_3d = get_3d_box(box_size, heading_angle, center_pred[i])

        heading_angle_label = class2angle(heading_class_label[i],
                                          heading_residual_label[i], NUM_HEADING_BIN)
        box_size_label = class2size(size_class_label[i], size_residual_label[i])
        corners_3d_label = get_3d_box(box_size_label,
                                      heading_angle_label, center_label[i])

        iou_3d, iou_2d = box3d_iou(corners_3d, corners_3d_label)
        iou3d_list.append(iou_3d)
        iou2d_list.append(iou_2d)
    return np.array(iou2d_list, dtype=np.float32), \
           np.array(iou3d_list, dtype=np.float32)


def from_prediction_to_label_format(center, angle_class, angle_res, \
                                    size_class, size_res, rot_angle):
    ''' Convert predicted box parameters to label format. '''
    l, w, h = class2size(size_class, size_res)
    ry = class2angle(angle_class, angle_res, NUM_HEADING_BIN) + rot_angle
    tx, ty, tz = rotate_pc_along_y(np.expand_dims(center, 0), -rot_angle).squeeze()
    #ty += h / 2.0
    return h, w, l, tx, ty, tz, ry


if __name__ == '__main__':
    import mayavi.mlab as mlab

    sys.path.append(os.path.join(ROOT_DIR, 'mayavi'))
    from viz_util import draw_lidar, draw_gt_boxes3d

    median_list = []
    dataset = FrustumDataset('pc_radar_2',npoints=3500, split='val',
    rotate_to_center=False, one_hot=True,all_batches = False, translate_radar_center=False, store_data=True, proposals_3 =False ,no_color=False)
    for i in range(len(dataset)):

        data = dataset[i]
        print("frame nbr", dataset.id_list[i])
        print(('Center: ', data[2], \
               'angle_class: ', data[3], 'angle_res:', data[4], \
               'size_class: ', data[5], 'size_residual:', data[6], \
               'real_size:', g_type_mean_size[g_class2type[data[5]]] + data[6]))
        print("radar_point", dataset.radar_point_list[i])
        print(('Frustum angle: ', dataset.frustum_angle_list[i]))
        median_list.append(np.median(data[0][:, 0]))
        print((data[2], dataset.box3d_list[i], median_list[-1]))
        box3d_from_label = get_3d_box(class2size(data[5], data[6]), class2angle(data[3], data[4], 12), data[2])
        print(box3d_from_label)
        print("angle: ",dataset.heading_list[i],class2angle(data[3], data[4], 12))
        ps = data[0]
        print("ps shape", ps.shape)
        seg = data[1]
        fig = mlab.figure(figure=None, bgcolor=(0.4, 0.4, 0.4), fgcolor=None, engine=None, size=(1000, 500))
        mlab.points3d(ps[:, 0], ps[:, 1], ps[:, 2], seg, mode='point', colormap='gnuplot', scale_factor=1, figure=fig)
        mlab.points3d(0, 0, 0, color=(1, 1, 1), mode='sphere', scale_factor=0.2, figure=fig)
        draw_gt_boxes3d([box3d_from_label], fig, color=(1, 0, 0))
        mlab.orientation_axes()
        raw_input()
    print(np.mean(np.abs(median_list)))

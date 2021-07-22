''' Evaluating Frustum PointNets.
Write evaluation results to KITTI format labels.
and [optionally] write results to pickle files.

Author: Charles R. Qi
Date: September 2017
'''
from __future__ import print_function
from datetime import datetime
import time
import os
import sys
import argparse
import importlib
import numpy as np
import tensorflow as tf
import cPickle as pickle
from shutil import copyfile
import json

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
ROOT_DIR = os.path.dirname(BASE_DIR)
sys.path.append(BASE_DIR)
# sys.path.append(os.path.join(ROOT_DIR, 'models'))
# sys.path.append(os.path.join(ROOT_DIR, '/train/log_v2/15-02-2020-23:59:37/'))
sys.path.append(os.path.join(ROOT_DIR, 'models'))
from model_util import NUM_HEADING_BIN, NUM_SIZE_CLUSTER
import  provider
from train_util import get_batch_test

parser = argparse.ArgumentParser()
parser.add_argument('--gpu', type=int, default=0, help='GPU to use [default: GPU 0]')
parser.add_argument('--num_point', type=int, default=1024, help='Point Number [default: 1024]')
parser.add_argument('--model', default='frustum_pointnets_v1', help='Model name [default: frustum_pointnets_v1]')
parser.add_argument('--model_path', default='log/model.ckpt',
                    help='model checkpoint file path [default: log/model.ckpt]')
parser.add_argument('--batch_size', type=int, default=32, help='batch size for inference [default: 32]')
parser.add_argument('--output', default='test_results', help='output file/folder name [default: test_results]')
parser.add_argument('--data_path', default=None, help='frustum dataset pickle filepath [default: None]')
parser.add_argument('--from_rgb_detection', action='store_true', help='test from dataset files from rgb detection.')
parser.add_argument('--idx_path', default=None,
                    help='filename of txt where each line is a data idx, used for rgb detection -- write <id>.txt for all frames. [default: None]')
parser.add_argument('--dump_result', action='store_true', help='If true, also dump results to .pickle file')
FLAGS = parser.parse_args()

# Set training configurations
BATCH_SIZE = FLAGS.batch_size
MODEL_PATH = FLAGS.model_path
GPU_INDEX = FLAGS.gpu
NUM_POINT = FLAGS.num_point
MODEL = importlib.import_module(FLAGS.model)
NUM_CLASSES = 2
NUM_CHANNEL = 4

pathsplit = MODEL_PATH.split('/')

OUTPUT_FILE = os.path.join('/', pathsplit[1], pathsplit[2], pathsplit[3], pathsplit[4], pathsplit[5], 'results/')
if not os.path.exists(OUTPUT_FILE):
    os.mkdir(OUTPUT_FILE)

# Load Frustum Datasets.
TEST_DATASET = provider.FrustumDataset('pc_radar_2',database="KITTI",npoints=NUM_POINT, split='val',rotate_to_center=False, one_hot=True,all_batches = True, translate_radar_center=False, store_data=True, proposals_3 =False ,no_color=True)


def get_session_and_ops(batch_size, num_point):
    ''' Define model graph, load model parameters,
    create session and return session handle and tensors
    '''
    with tf.Graph().as_default():
        with tf.device('/gpu:' + str(GPU_INDEX)):
            pointclouds_pl, one_hot_vec_pl, labels_pl, centers_pl, \
            heading_class_label_pl, heading_residual_label_pl, \
            size_class_label_pl, size_residual_label_pl, batch_mask_pl = \
                MODEL.placeholder_inputs_batch(batch_size, num_point)
            is_training_pl = tf.placeholder(tf.bool, shape=())
            end_points = MODEL.get_model(pointclouds_pl, one_hot_vec_pl,
                                         is_training_pl)
            loss = MODEL.get_loss_batch(labels_pl, batch_mask_pl, centers_pl,
                                        heading_class_label_pl, heading_residual_label_pl,
                                        size_class_label_pl, size_residual_label_pl, end_points)
            saver = tf.train.Saver()

        # Create a session
        config = tf.ConfigProto()
        config.gpu_options.allow_growth = True
        config.allow_soft_placement = True
        sess = tf.Session(config=config)

        # Restore variables from disk.
        saver.restore(sess, MODEL_PATH)
        ops = {'pointclouds_pl': pointclouds_pl,
               'one_hot_vec_pl': one_hot_vec_pl,
               'labels_pl': labels_pl,
               'centers_pl': centers_pl,
               'heading_class_label_pl': heading_class_label_pl,
               'heading_residual_label_pl': heading_residual_label_pl,
               'size_class_label_pl': size_class_label_pl,
               'size_residual_label_pl': size_residual_label_pl,
               'is_training_pl': is_training_pl,
               'logits': end_points['mask_logits'],
               'center': end_points['center'],
               'end_points': end_points,
               'loss': loss}
        return sess, ops


def softmax(x):
    ''' Numpy function for softmax'''
    shape = x.shape
    probs = np.exp(x - np.max(x, axis=len(shape) - 1, keepdims=True))
    probs /= np.sum(probs, axis=len(shape) - 1, keepdims=True)
    return probs


def inference(sess, ops, pc, one_hot_vec, batch_size):
    ''' Run inference for frustum pointnets in batch mode '''
    assert pc.shape[0] % batch_size == 0
    num_batches = pc.shape[0] / batch_size
    logits = np.zeros((pc.shape[0], pc.shape[1], NUM_CLASSES))
    centers = np.zeros((pc.shape[0], 3))
    heading_logits = np.zeros((pc.shape[0], NUM_HEADING_BIN))
    heading_residuals = np.zeros((pc.shape[0], NUM_HEADING_BIN))
    size_logits = np.zeros((pc.shape[0], NUM_SIZE_CLUSTER))
    size_residuals = np.zeros((pc.shape[0], NUM_SIZE_CLUSTER, 3))
    scores = np.zeros((pc.shape[0],))  # 3D box score

    ep = ops['end_points']
    for i in range(num_batches):
        print(i)
        feed_dict = { \
            ops['pointclouds_pl']: pc[i * batch_size:(i + 1) * batch_size, ...],
            ops['one_hot_vec_pl']: one_hot_vec[i * batch_size:(i + 1) * batch_size, :],
            ops['is_training_pl']: False}

        batch_logits, batch_centers, \
        batch_heading_scores, batch_heading_residuals, \
        batch_size_scores, batch_size_residuals = \
            sess.run([ops['logits'], ops['center'],
                      ep['heading_scores'], ep['heading_residuals'],
                      ep['size_scores'], ep['size_residuals']],
                     feed_dict=feed_dict)

        logits[i * batch_size:(i + 1) * batch_size, ...] = batch_logits
        centers[i * batch_size:(i + 1) * batch_size, ...] = batch_centers
        heading_logits[i * batch_size:(i + 1) * batch_size, ...] = batch_heading_scores
        heading_residuals[i * batch_size:(i + 1) * batch_size, ...] = batch_heading_residuals
        size_logits[i * batch_size:(i + 1) * batch_size, ...] = batch_size_scores
        size_residuals[i * batch_size:(i + 1) * batch_size, ...] = batch_size_residuals

        # Compute scores
        batch_seg_prob = softmax(batch_logits)[:, :, 1]  # BxN
        batch_seg_mask = np.argmax(batch_logits, 2)  # BxN
        mask_mean_prob = np.sum(batch_seg_prob * batch_seg_mask, 1)  # B,
        mask_mean_prob = mask_mean_prob / np.sum(batch_seg_mask, 1)  # B,
        heading_prob = np.max(softmax(batch_heading_scores), 1)  # B
        size_prob = np.max(softmax(batch_size_scores), 1)  # B,
        batch_scores = np.log(mask_mean_prob) + np.log(heading_prob) + np.log(size_prob)
        scores[i * batch_size:(i + 1) * batch_size] = batch_scores
        # Finished computing scores

    heading_cls = np.argmax(heading_logits, 1)  # B
    size_cls = np.argmax(size_logits, 1)  # B
    heading_res = np.array([heading_residuals[i, heading_cls[i]] \
                            for i in range(pc.shape[0])])
    size_res = np.vstack([size_residuals[i, size_cls[i], :] \
                          for i in range(pc.shape[0])])

    return np.argmax(logits, 2), centers, heading_cls, heading_res, \
           size_cls, size_res, scores


def write_detection_results(result_dir, id_list, type_list, box2d_list, center_list, \
                            heading_cls_list, heading_res_list, \
                            size_cls_list, size_res_list, \
                            rot_angle_list, score_list, ):
    ''' Write frustum pointnets results to KITTI format label files. '''
    if result_dir is None: return
    results = {}  # map from idx to list of strings, each string is a line (without \n)
    for i in range(len(center_list)):

        idx = id_list[i]
        output_str = type_list[i] + " -1 -1 -10 "
        box2d = box2d_list[i]
        output_str += "%f %f %f %f " % (box2d[0], box2d[1], box2d[2], box2d[3])
        h, w, l, tx, ty, tz, ry = provider.from_prediction_to_label_format(center_list[i],
                                                                           heading_cls_list[i], heading_res_list[i],
                                                                           size_cls_list[i], size_res_list[i],
                                                                           rot_angle_list[i])
        score = score_list[i]
        output_str += "%f %f %f %f %f %f %f %f" % (h, w, l, tx, ty, tz, ry, score)
        if idx not in results: results[idx] = []
        results[idx].append(output_str)

    # Write TXT files
    if not os.path.exists(result_dir): os.mkdir(result_dir)
    output_dir = os.path.join(result_dir, 'data')
    if not os.path.exists(output_dir): os.mkdir(output_dir)
    for idx in results:
        pred_filename = os.path.join(output_dir, '%06d.txt' % (idx))
        fout = open(pred_filename, 'w')
        for line in results[idx]:
            fout.write(line + '\n')
        fout.close()


def write_detection_results_test(result_dir, id_list, center_list, \
                                 heading_cls_list, heading_res_list, \
                                 size_cls_list, size_res_list, \
                                 rot_angle_list, score_list, segp_list):
    ''' Write frustum pointnets results to KITTI format label files. '''
    result_dir = OUTPUT_FILE
    if result_dir is None: return
    results = {}  # map from idx to list of strings, each string is a line (without \n)

    for i in range(len(segp_list)):
        if np.count_nonzero(segp_list[i] == 1) < 5:
            continue
        idx = id_list[i]

        output_str = "Pedestrian -1 -1 -10 "
        output_str += "0.0 0.0 0.0 0.0 "
        h, w, l, tx, ty, tz, ry = provider.from_prediction_to_label_format(center_list[i],
                                                                           heading_cls_list[i], heading_res_list[i],
                                                                           size_cls_list[i], size_res_list[i], 0.0)
        score = score_list[i]
        output_str += "%f %f %f %f %f %f %f %f" % (h, w, l, tx, ty, tz, ry, score)
        if idx not in results: results[idx] = []
        results[idx].append(output_str)

    # Write TXT files
    if not os.path.exists(result_dir): os.mkdir(result_dir)
    output_dir = os.path.join(result_dir, 'data')

    if not os.path.exists(output_dir): os.mkdir(output_dir)
    for idx in results:
        pred_filename = os.path.join(output_dir, '%06d.txt' % (idx))
        print(pred_filename)
        fout = open(pred_filename, 'w')
        for line in results[idx]:
            fout.write(line + '\n')
        fout.close()


def write_detection_results_3dbat(result_dir, id_list, center_list, \
                                  heading_cls_list, heading_res_list, \
                                  size_cls_list, size_res_list, \
                                  rot_angle_list, score_list):
    orig_path = "/root/3D_BoundingBox_Annotation_Tool_3D_BAT/input/NuScenes/ONE/"
    # copy files to 3d bbat
    i = 0
    for idx in id_list:
        # copy pcd file
        copyfile("/root/frustum-pointnets_RSC/dataset/KITTI/object/training/velodyne/%06d.pcd" % idx,
                 orig_path + "/pointclouds/%06d.pcd" % i)
        copyfile("/root/frustum-pointnets_RSC/dataset/KITTI/object/training/image_left/%06d.jpg" % idx,
                 orig_path + "/images/CAM_FRONT_LEFT/%06d.jpg" % i)
        copyfile("/root/frustum-pointnets_RSC/dataset/KITTI/object/training/image_right/%06d.jpg" % idx,
                 orig_path + "/images/CAM_FRONT/%06d.jpg" % i)
        i = i + 1

    # generate json annotation
    annotation_frame = []
    annotation = []
    for i in range(len(center_list)):
        h, w, l, tx, ty, tz, ry = provider.from_prediction_to_label_format(center_list[i],
                                                                           heading_cls_list[i], heading_res_list[i],
                                                                           size_cls_list[i], size_res_list[i],
                                                                           rot_angle_list[i])
        x = tz
        z = -ty
        y = -tx
        th = 2
        tw = 3
        tl = 4

        # if id_list[i] not in annotation[:]["frameIdx"]:
        #   annotation_frame.append(annotation)
        #   annotation = []
        annotation.append(
            {"class": "Pedestrian", "width": tw, "length": tl, "height": th, "x": x, "y": y, "z": z, "rotationY": ry,
             "trackId": 1, "frameIdx": id_list[i]})
    json_format = []
    annotation_frame = []
    annotation_frame.append(annotation[0])
    m = 0
    for i in range(1, len(annotation)):
        if annotation_frame[0]["frameIdx"] != annotation[i]["frameIdx"]:
            json_format.append(annotation_frame)
            annotation_frame = []
            m = m + 1
        annotation_frame.append(annotation[i])
    json_format.append(annotation_frame)
    for i in range(len(json_format)):
        for j in range(len(json_format[i])):
            json_format[i][j]["frameIdx"] = i
            json_format[i][j]["trackId"] = j
    print(json_format)
    with open(orig_path + "/annotations/NuScenes_ONE_annotations.txt", 'w') as outfile:
        json.dump(json_format, outfile)


def fill_files(output_dir, to_fill_filename_list):
    ''' Create empty files if not exist for the filelist. '''
    for filename in to_fill_filename_list:
        filepath = os.path.join(output_dir, filename)
        if not os.path.exists(filepath):
            fout = open(filepath, 'w')
            fout.close()


def test(output_filename, result_dir=None):
    ''' Test frustum pointnets with GT 2D boxes.
    Write test results to KITTI format label files.
    todo (rqi): support variable number of points.
    '''
    ps_list = []
    seg_list = []
    segp_list = []
    center_list = []
    heading_cls_list = []
    heading_res_list = []
    size_cls_list = []
    size_res_list = []
    rot_angle_list = []
    score_list = []

    test_idxs = np.arange(0, len(TEST_DATASET))
    batch_size = BATCH_SIZE
    num_batches = len(TEST_DATASET.idx_batch)
    total_seen_class = [0 for _ in range(NUM_CLASSES)]
    total_correct_class = [0 for _ in range(NUM_CLASSES)]
    sess, ops = get_session_and_ops(batch_size=batch_size, num_point=NUM_POINT)
    correct_cnt = 0
    t = []
    iou2ds_sum = 0.0
    iou3ds_sum = 0.0
    iou3d_correct_cnt = 0.0
    box_pred_nbr_sum = 0.0
    print(TEST_DATASET.idx_batch)
    print(TEST_DATASET.id_list)
    tss=0
    not_found=[]
    for batch_idx in range(num_batches):
        print('batch idx: %d' % (batch_idx))

        #start_idx = batch_idx * batch_size
        #end_idx = (batch_idx + 1) * batch_size
        indices = np.argwhere(np.asarray(TEST_DATASET.id_list) == TEST_DATASET.idx_batch[batch_idx])

        indices=indices.reshape(-1)
        print(indices)
        if len(indices)==0:
            not_found.append(TEST_DATASET.idx_batch[batch_idx])
            continue
        print(indices)
        start_idx = np.min(indices)
        end_idx = np.max(indices)
        print("start_idx: ", start_idx)
        print("end_idx:",  end_idx)
        batch_size = 1
        batch_data, batch_label, batch_center, \
        batch_hclass, batch_hres, batch_sclass, batch_sres, \
        batch_rot_angle, batch_one_hot_vec,batch_mask = \
            get_batch_test(TEST_DATASET, test_idxs, start_idx, end_idx,
                      NUM_POINT, NUM_CHANNEL)
        start_t = time.time()
        batch_output, batch_center_pred, \
        batch_hclass_pred, batch_hres_pred, \
        batch_sclass_pred, batch_sres_pred, batch_scores = \
            inference(sess, ops, batch_data,
                      batch_one_hot_vec, batch_size=1)
        end_t = time.time()

        t.append(end_t - start_t)
        print("inference time: ", end_t - start_t)
        correct_cnt += np.sum(batch_output == batch_label)

        iou2ds, iou3ds, box_pred_nbr = provider.compute_box3d_iou_batch_test1(batch_output, batch_center_pred,
                                                                             batch_hclass_pred, batch_hres_pred,
                                                                             batch_sclass_pred, batch_sres_pred,
                                                                             batch_center,
                                                                             batch_hclass, batch_hres,
                                                                             batch_sclass, batch_sres)
        for l in range(NUM_CLASSES):
            total_seen_class[l] += np.sum(batch_label==l)
            total_correct_class[l] += (np.sum((batch_output==l) & (batch_label==l)))

        iou2ds_sum += np.sum(iou2ds)
        iou3ds_sum += np.sum(iou3ds)
        iou3d_correct_cnt += np.sum(iou3ds >= 0.5)
        box_pred_nbr_sum += box_pred_nbr
        for i in range(batch_output.shape[0]):
            ps_list.append(batch_data[i, ...])
            seg_list.append(batch_label[i, ...])
            segp_list.append(batch_output[i, ...])
            center_list.append(batch_center_pred[i, :])
            heading_cls_list.append(batch_hclass_pred[i])
            heading_res_list.append(batch_hres_pred[i])
            size_cls_list.append(batch_sclass_pred[i])
            size_res_list.append(batch_sres_pred[i, :])
            rot_angle_list.append(batch_rot_angle[i])
            score_list.append(batch_scores[i])
    print('box IoU (ground/3D): %f / %f' % \
          (iou2ds_sum / max(float(box_pred_nbr_sum), 1.0), iou3ds_sum / max(float(box_pred_nbr_sum), 1.0)))
    print('box estimation accuracy (IoU=0.5): %f' % \
          (float(iou3d_correct_cnt) / max(float(box_pred_nbr_sum), 1.0)))
    print("Segmentation accuracy: %f" % \
          (correct_cnt / float(batch_size * num_batches * NUM_POINT)))
    print("average time", np.mean(t))
    print("not_found",not_found)
    print('eval segmentation avg class acc: %f' % \
        (np.mean(np.array(total_correct_class) / \
            np.array(total_seen_class,dtype=np.float))))

    # Write detection results for KITTI evaluation
    """ write_detection_results_test(result_dir, TEST_DATASET.id_list,
                                 center_list,
                                 heading_cls_list, heading_res_list,
                                 size_cls_list, size_res_list, rot_angle_list, score_list, segp_list)"""
    # write_detection_results_3dbat(result_dir,  TEST_DATASET.id_list, center_list, \
    #                             heading_cls_list, heading_res_list, \
    #                            size_cls_list, size_res_list, \
    #                           rot_angle_list, score_list)


if __name__ == '__main__':
    if FLAGS.from_rgb_detection:
        test_from_rgb_detection(FLAGS.output + '.pickle', FLAGS.output)
    else:
        test(FLAGS.output + '.pickle', FLAGS.output)

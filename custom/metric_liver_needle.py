from sklearn.linear_model import LinearRegression
import numpy as np
from pycocotools.mask import decode as m_decode
from pycocotools.coco import COCO
import math
import json
import pandas as pd
import os
import matplotlib.pyplot as plt
import seaborn as sns
import sys
import cv2
from mmdet.core import BboxOverlaps2D
import torch
import copy
import pickle
import itertools
from random import shuffle
from mmtrack.datasets.parsers import CocoVID
import matplotlib.pyplot as plt
import gc
from sklearn.metrics import confusion_matrix
from medpy.metric import dc, jc, hd, asd, hd95, sensitivity, specificity
from PIL import Image

data_dir = r'path/to/your/dataset'
resolution_dict = json.load(open(r'path/to/your/dataset/DataSet/Documents/resolution.json', 'r'))
frame_type_xlsx = pd.ExcelFile(r'path/to/your/dataset/DataSet/tip_type.xlsx')
sns.set_context("paper", rc={"font.size":20,"axes.titlesize":20,"axes.labelsize":20})   

# Ablation phase origin dict {v_name: (start_time, end_time)}
ablation_phase_origin_dict = {
}

# RFA list [p001, p002, ...]
RFA_list = []

class EvalCombine():
    def __init__(self, 
                 result_dir, 
                 anno_dir, 
                 fold_dict=None,
                 data_dir=None, 
                 save_path=None, 
                 top_n=3, 
                 IoU_thres=0.3, 
                 deg_thres=None, 
                 is_visu=False, 
                 filter_dict={'visu': -1, 'abla': -1},
                 is_ext=False,
                 is_save_mask=False) -> None:
        self.IoU_calculator = BboxOverlaps2D()
        self.line_reg = LinearRegression()
        self.top_n = top_n
        self.IoU_thres = IoU_thres
        self.deg_thres = deg_thres
        self.is_visu = is_visu
        self.result_dir = result_dir
        self.anno_dir = anno_dir
        self.fold_dict = fold_dict
        self.data_dir = data_dir
        self.save_path = save_path
        self.filter_dict = filter_dict
        self.is_ext = is_ext
        self.is_save_mask = is_save_mask
        self.metric_list = ['dis_bbox', 'dis_seg', 'dis_inter', 'error_angle']
        if is_save_mask:
            self.metric_mask_list = ['dice', 'iou', 'hd95', 'sen', 'spe']
        color_list =  [p for p in itertools.product([255, 128, 0], repeat=3)]
        color_list = [p for p in color_list if (sum(p) >= 128 * 3 and sum(p) != 255 *3)]
        shuffle(color_list)
        self.color_list = [[255, 255, 255]] + color_list
        
        self.ablation_phase_origin_dict = ablation_phase_origin_dict
        self.ablation_phase_dict = {}
        self.video_filter_dict = {}
        self.gt_video_dict = {}
        self.per_video_dict = {}
        self.score_dict = {}

    def eval_dataset(self, anno_dir=None, result_dir=None, is_save=True):
        anno_dir = anno_dir if anno_dir else self.anno_dir
        result_dir = result_dir if result_dir else self.result_dir
        assert os.path.isfile(anno_dir) and os.path.isfile(result_dir)
        coco_vid = CocoVID(anno_dir)
        v_ids = coco_vid.get_vid_ids()
        result_dict = np.load(result_dir, allow_pickle=True)
        if isinstance(result_dict, dict) and 'det_bboxes' in result_dict.keys() and 'det_masks' in result_dict.keys():
            bbox_list = result_dict['det_bboxes']
            mask_list = result_dict['det_masks']
        elif isinstance(result_dict, dict) and 'det_bboxes' in result_dict.keys():
            bbox_list = result_dict['det_bboxes']
            bbox_list = [[b] for b in bbox_list]
            mask_list = []
            for bbox_frame in bbox_list:
                mask_frame = [None for i in bbox_frame[0][0]]
                mask_list.append([mask_frame])
        elif isinstance(result_dict, dict) and 'track_bboxes' in result_dict.keys():
            bbox_list = [np.expand_dims(result_dict['track_bboxes'][i][0][:, 1:], axis=(0, 1)) for i in range(len(result_dict['track_bboxes']))]
            mask_list = result_dict['track_masks']
        elif isinstance(result_dict, list):
            bbox_list = [np.expand_dims(np.array(res[0]), axis=(0, 1)) for res in result_dict]
            mask_list = []
            for bbox_frame in bbox_list:
                mask_frame = [None for i in bbox_frame[0]]
                mask_list.append([mask_frame])
        else:
            bbox_list = [result_dict[i] for i in range(len(result_dict))]
            mask_list = [result_dict[i][1] for i in range(len(result_dict))]
        for vid in v_ids:
            video_name = coco_vid.videos[vid]['name']
            # if video_name not in ['p012_021']:
            #     continue
            image_ids = coco_vid.get_img_ids_from_vid(vid)
            # if video_name == 'p020_05':
            #     continue
            per_list, per_avg = self.eval_sigleton(video_name, image_ids, coco_vid, bbox_list, mask_list, is_save)
        del per_list
        del per_avg
        del coco_vid
        del result_dict
        gc.collect()
    
    def eval_all_fold(self, is_save=True):
        for idx, fold in enumerate(self.fold_dict['fold']):
            result_index = self.fold_dict['index'][idx]
            result_dir = self.result_dir.format(fold, result_index)
            anno_dir = self.anno_dir.format(fold, result_index)
            self.eval_dataset(anno_dir, result_dir, is_save)

    def eval_video(self, video_name, is_save=True):
        print(video_name)
        self.video_name = video_name
        result_path = self.result_dir.format(video_name)
        anno_path = self.anno_dir.format(video_name)
        coco = COCO(anno_path)
        image_ids = coco.getImgIds()
        res_dict = np.load(result_path, allow_pickle=True)
        if isinstance(res_dict, dict) and 'det_bboxes' in res_dict.keys():
            bbox_list = res_dict['det_bboxes']
            mask_list = res_dict['det_masks']
        elif isinstance(res_dict, dict) and 'track_bboxes' in res_dict.keys():
            bbox_list = [np.expand_dims(res_dict['track_bboxes'][i][0][:, 1:], axis=(0, 1)) for i in range(len(res_dict['track_bboxes']))]
            mask_list = res_dict['track_masks']
        else:
            bbox_list = [res_dict[i] for i in range(len(res_dict))]
            mask_list = [res_dict[i][1] for i in range(len(res_dict))]
        assert len(image_ids) == len(bbox_list)
        
        per_list, per_avg = self.eval_sigleton(image_ids, coco, bbox_list, mask_list, is_save)

    def eval_sigleton(self, 
                      video_name,
                      image_ids, 
                      anno_coco, 
                      bbox_list, 
                      mask_list, 
                      is_save=False):
        print(video_name)
        frame_type_df = pd.read_excel(frame_type_xlsx, video_name)
        frame_type_df = frame_type_df.sort_values(by=['NAME'])
        frame_type_list = frame_type_df.to_dict(orient='list')['TYPE']
        frame_type_list = frame_type_list[::2]
        frame_type_list = frame_type_list[: len(image_ids)]
        assert len(frame_type_list) == len(image_ids), f'TYPE_LEN: {len(frame_type_list)}, IMAGES_LEN:{len(image_ids)}'

        frame_type_filtered = []
        res_frame_list = []
        if self.is_save_mask:
            per_mask_list = []
        else:
            per_mask_list = None
        gt_info_list = []
        ablation_list = []
        bbox_last = bbox_list[0][0][0]
        mask_last = mask_list[0][0]
        for idx, image_id in enumerate(image_ids):
            image_info = anno_coco.loadImgs(image_id)[0]
            # print(image_info['file_name'])
            anno_ids = anno_coco.getAnnIds(imgIds=image_id, catIds=1)
            anno_infos = anno_coco.loadAnns(ids=anno_ids)
            if len(anno_infos) <= 0:
                continue
            
            anno_info = anno_infos[0]
            # if anno_info['iscrowd'] == 1 or anno_info['occluded'] or anno_info['ignore']:
            #     continue
            frame_type_filtered.append(frame_type_list[idx])
            if video_name not in self.ablation_phase_origin_dict.keys():
                ablation_list.append(1)
            else:
                if self.ablation_phase_origin_dict[video_name][0] <= idx <= self.ablation_phase_origin_dict[video_name][1]:
                    ablation_list.append(1)
                else:
                    ablation_list.append(0)
            bbox_gt = anno_info['bbox']
            bbox_gt = self.format_bbox(bbox_gt, is_gt=True)
            mask_gt = anno_coco.annToMask(anno_info)
            

            try:
                bbox_res = bbox_list[image_id-1][0][0]
                mask_res = mask_list[image_id-1][0]
            except:
                bbox_res = bbox_last
                mask_res = mask_last
                print('ZERO')
            if len(bbox_res) <= 0:
                bbox_res = bbox_last
                mask_res = mask_last
                print('ZERO')
            else:
                bbox_last = bbox_res
                mask_last = mask_res
            if isinstance(mask_res[0], dict):
                mask_res = [m_decode(mask) for mask in mask_res]
            points_gt = self.mask2points(mask_gt)
            slope_gt = self.line_reg.fit(points_gt['x'].reshape(-1, 1), points_gt['y']).coef_[0]
            if slope_gt <0:
                tip_gt = (bbox_gt['x1'], bbox_gt['y2'])
                enter_gt = (bbox_gt['x2'], bbox_gt['y1'])
            else:
                tip_gt = (bbox_gt['x2'], bbox_gt['y2'])
                enter_gt = (bbox_gt['x1'], bbox_gt['y1'])
            
            anno_tip_ids = anno_coco.getAnnIds(imgIds=image_id, catIds=2)
            anno_tip_infos = anno_coco.loadAnns(ids=anno_tip_ids)
            if len(anno_tip_infos) > 0:
                tip_bbox = anno_tip_infos[0]['bbox']
                tip_bbox = self.format_bbox(tip_bbox, is_gt=True)
                ext_gt = ((tip_bbox['x1'] + tip_bbox['x2']) / 2, (tip_bbox['y1'] + tip_bbox['y2']) /  2)
            else:
                ext_gt = None

            gt_info = {'enter_gt': enter_gt, 'tip_gt': tip_gt, 'ext_gt': ext_gt, 'slope_gt': slope_gt, 'mask_gt': mask_gt}
            gt_info_list.append(gt_info)
            if self.is_save_mask:
                mask_per_dict = {}
                points_frame, res_frame, masks = self.eval_frame_combine(bbox_res, mask_res, gt_info)
                for key in masks.keys():
                    mask_pred = masks[key]                
                   
                    mask_pred[mask_pred!=0] = 1
                    if np.max(mask_pred) == 0:
                        mask_per_dict[key] = {'dice': 0,
                                              'iou': 0,
                                              'hd95': 0,
                                              'sen': 0,
                                              'spe': specificity(masks[key], mask_gt)
                                            }
                    else:
                        mask_per_dict[key] = {'dice': dc(masks[key], mask_gt), 
                                            'iou': jc(masks[key], mask_gt), 
                                            'hd95': hd95(masks[key], mask_gt),
                                            'sen': sensitivity(masks[key], mask_gt),
                                            'spe': specificity(masks[key], mask_gt)
                                        }
                per_mask_list.append(mask_per_dict)               
            else:
                points_frame, res_frame = self.eval_frame_combine(bbox_res, mask_res, gt_info)

            res_frame['index'] = video_name+'_'+str(image_id)
            res_frame_list.append(res_frame)
            if self.is_visu and self.save_path is not None:
                # self.visu(image_info, gt_info, points_frame)
                self.visu_best(image_info, gt_info, points_frame, res_frame)
        self.video_filter_dict[video_name] = frame_type_filtered
        per_list, per_avg = self.avg_video_per(res_frame_list, per_mask_list, frame_type_filtered, ablation_list)
        self.per_video_dict[video_name] = (per_list, per_avg)
        self.gt_video_dict[video_name] = gt_info_list
        self.ablation_phase_dict[video_name] = ablation_list
        if is_save:
            if self.save_path is None:
                print('SAVE FALSE, NO SAVE PATH')
            with open(self.save_path.format(video_name+'_per_list'), 'wb') as handle:
                pickle.dump(per_list, handle)
            with open(self.save_path.format(video_name+'_per_avg'), 'wb') as handle:
                pickle.dump(per_avg, handle)
        # self.statistic_per(video_name)
        return per_list, per_avg

    def statistic_per(self, name_list, figure_path):
        os.makedirs(figure_path.replace('{}', ''), exist_ok=True)
        if not isinstance(name_list, list):
            name_list = [name_list]
        length_all = []
        slope_all = []
        dis_all = []
        angle_all = []
        ablation_all = []
        metric_key_list = self.metric_list
        if self.is_save_mask:
            metric_key_list = metric_key_list + self.metric_mask_list
        for v_name in name_list:
            best_per = sys.maxsize
            best_key = None
            for type_key in self.per_video_dict[v_name][1].keys():
                if type_key == 'LEN' or 'best' in type_key:
                    continue
                for metric_key in metric_key_list:
                    if not metric_key.startswith('dis_'):
                        continue
                    if metric_key not in self.per_video_dict[v_name][1][type_key].keys():
                        continue
                    dis_value = self.per_video_dict[v_name][1][type_key][metric_key]
                    if dis_value < best_per:
                        best_per = dis_value
                        best_key = (type_key, metric_key)
            try:
                dis_list = self.per_video_dict[v_name][0][best_key[0]][best_key[1]]
                angle_list = self.per_video_dict[v_name][0][best_key[0]]['error_angle']
            except:
                continue
            resolution = resolution_dict[v_name]['x_voxel'] if v_name in resolution_dict.keys() else resolution_dict['default']['x_voxel']
            
            dis_list = list(map(lambda x: x * resolution, dis_list))
            slope_list = []
            length_list = []
            gt_info_list = self.gt_video_dict[v_name]
            for gt_info in gt_info_list:
                slope_list.append(gt_info['slope_gt'])
                enter_point = gt_info['enter_gt']
                end_point = gt_info['ext_gt'] if (self.is_ext and gt_info['ext_gt'] is not None) else gt_info['tip_gt']
                length_list.append(np.sqrt((enter_point[0] - end_point[0]) ** 2 + (enter_point[1] - end_point[1]) ** 2) * resolution)
            # self.statistic_scatter(dis_list, angle_list, v_name)
            ablation_indicator = self.ablation_phase_dict[v_name]

            length_all.extend(length_list)
            slope_all.extend(slope_list)
            dis_all.extend(dis_list)
            angle_all.extend(angle_list)
            ablation_all.extend(ablation_indicator)
            # figure_ablation = self.statistic_histogram(length_list, slope_list, dis_list, angle_list, ablation_indicator, ablation_phase=1)
            # figure_no_ablation = self.statistic_histogram(length_list, slope_list, dis_list, angle_list, ablation_indicator, ablation_phase=0)
            # if figure_ablation:
            #     figure_ablation.savefig(self.figure_save_path.format(v_name)+'_violin_abla.png', dpi=300)
            # if figure_no_ablation:
            #     figure_no_ablation.savefig(self.figure_save_path.format(v_name)+'_violin_no_abla.png', dpi=300)
            del length_list, slope_list, dis_list, angle_list, ablation_indicator
            del self.per_video_dict[v_name]
            del self.gt_video_dict[v_name]
            # gc.collect()
        abla_value = str(self.filter_dict['abla'])
        if len(name_list)> 1:
            violin = self.statistic_violin(length_all, slope_all, dis_all, angle_all, ablation_all, ablation_phase=self.filter_dict['abla'], figure_path=figure_path)
            if violin:
                violin.savefig(figure_path.format('all')+f'_violin_abla={abla_value}.png', dpi=300)
            
            histogram = self.statistic_histogram(length_all, slope_all, dis_all, angle_all, ablation_all, ablation_phase=self.filter_dict['abla'], figure_path=figure_path)
            if histogram:
                histogram.savefig(figure_path.format('all')+f'_his_abla={abla_value}.png', dpi=300)

    def statistic_histogram(self, length_list, slope_list, dis_list, angle_list, ablation_indicator, ablation_phase=1, figure_path=None):
        ablation_indicator =np.array(ablation_indicator)
        if ablation_phase != -1:
            ablation_mask = ablation_indicator == ablation_phase
        else:
            ablation_mask = ablation_indicator == ablation_indicator
        if np.sum(ablation_mask) == 0:
            return None
        
        slope_list = np.array(slope_list)[ablation_mask]
        length_list = np.array(length_list)[ablation_mask]
        dis_list = np.array(dis_list)[ablation_mask]
        angle_list = np.array(angle_list)[ablation_mask]

        bins_length = np.linspace(np.min(length_list), np.max(length_list), 25)
        bins_slope = np.linspace(np.min(slope_list), np.max(slope_list), 25)
        dist_length_average = []
        angle_length_average = []
        for i in range(len(bins_length)-1):
            mask = np.logical_and(length_list >= bins_length[i], length_list < bins_length[i+1])
            dist_length_average.append(np.mean(dis_list[mask]))
            angle_length_average.append(np.mean(angle_list[mask]))
        
        dist_slope_average = []
        angle_slope_average = []
        for i in range(len(bins_slope)-1):
            mask = np.logical_and(slope_list >= bins_slope[i], slope_list < bins_slope[i+1])
            dist_slope_average.append(np.mean(dis_list[mask]))
            angle_slope_average.append(np.mean(angle_list[mask]))
        bins_length = list(map(lambda x: round(x, 2), bins_length))
        bins_slope = list(map(lambda x: round(x, 2), bins_slope))
        fig, axes = plt.subplots(2, 2, figsize=(60, 50))
        # data_df = pd.DataFrame({'LENGTH': length_list, 'SLOPE': slope_list, 'DIST': dis_list, 'ANGLE': angle_list, 'ABLA': ablation_indicator})
        sns.barplot(ax=axes[0, 0], data={'LENGTH': bins_length[:-1], 'DIST': dist_length_average}, x='LENGTH', y='DIST').set(title='length vs. tip')
        sns.barplot(ax=axes[0, 1], data={'LENGTH': bins_length[:-1], 'ANGLE': angle_length_average}, x='LENGTH', y='ANGLE').set(title='length vs. angle')
        sns.barplot(ax=axes[1, 0], data={'SLOPE': bins_slope[:-1], 'DIST': dist_slope_average}, x='SLOPE', y='DIST').set(title='angle vs. tip')
        sns.barplot(ax=axes[1, 1], data={'SLOPE': bins_slope[:-1], 'ANGLE': angle_slope_average}, x='SLOPE', y='ANGLE').set(title='angle vs. angle')
        # plt.show()
        pd.DataFrame({'LENGTH': bins_length[:-1], 'DIST': dist_length_average}).to_csv(figure_path.format('all_l_d')+f'_his_abla={ablation_phase}.csv', index=False)
        pd.DataFrame({'LENGTH': bins_length[:-1], 'ANGLE': angle_length_average}).to_csv(figure_path.format('all_l_a')+f'_his_abla={ablation_phase}.csv', index=False)
        pd.DataFrame({'SLOPE': bins_slope[:-1], 'DIST': dist_slope_average}).to_csv(figure_path.format('all_s_d')+f'_his_abla={ablation_phase}.csv', index=False)
        pd.DataFrame({'SLOPE': bins_slope[:-1], 'ANGLE': angle_slope_average}).to_csv(figure_path.format('all_s_a')+f'_his_abla={ablation_phase}.csv', index=False)
        figure = plt.gcf()
        return figure

    def statistic_violin(self, length_list, slope_list, dis_list, angle_list, ablation_indicator, ablation_phase=1, figure_path=None):
        ablation_indicator =np.array(ablation_indicator)
        if ablation_phase != -1:
            ablation_mask = ablation_indicator == ablation_phase
        else:
            ablation_mask = ablation_indicator == ablation_indicator
        if np.sum(ablation_mask) == 0:
            return None
        
        slope_list = np.array(slope_list)[ablation_mask]
        length_list = np.array(length_list)[ablation_mask]
        dis_list = np.array(dis_list)[ablation_mask]
        angle_list = np.array(angle_list)[ablation_mask]

        bins_length = np.linspace(np.min(length_list), np.max(length_list), 26)
        print(bins_length)
        bin_indices_length = np.digitize(length_list, bins_length)
        bins_slope = np.linspace(np.min(slope_list), np.max(slope_list), 26)
        print(bins_slope)
        bin_indices_slope = np.digitize(slope_list, bins_slope)

        fig, axes = plt.subplots(2, 2, figsize=(60, 50))
        binned_length_dist = {i: [] for i in range(1, len(bins_length)+1)}
        for i, bin_idx in enumerate(bin_indices_length):
            binned_length_dist[bin_idx].append(dis_list[i])
        data = [binned_length_dist[i] for i in range(1, len(bins_length)+1)]
        d = dict(zip(list(range(len(data))), [data[i] for i in range(len(data))]))
        pd.DataFrame(dict([(k,pd.Series(v)) for k,v in d.items() ])).to_csv(figure_path.format('all_l_d')+f'_vio_abla={ablation_phase}.csv', index=False)
        sns.violinplot(ax=axes[0, 0], data=data, orient='v').set(title='length vs. tip')


        binned_length_angle = {i: [] for i in range(1, len(bins_length)+1)}
        for i, bin_idx in enumerate(bin_indices_length):
            binned_length_angle[bin_idx].append(angle_list[i])
        data = [binned_length_angle[i] for i in range(1, len(bins_length)+1)]
        d = dict(zip(list(range(len(data))), [data[i] for i in range(len(data))]))
        pd.DataFrame(dict([(k,pd.Series(v)) for k,v in d.items() ])).to_csv(figure_path.format('all_l_a')+f'_vio_abla={ablation_phase}.csv', index=False)
        sns.violinplot(ax=axes[0, 1], data=data, orient='v').set(title='length vs. angle')


        binned_slope_dist = {i: [] for i in range(1, len(bins_slope)+1)}
        for i, bin_idx in enumerate(bin_indices_slope):
            binned_slope_dist[bin_idx].append(dis_list[i])
        data = [binned_slope_dist[i] for i in range(1, len(bins_slope)+1)]
        d = dict(zip(list(range(len(data))), [data[i] for i in range(len(data))]))
        pd.DataFrame(dict([(k,pd.Series(v)) for k,v in d.items() ])).to_csv(figure_path.format('all_s_d')+f'_vio_abla={ablation_phase}.csv', index=False)
        sns.violinplot(ax=axes[1, 0], data=data, orient='v').set(title='slope vs. tip')
        
        binned_slope_angle = {i: [] for i in range(1, len(bins_slope)+1)}
        for i, bin_idx in enumerate(bin_indices_slope):
            binned_slope_angle[bin_idx].append(angle_list[i])
        data = [binned_slope_angle[i] for i in range(1, len(bins_slope)+1)]
        d = dict(zip(list(range(len(data))), [data[i] for i in range(len(data))]))
        pd.DataFrame(dict([(k,pd.Series(v)) for k,v in d.items() ])).to_csv(figure_path.format('all_s_a')+f'_vio_abla={ablation_phase}.csv', index=False)
        sns.violinplot(ax=axes[1, 1], data=data, orient='v').set(title='slope vs. angle')
        # pd.DataFrame({'LENGTH': bins_length[:-1], 'DIST': dist_length_average}).to_csv(figure_path.format('all_l_d')+f'_his_abla={ablation_phase}.csv', index=False)
        # pd.DataFrame({'LENGTH': bins_length[:-1], 'ANGLE': angle_length_average}).to_csv(figure_path.format('all_l_a')+f'_his_abla={ablation_phase}.csv', index=False)
        # pd.DataFrame({'SLOPE': bins_length[:-1], 'DIST': dist_slope_average}).to_csv(figure_path.format('all_s_d')+f'_his_abla={ablation_phase}.csv', index=False)
        # pd.DataFrame({'SLOPE': bins_length[:-1], 'ANGLE': angle_slope_average}).to_csv(figure_path.format('all_s_a')+f'_his_abla={ablation_phase}.csv', index=False)
        figure = plt.gcf()
        return figure
        
    def statistic_scatter(self, dis_list, angle_list, v_name):
        os.makedirs(self.figure_save_path.replace('{}', ''), exist_ok=True)
        
        ablation_indicator = self.ablation_phase_dict[v_name]

        gt_info_list = self.gt_video_dict[v_name]
        slope_list = []
        length_list = []
        for gt_info in gt_info_list:
            slope_list.append(gt_info['slope_gt'])
            enter_point = gt_info['enter_gt']
            end_point = gt_info['ext_gt'] if (self.is_ext and gt_info['ext_gt'] is not None) else gt_info['tip_gt']
            length_list.append(np.sqrt((enter_point[0] - end_point[0]) ** 2 + (enter_point[1] - end_point[1]) ** 2) * resolution_dict[v_name]['x_voxel'])

        

        fig, axes = plt.subplots(2, 2)
        data_df = pd.DataFrame({'LENGTH': length_list, 'SLOPE': slope_list, 'DIST': dis_list, 'ANGLE': angle_list, 'ABLA': ablation_indicator})
        sns.scatterplot(ax=axes[0, 0], data=data_df, x='LENGTH', y='DIST', hue='ABLA', style='ABLA').set(title='shaft length vs. tip error')
        sns.scatterplot(ax=axes[0, 1], data=data_df, x='LENGTH', y='ANGLE', hue='ABLA', style='ABLA').set(title='shaft length vs. angle error')
        sns.scatterplot(ax=axes[1, 0], data=data_df, x='SLOPE', y='DIST', hue='ABLA', style='ABLA').set(title='shaft angle vs. tip error')
        sns.scatterplot(ax=axes[1, 1], data=data_df, x='SLOPE', y='ANGLE', hue='ABLA', style='ABLA').set(title='shaft angle vs. angle error')
        # plt.show()
        figure = plt.gcf()
        figure.savefig(self.figure_save_path.format(v_name)+'.png', dpi=100)

    def to_table(self, path):

        filter_visu, filter_abla = self.filter_dict['visu'], self.filter_dict['abla']
        list_writer = pd.ExcelWriter(path.format(f'per_list_{str(filter_visu)}_{str(filter_abla)}'))
        avg_writer = pd.ExcelWriter(path.format(f'per_avg_{str(filter_visu)}_{str(filter_abla)}'))
        avg_dist = {'VIDEO': [], 'RESO': [], 'LEN': []}
        for video_name, value in self.per_video_dict.items():
            resolution = resolution_dict[video_name]['x_voxel'] if video_name in resolution_dict.keys() else resolution_dict['default']['x_voxel']
            avg_dist['VIDEO'].append(video_name)
            # resolution = np.sqrt(resolution_dict[video_name]['x_voxel'] ** 2 + resolution_dict[video_name]['y_voxel'] ** 2)

            avg_dist['RESO'].append(resolution)
            list_dict = {}
            per_list, per_avg = copy.deepcopy(value)
            avg_dist['LEN'].append(per_avg['LEN'])

            metric_key_list = self.metric_list
            if self.is_save_mask:
                metric_key_list = metric_key_list + self.metric_mask_list

            for type_key in per_list.keys():
                if type_key == 'index':
                    list_dict[type_key] = per_list[type_key]
                    list_dict[type_key].append('AVG')
                    list_dict[type_key].append('STD')
                    continue

                for metric_key in metric_key_list:
                    if metric_key not in per_list[type_key].keys():
                        continue
                    tmp_key = f'{type_key}_{metric_key}'
                    list_dict[tmp_key] = per_list[type_key][metric_key]
                    try:
                        mean_value = np.nanmean(per_list[type_key][metric_key])
                        std_value = np.nanstd(per_list[type_key][metric_key])
                    except TypeError as e:
                        print(per_list[type_key][metric_key])
                        mean_value = np.nan
                        std_value = np.nan
                    list_dict[tmp_key].append(mean_value)
                    list_dict[tmp_key].append(std_value)

                    if tmp_key not in avg_dist.keys():
                        avg_dist[tmp_key] = []
                        if '_dis_' in tmp_key:
                            avg_dist[tmp_key+'_MM'] = []
                    avg_dist[tmp_key].append(per_avg[type_key][metric_key])
                    if '_dis_' in tmp_key:
                        if not np.isnan(per_avg[type_key][metric_key]):
                            avg_dist[tmp_key+'_MM'].append(per_avg[type_key][metric_key]*resolution)
                        else:
                            avg_dist[tmp_key+'_MM'].append(np.nan)
            
            list_df = pd.DataFrame(list_dict)
            cols = list_df.columns.to_list()
            cols = cols[-1:] + cols[:-1]
            list_df = list_df[cols]
            # gt_info = {'enter_gt': enter_gt, 'tip_gt': tip_gt, 'ext_gt': ext_gt, 'slope_gt': slope_gt, 'mask_gt': mask_gt}
            gt_info_list = self.gt_video_dict[video_name]

            list_df['ENTER'] = [f'{str(gt["enter_gt"][0])} {str(gt["enter_gt"][1])}' for gt in gt_info_list] + ['', '']
            list_df['TIP'] = [f'{str(gt["tip_gt"][0])} {str(gt["tip_gt"][1])}' for gt in gt_info_list] + ['', '']
            list_df['EXT'] = [f'{str(gt["ext_gt"][0])} {str(gt["ext_gt"][1])}' if gt["ext_gt"] else None for gt in gt_info_list] + ['', '']
            list_df['SLOPE'] = [gt['slope_gt'] for gt in gt_info_list] + ['', '']
            list_df.to_excel(list_writer, sheet_name=video_name, index=False)
        for key in avg_dist.keys():
            if key == 'VIDEO':
                avg_dist[key].append('MACRO_AVG')
                avg_dist[key].append('MICRO_AVG')
            elif key == 'RESO':
                avg_dist[key].extend((np.nan, np.nan))
            elif key != 'LEN':
                macro_avg = np.nanmean(avg_dist[key])
                sum_value = 0
                len_value = 0
                for idx in range(len(avg_dist[key])):
                    if avg_dist['LEN'][idx] > 0:
                        sum_value = sum_value +  avg_dist['LEN'][idx]*avg_dist[key][idx]
                        len_value = len_value + avg_dist['LEN'][idx]
                if len_value > 0:
                    micro_avg = sum_value / len_value
                else:
                    micro_avg = None
                avg_dist[key].extend((macro_avg, micro_avg))
            else:
                len_avg = np.mean(avg_dist['LEN'])
                avg_dist[key].extend((len_avg, len_avg))
        pd.DataFrame(avg_dist).to_excel(avg_writer, sheet_name='VIDEO AVG', index=False)
        list_writer.save()
        avg_writer.save()

    def avg_video_per(self, per_video, per_mask_video=None, frame_type_filtered=None, ablation_list=None):
        assert len(per_video) == len(frame_type_filtered) == len(ablation_list)
        frame_type_filtered = np.array(frame_type_filtered)
        ablation_list = np.array(ablation_list)

        filter_visu, filter_abla = self.filter_dict['visu'], self.filter_dict['abla']
        if filter_visu != -1:
            visu_mask = (frame_type_filtered == filter_visu)
        else:
            visu_mask = (frame_type_filtered == frame_type_filtered)
        if filter_abla != -1:
            abla_mask = (ablation_list == filter_abla)
        else:
            abla_mask = (ablation_list == ablation_list)
        mask_filter = visu_mask & abla_mask
        is_sample = True
        if not np.any(abla_mask):
            is_sample = False
        per_list = {}
        metric_dict = dict(zip(self.metric_list, [[] for i in self.metric_list]))
        if self.is_save_mask and per_mask_video is not None:
            metric_mask_dict = dict(zip(self.metric_mask_list, [[] for i in self.metric_mask_list]))
            for type_key in per_mask_video[0].keys():
                per_list[type_key] = copy.deepcopy(metric_mask_dict)
        for type_key in per_video[0].keys():
            if type_key == 'single_list':
                per_list['single_best'] = copy.deepcopy(metric_dict)
                per_list['single_mean'] = copy.deepcopy(metric_dict)
            else:
                per_list[type_key] = copy.deepcopy(metric_dict)
        per_list['index'] = []
        for idx, per_frame in enumerate(per_video):
            for key in per_frame.keys():
                if key == 'single_list':
                    for metric_key in self.metric_list:
                        if mask_filter[idx]:
                            per_list['single_best'][metric_key].append(min([single_per[metric_key] for single_per in per_frame[key]]))
                            per_list['single_mean'][metric_key].append(np.mean([single_per[metric_key] for single_per in per_frame[key]]))
                        else:
                            per_list['single_best'][metric_key].append(np.nan)
                            per_list['single_mean'][metric_key].append(np.nan)
                elif key == 'index':
                     per_list[key].append(per_frame[key])
                else:
                    for metric_key in self.metric_list:
                        if mask_filter[idx]:
                            per_list[key][metric_key].append(per_frame[key][metric_key])
                        else:
                            per_list[key][metric_key].append(np.nan)
            if self.is_save_mask and per_mask_video is not None:
                per_mask = per_mask_video[idx]
                for key in per_mask.keys():
                    for key_metric_mask in self.metric_mask_list:
                        per_list[key][key_metric_mask].append(per_mask[key][key_metric_mask])   
        
        per_avg = dict(zip(per_list.keys(), [copy.deepcopy(metric_dict) for i in range(len(per_list.keys()))]))
        if self.is_save_mask and per_mask_video is not None:
            for key in per_avg.keys():
                if key.startswith('mask_'):
                    per_avg[key] = dict(zip(self.metric_mask_list, [[] for i in self.metric_mask_list]))
        del per_avg['index']

        metric_key_list = self.metric_list
        if self.is_save_mask:
            metric_key_list = metric_key_list + self.metric_mask_list

        for key in per_avg.keys():
            for metric_key in metric_key_list:
                if metric_key not in per_list[key].keys():
                    continue
                if is_sample:
                    ma_array = np.nanmean(np.array(per_list[key][metric_key]))
                    per_avg[key][metric_key] = ma_array
                else:
                    per_avg[key][metric_key] = np.nan
        
        per_avg['LEN'] = (mask_filter * 1).sum()
                
        return per_list, per_avg
        
    def eval_frame_combine(self, bbox_list, mask_list, gt_info):
        """
        bbox_list (Tensor): bboxes have shape (m, 4) in <x1, y1, x2, y2>
                format, or shape (m, 5) in <x1, y1, x2, y2, score> format.
        """

        enter_gt, tip_gt, ext_gt, slope_gt = gt_info['enter_gt'], gt_info['tip_gt'], gt_info['ext_gt'], gt_info['slope_gt']
        angle_gt = math.atan(slope_gt) / math.pi * 180

        # if np.max(mask_list[0]) == 0:
        # if True:
        if len(bbox_list.shape) > 2:
            bbox_list = [b[0] for b in bbox_list]

        if mask_list[0] is None or not mask_list[0].any():
            mask_list = []
            for idx in range(len(bbox_list)):
                bbox = bbox_list[idx]
                bbox[2] = min(bbox[2], 800-1)
                bbox[3] = min(bbox[3], 700-1)
                mask_array = np.zeros_like(gt_info['mask_gt'])
                if slope_gt >= 0:
                    mask_array[int(bbox[1]), int(bbox[0])] = 1
                    mask_array[int(bbox[3]), int(bbox[2])] = 1
                else:
                    mask_array[int(bbox[1]), int(bbox[2])] = 1
                    mask_array[int(bbox[3]), int(bbox[0])] = 1
                mask_list.append(mask_array)
        
        # filter the result bbox
        bbox_list = bbox_list[: min(self.top_n, len(bbox_list))]
        mask_list = mask_list[: min(self.top_n, len(mask_list))]
        assert len(bbox_list) == len(mask_list)
        IoU_matrix = self.IoU_calculator(torch.tensor(bbox_list), torch.tensor(bbox_list)).numpy()
        IoU_avg = np.mean(IoU_matrix, axis=1)
        filter_index = IoU_avg > self.IoU_thres
        if np.sum(filter_index) == 0:
            filter_index[0] = True

        bbox_list = [bbox_list[i] for i in range(len(filter_index)) if filter_index[i]]
        mask_list = [mask_list[i] for i in range(len(filter_index)) if filter_index[i]]
        res_num = len(bbox_list)
        all_points_list = [self.all_points(res[0], res[1], slope_gt) for res in zip(bbox_list, mask_list)]
        if res_num == 2:
            # TODO, choose the longest one
            # all_points_list.sort(key=lambda x: (x['enter_bbox'][0] - x['tip_bbox'][0]) ** 2 + (x['enter_bbox'][1] - x['tip_bbox'][1]) ** 2, reverse=True)
            all_points_list = [all_points_list[0]]
        elif self.deg_thres and res_num >= 3:
            # remove the wrong bbox with odd slope
            deg_list = [p['angle'] for p in all_points_list]
            deg_list = np.broadcast_to(deg_list, (res_num, res_num))
            deg_error = np.abs(deg_list - deg_list.T)
            deg_error_avg = np.mean(deg_error, axis=1)
            filter_index = deg_error_avg < self.deg_thres
            if np.sum(filter_index) == 0:
                filter_index[0] = True
            all_points_list = [all_points_list[i] for i in range(res_num) if filter_index[i]]
            bbox_list = [bbox_list[i] for i in range(len(filter_index)) if filter_index[i]]
            mask_list = [mask_list[i] for i in range(len(filter_index)) if filter_index[i]]
            res_num = len(bbox_list)
        
        per_single_list = [self.point_error(all_points, tip_gt, ext_gt, angle_gt) for all_points in all_points_list]

        # every single bbox result end

        # combine result begin
        mask_union = np.sum(mask_list, axis=0)
        mask_vote_2 = copy.deepcopy(mask_union)
        ceil_value = math.ceil(len(per_single_list) / 2)
        mask_vote_2[mask_vote_2 < ceil_value] = 0
        mask_vote_all = copy.deepcopy(mask_union)
        mask_vote_all[mask_vote_all < len(mask_list)] = 0
        
        bbox_highest = self.format_bbox(bbox_list[0])
        bbox_union = copy.deepcopy(bbox_highest)
        bbox_biggest = copy.deepcopy(bbox_highest)
        bbox_avg = {'x1': 0, 'y1': 0, 'x2': 0, 'y2': 0}
        biggest_area = -1
        for bbox in bbox_list:
            bbox = self.format_bbox(bbox)
            bbox_union['x1'] =  min(bbox_union['x1'], bbox['x1'])
            bbox_union['y1'] =  min(bbox_union['y1'], bbox['y1'])
            bbox_union['x2'] =  max(bbox_union['x2'], bbox['x2'])
            bbox_union['y2'] =  max(bbox_union['y2'], bbox['y2'])
            area = (bbox['x2'] - bbox['x1']) * (bbox['y2'] - bbox['y1'])
            if area > biggest_area:
                biggest_area = area
                bbox_biggest = bbox
            bbox_avg['x1'] += bbox['x1']
            bbox_avg['y1'] += bbox['y1']
            bbox_avg['x2'] += bbox['x2']
            bbox_avg['y2'] += bbox['y2']
        bbox_avg['x1'] = bbox_avg['x1'] / res_num
        bbox_avg['y1'] = bbox_avg['y1'] / res_num
        bbox_avg['x2'] = bbox_avg['x2'] / res_num
        bbox_avg['y2'] = bbox_avg['y2'] / res_num

        # mask_bbox
        union_highest = self.all_points(bbox_highest, mask_union, slope_gt)
        union_union = self.all_points(bbox_union, mask_union, slope_gt)
        union_biggest = self.all_points(bbox_union, mask_union, slope_gt)
        union_avg = self.all_points(bbox_avg, mask_union, slope_gt)
        vote_2_highest = self.all_points(bbox_highest, mask_vote_2, slope_gt)
        vote_2_union = self.all_points(bbox_union, mask_vote_2, slope_gt)
        vote_2_biggest = self.all_points(bbox_biggest, mask_vote_2, slope_gt)
        vote_2_avg = self.all_points(bbox_avg, mask_vote_2, slope_gt)
        vote_all_highest = self.all_points(bbox_highest, mask_vote_all,slope_gt) 
        vote_all_union = self.all_points(bbox_union, mask_vote_all, slope_gt)
        vote_all_biggest = self.all_points(bbox_biggest, mask_vote_all, slope_gt)
        vote_all_avg = self.all_points(bbox_avg, mask_vote_all, slope_gt)
    
        points_list = {
            'single_list': all_points_list,
            'union_highest': union_highest,
            'union_union': union_union,
            'union_biggest': union_biggest,
            'union_avg': union_avg,
            'vote_2_highest': vote_2_highest,
            'vote_2_union': vote_2_union,
            'vote_2_biggest': vote_2_biggest,
            'vote_2_avg': vote_2_avg,
            'vote_all_highest': vote_all_highest,
            'vote_all_union': vote_all_union,
            'vote_all_biggest': vote_all_biggest,
            'vote_all_avg': vote_all_avg,
        }
        per_list = {
            'single_list': per_single_list,
            'union_highest': self.point_error(union_highest, tip_gt, ext_gt, angle_gt),
            'union_union': self.point_error(union_union, tip_gt, ext_gt, angle_gt),
            'union_biggest': self.point_error(union_biggest, tip_gt, ext_gt, angle_gt),
            'union_avg': self.point_error(union_avg, tip_gt, ext_gt, angle_gt),
            'vote_2_highest': self.point_error(vote_2_highest, tip_gt, ext_gt, angle_gt),
            'vote_2_union': self.point_error(vote_2_union, tip_gt, ext_gt, angle_gt),
            'vote_2_biggest': self.point_error(vote_2_biggest, tip_gt, ext_gt, angle_gt),
            'vote_2_avg': self.point_error(vote_2_avg, tip_gt, ext_gt, angle_gt),
            'vote_all_highest': self.point_error(vote_all_highest, tip_gt, ext_gt, angle_gt),
            'vote_all_union': self.point_error(vote_all_union, tip_gt, ext_gt, angle_gt),
            'vote_all_biggest': self.point_error(vote_all_biggest, tip_gt, ext_gt, angle_gt),
            'vote_all_avg': self.point_error(vote_all_avg, tip_gt, ext_gt, angle_gt),
        }

        if self. is_save_mask:
            mask_dict = {
                        'mask_highest': mask_list[0],
                        'mask_union': mask_union,
                        'mask_vote_2': mask_vote_2,
                        'mask_vote_all': mask_vote_all
                    }
            return points_list, per_list, mask_dict
        return points_list, per_list
        
    def visu(self, image_info, gt_info, all_points_list, is_gt=True, is_res=True):
        font = cv2.FONT_HERSHEY_SIMPLEX
        enter_gt, tip_gt = gt_info['enter_gt'], gt_info['tip_gt']
        enter_gt = list(map(int, enter_gt))
        tip_gt = list(map(int, tip_gt))
        image_path = os.path.join(self.data_dir, image_info['file_name'])
        image = cv2.imread(image_path)
        if is_gt:
            image = cv2.circle(image, tip_gt, radius=5, color=self.color_list[0], thickness=-1)       
            image = cv2.line(image, enter_gt, tip_gt, color=self.color_list[0], thickness=1)

        if is_res:
            points_list = all_points_list['single_list']
            for idx, all_points in enumerate(points_list):
                c = self.color_list[idx+1]
                enter = list(map(int, all_points['enter_inter']))
                tip = list(map(int, all_points['tip_inter']))
                image = cv2.circle(image, tip, radius=5, color=c, thickness=-1)       
                image = cv2.line(image, enter, tip, color=c, thickness=1)
                # cv2.putText(img, word, (20, offset), font, 1, (0, 255, 0), 3)
                cv2.putText(image, str(idx+1), (idx*25, 20), font, 1, c, 2)
        save_path = os.path.join(os.path.split(self.save_path)[0], image_info['file_name'])
        save_folder = os.path.split(save_path)[0]
        if not os.path.exists(save_folder):
            os.makedirs(save_folder)
        cv2.imwrite(save_path, image)  

    def visu_best(self, image_info, gt_info, all_points_list, all_per_list, is_gt=True, is_res=True):
        font = cv2.FONT_HERSHEY_SIMPLEX
        enter_gt, tip_gt = gt_info['enter_gt'], gt_info['tip_gt']
        enter_gt = list(map(int, enter_gt))
        tip_gt = list(map(int, tip_gt))
        image_path = os.path.join(self.data_dir, image_info['file_name'])
        image = cv2.imread(image_path)
        if is_gt:
            image = cv2.circle(image, tip_gt, radius=5, color=[255, 255, 255], thickness=-1)       
            image = cv2.line(image, enter_gt, tip_gt, color=self.color_list[0], thickness=1)
        best_key = None
        best_val = sys.maxsize
        
        v_name = image_info['file_name'].split('/')[0]
        resolution = resolution_dict[v_name]['x_voxel'] if v_name in resolution_dict.keys() else resolution_dict['default']['x_voxel']

        if is_res:
            for key in all_per_list.keys():
                if key in ['single_list', 'index']:
                    continue
                for sub_key in ['dis_bbox', 'dis_seg', 'dis_inter']:
                    if all_per_list[key][sub_key] < best_val:
                        best_key = [key, sub_key]
                        best_val = all_per_list[key][sub_key]
            best_angle = all_per_list[best_key[0]]['error_angle']
            best_val_mm = best_val * resolution
            all_points =  all_points_list[best_key[0]]
            enter = list(map(int, all_points['enter_inter']))
            tip = list(map(int, all_points['tip_inter']))
            image = cv2.circle(image, tip, radius=5, color=[0, 0, 255], thickness=-1)       
            image = cv2.line(image, enter, tip, color=[0, 0, 255], thickness=1)
            # cv2.putText(img, word, (20, offset), font, 1, (0, 255, 0), 3)
            text = 'Tip distance error: {} pixel, {} mm'.format(round(best_val, 2), round(best_val_mm, 2)) 
            cv2.putText(image, text, (35, 30), font, 1, [255, 255, 255], 1)
            text = 'Shaft direction error: {} degree'.format(round(best_angle, 2)) 
            cv2.putText(image, text, (35, 65), font, 1, [255, 255, 255], 1)

        save_path = os.path.join(os.path.split(self.save_path)[0], 'visu', image_info['file_name'])
        save_folder = os.path.split(save_path)[0]
        if not os.path.exists(save_folder):
            os.makedirs(save_folder)
        cv2.imwrite(save_path, image)  

    def point_error(self, points, tip_gt, ext_gt, angle_gt):
        dis_bbox = self.point_distance(points['tip_bbox'], tip_gt)
        dis_seg = self.point_distance(points['tip_seg'], tip_gt)
        dis_inter = self.point_distance(points['tip_inter'], tip_gt)
        if ext_gt is not None and self.is_ext:
            dis_bbox = min(self.point_distance(points['tip_bbox'], ext_gt), dis_bbox)
            dis_seg = min(self.point_distance(points['tip_seg'], ext_gt), dis_seg)
            dis_inter = min(self.point_distance(points['tip_inter'], ext_gt), dis_inter)
        error_angle = abs(points['angle'] - angle_gt)
        return {
                'dis_bbox': dis_bbox,
                'dis_seg': dis_seg,
                'dis_inter': dis_inter,
                'error_angle': error_angle
        }
    
    def all_points(self, bbox, mask, slope_gt):
        """
        get the slop of shaft, enter point and tip of niddle.
        """
        points = self.mask2points(mask)
        bbox = self.format_bbox(bbox)
        
        seg_enough = True
        if len(points['x']) < 10:
            seg_enough = False
            intercept = 0
            if slope_gt <= 0:
                slope = (bbox['y2'] - bbox['y1']) / (bbox['x2'] - bbox['x1']) * -1
            else:
                slope = (bbox['y2'] - bbox['y1']) / (bbox['x2'] - bbox['x1']) * 1
        else:
            self.line_reg.fit(points['x'].reshape(-1, 1), points['y'])
            slope = self.line_reg.coef_[0]
            intercept = self.line_reg.intercept_
        
        if slope == 0:
            slope = slope + 1e-5
        # bbox points
        if slope <=0:
            enter_bbox = (bbox['x2'], bbox['y1'])
            tip_bbox = ((bbox['x1'], bbox['y2']))
        else:
            enter_bbox = (bbox['x1'], bbox['y1'])
            tip_bbox = ((bbox['x2'], bbox['y2']))
        
        # seg points
        if seg_enough:
            enter_seg, tip_seg = self.points2niddle(points)
        else:
            enter_seg = enter_bbox
            tip_seg = tip_bbox

        # intersection points
        if seg_enough:
            x1, y1 = bbox['x1'], bbox['x1'] * slope + intercept
            x2, y2 = bbox['x2'], bbox['x2'] * slope + intercept
            x3, y3 = (bbox['y1'] - intercept) / slope, bbox['y1']
            x4, y4 = (bbox['y2'] - intercept) / slope, bbox['y2']
            xy_list = ((x1, y1), (x2, y2), (x3, y3), (x4, y4))
            xy_list = [xy for xy in xy_list if self.is_intersect(xy, bbox)]

            if len(xy_list) == 0:
                enter_inter = enter_bbox
                tip_inter = tip_bbox
                if slope_gt < 0:
                    slope = (bbox['y2'] - bbox['y1']) / (bbox['x2'] - bbox['x1']) * -1
                else:
                    slope = (bbox['y2'] - bbox['y1']) / (bbox['x2'] - bbox['x1']) * 1
            else:
                enter_inter, tip_inter = xy_list[0], xy_list[1]
                if enter_inter[1] > tip_inter[1]:
                    enter_inter, tip_inter = tip_inter, enter_inter
        else:
            enter_inter = enter_bbox
            tip_inter = tip_bbox
        
        return {
            'enter_bbox': enter_bbox,
            'tip_bbox': tip_bbox,
            'enter_seg': enter_seg,
            'tip_seg': tip_seg,
            'enter_inter': enter_inter,
            'tip_inter': tip_inter,
            'angle': math.atan(slope) / math.pi * 180,
            'slope': slope,
            'inter': intercept
        }

    def points2niddle(self, mask_points):
        y_enter, y_tip = np.min(mask_points['y']), np.max(mask_points['y'])
        y_enter_indx, y_tip_indx = np.where(mask_points['y']==y_enter), np.where(mask_points['y']==y_tip)
        x_enter, x_tip = np.mean(mask_points['x'][y_enter_indx]), np.mean(mask_points['x'][y_tip_indx])
        enter = (x_enter, y_enter)
        tip = (x_tip, y_tip)
        return enter, tip

    def mask2points(self, mask_array):
        points = np.where(mask_array>0)
        points = {'x': points[1], 'y': points[0]}
        return points

    def format_bbox(self, bbox, is_gt=False):
        if isinstance(bbox, dict) and 'x1' in bbox:
            return bbox
        if is_gt:
            bbox = {'x1': bbox[0], 'x2': bbox[0]+bbox[2], 'y1': bbox[1], 'y2': bbox[1]+bbox[3], 'score': bbox[-1]}
        else:
            assert bbox[2] >= bbox[0] and bbox[3] >= bbox[1]
            bbox = {'x1': bbox[0], 'x2': bbox[2], 'y1': bbox[1], 'y2': bbox[3], 'score': bbox[-1]}
        return bbox

    def is_intersect(self, point, bbox, error_tor=0.001):
        x, y = point[0], point[1]
        return (bbox['x1'] - error_tor <= x <= bbox['x2'] + error_tor) and \
                (bbox['y1'] - error_tor <= y <= bbox['y2'] + error_tor)

    def point_distance(self, p1, p2):
        dis = math.sqrt((p1[0] - p2[0])**2 + (p1[1] - p2[1])**2)
        return dis

    # @classmethod
    def hit_rate(self, table_path):
        filter_visu, filter_abla = self.filter_dict['visu'], self.filter_dict['abla']
        print(filter_abla)
        list_path = os.path.join(table_path.format(f'per_list_{str(filter_visu)}_{str(filter_abla)}'))
        xls_file = pd.ExcelFile(list_path)
        all_list = []
        for v_name in resolution_dict.keys():
            resolution = resolution_dict[v_name]['x_voxel'] if v_name in resolution_dict.keys() else resolution_dict['default']['x_voxel']
            list_dict = pd.read_excel(xls_file, v_name).to_dict(orient='list')
            series = list_dict['single_best_dis_bbox'][:-2]
            if np.all(np.isnan(series)):
                continue
            best_list = None
            best_value = sys.maxsize
            for col_name in list_dict.keys():
                if 'dis' not in col_name:
                    continue
                if list_dict[col_name][-2] < best_value:
                    best_list = list_dict[col_name][:-2]
                    best_value = list_dict[col_name][-2]
            mm = list(map(lambda x: x*resolution, best_list))
            all_list.extend(mm)
        all_list = np.array(all_list)
        all_list = all_list[~np.isnan(all_list)]
        all_len = len(all_list)
        rate_dict = {'MM': [], 'RATE': []}
        max_value = max(int(np.max(all_list)), 10)
        
        for i in range(0, max_value+1):
            less_num = np.sum((all_list <= i)*1.0)
            rate = less_num / all_len
            rate_dict['MM'].append(i)
            rate_dict['RATE'].append(round(rate * 100, 2))
            print(f'mm: {str(i)}, rate: {str(round(rate * 100, 2) )}%')
        pd.DataFrame(rate_dict).to_excel(table_path.format(f'hit_rate_{str(filter_visu)}_{str(filter_abla)}'), index=False)


   
def load_res():
    path = r'/srv/fenster/people/Ningtao/Project/ThyroidNodule/tracking/mmtracking/work_dirs/mask_selsa_faster_rcnn_r50_dc5_1x_c04_needle1020/result/094358_p008_v2.pkl'
    res_dict = np.load(path, allow_pickle=True)
    mask_list = res_dict['det_bboxes']
    mask_dict = mask_list[0][0][0]
    mask_array= m_decode(mask_dict)
    points = np.where(mask_array>0)
    print(res_dict)

def eval_combine():
    for eval in ['bbox','segm']:
        for abla in [-1,]:
            for visu in [-1,]:
                evaluator = EvalCombine(result_dir='/srv/fenster/people/Ningtao/Project/USVideo/tracking/mmtracking/work_dirs/mask_selsa_faster_rcnn_r50_dc5_1x_c04_needle1020_AFA_fold{}/result/{}/'+f'{eval}.pkl',
                                        data_dir=r'/srv/fenster/people/Ningtao/Dataset/LiverNeedle/1020/Data',
                                        save_path=f'/srv/fenster/people/Ningtao/Project/USVideo/tracking/mmtracking/work_dirs/all_folds_top2_1/result/{eval}'+'_{}.pkl',
                                        anno_dir='/srv/fenster/people/Ningtao/Dataset/LiverNeedle/1020/DataSet/coco_valid_0318_no_occu/{}/test.json',
                                        fold_dict={
                                                    'fold': ['1', '2', '3', '4', '5'],
                                                    # 'fold': ['1',],
                                                    'index': ['004507', '041441', '035857', '034112', '215646'],
                                                    # 'index': ['004507'],
                                                    },
                                        IoU_thres=0.3,
                                        deg_thres=15,
                                        is_visu=True,
                                        filter_dict={'visu': visu, 'abla': abla},
                                        is_ext=True,
                                        is_save_mask=False,
                                        top_n=2,
                            )
                # evaluator.hit_rate(table_path='/srv/fenster/people/Ningtao/Project/USVideo/tracking/mmtracking/custom/metric_doc/liver_needle/041441/bbox_{}_ext.xlsx')
                # evaluator.eval_dataset(is_save=False)
                evaluator.eval_all_fold(is_save=False)
                # statistic before to table
                evaluator.to_table(path=f'/srv/fenster/people/Ningtao/Project/USVideo/tracking/mmtracking/custom/metric_doc/liver_needle/all_folds_top2_1/{eval}'+'_{}_ext.xlsx')   
                # if visu == -1:
                #     evaluator.statistic_per(name_list=list(evaluator.per_video_dict.keys()), figure_path='/srv/fenster/people/Ningtao/Project/USVideo/tracking/mmtracking/custom/metric_doc/liver_needle/all_folds_top2/figure/{}')

def eval_combine_fold():
    for eval in ['bbox',]:
        for abla in [-1, 0, 1]:
            for visu in [-1, 1, 2, 3, 4]:
                evaluator = EvalCombine(result_dir='/srv/fenster/people/Ningtao/Project/USVideo/tracking/mmtracking/work_dirs/mask_selsa_faster_rcnn_r50_dc5_1x_c04_needle1020_AFA_half/bbox.pkl',
                                        data_dir=r'/srv/fenster/people/Ningtao/Dataset/LiverNeedle/1020/Data',
                                        save_path=f'/srv/fenster/people/Ningtao/Project/USVideo/tracking/mmtracking/work_dirs/MRCNN/result/{eval}'+'_{}.pkl',
                                        anno_dir='/srv/fenster/people/Ningtao/Dataset/LiverNeedle/1020/DataSet/coco_valid_0318_no_occu/1/test_half.json',
                                        fold_dict=None,
                                        IoU_thres=0.3,
                                        deg_thres=15,
                                        is_visu=True,
                                        filter_dict={'visu': visu, 'abla': abla},
                                        is_ext=True,
                                        is_save_mask=True,
                                        top_n=2
                            )
                # evaluator.hit_rate(table_path='/srv/fenster/people/Ningtao/Project/USVideo/tracking/mmtracking/custom/metric_doc/liver_needle/041441/bbox_{}_ext.xlsx')
                evaluator.eval_dataset(is_save=False)
                # evaluator.eval_all_fold(is_save=False)
                # statistic before to table
                evaluator.to_table(path=f'/srv/fenster/people/Ningtao/Project/USVideo/tracking/mmtracking/custom/metric_doc/liver_needle/half/{eval}'+'_{}_ext.xlsx')   
                if visu == -1:
                    evaluator.statistic_per(name_list=list(evaluator.per_video_dict.keys()), figure_path='/srv/fenster/people/Ningtao/Project/USVideo/tracking/mmtracking/custom/metric_doc/liver_needle/half/figure/{}')

def get_summary():
    per_table_dir = '/srv/fenster/people/Ningtao/Project/USVideo/tracking/mmtracking/custom/metric_doc/liver_needle'
    work_index = '041441'
    out_str = ''
    for eval in ['bbox', 'segm']:
        for visul in ['-1', '1', '2', '3', '4']:
            for phase in ['-1', '0', '1']:
                best_dict = {'MM': sys.maxsize, 'PIXEL': sys.maxsize, 'MM_KEY': None, 'ANGLE': sys.maxsize, 'ANGLE_KEY': None}
                file_path = os.path.join(per_table_dir, work_index, f'{eval}_per_avg_{visul}_{phase}_ext.xlsx')
                df_dict = pd.read_excel(file_path, sheet_name='VIDEO AVG').to_dict(orient='list')
                for key in df_dict.keys():
                    if 'single_best' in key or 'single_mean' in key or 'VIDEO' in key or 'RESO' in key or 'LEN' in key:
                        continue
                    min_val = min(*df_dict[key][-2:])
                    if '_MM' in key and min_val < best_dict['MM']:
                        best_dict['MM'] = min_val
                        best_dict['PIXEL'] = min(*df_dict[key.replace('_MM', '')][-2:])
                        best_dict['MM_KEY'] = key
                    if '_angle' in key and min_val < best_dict['ANGLE']:
                        best_dict['ANGLE'] = min_val
                        best_dict['ANGLE_KEY'] = key
                out_str += f'EVAL: {eval}, VISUL: {visul}, PHASE: {phase}\n'
                out_str += f'{best_dict["MM_KEY"]}: {best_dict["MM"]} mm, {best_dict["PIXEL"]} pixel; {best_dict["ANGLE_KEY"]}: {best_dict["ANGLE"]} degree\n'
    with open(os.path.join(per_table_dir, f'summary_{work_index}.txt'), mode='w') as f:
        f.write(out_str)
        f.flush()

def summary_describe():
    per_table_dir = '/srv/fenster/people/Ningtao/Project/USVideo/tracking/mmtracking/custom/metric_doc/liver_needle'
    work_index = '215646'
    out_str = ''
    for eval in ['bbox', 'segm']:
        for visul in ['-1', '1', '2', '3', '4']:
            for phase in ['-1', '0', '1']:
                best_dict = {'MM': sys.maxsize, 'PIXEL': sys.maxsize, 'MM_STD': None, 'PIXEL_STD': None, 'MM_KEY': None, 'ANGLE': sys.maxsize, 'ANGLE_STD': None, 'ANGLE_KEY': None}
                file_path = os.path.join(per_table_dir, work_index, f'{eval}_per_summary_{visul}_{phase}_ext.xlsx')
                df_dict = pd.read_excel(file_path).to_dict(orient='list')
                count = sum(df_dict['LEN'][:12])
                for key in df_dict.keys():
                    if 'single_best' in key or 'single_mean' in key or 'VIDEO' in key or 'RESO' in key or 'LEN' in key:
                        continue
                    min_val = round(min(*df_dict[key][12: 14]), 2)
                    if '_MM' in key and min_val < best_dict['MM']:
                        best_dict['MM'] = min_val
                        best_dict['MM_STD'] = round(min(*df_dict[key][14: 16]), 2)
                        best_dict['PIXEL'] = round(min(*df_dict[key.replace('_MM', '')][12: 14]), 2)
                        best_dict['PIXEL_STD'] = round(min(*df_dict[key.replace('_MM', '')][14: 16]), 2)
                        best_dict['MM_KEY'] = key
                    if '_angle' in key and min_val < best_dict['ANGLE']:
                        best_dict['ANGLE'] = min_val
                        best_dict['ANGLE_STD'] =  round(min(*df_dict[key][14: 16]), 2)
                        best_dict['ANGLE_KEY'] = key
                out_str += f'EVAL: {eval}, VISUL: {visul}, PHASE: {phase}, COUNT: {count}\n'
                out_str += f'{best_dict["MM_KEY"]}: {best_dict["MM"]}+{best_dict["MM_STD"]} mm, {best_dict["PIXEL"]}+{best_dict["PIXEL_STD"]} pixel; {best_dict["ANGLE_KEY"]}: {best_dict["ANGLE"]}+{best_dict["ANGLE_STD"]} degree\n'
    with open(os.path.join(per_table_dir, f'describe_summary_{work_index}.txt'), mode='w') as f:
        f.write(out_str)
        f.flush()


def summary_best():
    per_table_dir = '/srv/fenster/people/Ningtao/Project/USVideo/tracking/mmtracking/custom/metric_doc/liver_needle'
    work_index = 'half'
    evals = ['bbox',]

    video_log_dict = {
        'PHASE': [],
        'VISU': [],
        'VIDEO': [],
        'COL': [],
        'EVAL': [],
        'SHAFT': [],
        'TIP': []
    }


    patient_record_dict = {}
    out_dict = {'EVAL': [],
                'VISUAL': [],
                'PHASE': [],
                'COUNT': [],
                'MICRO_AVG': [],
                'MICRO_STD': [],
                'MICRO_MED': [],
                'MACRO_AVG': [],
                'MACRO_STD': [],
                'MACRO_MED': [],
                'MICRO_ANGLE_AVG': [],
                'MICRO_ANGLE_STD': [],
                'MICRO_ANGLE_MED': [],
                'MACRO_ANGLE_AVG': [],
                'MACRO_ANGLE_STD': [],
                'MACRO_ANGLE_MED': [],
                'HIT_RATE_3': [],
                'HIT_RATE_5': [],
                }
    patient_log_dict = {'P_NAME': [],
                'COUNT': [],
                'MICRO_AVG': [],
                'MICRO_STD': [],
                'MICRO_MED': [],
                'MACRO_AVG': [],
                'MACRO_STD': [],
                'MACRO_MED': [],
                'MICRO_ANGLE_AVG': [],
                'MICRO_ANGLE_STD': [],
                'MICRO_ANGLE_MED': [],
                'MACRO_ANGLE_AVG': [],
                'MACRO_ANGLE_STD': [],
                'MACRO_ANGLE_MED': [],
                'HIT_RATE_3': [],
                'HIT_RATE_5': [],
                }
    for visu in ['-1']:
        for phase in ['-1']:
            best_eval_dict = dict(zip(evals, [dict() for i in evals]))
            for eval in evals:
                list_file = pd.ExcelFile(os.path.join(per_table_dir, work_index, f'{eval}_per_list_{visu}_{phase}_ext.xlsx'), engine="openpyxl")
                v_names = list_file.sheet_names
                # v_RFA = list(filter(lambda x: x.split('_')[0].lower() in RFA_list, v_names))
                # v_RFA.extend([v_name for v_name in list_file.sheet_names if v_name.lower().startswith('p003_rep')])
                # v_names = [v_name for v_name in v_names if v_name not in v_RFA]
                # # v_names = v_RFA
                pd_list = [pd.read_excel(list_file, sheet_name=v_name)[:-2] for v_name in v_names]
                for idx, video_df in enumerate(pd_list):
                    v_name = v_names[idx]
                    resolution = resolution_dict[v_name]['x_voxel'] if v_name in resolution_dict.keys() else resolution_dict['default']['x_voxel']
                    min_avg = sys.maxsize
                    min_angle = np.nan
                    min_col = None
                    min_list =[np.nan]
                    min_angle_list = [np.nan]
                    for col in video_df.columns:
                        col_angle = '_'.join(col.split('_')[:-2] + ['error', 'angle'])
                        if 'angle' in col or 'mask' in col or 'single_' in col or col in ['index', 'ENTER', 'TIP', 'EXT', 'SLOPE']:
                            continue
                        col_list = list(video_df[col])
                        angle_list = list(video_df[col_angle])
                        average_value = np.nanmean(col_list[:-2])
                        # average_value = col_list[-2]
                        if average_value < min_avg:
                            min_avg = average_value
                            min_col = col
                            min_list = np.array(col_list[:-2])
                            min_angle_list = angle_list[:-2]
                            min_angle = np.nanmean(angle_list[:-2])
                    if min_avg < sys.maxsize:
                        best_eval_dict[eval][v_name] = (min_col, min_avg*resolution, min_list*resolution, min_angle, min_angle_list[:-2])
                    else:
                        best_eval_dict[eval][v_name] = (min_col, np.nan, min_list, np.nan, min_angle_list[:-2])
                    
            if evals[0] != ['bbox_segm']:
                best_eval_dict['bbox_segm'] = {}
            for v_name in v_names:
                if len(evals) > 1:
                    if best_eval_dict['bbox'][v_name][1] < best_eval_dict['segm'][v_name][1]:
                        best_eval_dict['bbox_segm'][v_name] = best_eval_dict['bbox'][v_name]
                        eval_best = 'bbox'
                    else:
                        best_eval_dict['bbox_segm'][v_name] = best_eval_dict['segm'][v_name]
                        eval_best = 'segm'
                else:
                    best_eval_dict['bbox_segm'][v_name] = best_eval_dict[evals[0]][v_name]
                    eval_best = evals[0]      
                # video_log += f'PHASE:{str(phase)}, VISU:{str(visu)}, VIDEO:{v_name}, COL:{best_eval_dict["bbox_segm"][v_name][0]}, EVAL: {eval_best}\n'
                video_log_dict['PHASE'].append(phase)
                video_log_dict['VISU'].append(visu)
                video_log_dict['VIDEO'].append(v_name)
                video_log_dict['COL'].append(best_eval_dict["bbox_segm"][v_name][0])
                video_log_dict['EVAL'].append(eval_best)
                video_log_dict['SHAFT'].append(np.nanmean(best_eval_dict['bbox_segm'][v_name][2]))
                video_log_dict['TIP'].append(np.nanmean(best_eval_dict['bbox_segm'][v_name][4]))
                if visu == '-1' and phase == '-1':
                    p_name = v_name.split('_')[0]
                    if p_name not in patient_record_dict.keys():
                        patient_record_dict[p_name] = {'tip_list': [],
                                                    'tip_avg': [],
                                                    'shaft_list': [],
                                                    'shaft_avg': []}
                    patient_record_dict[p_name]['tip_list'].append(best_eval_dict['bbox_segm'][v_name][2])
                    patient_record_dict[p_name]['tip_avg'].append(np.nanmean(best_eval_dict['bbox_segm'][v_name][2]))
                    patient_record_dict[p_name]['shaft_list'].append(best_eval_dict['bbox_segm'][v_name][4])
                    patient_record_dict[p_name]['shaft_avg'].append(np.nanmean(best_eval_dict['bbox_segm'][v_name][4]))
            for eval in evals + ['bbox_segm']:
                out_dict['EVAL'].append(eval)
                out_dict['VISUAL'].append(visu)
                out_dict['PHASE'].append(phase)
                per_list_all = np.concatenate([best_eval_dict[eval][v_name][2] for v_name in v_names])
                per_avg_all = np.array([np.nanmean(best_eval_dict[eval][v_name][2]) for v_name in v_names])
                per_list_all_angle = np.concatenate([best_eval_dict[eval][v_name][4] for v_name in v_names])
                per_avg_all_angle = np.array([np.nanmean(best_eval_dict[eval][v_name][4]) for v_name in v_names])
                # print(per_avg_all)
                count = np.sum(~np.isnan(per_list_all))
                list_value = copy.deepcopy(per_list_all)
                list_value[np.isnan(list_value)] = 1
                count_all = len(list_value)
                nan_rate = count/count_all
                out_dict['COUNT'].append(count)
                out_dict['MICRO_AVG'].append(np.nanmean(per_list_all))
                out_dict['MICRO_STD'].append(np.nanstd(per_list_all))
                out_dict['MICRO_MED'].append(np.nanmedian(per_list_all))
                out_dict['MACRO_AVG'].append(np.nanmean(per_avg_all))
                out_dict['MACRO_STD'].append(np.nanstd(per_avg_all))
                out_dict['MACRO_MED'].append(np.nanmedian(per_avg_all))
                out_dict['MICRO_ANGLE_AVG'].append(np.nanmean(per_list_all_angle))
                out_dict['MICRO_ANGLE_STD'].append(np.nanstd(per_list_all_angle))
                out_dict['MICRO_ANGLE_MED'].append(np.nanmedian(per_list_all_angle))
                out_dict['MACRO_ANGLE_AVG'].append(np.nanmean(per_avg_all_angle))
                out_dict['MACRO_ANGLE_STD'].append(np.nanstd(per_avg_all_angle))
                out_dict['MACRO_ANGLE_MED'].append(np.nanmedian(per_avg_all_angle))
                out_dict['HIT_RATE_3'].append(np.sum(list_value <= 3) /count_all * 100)
                out_dict['HIT_RATE_5'].append(np.sum(list_value <= 5) /count_all * 100)

    for p_name in patient_record_dict.keys():
        patient_log_dict['P_NAME'].append(p_name)
        per_list_all = np.concatenate(patient_record_dict[p_name]['tip_list'])
        per_avg_all = np.array(patient_record_dict[p_name]['tip_avg'])
        per_list_all_angle = np.concatenate(patient_record_dict[p_name]['shaft_list'])
        per_avg_all_angle = np.array(patient_record_dict[p_name]['shaft_avg'])
        # print(per_avg_all)
        count = np.sum(~np.isnan(per_list_all))
        list_value = copy.deepcopy(per_list_all)
        list_value[np.isnan(list_value)] = 1
        count_all = len(list_value)
        nan_rate = count/count_all
        patient_log_dict['COUNT'].append(count)
        patient_log_dict['MICRO_AVG'].append(np.nanmean(per_list_all))
        patient_log_dict['MICRO_STD'].append(np.nanstd(per_list_all))
        patient_log_dict['MICRO_MED'].append(np.nanmedian(per_list_all))
        patient_log_dict['MACRO_AVG'].append(np.nanmean(per_avg_all))
        patient_log_dict['MACRO_STD'].append(np.nanstd(per_avg_all))
        patient_log_dict['MACRO_MED'].append(np.nanmedian(per_avg_all))
        patient_log_dict['MICRO_ANGLE_AVG'].append(np.nanmean(per_list_all_angle))
        patient_log_dict['MICRO_ANGLE_STD'].append(np.nanstd(per_list_all_angle))
        patient_log_dict['MICRO_ANGLE_MED'].append(np.nanmedian(per_list_all_angle))
        patient_log_dict['MACRO_ANGLE_AVG'].append(np.nanmean(per_avg_all_angle))
        patient_log_dict['MACRO_ANGLE_STD'].append(np.nanstd(per_avg_all_angle))
        patient_log_dict['MACRO_ANGLE_MED'].append(np.nanmedian(per_avg_all_angle))
        patient_log_dict['HIT_RATE_3'].append(np.sum(list_value <= 3) /count_all * 100)
        patient_log_dict['HIT_RATE_5'].append(np.sum(list_value <= 5) /count_all * 100)
    pd.DataFrame(out_dict).to_excel(os.path.join(per_table_dir, work_index, f'summary_best_ext.xlsx'), index=False)
    pd.DataFrame(video_log_dict).to_excel(os.path.join(per_table_dir, work_index, f'video_log.xlsx'), index=False)
    pd.DataFrame(patient_log_dict).to_excel(os.path.join(per_table_dir, work_index, f'patient_log.xlsx'), index=False)
            
def concat_list():
    per_table_dir = '/srv/fenster/people/Ningtao/Project/USVideo/tracking/mmtracking/custom/metric_doc/liver_needle'
    work_index = 'all_folds_top2'
    visu, phase = '-1', '-1'
    video_log_df = pd.read_excel(os.path.join(per_table_dir, work_index, 'video_log.xlsx'))
    list_file_bbox = pd.ExcelFile(os.path.join(per_table_dir, work_index, f'bbox_per_list_{visu}_{phase}_ext.xlsx'))
    list_file_segm = pd.ExcelFile(os.path.join(per_table_dir, work_index, f'segm_per_list_{visu}_{phase}_ext.xlsx'))
    video_log_df = video_log_df[(video_log_df['PHASE'] == int(phase)) & (video_log_df['VISU'] == int(visu))]
    dis_all = []
    angle_all = []
    for idx, row in video_log_df.iterrows():
        v_name = row['VIDEO']
        col = row['COL']
        col_angle = '_'.join(col.split('_')[:-2] + ['error', 'angle'])
        eval = row['EVAL']
        resolution = resolution_dict[v_name]['x_voxel'] if v_name in resolution_dict.keys() else resolution_dict['default']['x_voxel']
        if eval == 'bbox':
            video_df = pd.read_excel(list_file_bbox, sheet_name=v_name)
        else:
            video_df = pd.read_excel(list_file_segm, sheet_name=v_name)
        dis_list = np.array(video_df[col])[:-2] * resolution
        angle_list = np.array(video_df[col_angle])[:-2]
        dis_all.append(dis_list[~np.isnan(dis_list)])
        angle_all.append(angle_list[~np.isnan(angle_list)])
    dis_all = np.concatenate(dis_all)
    angle_all = np.concatenate(angle_all)
    pd.DataFrame(
        {'DIS_ERROR': dis_all,
         'ANGLE_ERROR': angle_all}
    ).to_excel(os.path.join(per_table_dir, work_index, f'best_concat_{visu}_{phase}_ext.xlsx'))

def static_std():
    per_table_dir = '/srv/fenster/people/Ningtao/Project/USVideo/tracking/mmtracking/custom/metric_doc/liver_needle'
    work_index = '215646'

    for eval in ['bbox', 'segm']:
        for visu in ['-1', '1', '2', '3', '4']:
            for phase in ['-1', '0', '1']:
                list_file = pd.ExcelFile(os.path.join(per_table_dir, work_index, f'{eval}_per_list_{visu}_{phase}_ext.xlsx'))
                avg_dict = pd.read_excel(os.path.join(per_table_dir, work_index, f'{eval}_per_avg_{visu}_{phase}_ext.xlsx'), sheet_name='VIDEO AVG').to_dict(orient='list')
                pd_list = [pd.read_excel(list_file, sheet_name=v_name)[:-2] for v_name in list_file.sheet_names]
                for idx, video_df in enumerate(pd_list):
                    v_name = list_file.sheet_names[idx]
                    resolution = resolution_dict[v_name]['x_voxel'] if v_name in resolution_dict.keys() else resolution_dict['default']['x_voxel']
                    video_df.drop(columns=['index', 'ENTER', 'TIP', 'EXT'], inplace=True)
                    for col in video_df.columns:
                        if 'angle' not in col and col not in ['index', 'ENTER', 'TIP', 'EXT', 'SLOPE']:
                            video_df[col+'_MM'] = video_df[col] * resolution
                discrib_list = [video.describe(include='all') for video in pd_list]
                print(discrib_list[0])
                video_summary = pd.concat(pd_list, ignore_index=True)
                micro_describe = video_summary.describe(include='all')
                macro_describe = pd.concat([dis.loc['std'] for dis in  discrib_list], axis=1, ignore_index=True).T
                macro_describe.reset_index(drop=True)
                print(macro_describe)
                macro_describe = macro_describe.describe(include='all')
                print(macro_describe)
                for key in avg_dict.keys():
                    if key == 'VIDEO':
                        avg_dict[key].extend(['MACRO_STD', 'MICRO_STD', 'COUNT', 'MACRO_MEAN', 'MICRO_MEAN'])
                    elif key == 'RESO':
                        avg_dict[key].extend([None] * 5)
                    elif key == 'LEN':
                        avg_dict[key].extend([None] * 5)
                    else:
                        # TODO macro std
                        avg_dict[key].extend([macro_describe.at['mean', key], 
                                            micro_describe.at['std', key],
                                            micro_describe.at['count', key],
                                            macro_describe.at['mean', key],
                                            micro_describe.at['mean', key]])
                pd.DataFrame(avg_dict).to_excel(os.path.join(per_table_dir, work_index, f'{eval}_per_summary_{visu}_{phase}_ext.xlsx'), index=False)


def load_pkl():
    path = r'/srv/fenster/people/Ningtao/Project/USVideo/tracking/mmdetection/work_dirs/mask_rcnn_r50_caffe_c4_1x_needle_tip_RPN/result/230403/bbox.pkl'
    array = np.load(path, allow_pickle=True)
    print(array)


def error_analysis():
    result_dir = r'/srv/fenster/people/Ningtao/Project/USVideo/tracking/mmtracking/custom/metric_doc/liver_needle'
    work_index = 'all_folds_top2'
    result_dir = os.path.join(result_dir, work_index)
    video_log_all = pd.read_excel(os.path.join(result_dir, 'video_log.xlsx'))
    for phase in ['-1', '0', '1']:
        for visu in ['-1']:
            error_analy_dict = {
                'ENTER_X': [],
                'ENTER_Y': [],
                'TIP_X': [],
                'TIP_Y': [],
                'SLOPE': [],
                'TIP_MOTION': [],
                'SLOPE_MOTION': [],
                'DIS_ERROR': [],
                'DIS_ERROR_MM': [],
                'ANG_ERROR': []
            }
            list_bbox = pd.ExcelFile(os.path.join(result_dir, 'bbox_per_list_{}_{}_ext.xlsx'.format(visu, phase)))
            list_segm = pd.ExcelFile(os.path.join(result_dir, 'segm_per_list_{}_{}_ext.xlsx'.format(visu, phase)))
            video_log = video_log_all[(video_log_all['PHASE'] == int(phase)) & 
                                      (video_log_all['VISU'] == int(visu))]
            for idx, row in video_log.iterrows():
                v_name = row['VIDEO']
                reso = resolution_dict[v_name]['x_voxel'] if v_name in resolution_dict.keys() else resolution_dict['default']['x_voxel']
                eval = row['EVAL']
                col = row['COL']
                if pd.isna(col):
                    continue
                col_angle = '_'.join(col.split('_')[:-2] + ['error', 'angle'])
                if eval == 'bbox':
                    list_df = pd.read_excel(list_bbox, sheet_name=v_name)[:-2]
                else:
                    list_df = pd.read_excel(list_segm, sheet_name=v_name)[:-2]
                dis_error_series = np.array(list_df[col][1:])
                ang_error_series = np.array(list_df[col_angle][1:])
                
                enter_series = list_df['ENTER']
                tip_series = list_df['TIP']
                enter_x, enter_y = np.array(pd.to_numeric(enter_series.str.split(' ', expand=True)[0])), np.array(pd.to_numeric(enter_series.str.split(' ', expand=True)[1]))
                tip_x, tip_y = pd.to_numeric(tip_series.str.split(' ', expand=True)[0]), pd.to_numeric(tip_series.str.split(' ', expand=True)[1])
                
                tip_x_diff, tip_y_diff = np.array(tip_x.diff()[1:]), np.array(tip_y.diff()[1:])
                tip_motion_list = np.sqrt(np.power(tip_x_diff, 2) + np.power(tip_y_diff, 2))

                slope_series = list_df['SLOPE']
                slop_motion_list = np.array(slope_series.diff()[1:])

                enter_x = enter_x[1:]
                enter_y = enter_y[1:]
                tip_x = np.array(tip_x)[1:]
                tip_y = np.array(tip_y)[1:]
                slope_series = np.array(slope_series)[1:]
                
                non_empty_index = ~np.isnan(dis_error_series)
                error_analy_dict['ANG_ERROR'].extend(ang_error_series[non_empty_index].tolist())
                error_analy_dict['DIS_ERROR'].extend(dis_error_series[non_empty_index].tolist())
                error_analy_dict['DIS_ERROR_MM'].extend((dis_error_series[non_empty_index] * reso).tolist())
                error_analy_dict['ENTER_X'].extend(enter_x[non_empty_index].tolist())
                error_analy_dict['ENTER_Y'].extend(enter_y[non_empty_index].tolist())
                error_analy_dict['TIP_X'].extend(tip_x[non_empty_index].tolist())
                error_analy_dict['TIP_Y'].extend(tip_y[non_empty_index].tolist())
                error_analy_dict['SLOPE'].extend(slope_series[non_empty_index].tolist())
                error_analy_dict['TIP_MOTION'].extend(tip_motion_list[non_empty_index].tolist())
                error_analy_dict['SLOPE_MOTION'].extend(slop_motion_list[non_empty_index].tolist())
            pd.DataFrame(error_analy_dict).to_excel(os.path.join(result_dir, f'error_analysis_{visu}_{phase}.xlsx'))

def error_bins():
    result_dir = r'/srv/fenster/people/Ningtao/Project/USVideo/tracking/mmtracking/custom/metric_doc/liver_needle'
    work_index = 'all_folds'
    result_dir = os.path.join(result_dir, work_index)
    num_bins = 15
    x_list = ['SHAFT_LEN', 'SLOPE', 'TIP_MOTION', 'TIP_DEPTH', 'SLOPE_MOTION']
    x_bines_dict = {
        'SHAFT_LEN': [i for i in range(12, 73, 4)],
        'SLOPE': [i/10 for i in range(-16, 15, 2)],
        'TIP_MOTION': [i/10 for i in range(0, 121, 8)],
        'TIP_DEPTH': [i/10 for i in range(250, 626, 25)],
        'SLOPE_MOTION': [i/10 for i in range(0, 60, 4)]
    }
    y_list = ['DIS_ERROR', 'ANG_ERROR']
    col_map = {
        'SHAFT_LEN': 'Shaft Length',
        'SLOPE': 'Slope',
        'TIP_MOTION': 'Tip Motion Speed',
        'TIP_DEPTH': 'Tip Depth',
        'SLOPE_MOTION': 'Shaft Angular Speed',
        'DIS_ERROR': 'Tip Distance Error',
        'ANG_ERROR': 'Shaft Direction Error'
    }
    for phase in ['-1', '0', '1']:
        for visu in ['-1']:
            save_excel = pd.ExcelWriter(os.path.join(result_dir,  f'error_bin_{visu}_{phase}.xlsx'))
            error_list_df = pd.read_excel(os.path.join(result_dir, f'error_analysis_{visu}_{phase}.xlsx'))
            error_dict = dict(zip(error_list_df.columns, [np.array(error_list_df[col]) for col in error_list_df.columns]))
            for x_col in x_list:
                if x_col == 'SHAFT_LEN':
                    x_series = np.sqrt((error_dict['TIP_X'] - error_dict['ENTER_X']) **2 + (error_dict['TIP_Y'] - error_dict['ENTER_Y']) **2)
                elif x_col == 'TIP_DEPTH':
                    x_series = error_dict['TIP_Y']
                elif x_col == 'SLOPE_MOTION':
                    x_series = error_dict[x_col]
                    x_series = np.abs(np.arctan(x_series) * 180 / math.pi)
                else:
                    x_series = error_dict[x_col]    
                if 'SLOPE' not in x_col:
                    x_series = x_series*0.17
                
                # if x_col == 'TIP_MOTION':
                #     x_bins = [0, 0.81923, 1.63846, 2.4577, 3.27693, 4.09616, 4.91539, 5.73463, 6.55386, 7.37309, 8.19232, 9.01155, 9.83079, 10.65002, 10.65002, 11.46925]
                # elif x_col == 'SLOPE_MOTION':
                #     x_bins = [0, 0.42041, 0.84083, 1.26124, 1.68166, 2.10207, 2.52248, 2.9429, 3.36331, 3.78373, 4.20414, 4.62455, 5.04497, 5.46538, 5.8858]
                # else:
                #     x_bins = np.linspace(np.percentile(x_series, 5), np.percentile(x_series, 95), num_bins)
                x_bins = x_bines_dict[x_col]
                for y_col in y_list:
                    y_series = error_dict[y_col]
                    if 'DIS' in y_col:
                        y_series = y_series * 0.17
                    bin_dict = {col_map[x_col]: [], 'START': [], 'END': [], col_map[y_col]: [], 'STD': []}
                    for idx in range(len(x_bins)-1):
                        start, end = x_bins[idx], x_bins[idx+1]
                        bin_dict[col_map[x_col]].append(f'{str(round((start+end)/2, 2))}')
                        bin_dict['START'].append(start)
                        bin_dict['END'].append(end)
                        y_avg = np.nanmean(y_series[(x_series >= start) & (x_series < end)])
                        y_std = np.nanstd(y_series[(x_series >= start) & (x_series < end)])
                        bin_dict[col_map[y_col]].append(y_avg)
                        bin_dict['STD'].append(y_std)
                    pd.DataFrame(bin_dict).to_excel(save_excel, sheet_name=f'{x_col}_{y_col}', index=False)
            save_excel.save()

def score_label_ROC():
    result_dir = '/srv/fenster/people/Ningtao/Project/USVideo/tracking/mmtracking/work_dirs/mask_selsa_faster_rcnn_r50_dc5_1x_c04_needle1020_AFA_fold{}/result/{}/{}.pkl'
    anno_dir = '/srv/fenster/people/Ningtao/Dataset/LiverNeedle/1020/DataSet/coco_valid_0318_no_occu/{}/test.json'
    fold_dict={
                'fold': ['1', '2', '3', '4', '5'],
                'index': ['004507', '041441', '035857', '034112', '215646'],
                }
    label_list = []
    score_bbox_list = []
    score_segm_list = []
    score_best_list = []
    for idx, fold in enumerate(fold_dict['fold']):
        fold = str(fold)
        result_idx = fold_dict['index'][idx]
        anno_path = anno_dir.format(fold)
        anno_coco = COCO(anno_path)
        result_bbox = np.load(result_dir.format(fold, result_idx, 'bbox'), allow_pickle=True)['det_bboxes']
        result_segm = np.load(result_dir.format(fold, result_idx, 'segm'), allow_pickle=True)['det_bboxes']
        for img_id in anno_coco.getImgIds():
            ann_ids = anno_coco.getAnnIds(imgIds=img_id, catIds=1)
            score_bbox = result_bbox[img_id-1][0][0][0][-1]
            score_segm = result_segm[img_id-1][0][0][0][-1]
            if len(ann_ids) <= 0:
                label = 0
                score_best = min(score_bbox, score_segm)
            else:
                label = 1
                score_best = max(score_bbox, score_segm)
            label_list.append(label)
            score_bbox_list.append(score_bbox)
            score_segm_list.append(score_segm)
            score_best_list.append(score_best)
    score_df = pd.DataFrame({'LABEL': label_list,
                  'SCORE_BBOX': score_bbox_list,
                  'SCORE_SEGM': score_segm_list,
                  'SCORE_BEST': score_best_list})
    score_df.to_csv('/srv/fenster/people/Ningtao/Project/USVideo/tracking/mmtracking/custom/metric_doc/liver_needle/all_folds/score.csv', index=False)
    # best threshold 0.83428
    pred_list = (np.array(score_best_list) >= 0.83428) * 1
    cm = confusion_matrix(np.array(label_list), pred_list) 
    print(cm)
    # conf_matrix = pd.DataFrame(cm, index=['Neg.', 'Pos.'], columns=['Neg.', 'Pos.']) 

    # fig, ax = plt.subplots(figsize=(4.5, 3.5))
    # sns.heatmap(conf_matrix, annot=False, annot_kws={"size": 10}, cmap="Blues")
    # plt.ylabel('True label', fontsize=10)
    # plt.xlabel('Predicted label', fontsize=10)
    # plt.xticks(fontsize=10)
    # plt.xticks(fontsize=10)
    # plt.yticks(fontsize=10)
    # plt.savefig('/srv/fenster/people/Ningtao/Project/USVideo/tracking/mmtracking/custom/metric_doc/liver_needle/all_folds/confusion.pdf', format='pdf', bbox_inches='tight')

def add_relative_dis():
    doc_dir = '/srv/fenster/people/Ningtao/Project/USVideo/tracking/mmtracking/custom/metric_doc/liver_needle'
    work_dir = 'all_folds_top2_mask'

    def get_relative_dis(row, dis_eror):
        try:
            t_x, t_y = float(row['TIP'].split(' ')[0]), float(row['TIP'].split(' ')[1])
            e_x, e_y = float(row['ENTER'].split(' ')[0]), float(row['ENTER'].split(' ')[1])
            length = math.sqrt((t_x - e_x) ** 2 + (t_y - e_y) ** 2)
            relative_dis = round(dis_eror / length, 4)
        except:
            # print(row)
            relative_dis = None
        return relative_dis

    for eval in ['bbox', 'segm']:
        for phase in ['-1', '0', '1']:
            for visu in ['-1', '1', '2', '3', '4']:
                per_list_path = os.path.join(doc_dir, work_dir, f'{eval}_per_list_{visu}_{phase}_ext.xlsx')
                if not os.path.exists(per_list_path):
                    continue
                print(per_list_path)
                per_list = pd.ExcelFile(per_list_path, engine='openpyxl')
                per_list_rel = pd.ExcelWriter(os.path.join(doc_dir, work_dir, f'{eval}_per_list_{visu}_{phase}_ext_rel.xlsx'), engine='openpyxl')
                v_names = per_list.sheet_names
                for v_name in v_names:
                    df = pd.read_excel(per_list, sheet_name=v_name)
                    cols = df.columns
                    for col in cols:
                        if 'dis' not in col:
                            continue
                        df[f'{col}_rel'] = df.apply(lambda row: get_relative_dis(row, row[col]), axis=1)
                    df.drop(columns=[col for col in cols if '_rel' not in col], inplace=True)
                    df.to_excel(per_list_rel, sheet_name=v_name, index=False)
                per_list_rel.save()

def statitic_specific_per():
    doc_dir = 'path/to/your/metric_doc'
    work_dir = 'path/to/your/work_dir'
    col_key = 'mask'
    result_file = open(os.path.join(doc_dir, work_dir, f'{col_key}.txt'), mode='w')

    for eval in ['bbox', 'segm']:
        for phase in ['-1', '0', '1']:
            for visu in ['-1', '1', '2', '3', '4']:
                print(f'{eval}_{visu}_{phase}')

                avg_dict = {}
                list_dict = {}
                per_list_path = os.path.join(doc_dir, work_dir, f'{eval}_per_list_{visu}_{phase}_ext_rel.xlsx')
                if not os.path.exists(per_list_path):
                    print(per_list_path)
                    continue
                per_list_file = pd.ExcelFile(per_list_path)
                v_names = per_list_file.sheet_names
                for v_name in v_names:
                    df = pd.read_excel(per_list_file, sheet_name=v_name)
                    cols = df.columns
                    for col in cols:
                        if col_key not in col:
                            continue
                        if col not in avg_dict.keys():
                            avg_dict[col] = []
                            list_dict[col] = []
                        per_list = df[col].tolist()
                        per_mean = np.nanmean(per_list)
                        avg_dict[col].append(per_mean)
                        list_dict[col].extend(per_list)
                temp_best_micro = {}
                temp_best_macro = {}
                for key in list_dict.keys():
                    mean = np.nanmean(list_dict[key])
                    median = np.nanmedian(list_dict[key])
                    std = np.nanstd(list_dict[key])
                    temp_best_micro[key] = (mean, median, std)

                    mean_macro = np.nanmean(avg_dict[key])
                    median_macro = np.nanmedian(avg_dict[key])
                    std_macro = np.nanstd(avg_dict[key])
                    temp_best_macro[key] = (mean_macro, median_macro, std_macro)
                    
                best_micro = sorted(temp_best_micro.items(), key=lambda x: x[1][0], reverse=False)[0]
                best_macro = sorted(temp_best_macro.items(), key=lambda x: x[1][0], reverse=False)[0]

                print(f'{eval}_{phase}_{visu}', file=result_file)
                print(f'micro: {str(best_micro)}', file=result_file)
                print(f'macro: {str(best_macro)}', file=result_file)
                result_file.flush()
            

if __name__ == '__main__':
    # eval_vis()
    # load_res()
    # eval_combine()
    # eval_combine_fold()
    # load_pkl()
    # get_summary()
    # static_std()
    # summary_describe()
    # summary_best()
    # concat_list()
    # error_analysis()
    # error_bins()
    # score_label_ROC()
    # add_relative_dis()
    # statitic_specific_per()
    pass
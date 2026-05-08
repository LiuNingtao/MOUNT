import os
import json
import numpy as np
import matplotlib.pyplot as plt
import pydicom
from skimage.metrics import structural_similarity as ssim
from pycocotools.coco import COCO
from PIL import Image, ImageFilter
from tqdm import tqdm
import xml.etree.ElementTree as ET
from sklearn.model_selection import train_test_split
from pycocotools.mask import decode as m_decode
import re
import random
import copy
import cv2
import shutil
from sklearn.linear_model import LinearRegression
# from metric_liver_needle import fit_line
import math
import pandas as pd
from scipy import ndimage
from matplotlib.animation import FuncAnimation
import sys
from pycocotools.coco import COCO
from mmtrack.datasets.parsers import CocoVID
import json



random.seed(42)
resolution_dict = json.load(open(r'/srv/fenster/people/Ningtao/Dataset/LiverNeedle/1020/DataSet/Documents/resolution.json', 'r'))
frame_type_xlsx = pd.ExcelFile(r'/srv/fenster/people/Ningtao/Dataset/LiverNeedle/1020/DataSet/tip_type.xlsx')

ablation_phase_origin_dict = {
    "p003_v7": (390, 670),
    "p004_v1": (1026, 1200),
    "p004_v6": (490, 566),
    "p007_v1": (403, 488),
    "p007_v2": (869, 879),
    "p008_v1": (-1, -1),
    "p008_v2": (916, 1055),
    "p012_04": (150, 163),
    "p012_021": (748, 1132),
    "p020_03": (-1, -1),
    "p020_07": (-1, -1),
    "p020_08": (-1, -1),
    "p008_v4": (25, 102),
    "p018_06": (181, 348),
    "p015_04": (521, 613),
    "p013_06": (52, 114),
    "p018_01": (853, 1063),
    "p003_v1": (414, 610),
    "p012_01": (-1, -1),
    "p019_01": (-1, -1),
    "p003_v2": (65, 69),
    "p019_03": (585, 770),
    "p003_v8": (-1, -1),
    "p015_07": (20, 33)
}

patient_video_num_dict = {
    'p020': 7,
    'p019': 3,
    'p018': 7,
    'p017': 2,
    'p015': 6,
    'p013': 3,
    'p012': 5,
    'p011': 2,
    'p010': 3,
    'p008': 4,
    'p007': 2,
    'p004': 7,
    'p003': 11
}

def divide_into_groups(patients):
    total_cases = sum(patients.values())
    target_cases = total_cases // 5
    n = len(patients)
    dp = [[False] * (target_cases + 1) for _ in range(n + 1)]
    dp[0][0] = True

    for i in range(1, n + 1):
        for j in range(target_cases + 1):
            dp[i][j] = dp[i - 1][j] or (j >= list(patients.values())[i - 1] and dp[i - 1][j - list(patients.values())[i - 1]])

    if not dp[n][target_cases]:
        return []

    groups = []
    current_cases = target_cases
    for i in range(n, 0, -1):
        if dp[i][current_cases] and not dp[i - 1][current_cases]:
            groups.append(list(patients.keys())[i - 1])
            current_cases -= list(patients.values())[i - 1]
    groups.reverse()

    result = [groups]
    remaining_patients = list(patients.keys())[:len(groups)]
    while len(result) < 5:
        remaining_patients = remaining_patients[len(groups):]
        result.append([])
        for patient in remaining_patients:
            if sum(result[-1]) + patients[patient] <= target_cases:
                result[-1].append(patient)
    return result

def check_poly_len():
    anno_dir = r'/srv/fenster/people/Ningtao/Dataset/LiverNeedle/1020/DataSet'
    for task in ['train', 'val', 'test']:
        print(task)
        anno_dict = json.load(open(os.path.join(anno_dir, task+'.json')))
        for anno_info in anno_dict['annotations']:
            segs = anno_info['segmentation']
            for seg in segs:
                if len(seg) != 10:
                    print(anno_info['image_id'])


def get_ssim():
    anno_dir = r'M:\people\Ningtao\Dataset\LiverNeedle\1020\DataSet\coco'
    save_dir = r'M:\people\Ningtao\Dataset\LiverNeedle\1020\DataSet\coco_sim'
    data_dir = r'M:\people\Ningtao\Dataset\LiverNeedle\1020\Data'
    
    for mode in ['train', 'test', 'val']:
        anno_dict = json.load(open(os.path.join(anno_dir, f'{mode}.json'))) 
        image_list = anno_dict['images']
        for idx, image_info in tqdm(enumerate(image_list)):
            frame_id = image_info['frame_id']
            if frame_id == 0:
                ssim_value = 0
            else:
                previous_info = image_list[idx-1]
                previous_img =  Image.open(os.path.join(data_dir, previous_info['file_name'])).convert('L')
                previous_array = np.array(previous_img, dtype=np.float32)
                current_img = Image.open(os.path.join(data_dir, image_info['file_name'])).convert('L')
                current_array = np.array(current_img, dtype=np.float32)
                ssim_value = ssim(previous_array, current_array)
                previous_img.close()
                current_img.close()
            image_info['ssim'] = round(ssim_value, 2)
        json.dump(anno_dict, open(os.path.join(save_dir, f'{mode}.json'), mode='w'))


def enhance(img):
    # converting to LAB color space
    lab= cv2.cvtColor(img, cv2.COLOR_BGR2LAB)
    l_channel, a, b = cv2.split(lab)

    # Applying CLAHE to L-channel
    # feel free to try different values for the limit and grid size:
    clahe = cv2.createCLAHE(clipLimit=5.0, tileGridSize=(2,2))
    cl = clahe.apply(l_channel)

    # merge the CLAHE enhanced L-channel with the a and b channel
    limg = cv2.merge((cl,a,b))

    # Converting image from LAB Color model to BGR color spcae
    enhanced_img = cv2.cvtColor(limg, cv2.COLOR_LAB2BGR)
    plt.imsave('enhance.png', enhanced_img)
    # Stacking the original image with the enhanced image
    return enhanced_img

def detec_edge(file_path=None):
    # file_path = r'/srv/fenster/people/Ningtao/Dataset/LiverNeedle/1020/Data/task_p003_v4/images/000117.bmp'
    image = cv2.imread(file_path)
    # image = enhance(image)
    edge_image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    edge_image = cv2.GaussianBlur(edge_image, (3, 3), 1)
    edge_image = cv2.Canny(edge_image, 100, 200)
    edge_image = cv2.dilate(
        edge_image,
        cv2.getStructuringElement(cv2.MORPH_RECT, (5, 5)),
        iterations=1
    )
    edge_image = cv2.erode(
        edge_image,
        cv2.getStructuringElement(cv2.MORPH_RECT, (5, 5)),
        iterations=1
    )
    return edge_image

def generate_edge():
    root_dir = r'/srv/fenster/people/Ningtao/Dataset/LiverNeedle/1020/Data'
    for root, _, file_list in os.walk(root_dir):
        for file_name in file_list:
            if file_name.endswith('.bmp'):
                file_path = os.path.join(root, file_name)
                edge_image = detec_edge(file_path)
                Image.fromarray(edge_image).save(file_path.replace('.bmp', '_edge.png'))


def gen_edge_anno():
    origin_anno_dir = r'/srv/fenster/people/Ningtao/Dataset/LiverNeedle/1020/DataSet/cross_valid'
    save_dir = r'/srv/fenster/people/Ningtao/Dataset/LiverNeedle/1020/DataSet/coco_flow'
    for fold in range(1, 6):
        fold = str(fold)
        os.makedirs(os.path.join(save_dir, fold))
        for mode in ['train', 'test']:
            anno = json.load(open(os.path.join(origin_anno_dir, fold, mode+'.json')))
            for img_info in anno['images']:
                img_info['file_name'] = [img_info['file_name'], img_info['file_name'].replace('.bmp', '_flow.png')]
            json.dump(anno, open(os.path.join(save_dir, fold, mode+'.json'), mode='w'))

def difference_adj_frame():
    root_dir = r'/srv/fenster/people/Ningtao/Dataset/LiverNeedle/1020/Data'
    window_width = 1
    for video_name in os.listdir(root_dir):
        print(video_name)
        video_path = os.path.join(root_dir, video_name, 'images')
        frame_list = os.listdir(video_path)
        frame_list = list(filter(lambda x: re.match(r'\d+.bmp', x), frame_list))
        frame_list.sort()
        for i in range(window_width, len(frame_list), window_width):
            frame_pre = np.array(Image.open(os.path.join(video_path, frame_list[i-window_width])).filter(ImageFilter.BoxBlur(2)), dtype=np.float32)
            frame_pos = np.array(Image.open(os.path.join(video_path, frame_list[i])).filter(ImageFilter.BoxBlur(2)), dtype=np.float32)
            frame_diff = np.abs(frame_pos - frame_pre)
            frame_diff = np.array(frame_diff / np.max(frame_diff) * 255, dtype=np.uint8)
            frame_diff = Image.fromarray(frame_diff)
            frame_diff.save(os.path.join(video_path, frame_list[i].replace('.bmp', '_diff.bmp')))
            frame_diff.close()

def generate_mask():
    data_root = r'/srv/fenster/people/Ningtao/Dataset/LiverNeedle/1020/Data'
    data_set_dir = r'/srv/fenster/people/Ningtao/Dataset/LiverNeedle/1020/DataSet/coco_valid_0318_no_occu/5'
    save_dir = r'/srv/fenster/people/Ningtao/Dataset/LiverNeedle/1020/ytvis_2019/DataMask'

    for mode in ['train', 'test']:
        anno_path = os.path.join(data_set_dir, mode+'.json')
        anno_coco = COCO(anno_path)
        images = anno_coco.imgs
        for image_id, image_info in images.items():
            anno_ids = anno_coco.getAnnIds(imgIds=image_id, catIds=1)
            annos = anno_coco.loadAnns(ids=anno_ids)
            # masks = []
            # for anno in annos:
            #     mask = anno_coco.annToMask(anno)
            #     masks.append(mask)
            # if len(masks) > 0:
            #     mask = np.sum(np.array(masks), axis=0)
            if len(annos) > 0:
                mask = anno_coco.annToMask(annos[0])
            else:
                mask = np.zeros(shape=(image_info['height'], image_info['width']))
            mask = np.array(mask*255, dtype=np.uint8)
            # print(np.max(mask))
            mask[mask != 0] = 1
            save_path = os.path.join(save_dir, image_info['file_name'].replace('.bmp', '.png')).replace('images', '')
            if not os.path.exists(os.path.dirname(save_path)):
                os.makedirs(os.path.dirname(save_path))

            Image.fromarray(mask).convert('P').save(save_path)


def seg_dataset():
    anno_dir = r'/srv/fenster/people/Ningtao/Dataset/LiverNeedle/1020/DataSet'
    for mode in ['train', 'test', 'val']:
        anno_coco = COCO(os.path.join(anno_dir, 'coco', mode+'.json'))
        images = anno_coco.imgs
        image_list = [image_info['file_name'] for (id, image_info) in images.items()]
        with open(os.path.join(anno_dir, 'seg', mode+'.txt'), mode='w') as f:
            f.write('\n'.join(image_list))

def split_quality_dataset():
    quality_json = r'/srv/fenster/people/Ningtao/Dataset/LiverNeedle/1020/DataSet/split_by_quality/quality.json'
    root_dir = r'/srv/fenster/people/Ningtao/Dataset/LiverNeedle/1020/Data'
    save_dir = r'/srv/fenster/people/Ningtao/Dataset/LiverNeedle/1020/DataSet/split_by_quality'
    quality_dict = json.load(open(quality_json))
    # for key in quality_dict.keys():
    for key in ['mid']:
        os.makedirs(os.path.join(save_dir, key), exist_ok=True)
        video_list = quality_dict[key]
        video_list = [v['video_name'] for v in video_list]
        train_list, test_list = train_test_split(video_list,test_size=2)
        train_dataset_dict = convert_video_coco(root_dir, train_list)
        test_dataset_dict = convert_video_coco(root_dir, test_list)
        json.dump(train_dataset_dict, open(os.path.join(save_dir, key, 'train.json'), mode='w'))
        json.dump(test_dataset_dict, open(os.path.join(save_dir, key, 'test.json'), mode='w'))

def ablation_type_dataset():
    type_json = r'/srv/fenster/people/Ningtao/Dataset/LiverNeedle/1020/DataSet/Documents/case_statistic.json'
    data_dir = r'/srv/fenster/people/Ningtao/Dataset/LiverNeedle/1020/Data'
    save_dir = r'/srv/fenster/people/Ningtao/Dataset/LiverNeedle/1020/DataSet/RFA'
    case_list = json.load(open(type_json))['cases']
    case_list = [c['name'] for c in case_list if c['ablation_type'] == 'RFA']
    data_list = os.listdir(data_dir)
    filtered_lits = []
    for video_name in data_list:
        for case_name in case_list:
            if video_name.startswith(case_name.lower()):
                filtered_lits.append(video_name)
                break
    train_list, test_list = train_test_split(filtered_lits,test_size=0.4)
    test_list, val_list = train_test_split(test_list,test_size=0.5)
    train_dataset_dict = convert_video_coco(data_dir, train_list)
    test_dataset_dict = convert_video_coco(data_dir, test_list)
    val_dataset_dict = convert_video_coco(data_dir, val_list)
    json.dump(train_dataset_dict, open(os.path.join(save_dir, 'train.json'), mode='w'))
    json.dump(test_dataset_dict, open(os.path.join(save_dir, 'test.json'), mode='w'))
    json.dump(val_dataset_dict, open(os.path.join(save_dir, 'val.json'), mode='w'))


    pass

def split_dataset_random():
    root_dir = r'/srv/fenster/people/Ningtao/Dataset/LiverNeedle/1020/DataSet/0318'
    save_dir = r'/srv/fenster/people/Ningtao/Dataset/LiverNeedle/1020/DataSet/coco_valid_0318_no_occu'
    video_list = os.listdir(root_dir)
    random.shuffle(video_list)
    k = 5
    test_len = len(video_list) // k
    for fold in range(k):
        os.makedirs(os.path.join(save_dir, str(fold+1)), exist_ok=True)
        train_list = video_list[:test_len*fold] + video_list[test_len*(fold+1): ]
        test_list = video_list[test_len*fold: test_len*(fold+1)]
        train_anno_dict = convert_video_coco(root_dir, train_list, True) 
        test_anno_dict = convert_video_coco(root_dir, test_list, True)
        json.dump(train_anno_dict, open(os.path.join(save_dir, str(fold+1), 'train.json'), mode='w'))
        json.dump(test_anno_dict, open(os.path.join(save_dir, str(fold+1), 'test.json'), mode='w'))
        json.dump(test_anno_dict, open(os.path.join(save_dir, str(fold+1), 'val.json'), mode='w'))



def convert_video_coco(root_dir, sequence_list, ignore_occu=True):
    video_idx = 0
    image_idx = 0
    anno_idx = 0
    instance_idx = 0
    image_map_dict = {}

    video_list = []
    image_list = []
    anno_list = []
    instance_list = []

    for video_name in sequence_list:
        print(video_name)
        video_idx += 1
        video_list.append(dict(id=video_idx, name=video_name))
        origin_anno = json.load(open(os.path.join(root_dir, video_name, 'annotations', 'instances_coco.json'), mode='r'))
        for frame_idx, image_info in enumerate(origin_anno['images']):
            image_info = copy.deepcopy(image_info)
            image_idx += 1
            origin_id = image_info['id']
            image_info['id'] = image_idx
            image_info['file_name'] = video_name + '/images/' + image_info['file_name']
            image_map_dict[f'{str(video_idx)}_{str(origin_id)}'] = image_idx
            image_info['video_id'] = video_idx
            image_info['frame_id'] = frame_idx
            image_list.append(image_info)
        for anno_info in origin_anno['annotations']:
            if anno_info['attributes']['occluded'] and ignore_occu:
                continue
            anno_info = copy.deepcopy(anno_info)
            anno_idx += 1
            anno_info['id'] = anno_idx
            anno_info['image_id'] = image_map_dict[str(video_idx)+'_'+str(anno_info['image_id'])]
            anno_info['video_id'] = video_idx
            try:
                track_id = str(video_idx) + '_' + str(anno_info['attributes']['track_id'])
            except:
                print(track_id)
                print(anno_info)
            if track_id not in instance_list:
                instance_idx += 1
                instance_list.append(track_id)
            anno_info['instance_id'] = instance_idx
            for attr_key in anno_info['attributes'].keys():
                anno_info[attr_key] = anno_info['attributes'][attr_key]
            anno_info['truncated'] = False
            anno_info['iscrowd'] = 0
            anno_info['ignore'] = False if not anno_info['attributes']['occluded'] else True
            anno_info['visibility'] = 1.0
            del anno_info['attributes']
            anno_list.append(anno_info)
    new_anno_dict = {'categories': origin_anno['categories'],
                     'videos': video_list,
                     'images': image_list,
                     'annotations': anno_list}
    return new_anno_dict

def init_new_video(video_dir):
    scope = 'anno'
    cate_list =  [{
            "id": 1,
            "name": "Needle",
            "supercategory": ""
        },
        {
            "id": 2,
            "name": "Tip",
            "supercategory": ""
        }
    ]
    cate_map_dict = {1: 1, 2: 1, 3: 1, 4: 2}
    anno_dict = json.load(open(os.path.join(video_dir, 'annotations', 'instances_default.json')))
    
    if scope == 'data':
        image_list = os.listdir(os.path.join(video_dir, 'images'))
    else:
        image_list = [image_info['file_name'] for image_info in anno_dict['images']]
    image_list.sort()
    image_name_map = {}
    for idx, file_name in enumerate(image_list):
        new_file_name = format(idx+1, '06d')+ '.' +file_name.split('.')[-1]
        image_name_map[file_name] = new_file_name
        if scope == 'data':
            os.rename(os.path.join(video_dir, 'images', file_name), os.path.join(video_dir, 'images', new_file_name))
    for image_info in anno_dict['images']:
        image_info['file_name'] = image_name_map[image_info['file_name']]
    for anno_info in anno_dict['annotations']:
        anno_info['category_id'] = cate_map_dict[anno_info['category_id']]
    anno_dict['categories'] = cate_list
    json.dump(anno_dict, open(os.path.join(video_dir, 'annotations', 'instances_coco.json'), mode='w'))

def init_inference_video():
    video_dir = r'/srv/fenster/people/Ningtao/Dataset/LiverNeedle/1020/Pantom/4'
    cate_list =  [{
            "id": 1,
            "name": "Needle",
            "supercategory": ""
        },
        {
            "id": 2,
            "name": "Tip",
            "supercategory": ""
        }
    ]
    video_list = [
        {
            "id": 1,
            "name": "03_01"
        }
    ]

    image_list = os.listdir(os.path.join(video_dir))
    image_name_map = {}
    image_annos = []
    for idx, file_name in enumerate(image_list):
        new_file_name = format(idx+1, '06d')+ '.' +file_name.split('.')[-1]
        image_name_map[file_name] = new_file_name
        os.rename(os.path.join(video_dir, file_name), os.path.join(video_dir, new_file_name))
        image_annos.append({
            "id": idx+1,
            "width": 900,
            "height": 700,
            "file_name": new_file_name,
            "license": 0,
            "flickr_url": "",
            "coco_url": "",
            "date_captured": 0,
            "video_id": 1,
            "frame_id": idx
        })
    json.dump({
        "videos": video_list,
        "categories": cate_list,
        "images": image_annos,
        "annotations": []
    },
    open(r'/srv/fenster/people/Ningtao/Dataset/LiverNeedle/1020/Pantom/4.json', 'w'))
def init_dataset():
    data_dir = r'/srv/fenster/people/Ningtao/Dataset/LiverNeedle/1020/DataSet/0318'
    for video_name in os.listdir(data_dir):
        if '.zip' in video_name:
            continue
        init_new_video(os.path.join(data_dir, video_name))
        print(video_name)

def get_tip_region():
    root_dir = r'/srv/fenster/people/Ningtao/Dataset/LiverNeedle/1020/DataSet/coco_valid_0318_no_occu'
    for fold in range(1, 2):
        fold = str(fold)
        for mode in ['train', 'test']:
            json_path = os.path.join(root_dir, fold, mode+'.json')
            anno_coco = COCO(json_path)
            origin_anno = json.load(open(json_path))

            image_with_tip = [anno['image_id'] for anno in origin_anno['annotations'] if anno['category_id'] == 2]
            image_with_tip = set(image_with_tip)

            anno_indx = 1
            new_anno_list = []
            for anno_info in origin_anno['annotations']:
                id = anno_info['id']
                image_id = anno_info['image_id']
                
                anno_info['id'] = anno_indx
                new_anno_list.append(anno_info)
                anno_indx += 1
                if (anno_info['category_id'] != 1) or (image_id in image_with_tip):
                    continue

                anno = anno_coco.loadAnns(id)[0]
                mask = anno_coco.annToMask(anno)
                # Image.fromarray(np.array(mask*128, dtype=np.uint8)).show()
                mask_index = np.nonzero(mask)
                h_index = np.max(mask_index[0])
                t = np.where(mask_index[0]==h_index)[0]
                w_index = mask_index[1][t[len(t)//2]]
                # print(f'h_index: {str(h_index)}, w_index: {str(w_index)}')
                assert mask[h_index, w_index] > 0  

                tip_anno = copy.deepcopy(anno_info)
                tip_anno['id'] = anno_indx
                anno_indx += 1
                tip_anno['category_id'] = 2
                # del tip_anno['segmentation']
                tip_anno['area'] = 400.0
                tip_anno['segmentation'] = [[float(w_index-10), float(h_index-10)+10,
                                            float(w_index-10)+10, float(h_index-10), 
                                            float(w_index-10)+20, float(h_index-10)+10,
                                            float(w_index-10)+10, float(h_index-10)+20
                                            ]]
                tip_anno['bbox'] = [float(w_index-10), float(h_index-10), 20.0, 20.0]
                new_anno_list.append(tip_anno)
            origin_anno['annotations'] = new_anno_list
            json.dump(origin_anno, open(os.path.join(root_dir, fold, mode+'_fake_tip.json'), mode='w'))


def visu_result_image(image, det_bbox, tip_bbox, masks=None, top_n=1):
    image = cv2.imread(image)
    color = np.array([0,255,0], dtype='uint8')
    top_n_b = min(top_n, len(det_bbox))
    top_n_t = min(top_n, len(tip_bbox))
    # top_n_t = 0
    for i in range(top_n_b):
        x1, y1, x2, y2, score = det_bbox[i][:]
        if score < 0.5:
            continue
        cv2.rectangle(image, (int(x1), int(y1)), (int(x2), int(y2)), (36,255,12), 1)
        cv2.putText(image, str(round(score, 2)), (int(x1), int(y1-10)), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (36,255,12), 1)

        if masks:
            mask = masks[i]
            mask = np.where(mask[...,None], color, image)
            image = cv2.addWeighted(image, 0.8, mask, 0.2,0)
    
    for i in range(top_n_t):
        x1, y1, x2, y2, score = tip_bbox[i][:]
        cv2.rectangle(image, (int(x1), int(y1)), (int(x2), int(y2)), (255, 36,12), 1)
        cv2.putText(image, str(round(score, 2)), (int(x1), int(y1-10)), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 36,12), 1)

        if masks:
            mask = masks[i]
            mask = np.where(mask[...,None], color, image)
            image = cv2.addWeighted(image, 0.8, mask, 0.2,0)
    # cv2.imshow('result', image)
    # cv2.waitKey(0)
    return image

def visu_result_video():
    result_dir = r'/srv/fenster/people/Ningtao/Dataset/LiverNeedle/1020/Pantom/result/181821'
    anno_dir = r'/srv/fenster/people/Ningtao/Dataset/LiverNeedle/1020/Pantom'
    data_dir = r'/srv/fenster/people/Ningtao/Dataset/LiverNeedle/1020/Pantom'
    save_dir = r'/srv/fenster/people/Ningtao/Dataset/LiverNeedle/1020/Pantom/result/181821'
    # for video_info in json.load(open(os.path.join(anno_dir, 'test_fake_tip.json')))['videos']:
    for video_name in ['1', '2', '3', '4']:
        # video_name = video_info['name']
        # if video_name != "p012_021":
        #     continue
        os.makedirs(os.path.join(save_dir, video_name, 'images'), exist_ok=True)
        anno_coco = COCO(os.path.join(anno_dir, video_name+'.json'))
        result_array = np.load(os.path.join(result_dir, video_name+'.pkl'), allow_pickle=True)
        image_list = anno_coco.imgs 
        for image_id in tqdm(image_list.keys()):
            image_info = image_list[image_id]
            if isinstance(image_info['file_name'], list):
                file_name = image_info['file_name'][0]
            else:
                file_name = image_info['file_name']
            image_path = os.path.join(data_dir, video_name, file_name)
            try:
                bbox_list = result_array['det_bboxes'][image_id-1][0][0]
                tip_list = result_array['det_bboxes'][image_id-1][0][1]
                mask_list = result_array['det_masks'][image_id-1][0][0]
            except:
                continue
            try:
                image_result = visu_result_image(image_path, bbox_list, tip_list, mask_list)
            except Exception as e:
                print(e)
                continue
            anno_idxs = anno_coco.getAnnIds(image_id, catIds=2)
            for anno_id in anno_idxs:
                anno_info = anno_coco.loadAnns(anno_id)[0]
                x, y, w, h = anno_info['bbox']
                cv2.rectangle(image_result, (int(x), int(y)), (int(x+w), int(y+h)), (20,20,255), 1)
                # cv2.imshow('result', image_result)
                # cv2.waitKey(0)
            cv2.imwrite(os.path.join(save_dir, video_name, 'images', file_name), image_result)
        print(video_name)  

def generate_video_singleton():
    origin_path = r'/srv/fenster/people/Ningtao/Dataset/LiverNeedle/1020/DataSet/cross_valid_0318_flow/1/test.json'
    save_dir = r'/srv/fenster/people/Ningtao/Dataset/LiverNeedle/1020/DataSet/cross_valid_0318_flow/1'
    origin_anno = json.load(open(origin_path))
    video_singleton_dict = {}
    video_idx = 0
    for image_info in origin_anno['images']:
        video_id = image_info['video_id']
        if video_id not in video_singleton_dict:
            video_singleton_dict[video_id] = {'images': [], 'image_id_map': {}, 'image_idx': 0, 'new_id': video_idx}
        video_singleton_dict[video_id]['image_idx'] += 1
        origin_id = image_info['id']
        image_info['id'] = video_singleton_dict[video_id]['image_idx']
        video_singleton_dict[video_id]['image_id_map'][origin_id] = video_singleton_dict[video_id]['image_idx']
        image_info['video_id'] = 1
        video_singleton_dict[video_id]['images'].append(image_info)
    for anno_info in origin_anno['annotations']:
        origin_video_id = anno_info['video_id']
        origin_image_id = anno_info['image_id']
        origin_instance_id = anno_info['instance_id']
        if 'anno_idx' not in video_singleton_dict[origin_video_id]:
            video_singleton_dict[origin_video_id]['anno_idx'] = 0
            video_singleton_dict[origin_video_id]['annos'] = []
            video_singleton_dict[origin_video_id]['instance'] = []
        if origin_instance_id not in video_singleton_dict[origin_video_id]['instance']:
            video_singleton_dict[origin_video_id]['instance'].append(origin_instance_id)
        video_singleton_dict[origin_video_id]['anno_idx'] += 1
        anno_info['id'] = video_singleton_dict[origin_video_id]['anno_idx']
        anno_info['image_id'] = video_singleton_dict[origin_video_id]['image_id_map'][origin_image_id]
        anno_info['video_id'] = 1
        video_singleton_dict[origin_video_id]['annos'].append(anno_info)
    
    def instance_map(anno_info, map_dict):
        anno_info['instance_id'] = map_dict[anno_info['instance_id']]+1
        return anno_info
 
    for video_info in origin_anno['videos']:
        video_id = video_info['id']
        video_name = video_info['name']
        try:
            instance_list = list(set(video_singleton_dict[video_id]['instance']))
        except:
            continue
        instance_list.sort()
        instance_map_dict = dict(zip(instance_list, list(range(len(instance_list)))))
        anno = list(map(lambda x: instance_map(x, instance_map_dict), video_singleton_dict[video_id]['annos']))
        new_anno_json = {
            'videos': [dict(id=1, name=video_name)],
            'categories': origin_anno['categories'],
            'images': video_singleton_dict[video_id]['images'],
            'annotations': anno
        }
        json.dump(new_anno_json, open(os.path.join(save_dir, 'test_'+video_name+'.json'), mode='w'))

def move_flow():
    """
    move optical flow image to origin data folder
    """
    data_dir = r'/srv/fenster/people/Ningtao/Dataset/LiverNeedle/1020/Data'
    flow_dir = r'/srv/fenster/people/Ningtao/Dataset/LiverNeedle/1020/OpticalFlow'
    for video_name in os.listdir(data_dir):
        last_flow = None
        video_path = os.path.join(data_dir, video_name, 'images')
        frame_list = os.listdir(video_path)
        frame_list.sort()
        for file_name in frame_list:
            if os.path.exists(os.path.join(flow_dir, video_name, file_name.replace('.bmp', '.png'))):
                shutil.copy(os.path.join(flow_dir, video_name, file_name.replace('.bmp', '.png')),
                            os.path.join(video_path, file_name.replace('.bmp', '_flow.png')))
                last_flow = file_name.replace('.bmp', '_flow.png')
            else:
                shutil.copy(os.path.join(video_path, last_flow),
                            os.path.join(video_path, file_name.replace('.bmp', '_flow.png')))

def cnvert_volum_to_sequence():
    data_path = r'/srv/fenster/people/Ningtao/Dataset/LiverNeedle/1020/Pantom/Dicom/IM_0004'
    sav_dir = r'/srv/fenster/people/Ningtao/Dataset/LiverNeedle/1020/Pantom/4'
    image_data = pydicom.dcmread(data_path)
    img_array = image_data.pixel_array
    for i in range(img_array.shape[0]):
        frame_array = img_array[i,120:750, ...]
        Image.fromarray(frame_array).convert('L').save(os.path.join(sav_dir, f'{format(i+1, "06d")}.bmp'))

def get_tip(mask, bbox, reg):
    points = np.where(mask)
    x = points[1]
    y = points[0]
    reg = fit_line(reg, x, y)
    if reg.coef_[0] < 0:
        tip_point = (bbox[0], bbox[1]+bbox[3])
    else:
        tip_point = (bbox[0]+bbox[2], bbox[1]+bbox[3])
    return tip_point

def tip_move_speed():
    anno_dir = '/srv/fenster/people/Ningtao/Dataset/LiverNeedle/1020/DataSet/coco_valid_0318_no_occu/1'  
    save_dir = r'/srv/fenster/people/Ningtao/Project/USVideo/tracking/mmtracking/custom/metric_doc/liver_needle'
    line_reg = LinearRegression()
    speed_dict = {}
    for video_name in ['p020_03', 'p008_v2', 'p007_v2', 'p004_v6', 'p007_v1', 'p004_v1', 'p012_021', 'p012_04', 'p020_07', 'p020_08', 'p008_v1', 'p003_v7']:
        resolution = resolution_dict[video_name]['x_voxel']
        coco = COCO(os.path.join(anno_dir, f'test_{video_name}.json'))
        img_ids = coco.imgs.keys()
        previous_tip = (0, 0)
        speed_list = []
        for img_id in img_ids:
            anno_id = coco.getAnnIds(imgIds=img_id, catIds=1)
            anno_info = coco.loadAnns(ids=anno_id)
            if len(anno_info) == 0:
                print(img_id)
                continue
            anno_info = anno_info[0]
            mask_gt = coco.annToMask(anno_info)
            bbox_gt = anno_info['bbox']
            gt_tip = get_tip(mask_gt, bbox_gt, line_reg)
            speed = math.sqrt((gt_tip[0]-previous_tip[0])**2+(gt_tip[1]-previous_tip[1])**2)
            speed_list.append(speed*resolution)
            previous_tip = gt_tip
        speed_dict[video_name] = speed_list
    pd.DataFrame(dict([(key, pd.Series(value)) for key, value in speed_dict.items()])).to_csv(os.path.join(save_dir, 'tip_speed.csv'))

def optical_flow_speed():
    anno_dir = r'/srv/fenster/people/Ningtao/Dataset/LiverNeedle/1020/DataSet/coco_valid_0318_no_occu'
    data_dir = r'/srv/fenster/people/Ningtao/Dataset/LiverNeedle/1020/Data'
    save_dir = r'/srv/fenster/people/Ningtao/Dataset/LiverNeedle/1020/DataSet/cross_valid_0318_flow'
    for fold in range(1, 6):
        fold = str(fold)
        for dataset in ['train.json', 'test.json']:
            anno_path = os.path.join(anno_dir, fold, dataset)
            anno_dict = copy.deepcopy(json.load(open(anno_path)))
            for image_info in tqdm(anno_dict['images']):
                image_pth = os.path.join(data_dir, image_info['file_name'].replace('.bmp', '_flow.png'))
                image_array = np.array(Image.open(image_pth).convert('L'))
                mean_gray = np.mean(image_array)/255
                image_info['flow_gray'] = mean_gray
            json.dump(anno_dict, open(os.path.join(save_dir, fold, dataset), mode='w'))


def pick_top():
    anno_path = r'/srv/fenster/people/Ningtao/Dataset/LiverNeedle/1020/DataSet/coco_valid_0318_no_occu/1/test.json'
    per_path = r'/srv/fenster/people/Ningtao/Project/USVideo/tracking/mmtracking/custom/metric_doc/liver_needle/190340_per_list.xlsx'
    result_dir = r'/srv/fenster/people/Ningtao/Project/USVideo/tracking/mmtracking/work_dirs/mask_selsa_faster_rcnn_r50_dc5_1x_c04_needle1020_ignore_occu_iou_aggre/result/190340'
    video_list = json.load(open(anno_path))['videos']
    for video_info in video_list:
        v_name = video_info['name']
        image_dir = os.path.join(result_dir, v_name, 'images')
        os.makedirs(os.path.join(result_dir, v_name, 'top'), exist_ok=True)
        per_df = pd.read_excel(per_path, sheet_name=v_name)
        per_df = per_df.sort_values('single_best_dis_inter', ascending=False)
        per_df = per_df.head(min(50, len(per_df)))
        for idx, row in per_df.iterrows():
            try:
                image_idx = int(row['index'].split('_')[-1])
            except:
                continue
            image_name = format(image_idx, '06d')+'.bmp'
            shutil.copy(os.path.join(image_dir, image_name), os.path.join(result_dir, v_name, 'top', image_name))
            # res_frame['index'] = video_name+'_'+str(image_id)

def frame_type_process():
    origin_path = r'/srv/fenster/people/Ningtao/Dataset/LiverNeedle/1020/DataSet/tip_type_demo.xlsx'
    xlsx = pd.ExcelFile(origin_path)
    writer = pd.ExcelWriter(origin_path.replace('tip_type_demo.', 'tip_type.'))  
    for s_name in xlsx.sheet_names[1: ]:
        df = pd.read_excel(xlsx, sheet_name=s_name, header=None)
        dict = df.to_dict(orient='list')
        new_dict = {'NAME': dict[0], 'TYPE': dict[1]}
        pd.DataFrame(new_dict).to_excel(writer, sheet_name=s_name.lower(), index=False)
    writer.close()


    
def needle_contras():
    anno_path = r''
    save_path = ''
    data_dir = r''
    def gray_block(image_array: np.ndarray, mask_array: np.ndarray):
        H, W = mask_array.shape[-2:]
        h_end = np.max(np.where(mask_array)[0])
        gray_list_all = []
        needle_index_list = []
        for i in range(h_end):
            mask_line = mask_array[i]
            mask_index = np.where(mask_line)
            needle_begin, needle_end = np.min(mask_index), np.max(mask_index)+1
            needle_width = needle_end - needle_begin
            begin = needle_begin % needle_width
            gray_list = []
            needle_block_index = -1
            for idx, j in enumerate(range(begin, W, needle_width)):
                blcok_gray = np.mean(image_array[i, j: j+needle_width])
                gray_list.append(blcok_gray)
                if j == needle_begin:
                    needle_block_index = idx
            gray_list_all.append(gray_list)
            needle_index_list = needle_index_list.append(needle_block_index)
        return gray_list_all, needle_index_list

    anno_coco = COCO(anno_path)
    image_ids = anno_coco.imgs
    for image_id in image_ids:
        image_info  = anno_coco.loadImgs(image_id)[0]
        image_array = np.array(Image.open(os.path.join(data_dir, image_info['file_name'])))
        anno_ids = anno_coco.getAnnIds(imgIds=image_id, catIds=1)[0]
        mask_array = anno_coco.loadAnns(anno_ids)[0]['segmentation']
        mask_array = m_decode(mask_array)
        gray_list, needle_index = gray_block(image_array, mask_array)
        pass

def merge_per_list():
    source_dict = {
        '-1_0': ('004507/bbox_per_list_{}_ext.xlsx', 'index'),
        '-1_1': ('004507/bbox_per_list_{}_ext.xlsx', 'index'),
        '1_0': ('095116/bbox_per_list_{}_ext.xlsx', 'union_highest_dis_seg_MM'),
        '2_0': ('004507/bbox_per_list_{}_ext.xlsx', 'vote_all_highest_dis_inter_MM'),
        '3_0': ('004507/bbox_per_list_{}_ext.xlsx', 'vote_2_highest_dis_bbox_MM'),
        '4_0': ('004507/bbox_per_list_{}_ext.xlsx', 'vote_all_highest_dis_inter_MM'),
        '1_1': ('004507/segm_per_list_{}_ext.xlsx', 'vote_2_highest_error_angle'),
        '2_1': ('004507/segm_per_list_{}_ext.xlsx', 'vote_2_highest_error_angle'),
        '3_1': ('095656/bbos_segm_per_list_{}_ext.xlsx', 'vote_all_avg_error_angle'),
        '4_1': ('004507/bbox_per_list_{}_ext.xlsx', 'l ')
    }
    per_dir = r'/srv/fenster/people/Ningtao/Project/USVideo/tracking/mmtracking/custom/metric_doc/liver_needle'
    series_dict ={}
    for key in source_dict.keys():
        series_dict[key] = pd.ExcelFile(os.path.join(per_dir, source_dict[key][0].format('-1_-1')))
    anno_dir = r'/srv/fenster/people/Ningtao/Dataset/LiverNeedle/1020/DataSet/coco_valid_0318_no_occu/1/test.json'

    xlsx_writer = pd.ExcelWriter(os.path.join(per_dir, 'summary.xlsx'))

    v_names = [v['name'] for v in json.load(open(anno_dir))['videos']]
    v_lens = [(max(pd.read_excel(os.path.join(per_dir, source_dict['-1_0'][0].format('-1_0')), v_name)['single_best_dis_bbox'].count()-2, 0),
                max(pd.read_excel(os.path.join(per_dir, source_dict['-1_1'][0].format('-1_1')), v_name)['single_best_dis_bbox'].count()-2, 0)) for v_name in v_names]
    len_dict = dict(zip(v_names, v_lens))
    value_dict = {}
    for v_name in v_names:
        value_dict[v_name] = [[], []]
        type_list = pd.read_excel(frame_type_xlsx, v_name).to_dict(orient='list')['TYPE']
        for phase in ['0', '1']:
            phase_list = [0 for i in type_list]
            if ablation_phase_origin_dict[v_name] != (-1, -1):
                begin, end = ablation_phase_origin_dict[v_name]
                for j in range(begin, end+1):
                    phase_list[j] = 1
            for visu in ['1', '2', '3', '4']:
                abla_mask = np.array(phase_list) == int(phase)
                visu_mask = np.array(type_list) == int(visu)
                mask_filter = visu_mask & abla_mask
                col_name = source_dict[f'{visu}_{phase}'][1]            
                series = pd.read_excel(series_dict[f'{visu}_{phase}'], v_name)[col_name.replace('_MM', '')].tolist()
                len_min = min(len(series), len(mask_filter))
                values = [series[k] for k in range(len_min) if mask_filter[k]]
                if '_MM' in col_name:
                    values = [v*resolution_dict[v_name]['x_voxel'] for v in values]
                value_dict[v_name][int(phase)].extend(values)
        # assert abs(len(value_dict[v_name][0]) - len_dict[v_name][0]) < 5 and \
        #       abs(len(value_dict[v_name][1]) - len_dict[v_name][1]) < 5, \
        #         f'{v_name}: {str(len(value_dict[v_name][0]))}_{str(len_dict[v_name][0])} {str(len(value_dict[v_name][1]))}_{str(len_dict[v_name][1])}'
        df = pd.DataFrame.from_dict({'MM': value_dict[v_name][0], 'ANGLE': value_dict[v_name][1]}, orient='index').T
        df.to_excel(xlsx_writer, sheet_name=v_name, index=False)

def merge_video():
    per_dir = r'/srv/fenster/people/Ningtao/Project/USVideo/tracking/mmtracking/custom/metric_doc/liver_needle'
    xlsx_file = pd.ExcelFile(os.path.join(per_dir, 'summary_shaft.xlsx'))
    summary_dict = {'VIDEO': [], 'INDEX': [], 'ANGLE': [], 'ENTER_X': [], 'ENTER_Y': [], 'TIP_X': [], 'TIP_Y': [], 'SLOPE': []}

    for sheet_name in xlsx_file.sheet_names:
        df_video = pd.read_excel(xlsx_file, sheet_name)
        for idx, row in df_video.iterrows():
            if pd.isna(row['ANGLE_ERROR']) or row['index'] in ['AVG', 'STD']:
                continue
            summary_dict['VIDEO'].append(sheet_name)
            frame_idx = row['index'].split('_')[-1]
            summary_dict['INDEX'].append(frame_idx)
            summary_dict['ANGLE'].append(row['ANGLE_ERROR'])
            try:
                enter_x, enter_y = row['ENTER'].split(' ')
            except:
                print(row)
                print(row['ENTER'])
            summary_dict['ENTER_X'].append(enter_x)
            summary_dict['ENTER_Y'].append(enter_y)
            try:
                tip_x, tip_y = row['TIP'].split(' ')
            except:
                print(row)
                print(row['TIP'])
            summary_dict['TIP_X'].append(tip_x)
            summary_dict['TIP_Y'].append(tip_y)
            summary_dict['SLOPE'].append(row['SLOPE'])
    pd.DataFrame(summary_dict).to_excel(os.path.join(per_dir, 'shaft_all.xlsx'), index=False)

def video_frame_num():
    data_dir = r'/srv/fenster/people/Ningtao/Dataset/LiverNeedle/1020/Data'
    f_num_dict = {'v_name': [], 'num_frame': []}
    for v_name in os.listdir(data_dir):
        f_list = os.listdir(os.path.join(data_dir, v_name, 'images'))
        f_list = list(filter(lambda x: x.endswith('.bmp') and 'flow' not in x, f_list))
        f_num_dict['v_name'].append(v_name)
        f_num_dict['num_frame'].append(len(f_list))
    pd.DataFrame(f_num_dict).to_csv(os.path.join(data_dir, 'video_frame_num.csv'), index=False)
        
def generate_video():
    # plt.rcParams['font.family'] = 'Arial'
    plt.rcParams['font.size'] = 16
    plt.rcParams['lines.linewidth'] = 2
    exp_name = 'all_folds_top2_1'
    per_list_path = f'/srv/fenster/people/Ningtao/Project/USVideo/tracking/mmtracking/custom/metric_doc/liver_needle/{exp_name}/bbox_per_list_-1_-1_ext.xlsx'
    frame_dir = f'/srv/fenster/people/Ningtao/Project/USVideo/tracking/mmtracking/work_dirs/{exp_name}/result/visu'

    per_cols = ['single_mean_dis_bbox', 'single_mean_dis_seg', 'single_mean_dis_inter',  
                'union_highest_dis_bbox', 'union_highest_dis_seg', 'union_highest_dis_inter', 
                'union_union_dis_bbox', 'union_union_dis_seg', 'union_union_dis_inter', 
                'union_biggest_dis_bbox', 'union_biggest_dis_seg', 'union_biggest_dis_inter', 
                'union_avg_dis_bbox', 'union_avg_dis_seg', 'union_avg_dis_inter', 
                'vote_2_highest_dis_bbox', 'vote_2_highest_dis_seg', 'vote_2_highest_dis_inter', 
                'vote_2_union_dis_bbox', 'vote_2_union_dis_seg', 'vote_2_union_dis_inter', 
                'vote_2_biggest_dis_bbox', 'vote_2_biggest_dis_seg', 'vote_2_biggest_dis_inter', 
                'vote_2_avg_dis_bbox', 'vote_2_avg_dis_seg', 'vote_2_avg_dis_inter', 
                'vote_all_highest_dis_bbox', 'vote_all_highest_dis_seg', 'vote_all_highest_dis_inter', 
                'vote_all_union_dis_bbox', 'vote_all_union_dis_seg', 'vote_all_union_dis_inter', 
                'vote_all_biggest_dis_bbox', 'vote_all_biggest_dis_seg', 'vote_all_biggest_dis_inter', 
                'vote_all_avg_dis_bbox', 'vote_all_avg_dis_seg', 'vote_all_avg_dis_inter']

    for v_name in ['p020_02']:
        print(v_name)
        frame_lim = 150_400
        resolution = resolution_dict[v_name]['x_voxel'] if v_name in resolution_dict.keys() else resolution_dict['default']['x_voxel']
        per_list = pd.read_excel(per_list_path, sheet_name=v_name)
        per_list = per_list.iloc[:-2]
        per_tip_list = []
        per_angle_list = []
        tip_post_list = []
        slope_list = []
        for idx, row in per_list.iterrows():
            per_tip = sys.maxsize
            best_col = ''   
            for col in per_cols:
                if row[col] < per_tip:
                    per_tip = row[col]
                    best_col = col
            per_angle = float(row[best_col.replace('dis_seg', '').replace('dis_bbox', '').replace('dis_inter', '')+'error_angle'])
            per_tip_list.append(per_tip)
            per_angle_list.append(per_angle)

            tip_x, tip_y = row['TIP'].split(' ')
            slope = row['SLOPE']
            tip_post_list.append((float(tip_x), float(tip_y)))
            slope_list.append(slope)
        per_tip_list = list(map(lambda x: x*resolution, per_tip_list))
        slope_list = list(map(lambda x: math.atan(x)*180/math.pi, slope_list))

        tip_speed_list = [math.sqrt((tip_post_list[i+1][0] - tip_post_list[i][0])**2 + (tip_post_list[i+1][1] - tip_post_list[i][1])**2) * resolution for i in range(len(tip_post_list)-1)]
        tip_speed_list.insert(0, tip_speed_list[0])
        
        angle_speed_list = [abs(slope_list[i+1] - slope_list[i]) for i in range(len(slope_list)-1)]
        angle_speed_list.insert(0, angle_speed_list[0])

        angle_speed_list = angle_speed_list[:150] + angle_speed_list[400:] 
        tip_speed_list = tip_speed_list[:150] + tip_speed_list[400:]
        per_angle_list = per_angle_list[:150] + per_angle_list[400:]
        per_tip_list = per_tip_list[:150] + per_tip_list[400:]
        # 读取第一帧以获取帧的尺寸
        frame_list = os.listdir(os.path.join(frame_dir, v_name, 'images'))
        frame_list = frame_list[:150] + frame_list[400:]
        total_frames = len(frame_list)
        first_frame = cv2.imread(os.path.join(frame_dir, v_name, 'images', frame_list[0]))
        height, width, _ = first_frame.shape

        # 创建视频写入器
        fourcc = cv2.VideoWriter_fourcc(*'mp4v')  # mp4编码格式
        out = cv2.VideoWriter(os.path.join(frame_dir, v_name+f'_{str(frame_lim)}.mp4'), fourcc, 15, (width, height))

        # 创建两个空白图像用于左下角和右下角显示误差曲线
        fig1, ax1 = plt.subplots()
        ax1.set_xlim(0, total_frames)
        ax1.set_xlabel('Frame')

        # 初始化图1：针柄角速度和角度误差
        ax1.set_ylim(0, np.max(angle_speed_list) * 1.2)
        ax1.set_ylabel('Shaft angle speed (degrees/f)', color='blue')
        line_angle_speed, = ax1.plot([], [], label='Shaft angle speed (degrees/f)', color='blue')
        ax1.tick_params(axis='y', labelcolor='blue')
        # ax1.legend(loc='upper left')

        ax1_2 = ax1.twinx()  # 创建第二个y轴
        # ax1_2.set_ylim(0, np.max(per_angle_list) * 1.2)
        ax1_2.set_ylim(0, 8)
        ax1_2.set_ylabel('Shaft direction error (degree)', color='red')
        line_angle_error, = ax1_2.plot([], [], label='Shaft direction error (degree)', color='red')
        ax1_2.tick_params(axis='y', labelcolor='red')
        # ax1_2.legend(loc='upper right')
        fig1.tight_layout()

        # 初始化图2：针尖运动速度和针尖距离误差
        fig2, ax2 = plt.subplots()
        ax2.set_xlim(0, total_frames)
        ax2.set_xlabel('Frame')

        ax2.set_ylim(0, np.max(tip_speed_list) * 1.2)
        ax2.set_ylabel('Tip motion speed (mm/f)', color='green')
        line_tip_speed, = ax2.plot([], [], label='Tip motion speed (mm/f)', color='green')
        ax2.tick_params(axis='y', labelcolor='green')
        # ax2.legend(loc='upper left')

        ax2_2 = ax2.twinx()  # 创建第二个y轴
        # ax2_2.set_ylim(0, np.max(per_tip_list) * 1.2)
        ax2_2.set_ylim(0, 30)
        ax2_2.set_ylabel('Tip distance error (mm)', color='purple')
        line_distance_error, = ax2_2.plot([], [], label='Tip distance error (mm)', color='purple')
        ax2_2.tick_params(axis='y', labelcolor='purple')
        # ax2_2.legend(loc='upper right')
        fig2.tight_layout()
    
        # 处理每一帧图像并叠加检测结果和误差图
        for frame_idx, image_file in enumerate(frame_list):
            
            frame_path = os.path.join(frame_dir, v_name, 'images', image_file)
            
            # 读取当前帧图像
            frame = cv2.imread(frame_path)     
            
            # 更新图1
            x_data = np.arange(0, frame_idx + 1)
            line_angle_speed.set_data(x_data, angle_speed_list[:frame_idx + 1])
            line_angle_error.set_data(x_data, per_angle_list[:frame_idx + 1])
            fig1.canvas.draw()
            
            # 更新图2
            line_tip_speed.set_data(x_data, tip_speed_list[:frame_idx + 1])
            line_distance_error.set_data(x_data, per_tip_list[:frame_idx + 1])
            fig2.canvas.draw()

            # 将图1和图2保存为图像
            graph1_img = np.frombuffer(fig1.canvas.tostring_rgb(), dtype=np.uint8)
            graph1_img = graph1_img.reshape(fig1.canvas.get_width_height()[::-1] + (3,))
            graph2_img = np.frombuffer(fig2.canvas.tostring_rgb(), dtype=np.uint8)
            graph2_img = graph2_img.reshape(fig2.canvas.get_width_height()[::-1] + (3,))
            
            # 缩放图像以适合左下角和右下角
            graph1_img = cv2.resize(graph1_img, (round(width / 2.1), height // 3))
            graph2_img = cv2.resize(graph2_img, (round(width / 2.1), height // 3))
            
            # 叠加图1到左下角
            frame[-graph1_img.shape[0]:, 10 :graph1_img.shape[1]+10] = graph1_img
            
            # 叠加图2到右下角
            frame[-graph2_img.shape[0]:, -graph2_img.shape[1]-10: -10] = graph2_img
            
            # 写入修改后的帧到视频
            out.write(frame)

        # 释放资源
        out.release()
        plt.close()

def frame_to_video(fps=15):
    frame_dir = r'/srv/fenster/people/Ningtao/Dataset/LiverNeedle/1020/bbox_tip/p012_021'
    for video_name in ['images']:
        # video_name = 'p008_v4_6'
        frame_list = sorted(os.listdir(os.path.join(frame_dir, video_name,)))[:300]
        frame_list = [f for f in frame_list if f.endswith('.png')]
        first_frame = cv2.imread(os.path.join(frame_dir, video_name, frame_list[0]))  # 获取第一帧图像的宽度和高度
        height, width, _ = first_frame.shape
        fourcc = cv2.VideoWriter_fourcc(*'mp4v')  # 定义编码器
        out = cv2.VideoWriter(os.path.join(frame_dir, video_name, video_name + '.mp4'), fourcc, fps, (width, height)) 
        
        for frame in frame_list:
            img = cv2.imread(os.path.join(frame_dir, video_name, frame))
            out.write(img)  
            print(frame)
        out.release()
        print('done')

def make_bbox_and_tip():
    free_degree = 10
    anno_path =  r'/srv/fenster/people/Ningtao/Dataset/LiverNeedle/1020/DataSet/coco_valid_0318_no_occu/1/test.json'
    data_dir = r'/srv/fenster/people/Ningtao/Dataset/LiverNeedle/1020/Data'
    save_dir = r'/srv/fenster/people/Ningtao/Dataset/LiverNeedle/1020/bbox_tip'
    coco = CocoVID(anno_path)
    vids = coco.get_vid_ids()
    for vid in vids:
        v_name = coco.load_vids([vid])[0]['name']
        if v_name not in ['p012_021']:
            continue
        img_ids = coco.get_img_ids_from_vid(vidId=vid)
        anno_last = { "bbox": [
                432.52,
                110.6,
                176.71,
                144.76
            ]}
        for img_id in img_ids[:300]:
            img_info = coco.load_imgs([img_id])[0]
            img_path = os.path.join(data_dir, img_info['file_name'])
            anno_id = coco.get_ann_ids(img_ids=[img_id], cat_ids=[1])
            if len(anno_id) == 0:
                anno = anno_last
            else:
                anno = coco.load_anns(anno_id)[0]
                anno_last = anno
            bbox = anno['bbox']
            bbox = {'x1': bbox[0], 'x2': bbox[0]+bbox[2], 'y1': bbox[1], 'y2': bbox[1]+bbox[3], 'score': bbox[-1]}
            tip = (bbox['x1'], bbox['y2'])
            img = cv2.imread(img_path)
            for i in range(10):
                bbox_x_diff = np.random.normal(bbox['x1'], (bbox['x2'] - bbox['x1']) / free_degree)
                bbox_y_diff = np.random.normal(bbox['y1'], (bbox['y2'] - bbox['y1']) / free_degree)
                bbox_width_diff = np.random.normal((bbox['x2'] - bbox['x1']), (bbox['x2'] - bbox['x1']) / free_degree*2)
                bbox_height_diff = np.random.normal((bbox['y2'] - bbox['y1']), (bbox['y2'] - bbox['y1']) / free_degree*2)

                img = cv2.rectangle(img, (int(bbox_x_diff), int(bbox_y_diff)), (int(bbox_x_diff + bbox_width_diff), int(bbox_y_diff + bbox_height_diff)), (0, 255, 255), 1)
            for _ in range((4*3*4+1)*7*7-50):
                gap_w = (bbox['x2'] - bbox['x1']) / free_degree*0.8
                gap_h = (bbox['y2'] - bbox['y1']) / free_degree*0.8
                # tip_x_diff = np.random.uniform(int(tip[0]-gap_w), int(tip[0]+gap_w))
                # tip_y_diff = np.random.uniform(int(tip[1]-gap_h), int(tip[1]+gap_h))
                tip_x_diff = np.random.normal(tip[0], gap_w)
                tip_y_diff = np.random.normal(tip[1], gap_h)
                img = cv2.circle(img, (int(tip_x_diff), int(tip_y_diff)), 3, (0, 0, 255), -1)
            for _ in range(30):
                gap_w = (bbox['x2'] - bbox['x1']) / free_degree*2
                gap_h = (bbox['y2'] - bbox['y1']) / free_degree*2 

                # tip_x_diff = np.random.uniform(int(tip[0]-gap_w), int(tip[0]+gap_w))
                # tip_y_diff = np.random.uniform(int(tip[1]-gap_h), int(tip[1]+gap_h))
                tip_x_diff = np.random.normal(tip[0], gap_w)
                tip_y_diff = np.random.normal(tip[1], gap_h)
                img = cv2.circle(img, (int(tip_x_diff), int(tip_y_diff)), 3, (0, 0, 255), -1)
            for _ in range(20):
                gap_w = (bbox['x2'] - bbox['x1']) / free_degree*5
                gap_h = (bbox['y2'] - bbox['y1']) / free_degree*5 

                # tip_x_diff = np.random.uniform(int(tip[0]-gap_w), int(tip[0]+gap_w))
                # tip_y_diff = np.random.uniform(int(tip[1]-gap_h), int(tip[1]+gap_h))
                tip_x_diff = np.random.normal(tip[0], gap_w)
                tip_y_diff = np.random.normal(tip[1], gap_h)
                img = cv2.circle(img, (int(tip_x_diff), int(tip_y_diff)), 3, (0, 0, 255), -1)
            cv2.imwrite(os.path.join(save_dir, img_info['file_name'].replace('.bmp', '.png')), img) 

def downsample_video():
    # 1/2 sample the frame in video, and reindex the image_id
    josn_file = r'/srv/fenster/people/Ningtao/Dataset/LiverNeedle/1020/DataSet/coco_valid_0318_no_occu/1/test.json'
    save_path = r'/srv/fenster/people/Ningtao/Dataset/LiverNeedle/1020/DataSet/coco_valid_0318_no_occu/1/test_half.json'    
    coco = CocoVID(josn_file)
    global_image_idx = 1
    image_list = []
    global_anno_idx = 1
    anno_list = []
    global_frame_idx = 0

    img_ids = coco.get_img_ids()
    img_ids.sort()
    img_ids = img_ids[::2]

    v_id = 1
    for img_id in img_ids:
        img_info = coco.load_imgs([img_id])[0]
        v_id_tmp = img_info['video_id']
        if v_id_tmp != v_id:
            v_id = v_id_tmp
            global_frame_idx = 0
        img_info['id'] = global_image_idx
        img_info['frame_id'] = global_frame_idx
        image_list.append(img_info)
        anno_id = coco.get_ann_ids(img_ids=[img_id])
        for anno_idx in anno_id:
            anno = coco.load_anns([anno_idx])[0]
            anno['id'] = global_anno_idx
            anno['image_id'] = global_image_idx
            anno_list.append(anno)
            global_anno_idx += 1
        global_image_idx += 1
        global_frame_idx += 1
    cat_ids = coco.get_cat_ids()
    cat_list = [
        {
            'supercategory': coco.cats[cat_id]['supercategory'],
            'id': cat_id,
            'name': coco.cats[cat_id]['name']
        } for cat_id in cat_ids
    ]
    v_ids = coco.get_vid_ids()
    video_list = [
        {
            'id': v_id,
            'name': coco.videos[v_id]['name'],
        } for v_id in v_ids
    ]
    new_dict = {
        'categories': cat_list,
        'images': image_list,
        'annotations': anno_list,
        'videos': video_list
    }
    with open(save_path, 'w') as f:
        json.dump(new_dict, f)

if __name__ == '__main__':
    # check_poly_len()
    # get_ssim()
    # detec_edge()
    # generate_edge()
    # gen_edge_anno()
    # difference_adj_frame()
    # generate_mask()
    # seg_dataset()
    # split_quality_dataset()
    # init_dataset()
    # split_dataset_random()
    # get_tip_region()
    # visu_result_video()
    # ablation_type_dataset()
    # generate_video_singleton()
    # move_flow()
    # cnvert_volum_to_sequence()
    # tip_move_speed()
    # init_inference_video()
    # optical_flow_speed()
    # frame_type_process()
    # merge_per_list()
    # merge_video()
    # video_frame_num()
    # generate_video()
    # make_bbox_and_tip()
    # frame_to_video()
    # groups = divide_into_groups(patient_video_num_dict)
    # for i, group in enumerate(groups):
    #     print(f"Group {i + 1}: {group}, Cases: {sum(patient_video_num_dict[patient] for patient in group)}")
    # pass
    downsample_video()
    
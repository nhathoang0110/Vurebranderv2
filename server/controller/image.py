import datetime
import logging
import os
import time
from pathlib import Path
from typing import List
from urllib.request import Request, urlopen
from PIL import Image

import cv2
import numpy as np
import requests
from remove_watermark import remove_watermark_func

from server.config import UPLOAD_FOLDER, RESULT_FOLDER, get_config
from server.service import google_cloud
from server.utils import util_url
from utils import padding_img, rebrander_img, detect_car_plate

config = get_config()

if not os.path.exists(UPLOAD_FOLDER):
    path = Path(UPLOAD_FOLDER)
    path.mkdir(parents=True)

if not os.path.exists(RESULT_FOLDER):
    path = Path(RESULT_FOLDER)
    path.mkdir(parents=True)


def re_brander(outputs: list,bg_remove=False):
    success_image = []

    print(len(outputs))
    try:
        index_min = min(range(len(outputs)), key=lambda i: outputs[i][-1])
        file_name_min = outputs[index_min][2]
    except:
        return [],0

    def process(idx):
        # img_main = rebrander_img(type=1 if idx == index_min else 0,
        img_main = rebrander_img(type=1,
                                 image_crop=outputs[idx][1],
                                 box_plate_final=outputs[idx][0],
                                 logo_path=config.logo_path,
                                 logo_replacement_path=config.logo_replacement_path,
                                 rmbg=bg_remove)
        result_path = os.path.join(RESULT_FOLDER, outputs[idx][2])
        cv2.imwrite(result_path, img_main)
        success_image.append(result_path)

    for idx in range(len(outputs)):
        process(idx)

    return success_image, file_name_min

def re_brander_interior(outputs: list):
    success_image = []

    def process(idx):
        image_rgb=outputs[idx][1].copy()
        image_pil = Image.fromarray(image_rgb)
        img_main= padding_img(image_pil,"images/logo.webp")
        img_main = np.array(img_main)
        img_main = cv2.cvtColor(img_main, cv2.COLOR_BGR2RGB)
        result_path = os.path.join(RESULT_FOLDER, outputs[idx][2])
        cv2.imwrite(result_path, img_main)
        success_image.append(result_path)

    for idx in range(len(outputs)):
        process(idx)

    return success_image

def read_image_for_watermark(outputs: list):
    success_image = []

    def process(idx):
        image_rgb=outputs[idx][1].copy()
        # image_pil = Image.fromarray(image_rgb)
        # img_main= padding_img(image_pil,"images/logo.webp")
        # img_main = np.array(img_main)
        # img_main = cv2.cvtColor(img_main, cv2.COLOR_BGR2RGB)
        result_path = os.path.join(RESULT_FOLDER, outputs[idx][2])
        cv2.imwrite(result_path, image_rgb)
        success_image.append(result_path)

    for idx in range(len(outputs)):
        process(idx)

    return success_image


def read_image_url(list_image_url: List[str],watermark_remove,device_id) -> list:
    outputs = []
    for url in sorted(list_image_url):
        s = time.time()
        file_name = util_url.get_name_image_from_url(img_url=url)
        if not file_name:
            logging.error(f"cannot regex file name {file_name}")
            continue
        file_name = file_name[0]
        url = url.replace(" ", "%20")
        try:
            req = urlopen(Request(url, headers={'User-Agent': 'XYZ/3.0'}))
            arr = np.asarray(bytearray(req.read()), dtype=np.uint8)
            img = cv2.imdecode(arr, -1)
            
            if watermark_remove:
                img=remove_watermark_func(img,device_id)
            
            file_type = file_name.split('.')[-1]
            file_name = ''.join(file_name.split('.')[:-1])
            filename = f'{file_name}_{int(datetime.datetime.now().timestamp() * 1000)}.{file_type}'
            logging.info(f"time read image: {time.time() - s} === {filename} ")
            outputs.append([filename, img, url])
        except:
            continue
    return outputs


# def re_brander_preprocess_image(list_info_img: list):
#     ret = []
#     for filename, img, url in list_info_img:
#         ok, box_plate_final, image_crop, distance = detect_car_plate(config.car_detector, config.plate_detector_yolov8,
#                                                                      img, cv2_format=True)
#         if ok:
#             ret.append([box_plate_final, image_crop, filename, distance])
#     return ret

def re_brander_preprocess_image(list_info_img: list,bg_remove=False):
    ret = []
    interior=[]
    for filename, img, url in list_info_img:
        img1 = img.copy()        
        label = config.classify_model.classify(img1)
        if label==1:
            interior.append([None,img1,filename,None])
        else:
            if bg_remove:
                ok, box_plate_final, image_crop, distance = detect_car_plate(config.car_detector, config.plate_detector_yolov8,
                                                                            img, cv2_format=True, conf=0.2,extent_car=False)
            else:
                ok, box_plate_final, image_crop, distance = detect_car_plate(config.car_detector, config.plate_detector_yolov8,
                                                                            img, cv2_format=True, conf=0.2,extent_car=True)
            if ok:
                ret.append([box_plate_final, image_crop, filename, distance])
    return ret,interior



async def api_re_brander(_task_id: str, result_read_image: list, webhook: str,
                         hide_license_plate,bg_remove,watermark_remove):
    _start_time = time.time()
    if not hide_license_plate:
        list_url=[]
        interior=[]
        for filename, img, url in result_read_image:
            interior.append([0,img,filename,0])
            
        list_result= re_brander_interior(outputs=interior)
        s1 = time.time()
        logging.info(f"time handle image: {s1 - _start_time}")
        list_url = await google_cloud.upload_to_gcloud(list_images=list_result, task_id=_task_id)

    else:     
        interior=[]
        # preprocess
        output_preprocess,interior = re_brander_preprocess_image(list_info_img=result_read_image,bg_remove=bg_remove)
        # process
        s1 = time.time()
        if output_preprocess!=[]:
            list_result, filename_min = re_brander(outputs=output_preprocess,bg_remove=bg_remove)
            logging.info(f"time handle image: {s1 - _start_time}")

            # upload
            list_url = await google_cloud.upload_to_gcloud(list_images=list_result, task_id=_task_id)
            list_url = sorted(list_url, key=lambda x: filename_min in x, reverse=True)
        else:
            list_url=[]
        list_result= re_brander_interior(outputs=interior)
        list_result_interior = await google_cloud.upload_to_gcloud(list_images=list_result, task_id=_task_id)
        list_url=list_url+list_result_interior
    s2 = time.time()
    logging.info(f"time upload image to cloud: {s2 - s1}")

    string_list_url = '\n'.join(list_url)
    logging.info(f"uploaded {len(list_url)} images: \n{string_list_url}")

    # send request to webhook with task_id
    headers = {
        'x-api-key': os.getenv("X_API_KEY"),
        'Content-Type': 'application/json'
    }
    body = dict(taskId=_task_id, urls=list_url)
    response_webhook = requests.post(url="https://dev.vucar.vn"+webhook, headers=headers, json=body)
    logging.info(f"respond webhook rebrander: {response_webhook.status_code} - description: {response_webhook.text}")
    logging.info(f"total time handle background: {time.time() - _start_time} second(s)")
    





async def api_watermark(_task_id: str, result_read_image: list, webhook: str):
    _start_time = time.time()
    
    list_url=[]
    interior=[]
    for filename, img, url in result_read_image:
        interior.append([0,img,filename,0])
        
    list_result= read_image_for_watermark(outputs=interior)
    s1 = time.time()
    logging.info(f"time handle image: {s1 - _start_time}")
    list_url = await google_cloud.upload_to_gcloud(list_images=list_result, task_id=_task_id)

    s2 = time.time()
    logging.info(f"time upload image to cloud: {s2 - s1}")

    string_list_url = '\n'.join(list_url)
    logging.info(f"uploaded {len(list_url)} images: \n{string_list_url}")

    # send request to webhook with task_id
    headers = {
        'x-api-key': os.getenv("X_API_KEY"),
        'Content-Type': 'application/json'
    }
    body = dict(taskId=_task_id, urls=list_url)
    response_webhook = requests.post(url="https://dev.vucar.vn"+webhook, headers=headers, json=body)
    logging.info(f"respond webhook rebrander: {response_webhook.status_code} - description: {response_webhook.text}")
    logging.info(f"total time handle background: {time.time() - _start_time} second(s)")

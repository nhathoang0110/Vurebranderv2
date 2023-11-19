import os

from dotenv import load_dotenv
from ultralytics import YOLO
from modules.classification.tinyvit import Classifier

from modules.loader.yaml_loader import get_cfg

load_dotenv()
ROOT_PATH = __file__
ROOT_DIR = '/'.join(ROOT_PATH.split('/')[:-2])

UPLOAD_FOLDER = './data/images'
RESULT_FOLDER = './data/result'
BUCKET_NAME = 'vucar-production'

GCLOUD_FILE_CREDENTIAL = os.getenv("GG_CLOUD_CREDENTIAL")
GCLOUD_FOLDER_PROCESSED_IMAGE = 'public/rebrander'
IMG_FORMATS = ['bmp', 'dng', 'jpeg', 'jpg', 'mpo', 'png', 'tif', 'tiff', 'webp', 'pfm','JPEG','JPG','PNG','WEBP']  # include image suffixes
# VU_WEBHOOK = 'https://api.vucar.vn/webhook/rebrander'
GCLOUD_STORAGE_BASE_URL = 'https://storage.googleapis.com'


class Config:
    __instance = None

    def __init__(self):
        if Config.__instance is not None:
            raise Exception("This class is a singleton, use Config.create()")
        else:
            Config.__instance = self
        self.cfg = get_cfg("configs/config.yaml")

        self.car_detector = YOLO(self.cfg['DETECTION']['model_path'], task='detect')
        self.plate_detector_yolov8 = YOLO(self.cfg['PLATE_DETECTION']['model_path'], task='detect')
        self.classify_model = Classifier(self.cfg['CLASSIFY'])
        self.logo_path = self.cfg['IMAGE']['LOGO']
        self.logo_replacement_path = self.cfg['IMAGE']['LOGO_REPLACEMENT']
        # logger.info("loaded configs successfully")

    @staticmethod
    def create():
        if Config.__instance is None:
            Config.__instance = Config()
        return Config.__instance


def get_config():
    return Config.create()

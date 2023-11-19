import gradio as gr
import cv2
from zipfile import ZipFile
import os
from ultralytics import YOLO
from modules.loader.yaml_loader import get_cfg
from modules.classification.tinyvit import Classifier
from utils import detect_car_plate, padding_img, rebrander_img
import shutil
from PIL import Image
import numpy as np


cfg = get_cfg("configs/config.yaml")
car_detector = YOLO(cfg['DETECTION']['model_path'],task='detect')
plate_detector_yolov8=YOLO(cfg['PLATE_DETECTION']['model_path'],task='detect')
classify_model = Classifier(cfg['CLASSIFY'])

logo_path=cfg['IMAGE']['LOGO']
logo_replacement_path=cfg['IMAGE']['LOGO_REPLACEMENT']


def rebrander_image(img,car_detector,plate_detector,logo_path,logo_replacement_path, conf=0.2):
    
    label = classify_model.classify(img)
    if label==1:
        # image_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        # Create a PIL Image from the RGB image
        image_rgb=img.copy()
        image_pil = Image.fromarray(image_rgb)
        new_image= padding_img(image_pil,"images/logo.webp")
        new_image = np.array(new_image)
        new_image = cv2.cvtColor(new_image, cv2.COLOR_BGR2RGB)
        return new_image
    
    ok,box_plate_final, image_crop, distance = detect_car_plate(car_detector,plate_detector,img, cv2_format=True, conf=conf)
    if ok:
        img_final=rebrander_img(1,image_crop,box_plate_final,logo_path,logo_replacement_path)
        return img_final
    else:
        return img
    


def process_images(images,conf):
    processed_images = []
    for image in images:
        img = cv2.imread(image.name)
        if img is None:
            continue
        image_result= rebrander_image(img,car_detector,plate_detector_yolov8,logo_path,logo_replacement_path,conf)
        processed_images.append(image_result)
    return processed_images


# def app(images):
#     processed_images = process_images(images)
#     output_images = []
#     for i, img in enumerate(processed_images):
#         temp_filename = f"outputs_gradio/output_image_{i}.jpg"
#         cv2.imwrite(temp_filename, img)
#         output_images.append(temp_filename)
#     return output_images

# iface = gr.Interface(fn=app, inputs=gr.inputs.File(file_count="multiple", label="Upload Images"), outputs="files",capture_session=True,title="Rebrander",)

# iface.launch()

def app(images, output_folder, conf):
    conf=float(conf)
    processed_images = process_images(images,conf)
    output_images = []
    shutil.rmtree("outputs_gradio")
    os.makedirs("outputs_gradio")
    for i, img in enumerate(processed_images):
        temp_filename = f"outputs_gradio/output_image_{i}.jpg"
        cv2.imwrite(temp_filename, img)
        output_images.append(temp_filename)

    # zip_filename = "processed_images.zip"
    zip_filename = f"{output_folder}.zip"
    with ZipFile(zip_filename, 'w') as zipf:
        for img_filename in os.listdir('outputs_gradio'):
            img_path = os.path.join('outputs_gradio', img_filename)
            zipf.write(img_path, img_filename)
    
    show_images=[cv2.cvtColor(img, cv2.COLOR_RGB2BGR) for img in processed_images]

    
    return show_images,output_images, zip_filename


if __name__ == "__main__":

    iface = gr.Interface(
        fn=app,
        # inputs=gr.inputs.File(file_count="multiple", label="Upload Images"),
        inputs=[
            gr.File(file_count="multiple", label="Upload Images"),
            gr.Textbox(label="Output Folder Name", value="processed_images"),
            gr.Textbox(label="Score", value=0.3)
        ],
        outputs=[
            gr.Gallery(
            label="Show Images", show_label=False, elem_id="gallery", columns=[2], rows=[2], object_fit="contain", height="auto"),
            gr.Files(label="Processed Images"),
            gr.File(label="Download All")
        ],
        title="Rebrander"
    )
    iface.queue()
    iface.launch(server_name='0.0.0.0',auth=("ma", "123"))


    # for img_path in os.listdir("outputs_gradio"):
    #     img= cv2.imread(os.path.join("outputs_gradio",img_path))
    #     image_result= rebrander_image(img,car_detector,plate_detector_yolov8,logo_path,logo_replacement_path)
    #     cv2.imwrite(os.path.join("outputs",img_path),image_result)






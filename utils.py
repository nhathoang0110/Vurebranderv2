import cv2
import numpy as np
from PIL import Image
from rembg import remove


def padding_img(img, logo_path):
    w, h = img.size
    # new_h = int(1.5 * h)
    # new_w = int(new_h * 3 / 2)
    # if new_w < w:
    #     new_w = int(1.1 * w)
    #     new_h = int(new_w * 2 / 3)

    # pad_top = int((new_h - h) / 2)
    # pad_bottom = new_h - h - pad_top
    # pad_left = int((new_w - w) / 2)
    # pad_right = new_w - w - pad_left

    # # Convert the PIL image to a numpy array and add padding using numpy.pad
    # img_array = np.array(img)
    # img_array_with_padding = np.pad(img_array, ((pad_top, pad_bottom), (pad_left, pad_right), (0, 0)), mode='constant',
    #                                 constant_values=255)
    # img_with_padding = Image.fromarray(img_array_with_padding)
    
    # img_with_padding = img_with_padding.convert('RGB')
    # R, G, B = img_with_padding.split()
    # img_with_padding = Image.merge("RGB", (B, G, R))
    
    img_with_padding=img.copy()
    new_w,new_h=img_with_padding.size
    img_with_padding=img_with_padding.convert('RGB')
    R, G, B = img_with_padding.split()
    img_with_padding = Image.merge("RGB", (B, G, R))
    
    logoIm = Image.open(logo_path)
    basewidth=int(w/5)
    wpercent = (basewidth/float(logoIm.size[0]))
    hsize = int((float(logoIm.size[1])*float(wpercent)))
    logoIm = logoIm.resize((basewidth,hsize), Image.Resampling.LANCZOS)

    # logoIm = Image.open(logo_path)
    # logoIm = logoIm.resize((int(logoIm.size[0] * pad_bottom / (2 * logoIm.size[1])), int(pad_bottom / 2)),
                        #    Image.Resampling.LANCZOS)
    logoWidth, logoHeight = logoIm.size
    img_with_padding.paste(logoIm, (new_w - logoWidth, new_h - logoHeight), logoIm)

    return img_with_padding


def car_detection(car_detector, img,extent_car=True):
    s_max = 0
    car_box = []
    results = car_detector(source=img, device='cpu', classes=[2, 5, 7], verbose=False)
    boxes = results[0].boxes
    for b in boxes:
        xyxy = b.xyxy.view(1, 4).clone().view(-1).tolist()
        s_ = abs(xyxy[0] - xyxy[2]) * abs(xyxy[1] - xyxy[3])
        if s_ > s_max:
            s_max = s_
            car_box = xyxy

    if car_box==[]:  
        return []  
    
    if extent_car:
        height, width, _ = img.shape
        xmin,ymin,xmax,ymax=car_box[0],car_box[1],car_box[2],car_box[3]
        
        bbox_width = xmax - xmin
        bbox_height = ymax - ymin

        # Tính toán tâm của bbox
        center_x = xmin + bbox_width // 2
        center_y = ymin + bbox_height // 2

        # Tính toán scale lớn nhất có thể mà không vượt qua khung hình
        max_scale_x = min(center_x / (bbox_width / 2), (width - center_x) / (bbox_width / 2), 1.5)
        
        scale_top = min((height - center_y) / (bbox_height / 2), 1.5)
        scale_bottom = min(center_y / (bbox_height / 2), 1.5)
        scale_y_top = scale_top if center_y + bbox_height / 2 * scale_top <= height else (height - center_y) / (bbox_height / 2)
        scale_y_bottom = scale_bottom if center_y - bbox_height / 2 * scale_bottom >= 0 else center_y / (bbox_height / 2)

        
        # Tính toán width và height mới sau khi nới rộng
        new_width = int(bbox_width * max_scale_x)
        new_height_top = int(bbox_height / 2 * scale_y_top)
        new_height_bottom = int(bbox_height / 2 * scale_y_bottom)


        # Tính toán tọa độ mới
        new_xmin = center_x - new_width // 2
        new_ymin = center_y - new_height_bottom
        new_xmax = center_x + new_width // 2
        new_ymax = center_y + new_height_top
        
        car_box=[new_xmin,new_ymin,new_xmax,new_ymax]
    
    
    return car_box


def plate_detection(plate_detector, img_input, conf):    
    # img1=img_input.copy()
    # image_crop2 = Image.fromarray(img1)
    # img1 = remove(image_crop2)
    # new_image = Image.new("RGBA", img1.size, "WHITE")  # Create a white rgba background
    # new_image.paste(img1, (0, 0), img1)
    # img = np.array(new_image)
    # img = img[:, :, :3]

    # print(img.shape)
    # exit()
    # new_image = cv2.cvtColor(new_image, cv2.COLOR_BGR2RGB)
    img=img_input.copy()
    s_max = 0
    conf_max=0
    box_plate_final = []
    box_plate = plate_detector(img, conf=conf, device='cpu', verbose=False)
    box_plates = box_plate[0].boxes
    # if len(box_plate)==0:
    #     box_plate = plate_detector(img, conf=0.01, device='cpu', verbose=False)
    #     box_plates = box_plate[0].boxes
    for b in box_plates:
        xyxy = b.xyxy.view(1, 4).clone().view(-1).tolist()
        if (xyxy[3] < img.shape[0]/3):
            continue
        
        # if abs(xyxy[0] - xyxy[2])*1.7<abs(xyxy[1] - xyxy[3]):
        #     continue
        
        # conf= b.conf.numpy()[0]
        # if conf>conf_max:
        #     conf_max=conf
        #     box_plate_final=xyxy
        
        s_ = abs(xyxy[0] - xyxy[2]) * abs(xyxy[1] - xyxy[3])
        if s_ > s_max:
            s_max = s_
            box_plate_final = xyxy
    return box_plate_final


def detect_car_plate(car_detector, plate_detector, img_path, cv2_format=False, conf=0.2, extent_car=True):
    if not cv2_format:
        img = cv2.imread(img_path)
        if img is None:
            return 0, 0, 0, 0
        img1 = img.copy()
    else:
        img1 = img_path.copy()
        if img1 is None:
            return 0, 0, 0, 0

    # yolov8 detection
    box = car_detection(car_detector, img1, extent_car=extent_car)
    if not box:
        return 0, 0, 0, 0

    xmin, ymin, xmax, ymax = int(box[0]), int(box[1]), int(box[2]), int(box[3])
    image_crop = img1[ymin:ymax, xmin:xmax]
    height_crop, width_crop, _ = image_crop.shape

    box_plate_final = plate_detection(plate_detector, image_crop, conf=conf)
    if not box_plate_final:
        return 1, [], image_crop, 100000

    xmin, ymin, xmax, ymax = int(box_plate_final[0]), int(box_plate_final[1]), int(box_plate_final[2]), int(
        box_plate_final[3])

    distance = abs(width_crop / 2 - (xmax - xmin))
    return 1, box_plate_final, image_crop, distance


def rebrander_img(type, image_crop, box_plate_final, logo_path, logo_replacement_path,rmbg=False):
    # remove background
    # image_crop2 = Image.fromarray(image_crop)
    # img = remove(image_crop2)
    # new_image = Image.new("RGBA", img.size, "WHITE")  # Create a white rgba background
    # new_image.paste(img, (0, 0), img)
    
    if rmbg:
        image_crop2=Image.fromarray(image_crop)
        img = remove(image_crop2)
        w,h = image_crop2.size
        new_h= int(h+0.5*h+1/6*h)
        new_w = int(w * 3/2)
        x= int(1/6 * new_w)
        y= int(1/3 * new_h) 
        if new_w<w:
            new_w= int(1.1 * w)
            new_h=int(new_w*2/3)
            x=int(0.05*w)
            y=int(new_h-h-0.1*new_h)
            
        new_image=Image.open('AdobeStock_127158488_v2.jpg')
        new_image= new_image.resize((new_w,new_h))
        new_image.paste(img, (x, y), img)
    else:
        new_image= Image.fromarray(image_crop)
    
    height_crop, width_crop, _ = image_crop.shape

    if box_plate_final:
        if rmbg:
            box_plate_final=box_plate_final+np.array([x,y,x,y])
        xmin, ymin, xmax, ymax = int(box_plate_final[0]), int(box_plate_final[1]), int(box_plate_final[2]), int(
            box_plate_final[3])
        if type == 1:
            image = Image.open(logo_replacement_path)
            
            corner_color = image.getpixel((0, 0))
            aspect_ratio= (xmax-xmin)/ (ymax-ymin)
            width, height = image.size
            desired_height= int(width/aspect_ratio)
            if desired_height < height:
                desired_height= height
            desired_width= width
            new_logo_image = Image.new('RGB', (desired_width, desired_height), corner_color)
            x_offset = (desired_width - width) // 2
            y_offset = (desired_height - height) // 2
            new_logo_image.paste(image, (x_offset, y_offset))
            new_logo_image.save('intermediate.jpg')
            image.close()
            
            logo_img = cv2.imread('intermediate.jpg')
            logo_img = cv2.resize(logo_img, (xmax - xmin, ymax - ymin))

            new_image = np.array(new_image)
            new_image = cv2.cvtColor(new_image, cv2.COLOR_RGBA2RGB)

            new_image2 = new_image.copy()
            new_image2[ymin:ymax, xmin:xmax] = logo_img
            # hsv_img1 = cv2.cvtColor(new_image, cv2.COLOR_BGR2HSV)
            # lower_white = (0, 0, 255)
            # upper_white = (0, 0, 255)
            # white_mask = cv2.inRange(hsv_img1, lower_white, upper_white)
            # new_image2[white_mask == 255] = (255, 255, 255)
            new_image = Image.fromarray(new_image2)

        else:
            bbox_img = image_crop[max(0, ymin - 10):min(ymax + 10, height_crop),
                       max(0, xmin - 10):min(width_crop, xmax + 10)]
            pad = int((ymax - ymin) / 2)
            left = image_crop[ymin - 10:ymax - 10, max(0, xmin - pad - 10):xmin - 10]
            right = image_crop[ymin - 10:ymax - 10, xmax + 10:min(width_crop, xmax + pad + 10)]
            left_avg_color = np.mean(left, axis=(0, 1))
            right_avg_color = np.mean(right, axis=(0, 1))
            mean_color = np.mean(np.vstack((left_avg_color, right_avg_color)), axis=0)
            blurred_bbox_img = np.zeros((bbox_img.shape[0], bbox_img.shape[1], 3), dtype=np.uint8)
            blurred_bbox_img[:, :] = mean_color
            blurred_bbox_img = cv2.GaussianBlur(blurred_bbox_img, (53, 53), 0)

            new_image = np.array(new_image)
            new_image = cv2.cvtColor(new_image, cv2.COLOR_RGBA2RGB)

            new_image2 = new_image.copy()
            new_image2[max(0, ymin - 10):min(ymax + 10, height_crop),
            max(0, xmin - 10):min(width_crop, xmax + 10)] = blurred_bbox_img
            hsv_img1 = cv2.cvtColor(new_image, cv2.COLOR_BGR2HSV)
            lower_white = (0, 0, 255)
            upper_white = (0, 0, 255)
            white_mask = cv2.inRange(hsv_img1, lower_white, upper_white)
            new_image2[white_mask == 255] = (255, 255, 255)
            new_image = Image.fromarray(new_image2)

    # padding and add logo vucar
    new_image = padding_img(new_image, logo_path)
    new_image = np.array(new_image)
    new_image = cv2.cvtColor(new_image, cv2.COLOR_BGR2RGB)
    return new_image

import base64
def decode_base64_image(base64_string):
    if ',' in base64_string:
        base64_string = base64_string.split(',')[-1]
    imgdata = base64.b64decode(base64_string)
    image = np.frombuffer(imgdata, np.uint8)
    image = cv2.imdecode(image, cv2.IMREAD_COLOR)
    return image


# def rebrander_img(car_detector, plate_detector, img_path, logo_path, logo_replacement_path, cv2_format=False):
#     if not cv2_format:
#         img = cv2.imread(img_path)
#     else:
#         img = img_path
#     img1 = img.copy()

#     # yolov8 detection
#     s_max = 0
#     results = car_detector(source=img1, device='cpu', classes=[2, 5, 7], verbose=False)
#     boxes = results[0].boxes
#     for b in boxes:
#         xyxy = b.xyxy.view(1, 4).clone().view(-1).tolist()
#         s_ = abs(xyxy[0] - xyxy[2]) * abs(xyxy[1] - xyxy[3])
#         if s_ > s_max:
#             s_max = s_
#             box = xyxy

#     if box == []:
#         return None

#     xmin, ymin, xmax, ymax = int(box[0]), int(box[1]), int(box[2]), int(box[3])
#     # image_crop = img1[max(0,ymin-20):min(ymax+20,img1.shape[0]-1),max(0,xmin-20):min(xmax+20,img1.shape[1]-1)]
#     image_crop = img1[ymin:ymax, xmin:xmax]
#     height_crop, width_crop, _ = image_crop.shape

#     s_max = 0
#     box_plate_final = []
#     box_plate = plate_detector(image_crop, conf=0.2, device='cpu', verbose=False)
#     box_plates = box_plate[0].boxes
#     for b in box_plates:
#         xyxy = b.xyxy.view(1, 4).clone().view(-1).tolist()
#         s_ = abs(xyxy[0] - xyxy[2]) * abs(xyxy[1] - xyxy[3])
#         if s_ > s_max:
#             s_max = s_
#             box_plate_final = xyxy

#     if box_plate_final != []:
#         xmin, ymin, xmax, ymax = int(box_plate_final[0]), int(box_plate_final[1]), int(box_plate_final[2]), int(
#             box_plate_final[3])

#         # bbox_img = image_crop[max(0,ymin-10):min(ymax+10,height_crop), max(0,xmin-10):min(width_crop,xmax+10)]
#         # blurred_bbox_img = bbox_img.copy()
#         # blurred_bbox_img = cv2.GaussianBlur(blurred_bbox_img, (201, 201), 0)
#         # image_crop[max(0,ymin-10):min(ymax+10,height_crop), max(0,xmin-10):min(width_crop,xmax+10)] = blurred_bbox_img
#         new_img = cv2.imread(logo_replacement_path)
#         new_img = cv2.resize(new_img, (xmax - xmin, ymax - ymin))
#         image_crop[ymin:ymax, xmin:xmax] = new_img

#         # remove background
#     image_crop = Image.fromarray(image_crop)
#     img = remove(image_crop)
#     new_image = Image.new("RGBA", img.size, "WHITE")  # Create a white rgba background
#     new_image.paste(img, (0, 0), img)

#     # padding and add logo vucar
#     new_image = padding_img(new_image, logo_path)
#     new_image = np.array(new_image)
#     new_image = cv2.cvtColor(new_image, cv2.COLOR_BGR2RGB)
#     # new_image=new_image[20:new_image.shape[0]-20,20:new_image.shape[1]-20]

#     return new_image


def rebrander_image(img,classify_model,car_detector,plate_detector,logo_path,logo_replacement_path, conf=0.2,extent_car=True,rmbg=False):
    label = classify_model.classify(img)
    if label==1:
        image_rgb=img.copy()
        image_pil = Image.fromarray(image_rgb)
        new_image= padding_img(image_pil,"images/logo.webp")
        new_image = np.array(new_image)
        new_image = cv2.cvtColor(new_image, cv2.COLOR_BGR2RGB)
        return new_image
    
    ok,box_plate_final, image_crop, distance = detect_car_plate(car_detector,plate_detector,img, cv2_format=True, conf=conf,extent_car=extent_car)
    if ok:
        img_final=rebrander_img(1,image_crop,box_plate_final,logo_path,logo_replacement_path,rmbg=rmbg)
        return img_final
    else:
        return img
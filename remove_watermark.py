import argparse

import os


from WatermarkRemovalPytorch.api import remove_watermark
import cv2
import os
from PIL import Image, ImageOps
import numpy as np

# parser = argparse.ArgumentParser(description = 'Removing Watermark')
# parser.add_argument('--image-path', type = str, default = 'inputs/1e2dd0724ac2039ea7c7030d4acb89b2-2840940181991627209.jpg', help = 'Path to the "watermarked" image.')
# parser.add_argument('--mask-path', type = str, default = 'mask.jpg', help = 'Path to the "watermark" image.')
# parser.add_argument('--input-depth', type = int, default = 64, help = 'Max channel dimension of the noise input. Set it based on gpu/device memory you have available.')
# parser.add_argument('--lr', type = float, default = 0.01, help = 'Learning rate.')
# parser.add_argument('--training-steps', type = int, default = 500, help = 'Number of training iterations.')
# parser.add_argument('--show-step', type = int, default = 200, help = 'Interval for visualizing results.')
# parser.add_argument('--reg-noise', type = float, default = 0.03, help = 'Hyper-parameter for regularized noise input.')
# parser.add_argument('--max-dim', type = float, default = 1024, help = 'Max dimension of the final output image')

# args = parser.parse_args()





def remove_watermark_func(img_cv2,device_id):
    # width = 720
    # height = 360
    #convert img_cv2 to PIL image
    img = cv2.cvtColor(img_cv2.copy(), cv2.COLOR_BGR2RGB)
    img_goc = Image.fromarray(img)

    # img_goc = Image.open(img_path)
    img_width, img_height = img_goc.size

    width=int(1/3*img_width)
    height=int(width/2)

    if img_height< height:
        height=img_height
        width=height*2
        if width>img_width:
            width=img_width

    print(width,height)

    left = (img_width - width) / 2
    top = (img_height - height) / 2
    right = (img_width + width) / 2
    bottom = (img_height + height) / 2

    cropped_img = img_goc.crop((left, top, right, bottom))
    
    # cropped_img_width, cropped_img_height = cropped_img.size
    # ratio_cropped=int(cropped_img_width/cropped_img_height)
    # new_size = (128, int(128/ratio_cropped))
    # cropped_img = cropped_img.resize(new_size)
    
    # cropped_img.save('cropped.jpg')
    # cropped_img.close()

    logo_path = 'images/chotot.jpg'
    background_width, background_height = cropped_img.size

    background = Image.new('RGB', (background_width, background_height), 'white')

    logo = Image.open(logo_path).convert('RGBA')
    width, height = logo.size
    new_logo_width=int(width/2)
    new_logo_height=int(height/2)
    logo = logo.resize((new_logo_width, new_logo_height), Image.ANTIALIAS)

    x_offset = (background_width - logo.width) // 2
    y_offset = (background_height - logo.height) // 2

    background.paste(logo, (x_offset, y_offset), logo)
    
    # background.save('mask.jpg')
    # background.close()
    logo.close()

    new_img= remove_watermark(
        cropped_img,
        background,
        device_id,
        # image_path = 'cropped.jpg',
        # mask_path = 'mask.jpg',
        max_dim = 1024,
        show_step = 200,
        reg_noise = 0.03,
        input_depth = 128,
        lr = 0.01,
        training_steps =500,
        
    )
    
    ####
    # new_img=new_img.resize((cropped_img_width, cropped_img_height))

    # new_img=  Image.open('cropped-output.jpg')
    img_goc.paste(new_img, (int(left), int(top)))
    
    # img_goc.close()
    del new_img
    
    img_goc = np.array(img_goc)
    img_goc = cv2.cvtColor(img_goc, cv2.COLOR_RGB2BGR)
    
    return img_goc
    
    # img_goc.save("output.jpg")

# import time
# t1=time.time()
# img_path= "images/fa29e2a8866403c980b4753644d4ca91-2846844909261545049.jpg"

# img=cv2.imread(img_path)
# result=remove_watermark_func(img,0)

# cv2.imwrite("out2.jpg",result)

# print(time.time()-t1)
import logging
import os
from typing import Union
from remove_watermark import remove_watermark_func
from utils import decode_base64_image, rebrander_image
from fastapi.middleware.cors import CORSMiddleware

import uvicorn
from dotenv import load_dotenv
from fastapi import FastAPI, Header, BackgroundTasks
from starlette.responses import JSONResponse
from typing_extensions import Annotated

from server.config import get_config
from server.schemas.image import ReBranderRequest, RemoveWatermarkRequest
from server.controller import image

import numpy as np
from PIL import Image
import cv2
import base64
from io import BytesIO

from pydantic import BaseModel
class ImageData(BaseModel):
    image_base64: str



config_variables = get_config()

load_dotenv()

app = FastAPI()

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)



@app.post("/rebrander")
async def upload_file(re_brander_request: ReBranderRequest, x_api_key: Annotated[Union[str, None], Header()],
                      background_tasks: BackgroundTasks):
    logging.info("calling API /rebrander")
    logging.info("taskid: ",re_brander_request.taskId)
    try:
        if x_api_key != os.getenv('API_KEY'):
            return JSONResponse(status_code=400, content=dict(description='invalid request'))

        if not re_brander_request.urls:
            return JSONResponse(status_code=400, content=dict(description='empty list images'))

        if len(re_brander_request.taskId) <= 4:
            return JSONResponse(status_code=400, content=dict(description='invalid task id'))

        outputs_image_url = image.read_image_url(list_image_url=re_brander_request.urls,
                                                 watermark_remove= re_brander_request.watermark_remove)

        logging.info(f"Calling webhook: {re_brander_request.webhook}")
        background_tasks.add_task(image.api_re_brander,
                                    _task_id=re_brander_request.taskId,
                                    result_read_image=outputs_image_url,
                                    webhook= re_brander_request.webhook,
                                    hide_license_plate=re_brander_request.hide_license_plate,
                                    bg_remove= re_brander_request.bg_remove,
                                    watermark_remove= re_brander_request.watermark_remove
                                    )

        return JSONResponse(status_code=200, content=dict(taskId=re_brander_request.taskId))
    except Exception as error:
        return JSONResponse(status_code=400, content=dict(description=str(error)))






@app.post("/rebrand")
async def rebrand(inp: ImageData, bg_remove: bool=False, watermark_remove: bool=False):
    logging.info("calling API /rebrander")
    try:
        img =  decode_base64_image(str(inp.image_base64))
        image_result = img.copy()
        
        if not bg_remove and not watermark_remove:
            image_result= rebrander_image(image_result,config_variables.classify_model, \
                                          config_variables.car_detector,config_variables.plate_detector_yolov8, \
                                          config_variables.logo_path,config_variables.logo_replacement_path,0.3, \
                                        extent_car=True)
            
        
        if bg_remove and not watermark_remove:
            image_result= rebrander_image(image_result,config_variables.classify_model, \
                                        config_variables.car_detector,config_variables.plate_detector_yolov8, \
                                        config_variables.logo_path,config_variables.logo_replacement_path,0.3, \
                                        extent_car=False, rmbg=True)
            
        if not bg_remove and watermark_remove:
            
            image_result=remove_watermark_func(image_result)
            image_result= rebrander_image(image_result,config_variables.classify_model, \
                                          config_variables.car_detector,config_variables.plate_detector_yolov8, \
                                          config_variables.logo_path,config_variables.logo_replacement_path,0.3, \
                                        extent_car=True)
            

        pil_img = Image.fromarray(cv2.cvtColor(image_result, cv2.COLOR_BGR2RGB))
        buf = BytesIO()
        pil_img.save(buf, format='PNG')
        byte_im = buf.getvalue()
        result_base64 = base64.b64encode(byte_im)

        return JSONResponse(status_code=200, content=dict(img_base64=result_base64.decode()))
    except Exception as error:
        return JSONResponse(status_code=400, content=dict(description=str(error)))



@app.post("/remove_watermark_v1")
async def remove_watermark_v1(watermark_request: RemoveWatermarkRequest, x_api_key: Annotated[Union[str, None], Header()],
                      background_tasks: BackgroundTasks):

    logging.info("calling API /rebrander")
    # logging.info("taskid: ",watermark_request.taskId)
    try:
        if x_api_key != os.getenv('API_KEY'):
            return JSONResponse(status_code=400, content=dict(description='invalid request'))

        if not watermark_request.urls:
            return JSONResponse(status_code=400, content=dict(description='empty list images'))

        if len(watermark_request.taskId) <= 4:
            return JSONResponse(status_code=400, content=dict(description='invalid task id'))

        outputs_image_url = image.read_image_url(list_image_url=watermark_request.urls,
                                                 watermark_remove= True,device_id=0)
        
        

        logging.info(f"Calling webhook: {watermark_request.webhook}")
        background_tasks.add_task(image.api_watermark,
                                    _task_id=watermark_request.taskId,
                                    result_read_image=outputs_image_url,
                                    webhook= watermark_request.webhook
                                    
                                    )
        print("hihi1")

        return JSONResponse(status_code=200, content=dict(taskId=watermark_request.taskId))
    except Exception as error:
        return JSONResponse(status_code=400, content=dict(description=str(error)))




@app.post("/remove_watermark_v2")
async def remove_watermark_v2(watermark_request: RemoveWatermarkRequest, x_api_key: Annotated[Union[str, None], Header()],
                      background_tasks: BackgroundTasks):

    logging.info("calling API /rebrander")
    # logging.info("taskid: ",watermark_request.taskId)
    try:
        if x_api_key != os.getenv('API_KEY'):
            return JSONResponse(status_code=400, content=dict(description='invalid request'))

        if not watermark_request.urls:
            return JSONResponse(status_code=400, content=dict(description='empty list images'))

        if len(watermark_request.taskId) <= 4:
            return JSONResponse(status_code=400, content=dict(description='invalid task id'))

        outputs_image_url = image.read_image_url(list_image_url=watermark_request.urls,
                                                 watermark_remove= True,device_id=1)
        
        

        logging.info(f"Calling webhook: {watermark_request.webhook}")
        background_tasks.add_task(image.api_watermark,
                                    _task_id=watermark_request.taskId,
                                    result_read_image=outputs_image_url,
                                    webhook= watermark_request.webhook
                                    
                                    )
        print("hihi2")

        return JSONResponse(status_code=200, content=dict(taskId=watermark_request.taskId))
    except Exception as error:
        return JSONResponse(status_code=400, content=dict(description=str(error)))



@app.post("/remove_watermark_v3")
async def remove_watermark_v3(watermark_request: RemoveWatermarkRequest, x_api_key: Annotated[Union[str, None], Header()],
                      background_tasks: BackgroundTasks):

    logging.info("calling API /rebrander")
    # logging.info("taskid: ",watermark_request.taskId)
    try:
        if x_api_key != os.getenv('API_KEY'):
            return JSONResponse(status_code=400, content=dict(description='invalid request'))

        if not watermark_request.urls:
            return JSONResponse(status_code=400, content=dict(description='empty list images'))

        if len(watermark_request.taskId) <= 4:
            return JSONResponse(status_code=400, content=dict(description='invalid task id'))

        outputs_image_url = image.read_image_url(list_image_url=watermark_request.urls,
                                                 watermark_remove= True,device_id=2)
        
        

        logging.info(f"Calling webhook: {watermark_request.webhook}")
        background_tasks.add_task(image.api_watermark,
                                    _task_id=watermark_request.taskId,
                                    result_read_image=outputs_image_url,
                                    webhook= watermark_request.webhook    
                                                                    )

        return JSONResponse(status_code=200, content=dict(taskId=watermark_request.taskId))
    except Exception as error:
        return JSONResponse(status_code=400, content=dict(description=str(error)))




@app.post("/remove_watermark_v4")
async def remove_watermark_v4(watermark_request: RemoveWatermarkRequest, x_api_key: Annotated[Union[str, None], Header()],
                      background_tasks: BackgroundTasks):

    logging.info("calling API /rebrander")
    # logging.info("taskid: ",watermark_request.taskId)
    try:
        if x_api_key != os.getenv('API_KEY'):
            return JSONResponse(status_code=400, content=dict(description='invalid request'))

        if not watermark_request.urls:
            return JSONResponse(status_code=400, content=dict(description='empty list images'))

        if len(watermark_request.taskId) <= 4:
            return JSONResponse(status_code=400, content=dict(description='invalid task id'))

        outputs_image_url = image.read_image_url(list_image_url=watermark_request.urls,
                                                 watermark_remove= True,device_id=3)
        
        

        logging.info(f"Calling webhook: {watermark_request.webhook}")
        background_tasks.add_task(image.api_watermark,
                                    _task_id=watermark_request.taskId,
                                    result_read_image=outputs_image_url,
                                    webhook= watermark_request.webhook
                                    )

        return JSONResponse(status_code=200, content=dict(taskId=watermark_request.taskId))
    except Exception as error:
        return JSONResponse(status_code=400, content=dict(description=str(error)))





if __name__ == '__main__':
    uvicorn.run("api:app", host='0.0.0.0', port=5556, log_config='server/config_log.ini',reload=False)

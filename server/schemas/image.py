from typing import List

from pydantic import BaseModel


class ReBranderRequest(BaseModel):
    taskId: str
    urls: List[str]
    webhook: str
    hide_license_plate: bool=True
    bg_remove: bool=False
    watermark_remove: bool=False
    
class RemoveWatermarkRequest(BaseModel):
    taskId: str
    urls: List[str]
    webhook: str

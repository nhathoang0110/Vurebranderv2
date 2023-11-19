from typing import List

import aiohttp
from aiofile import AIOFile
# from gcloud.aio.storage import Storage
from google.cloud import storage



from server.config import BUCKET_NAME, GCLOUD_FILE_CREDENTIAL, GCLOUD_FOLDER_PROCESSED_IMAGE, ROOT_DIR, \
    GCLOUD_STORAGE_BASE_URL


# async def upload_to_gcloud(list_images: List[str], task_id: str) -> List[str]:
#     result = []

#     async def async_upload_to_bucket(blob_file, folder=GCLOUD_FOLDER_PROCESSED_IMAGE):
#         async with AIOFile(blob_file, mode='rb') as afp:
#             file_obj = await afp.read()
#         blob_name = blob_file.split('/')[-1]
#         async with aiohttp.ClientSession() as session:
#             storage = Storage(service_file=ROOT_DIR + "/" + GCLOUD_FILE_CREDENTIAL, session=session)
#             status = await storage.upload(BUCKET_NAME, f'{folder}/{task_id}/{blob_name}', file_obj)
#             result.append(status['name'])

#     for img in list_images:
#         await async_upload_to_bucket(img)

#     return list(map(lambda x: f'https://cdn.vucar.vn/{BUCKET_NAME}/{x}', result))

async def upload_to_gcloud(list_images: List[str], task_id: str) -> List[str]:
    result = []

    async def async_upload_to_bucket(blob_file, folder=GCLOUD_FOLDER_PROCESSED_IMAGE):
        async with AIOFile(blob_file, mode='rb') as afp:
            file_obj = await afp.read()
        blob_name = blob_file.split('/')[-1]
        async with aiohttp.ClientSession() as session:
            client = storage.Client.from_service_account_json(GCLOUD_FILE_CREDENTIAL)
            bucket = client.get_bucket(BUCKET_NAME)
            blob = bucket.blob(f'{folder}/{task_id}/{blob_name}')
            blob.upload_from_string(file_obj)
            blob.make_public()
            result.append(blob.name)

    for img in list_images:
        await async_upload_to_bucket(img)

    return list(map(lambda x: f'https://cdn.vucar.vn/{x}', result))

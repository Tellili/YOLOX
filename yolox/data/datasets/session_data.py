import json
import logging
from io import BytesIO
from typing import Union

import PIL
import requests
import numpy as np
import cv2
from azure.identity import DefaultAzureCredential
from azure.storage.blob import BlobClient
from PIL import Image

logger = logging.getLogger(__name__)


def get_session_content(
    ocp_key: str,
    username: str,
    password: str,
    session_id: Union[str, int],
    tenant: str,
    token_url: str,
    session_content_url: str,
) -> Union[dict, list, Exception]:
    """
    Function retrieving session annotations. It makes two requests: for authorization and session content retrieval;
    :param ocp_key: key for authorization
    :param username: session participant/owner username
    :param password: session participant/owner password
    :param session_id: id of the collaborate session
    :param tenant: tenant of the session
    :param token_url: endpoint for authorization
    :param session_content_url: endpoint for content retrieval
    :return: session content output
    """
    # ------------------------------------- Get authorization token ----------------------------------------------------
    try:
        # TODO: remove authentication from trusted source of the backend API side

        headers = {
            "Ocp-Apim-Subscription-Key": ocp_key,
        }
        data = {
            "UserName": username,
            "Password": password,
            "Subdomain": tenant,
        }

        response = requests.post(token_url, headers=headers, data=data)
        logger.info(f"Response from token url {token_url}: {response.status_code}.")

        if response.status_code != 200:
            raise Exception(f"Failed to authorize.\n{response.reason}")
        token = json.loads(response.content)["access_token"]
        logger.info("Successful authorization.")

        # ------------------------------------- Get session content ----------------------------------------------------
        headers = {**headers, "Authorization": f"Bearer {token}"}
        session_content_url = f"{session_content_url.strip('/')}/{session_id}"

        response = requests.get(session_content_url, headers=headers)
        if response.status_code != 200:
            raise Exception(f"Failed to retrieve session {session_id} content.\n{response.reason}")
        session_content = json.loads(response.content)
        logger.info("Successful session content retrieval.")

        return session_content

    except Exception as e:
        logger.error(e)
        return e


def download_image_blob(blob_path: str) -> Union[PIL.Image.Image, Exception]:
    """
    Function that downloads a blob, and opens the downloaded image bytes as a PIL.Image.Image object
    :param blob_path: blob url with or without a sas token
    :return: PIL.Image.Image or Exception if blob download/image open fails
    """
    if "?skoid=" in blob_path:
        blob_client = BlobClient.from_blob_url(blob_path)
    else:
        splits = blob_path.split("/")
        storage_account_url = f"https://{splits[-3]}"
        container_name = splits[-2]
        file_name = splits[-1]
        blob_client = BlobClient(
            storage_account_url,
            container_name,
            file_name,
            credential=DefaultAzureCredential(),
        )

    try:
        bytestring = blob_client.download_blob().readall()
    except Exception as e:
        logger.error(f"Failed to download image blob: {blob_path}\n{e}")
        return e
    try:
        pil_image = Image.open(BytesIO(bytestring)).convert("RGB")
        pil_image_array = np.array(pil_image)

        image = cv2.cvtColor(pil_image_array, cv2.COLOR_RGB2BGR)

        return image
    except Exception as e:
        logger.error(f"Failed to load image: {blob_path}\n{e}")
        return e


if __name__ == "__main__":

    path = "https://tabeebprod.blob.core.windows.net/assets/CCT7P7zf9ezihl_OGjjFS_IMG0028.JPG"
    url = "https://tabeebprod.blob.core.windows.net/assets/8ff2aba2e4424fa0aebe0a9292980d13.JPG?skoid=1d0ccceb-33af-4c6f-8ca4-cd3e5a6b71a1&sktid=78640a62-55b2-4751-97e3-f421dfe38721&skt=2023-05-04T19%3A53%3A03Z&ske=2023-05-11T19%3A53%3A03Z&sks=b&skv=2021-06-08&sv=2021-06-08&st=2023-05-05T12%3A25%3A56Z&se=2023-06-02T13%3A30%3A56Z&sr=b&sp=r&sig=2iHcBPmcoelJ%2BsMmPjnzxiisKNibOpvDJyEXDptTAE0%3D"
    new_path = url.split('?skoid=')[0]
    x = download_image_blob(new_path)

    print(x)

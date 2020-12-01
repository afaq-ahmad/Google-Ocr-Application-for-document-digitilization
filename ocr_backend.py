import os
import os.path
import time
import re
import io

from google.cloud import vision
from google.cloud import storage

from collections import defaultdict


os.environ["GOOGLE_APPLICATION_CREDENTIALS"] = "phonebook-ocr-3512b1363e4f.json"
bucket_name = "phonebook-ocr-bucket"


def upload(path):
    '''
    Upload the PDF at 'path' and return the new Google Cloud
    Storage(GCS) location
    '''

    basename = os.path.basename(path)
    blob_name = "{0}-{1}".format(time.time(), basename)

    storage_client = storage.Client()

    #slow internet speeds sometimes cause GCS to time out
    #storage_client.blob._DEFAULT_CHUNKSIZE = 5 * 1024* 1024 # 5 MB

    bucket = storage_client.bucket(bucket_name)
    blob = bucket.blob(blob_name)

    #slow internet speeds sometimes cause GCS to time out
    #blob.chunk_size = 5 * 1024 * 1024 # Set 5 MB blob size

    blob.upload_from_filename(path)

    return "gs://{0}/{1}".format(bucket_name, blob_name)

def extract_text(gcs_path):
    '''
    Extract the text in the image saved at
    'gcs_path'
    '''

    vision_client = vision.ImageAnnotatorClient()

    image = vision.types.Image()
    image.source.image_uri = gcs_path

    response = vision_client.text_detection(image=image)

    return response


def extract_text_local(image_path):
    '''
    Extract text from the image file 'image_path'
    and return it
    '''

    vision_client = vision.ImageAnnotatorClient()

    image = vision.types.Image()

    content = None

    with io.open(image_path, 'rb') as image_file:
        content = image_file.read()

    image = vision.types.Image(content=content)

    response = vision_client.text_detection(image=image)

    return response

def delete_blob(gcs_path):
    '''
    Do some house cleaning: delete the blob at 'gcs_path'
    '''

    gcs_re = re.compile("gs://[^/]+/(.+)")

    gcs_re_matched = re.match(gcs_re, gcs_path)

    if ( gcs_re_matched is not None ):

        blob_name = gcs_re_matched.group(1)

        storage_client = storage.Client()

        bucket = storage_client.bucket(bucket_name)

        blob = bucket.blob(blob_name)

        blob.delete()





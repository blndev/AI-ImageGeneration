import json
import os
from time import sleep
import requests                 # for downloads of files
from datetime import datetime   # for timestamp
from PIL import Image           # for image handling
from hashlib import sha1        # generate image hash
import logging

# Set up module logger
logger = logging.getLogger(__name__)


def get_date_subfolder():
    return datetime.now().strftime("%Y-%m-%d")

def get_all_local_models(model_folder: str, extension: str = ".safetensors"):
    """find all local flux models"""
    safetensors_files = []
    try:
        for root, _, files in os.walk(model_folder, followlinks=True):
            for file in files:
                if file.endswith(extension) and "flux" in file.lower():
                    relative_path = "./" + os.path.relpath(os.path.join(root, file))
                    safetensors_files.append(relative_path)
        logger.debug("Found safetensors files: %s", safetensors_files)
    except Exception as e:
        logger.error("Error listing safetensors files: %s", str(e))
        logger.debug("Exception details:", exc_info=True)
    return safetensors_files


def download_file_if_not_existing(url, local_path):
    # Check if the file already exists
    if not os.path.exists(local_path):
        logger.info("Downloading %s... this can take some minutes", local_path)
        response = requests.get(url)
        response.raise_for_status()  # Check for download errors

        # Create directory if it does not exist
        os.makedirs(os.path.dirname(local_path), exist_ok=True)

        # Write the file to the specified path
        with open(local_path, 'wb') as file:
            file.write(response.content)
        logger.info("Downloaded %s successfully", local_path)
    else:
        logger.info("File %s already exists", local_path)


def save_image_as_png(image: Image.Image, dir: str, filename: str = None):
    """
    saves a image as PNG to the given directory and uses the SHA1 as filename if not filename is provided
    return value is the full filename
    """
    try:
        if not os.path.exists(dir):
            os.makedirs(dir)
        # if we use imageEditor from Gradio:
        # try:
        #     image = image['background'] # if we use editor field
        # except:
        #     print("seems no background in image dict")
        #     print (image)
        # Convert the image to bytes and compute the SHA-1 hash
        if filename is None:
            image_bytes = image.tobytes()
            hash = sha1(image_bytes).hexdigest()
            filetype = "png"
            filename = hash + "." + filetype

        file_path = os.path.join(dir, filename)

        if not os.path.exists(file_path):
            image.save(file_path, format="PNG")

        logger.debug("Image saved to %s", file_path)
        return file_path
    except Exception as e:
        logger.error("Error while saving image to cache: %s", str(e))
        logger.debug("Exception details:", exc_info=True)
        return None


def save_image_with_timestamp(image, folder_path, ignore_errors=False, reference="", appendix: str = "", generation_details: dict = None):
    """
    saves a image in a given folder and returns the used path
    reference: could be the SHA1 from source image to make a combined filename
    generation_details: if given, it wil be save as imagefilename.txt
    """
    try:
        # Create the folder if it does not exist
        os.makedirs(folder_path, exist_ok=True)

        # Generate a timestamp
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S.%f")[:-3]
        sleep(0.2)  # make sure we have unique filenames
        separator = "" if reference == "" else "-"
        # Create the filename with the timestamp
        filename = f"{reference}{separator}{timestamp}{appendix}.png"  # Change the extension as needed

        # Full path to save the image
        file_path = os.path.join(folder_path, filename)

        # Save the image
        image.save(file_path)
        if generation_details != None:
            try:
                with open(file_path+".txt", "w") as f:
                    generation_details
                    json.dump(obj=generation_details,fp=f,indent=4)
            except Exception as e:
                logger.error(f"Error while loading uploaded_images.json: {e}")

        return file_path
    except Exception as e:
        logger.error("Save image failed: %s", str(e))
        logger.debug("Exception details:", exc_info=True)
        if not ignore_errors:
            raise e

from PIL import Image, ImageOps # for image handling
import numpy as np              # converting PIL to CV2
import logging
import threading

from app.utils.singleton import singleton

import cv2                      # prepare images for face recognition
from insightface.app import FaceAnalysis    # face boxes detection

# Set up module logger
logger = logging.getLogger(__name__)

# src reference: https://github.com/cubiq/ComfyUI_IPAdapter_plus/issues/165
class FaceAnalysisEnhanced(FaceAnalysis):
    # NOTE: allows setting det_size for each detection call.
    # the model allows it but the wrapping code from insightface
    # doesn't show it, and people end up loading duplicate models
    # for different sizes where there is absolutely no need to
    def get(self, img, max_num=0, det_size=(640, 640)):
        if det_size is not None:
            self.det_model.input_size = det_size

        return super().get(img, max_num)

@singleton
class FaceDetector():
    def __init__(self):
        logger.info("Initializing FaceDetector")
        self.thread_lock = threading.Lock()

        #https://github.com/onnx/models/blob/main/validated/vision/body_analysis/emotion
        # _ferplus/model/emotion-ferplus-2.onnx
        #ctx_id =0 GPU, -1=CPU, 1,2, select GPU to be used
        self.ctx_id = -1 #to save gpu memory
        #providers = ['CUDAExecutionProvider', 'CPUExecutionProvider']
        providers = ['CPUExecutionProvider']
        try:
            #smallest: buffalo_sc but problems for some faces (blurry, mirror etc)
            # buffalo_l: much better detection, but don't trust age detection
            #FIXME: antelopev2 does not work (git issue is open) https://github.com/deepinsight/insightface/issues/2725
            self.face_detector = FaceAnalysisEnhanced(name="buffalo_l", providers=providers)  # https://github.com/deepinsight/insightface/tree/master/model_zoo
            self.face_detector.prepare(ctx_id=self.ctx_id, det_size=(512,512))
            logger.debug("FaceDetector initialization done")
        except Exception as e:
            logger.error("Error while initializing FaceDetector: %s", str(e))
            logger.debug("Exception details:", exc_info=True)

    def get_faces(self, pil_image: Image):
        """ return values are a list of dictionaries. if len=0, then no face was detected"""
        retVal = []
        cv2_image = None
        if self.face_detector == None: return retVal, cv2_image
        try:
            # reduce size if it is a big image to process it faster
            max_size = 1024
            pil_image.thumbnail((max_size, max_size))
            # correct EXIF-Orientation!! very important
            pil_image = ImageOps.exif_transpose(pil_image)
            # face analysis needs a base RGB format
            cv2_image = np.array(pil_image.convert("RGB"))

            # convert to OpenCV conforme colors
            cv2_image = cv2.cvtColor(cv2_image, cv2.COLOR_RGBA2RGB)

            if len(cv2_image.shape) == 2:  # if gray make RGB
                cv2_image = cv2.cvtColor(cv2_image, cv2.COLOR_GRAY2BGR)

            #size = scaling to for face detection (smaller = faster)
            #if size is bigger then the image size, we got no detection so 512x512 is fine
            with self.thread_lock:
                faces = self.face_detector.get(cv2_image)

            if len(faces)==0: faces = self._reduced_detection_site_detection(cv2_image)
            for face in faces:
                #print ("Face bbox", face['bbox'])
                x1, y1, x2, y2 = map(int, face.bbox)
                cropped_face = cv2_image[y1:y2, x1:x2]
                retVal.append(face)
                #cv2.imwrite(img=cropped_face, filename=f"./models/face_{x1}.jpg")
            logger.debug(f"Detected Faces {len(retVal)}")
        except Exception as e:
            logger.error("Error while detecting face: %s", str(e))
            logger.debug("Exception details:", exc_info=True)
        return retVal, cv2_image


    def _reduced_detection_site_detection(self, cv2_image):
        """ return values are a list of dictionaries. if len=0, then no face was detected"""
        try:
            logger.debug("startin enhanced detection with different detection sizes")
            detection_sizes = [None] + [(size, size) for size in range(640, 256, -64)] + [(256, 256)]
            with self.thread_lock:
                for size in detection_sizes:
                    faces = self.face_detector.get(cv2_image, det_size=size)
                    if len(faces) > 0:
                        logger.debug(f"Detected  {len(faces)} Faces with reduced detection size of {size}")
                        return faces
            
        except Exception as e:
            logger.error("Error while detecting face with reduced_detection_site_detection: %s", str(e))
            logger.debug("Exception details:", exc_info=True)
        return []

    def get_face_picture(self, cv2_image: np.array, face, filename: str = None):
        """ returns a cv2 image containing the face to work with it
        if filename is provided, the image is saved as jpg
        """
        try:
            x1, y1, x2, y2 = map(int, face.bbox)
            cropped_face = cv2_image[y1:y2, x1:x2]
            if not filename is None:
                cv2.imwrite(img=cropped_face, filename=filename)
            return cropped_face
        except Exception as e:
            logger.error("Error while detecting face: %s", str(e))
            logger.debug("Exception details:", exc_info=True)
        return None
    
    
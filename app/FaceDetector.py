from PIL import Image, ImageOps # for image handling
import numpy as np              # converting PIL to CV2
import logging
from app.utils.singleton import singleton

import cv2                      # prepare images for face recognition
from insightface.app import FaceAnalysis    # face boxes detection

# Set up module logger
logger = logging.getLogger(__name__)



@singleton
class FaceDetector():
    def __init__(self):
        logger.info("Initializing FaceDetector")
        #https://github.com/onnx/models/blob/main/validated/vision/body_analysis/emotion
        # _ferplus/model/emotion-ferplus-2.onnx
        # FIXME V3: we can have a switch here if GPU Memory >20GB(?) then onnx can run on GPU as well
        #ctx_id =0 GPU, -1=CPU, 1,2, select GPU to be used
        self.ctx_id = -1 #to save gpu memory
        #providers = ['CUDAExecutionProvider', 'CPUExecutionProvider']
        providers = ['CPUExecutionProvider']
        try:
            self.face_detector = FaceAnalysis(name="buffalo_sc", providers=providers)  # https://github.com/deepinsight/insightface/tree/master/model_zoo
            self.face_detector.prepare(ctx_id=self.ctx_id, det_size=(512,512))
            logger.debug("FaceDetector initialization done")
        except Exception as e:
            logger.error("Error while initializing FaceDetector: %s", str(e))
            logger.debug("Exception details:", exc_info=True)

    def get_faces(self, pil_image: Image):
        """ return values are a list of dictionaries. if len=0, then no face was detected"""
        retVal = []
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
            #TODO: with lock:
            faces = self.face_detector.get(cv2_image)

            for face in faces:
                #print ("Face bbox", face['bbox'])
                x1, y1, x2, y2 = map(int, face.bbox)
                cropped_face = cv2_image[y1:y2, x1:x2]
                retVal.append(cropped_face)
                #cv2.imwrite(img=cropped_face, filename=f"./models/face_{x1}.jpg")
            logger.debug(f"Detected Faces {len(retVal)}")
        except Exception as e:
            logger.error("Error while detecting face: %s", str(e))
            logger.debug("Exception details:", exc_info=True)
        return retVal

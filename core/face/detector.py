from ultralytics import YOLO
import os
#import requests

class FaceDetector:
    def __init__(self):
        yolo_weights_filename = os.path.join('weights', 'yolov8n-face.pt')
        '''
        #if not os.path.exists(yolo_weights_filename):
        #    response = requests.get(weights_url)
        #    with open(yolo_weights_filename, 'wb') as file:
        #        file.write(response.content)
        '''
        self.model = YOLO(yolo_weights_filename)

    def detect(self, frame, face_det_tresh):
        outputs = self.model(frame, verbose=False)
        faces = []
        for box in outputs[0].boxes:
            if float(box.conf) >= face_det_tresh:
                x, y, w, h = [int(coord) for coord in box.xywh[0]]
                x_center, y_center = x + w / 2, y + h / 2
                x1 = int(x_center - w)
                y1 = int(y_center - h)
                crop_img = frame[y1:y1+h, x1:x1+w]
                faces.append((crop_img, [x1, y1, w, h]))

        return faces
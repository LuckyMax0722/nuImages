from ultralytics import YOLO
import os
from lib.config import CONF

# Load a model
model = YOLO('yolov8x.pt')  # pretrained YOLOv8n model

# Run batched inference on a list of images
results = model('/home/jiachen/nuImages/data/nuimages/samples/CAM_FRONT/n003-2018-01-02-11-48-43+0800__CAM_FRONT__1514865067391098.jpg')  # return a list of Results objects

# Process results list
for result in results:
    boxes = result.boxes  # Boxes object for bounding box outputs
    masks = result.masks  # Masks object for segmentation masks outputs
    keypoints = result.keypoints  # Keypoints object for pose outputs
    probs = result.probs  # Probs object for classification outputs
    result.show()  # display to screen
    save_dir = os.path.join(CONF.PATH.BASE, 'YOLOv8/runs/detect/inference')

    if not os.path.exists(save_dir):
        os.makedirs(save_dir)

    result.save(filename=os.path.join(save_dir, 'result.jpg'))  # save to disk
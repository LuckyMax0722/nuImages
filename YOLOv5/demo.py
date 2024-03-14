import torch

# Model
model = torch.hub.load('ultralytics/yolov5', 'yolov5x', pretrained=True)

# Images
imgs = ['/home/jiachen/nuImages/YOLOv5/datasets/nuImages/images/train/n003-2018-01-03-12-03-23+0800__CAM_BACK__1514952316316487.jpg']  # batch of images

# Inference
results = model(imgs)

# Results
results.print()
results.save()  # or .show()

results.xyxy[0]  # img1 predictions (tensor)
results.pandas().xyxy[0]  # img1 predictions (pandas)

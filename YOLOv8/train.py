from ultralytics import YOLO

# Load a model
#model = YOLO('yolov8n.yaml')  # build a new model from YAML
#model = YOLO('yolov8n.pt')  # load a pretrained model (recommended for training)
model = YOLO('yolov8x.yaml').load('weights/yolov8x.pt')  # build from YAML and transfer weights

# Train the model
results = model.train(data='/home/jiachen/nuImages/YOLOv8/data/yolov8_nuImages_train.yaml',
                      epochs=300,
                      imgsz=640,
                      batch=4
                      )
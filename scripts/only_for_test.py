from models.yolo import Model

import torch
from PIL import Image
import torchvision.transforms as transforms
from models.common import DetectMultiBackend
from utils.general import check_img_size, Profile, increment_path, non_max_suppression, scale_boxes, xyxy2xywh
from utils.dataloaders import LoadImages
from pathlib import Path

from utils.plots import Annotator, colors, save_one_box
import cv2

def image_to_tensor():
    # 读取本地的 JPG 图像
    image_path = "/home/jiachen/nuImages/datasets_mini/nuImages/images/train/n003-2018-01-03-12-03-23+0800__CAM_BACK__1514952316316487.jpg"
    image = Image.open(image_path)

    # 转换图像为 PyTorch 张量，并进行大小调整和归一化
    transform = transforms.Compose([
        transforms.Resize((640, 640)),  # 将图像大小调整为 (640, 640)
        transforms.ToTensor(),  # 将图像转换为张量，并归一化到 [0, 1]
    ])

    # 应用变换并添加批次维度
    image_tensor = transform(image).unsqueeze(0)

    im = image_tensor.to(device)

    return im


if __name__ == '__main__':
    if torch.cuda.is_available():
        # 如果有 CUDA 设备可用，则将张量移动到第一个 CUDA 设备上
        device = torch.device("cuda")

    im = image_to_tensor()

    weights = '/home/jiachen/nuImages/YOLOv9/weights/yolov9-e-converted.pt'
    data = '/home/jiachen/nuImages/YOLOv9/data/coco.yaml'
    imgsz = (640, 640)
    source = '/home/jiachen/nuImages/datasets_mini/nuImages/images/train/n003-2018-01-03-12-03-23+0800__CAM_BACK__1514952316316487.jpg'
    project = '/home/jiachen/nuImages/runs/detect'
    name = 'exp'
    exist_ok = False
    save_txt = False
    augment = False
    visualize = False
    conf_thres = 0.25  # confidence threshold
    iou_thres = 0.45  # NMS IOU threshold
    classes = None  # filter by class: --class 0, or --class 0 2 3
    agnostic_nms = False
    max_det = 1000

    save_crop = False
    line_thickness = 1  # bounding box thickness (pixels)
    hide_labels = False  # hide labels
    hide_conf = False  # hide confidences
    save_img = True

    # Directories
    save_dir = increment_path(Path(project) / name, exist_ok=exist_ok)  # increment run
    (save_dir / 'labels' if save_txt else save_dir).mkdir(parents=True, exist_ok=True)  # make dir

    # Load model
    model = DetectMultiBackend(weights, device=device, data=data)
    stride, names, pt = model.stride, model.names, model.pt
    imgsz = check_img_size(imgsz, s=stride)  # check image size

    # Dataloader
    bs = 1  # batch_size
    dataset = LoadImages(source, img_size=imgsz, stride=stride, auto=pt, vid_stride=1)
    vid_path, vid_writer = [None] * bs, [None] * bs

    # Run inference
    model.warmup(imgsz=(1 if pt or model.triton else bs, 3, *imgsz))  # warmup
    seen, windows, dt = 0, [], (Profile(), Profile(), Profile())

    normalized_obj_bbox = []

    for path, im, im0s, vid_cap, s in dataset:
        with dt[0]:
            im = torch.from_numpy(im).to(model.device)
            im = im.half() if model.fp16 else im.float()  # uint8 to fp16/32
            im /= 255  # 0 - 255 to 0.0 - 1.0
            if len(im.shape) == 3:
                im = im[None]  # expand for batch dim

        # Inference
        with dt[1]:
            visualize = increment_path(save_dir / Path(path).stem, mkdir=True) if visualize else False
            pred = model(im, augment=augment, visualize=visualize)

        # NMS
        with dt[2]:
            pred = non_max_suppression(pred, conf_thres, iou_thres, classes, agnostic_nms, max_det=max_det)

    #print(pred)
        # Process predictions
        for i, det in enumerate(pred):  # per image
            seen += 1
            p, im0, frame = path, im0s.copy(), getattr(dataset, 'frame', 0)

            p = Path(p)  # to Path
            save_path = str(save_dir / p.name)  # im.jpg
            print()
            txt_path = str(save_dir / 'labels' / p.stem) + ('' if dataset.mode == 'image' else f'_{frame}')  # im.txt
            s += '%gx%g ' % im.shape[2:]  # print string
            gn = torch.tensor(im0.shape)[[1, 0, 1, 0]]  # normalization gain whwh
            imc = im0.copy() if save_crop else im0  # for save_crop
            annotator = Annotator(im0, line_width=line_thickness, example=str(names))

            if len(det):
                # Rescale boxes from img_size to im0 size
                det[:, :4] = scale_boxes(im.shape[2:], det[:, :4], im0.shape).round()

                # Print results
                for c in det[:, 5].unique():
                    n = (det[:, 5] == c).sum()  # detections per class
                    s += f"{n} {names[int(c)]}{'s' * (n > 1)}, "  # add to string

                # Write results
                for *xyxy, conf, cls in reversed(det):
                    c = int(cls)  # integer class
                    label = None if hide_labels else (names[c] if hide_conf else f'{names[c]} {conf:.2f}')
                    annotator.box_label(xyxy, label, color=colors(c, True))

            if save_img:
                if dataset.mode == 'image':
                    cv2.imwrite(save_path, im0)
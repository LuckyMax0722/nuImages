import os
import cv2
from multiprocessing import Pool, Manager
from tqdm import tqdm

from lib.config import CONF


def flip_labels(label_path, output_label_path):
    with open(label_path, 'r') as f:
        lines = f.readlines()

    flipped_lines = []
    for line in lines:
        label_info = line.strip().split()
        if len(label_info) == 5:
            class_id = label_info[0]
            x_center = float(label_info[1])
            y_center = float(label_info[2])
            width = float(label_info[3])
            height = float(label_info[4])

            # 水平翻转
            x_center = 1 - x_center

            flipped_lines.append(f"{class_id} {x_center} {y_center} {width} {height}\n")

    # 写入翻转后的标签信息
    with open(output_label_path, 'w') as f:
        f.writelines(flipped_lines)


def flip_image(args):
    image_path, label_path, output_images_folder, output_labels_folder, progress = args

    if os.path.basename(image_path).split('.')[0] != os.path.basename(label_path).split('.')[0]:
        print(f'Image {os.path.basename(image_path)} not match label {os.path.basename(label_path)}')
        return False

    # 读取图片
    img = cv2.imread(image_path)

    # 水平翻转
    flipped_img = cv2.flip(img, 1)

    # 保存翻转后的图片
    image_filename = os.path.basename(image_path)
    image_filename = image_filename.split('.')[0] + '_Flipped.' + image_filename.split('.')[1]
    output_path = os.path.join(output_images_folder, image_filename)
    cv2.imwrite(output_path, flipped_img)

    # 翻转标签
    label_filename = os.path.basename(label_path)
    label_filename = label_filename.split('.')[0] + '_Flipped.' + label_filename.split('.')[1]
    flip_labels(label_path, os.path.join(output_labels_folder, label_filename))

    # 进度+1
    progress.value += 1


def flip_images_in_folder(image_path, label_path, output_images_folder, output_labels_folder):
    # 确保输出图片文件夹存在
    if not os.path.exists(output_images_folder):
        os.makedirs(output_images_folder)

    # 确保输出标签文件夹存在
    if not os.path.exists(output_labels_folder):
        os.makedirs(output_labels_folder)

    # 使用共享变量来跟踪进度
    manager = Manager()
    progress = manager.Value('i', 0)

    # 使用多进程处理图片
    with Pool() as pool:
        for _ in tqdm(pool.imap_unordered(flip_image, [
            (image_path, label_path, output_images_folder, output_labels_folder, progress) for image_path, label_path in
            zip(image_paths, label_paths)]), total=len(image_paths), desc="Image Flipping"):
            pass


if __name__ == '__main__':
    input_images_folder = CONF.datasets.images_train
    output_images_folder = CONF.datasets.images_train

    input_lables_folder = CONF.datasets.labels_train
    output_labels_folder = CONF.datasets.labels_train

    # 获取文件夹中的所有图片路径
    image_paths = [os.path.join(input_images_folder, filename) for filename in sorted(os.listdir(input_images_folder))
                   if filename.endswith(('.jpg', '.jpeg', '.png', '.bmp'))]

    # 获取文件夹中的所有标签文件路径
    label_paths = [os.path.join(input_lables_folder, filename) for filename in sorted(os.listdir(input_lables_folder))
                   if filename.endswith('.txt')]

    # 对文件夹中的图片进行水平翻转操作
    flip_images_in_folder(image_paths, label_paths, output_images_folder, output_labels_folder)

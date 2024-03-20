from nuimages import NuImages
from lib.config import CONF

import os

import shutil
from tqdm import tqdm
import random

token_to_category = {
    '63a94dfa99bb47529567cd90d3b58384': 0,
    'a86329ee68a0411fb426dcad3b21452f': 1,
    '1fa93b757fc74fb197cdd60001ad8abf': 2,
    'b1c6de4c57f14a5383d9f963fbdcb5cb': 3,
    '909f1237d34a49d6bdd27c2fe4581d79': 4,
    '403fede16c88426885dd73366f16c34a': 5,
    'e3c7da112cd9475a9a10d45015424815': 6,
    '6a5888777ca14867a8aee3fe539b56c4': 7,
    'b2d7c6c701254928a9e4d6aac9446d79': 8,
    '653f7efbb9514ce7b81d44070d6208c1': 9,
    '063c5e7f638343d3a7230bc3641caf97': 10,
    'd772e4bae20f493f98e15a76518b31d7': 11,
    '85abebdccd4d46c7be428af5a6173947': 12,
    '0a30519ee16a4619b4f4acfe2d78fb55': 13,
    'fc95c87b806f48f8a1faea2dcc2222a4': 14,
    '003edbfb9ca849ee8a7496e9af3025d4': 15,
    'fedb11688db84088883945752e480c2c': 16,
    'fd69059b62a3469fbaef25340c0eab7f': 17,
    '5b3cd6f2bca64b83aa3d0008df87d0e4': 18,
    '7754874e6d0247f9855ae19a4028bf0e': 19,
    '732cce86872640628788ff1bb81006d4': 20,
    '7b2ff083a64e4d53809ae5d9be563504': 21,
    'dfd26f200ade4d24b540184e16050022': 22,
    '90d0f6f8e7c749149b1b6c3a029841a8': 23,
    '6021b5187b924d64be64a702e5570edf': 24
}

category_to_name = {
    0: 'animal',
    1: 'flat.driveable_surface',
    2: 'human.pedestrian.adult',
    3: 'human.pedestrian.child',
    4: 'human.pedestrian.construction_worker',
    5: 'human.pedestrian.personal_mobility',
    6: 'human.pedestrian.police_officer',
    7: 'human.pedestrian.stroller',
    8: 'human.pedestrian.wheelchair',
    9: 'movable_object.barrier',
    10: 'movable_object.debris',
    11: 'movable_object.pushable_pullable',
    12: 'movable_object.trafficcone',
    13: 'static_object.bicycle_rack',
    14: 'vehicle.bicycle',
    15: 'vehicle.bus.bendy',
    16: 'vehicle.bus.rigid',
    17: 'vehicle.car',
    18: 'vehicle.construction',
    19: 'vehicle.ego',
    20: 'vehicle.emergency.ambulance',
    21: 'vehicle.emergency.police',
    22: 'vehicle.motorcycle',
    23: 'vehicle.trailer',
    24: 'vehicle.truck'
}


def get_sample(sample):
    key_camera_token = sample['key_camera_token']
    return key_camera_token


def get_samples_data(sample, key_camera_token):
    sample_data = nuim.get('sample_data', sample['key_camera_token'])
    return sample_data


def get_file_path(sample_data):
    file_name = sample_data['filename']
    file_path = os.path.join(CONF.PATH.DATA, file_name)
    return file_path


def get_object_tokens(sample_data):
    object_tokens, _ = nuim.list_anns(sample_data['sample_token'], verbose=False)
    return object_tokens


def convert_bbox_to_normalized(bbox, image_shape):
    """
    Convert bounding box format from (xmin, ymin, xmax, ymax) to normalized (x_center, y_center, width, height).
    """
    xmin, ymin, xmax, ymax = bbox
    img_height, img_width = image_shape
    x_center = (xmin + xmax) / (2 * img_width)
    y_center = (ymin + ymax) / (2 * img_height)
    width = (xmax - xmin) / img_width
    height = (ymax - ymin) / img_height

    # Round values to keep only 6 decimal places
    x_center = round(x_center, 6)
    y_center = round(y_center, 6)
    width = round(width, 6)
    height = round(height, 6)

    return x_center, y_center, width, height


def get_object(object_tokens):
    obj_cat_list = []
    normalized_obj_bbox_list = []

    for i in range(len(object_tokens)):
        object = nuim.get('object_ann', object_tokens[i])

        obj_cat_token = object['category_token']
        obj_bbox = object['bbox']
        if object['mask'] != None:
            img_size = object['mask']['size']
        else:
            img_size = [900, 1600]

        normalized_obj_bbox = convert_bbox_to_normalized(obj_bbox, img_size)
        obj_cat = token_to_category[obj_cat_token]

        obj_cat_list.append(obj_cat)
        normalized_obj_bbox_list.append(normalized_obj_bbox)

    return obj_cat_list, normalized_obj_bbox_list


def crete_datasets(file_path, obj_cat_list, normalized_obj_bbox_list, data_version):
    if data_version == 'v1.0-mini':
        # 确定图片应该放置的目标文件夹
        if random.random() < CONF.datasets.split_ratio:
            images_dataset_folder = CONF.datasets.images_train
            labels_dataset_folder = CONF.datasets.labels_train
        else:
            images_dataset_folder = CONF.datasets.images_val
            labels_dataset_folder = CONF.datasets.labels_val

        # jpg文件的拷贝
        os.chmod(file_path, 0o644)  # 修改原文件的只读权限，以防止多次拷贝出现报错
        shutil.copy(file_path, images_dataset_folder)
        os.chmod(file_path, 0o444)  # 恢复权限为444，表示所有用户只有读权限

        # txt文件的拷贝
        file_name = os.path.splitext(os.path.basename(file_path))[0]  # 获取文件名
        target_path = os.path.join(labels_dataset_folder, file_name + '.txt')

        with open(target_path, 'w') as f:
            for obj_cat, normalized_obj_bbox_list in zip(obj_cat_list, normalized_obj_bbox_list):
                # 构建要写入的行内容
                line = (f"{obj_cat} {normalized_obj_bbox_list[0]} {normalized_obj_bbox_list[1]} "
                        f"{normalized_obj_bbox_list[2]} {normalized_obj_bbox_list[3]}\n")

                # 将行写入文件
                f.write(line)

    elif data_version == 'v1.0-train':
        images_dataset_folder = CONF.datasets.images_train
        labels_dataset_folder = CONF.datasets.labels_train

        # jpg文件的拷贝
        os.chmod(file_path, 0o644)  # 修改原文件的只读权限，以防止多次拷贝出现报错
        shutil.copy(file_path, images_dataset_folder)
        os.chmod(file_path, 0o444)  # 恢复权限为444，表示所有用户只有读权限

        # txt文件的拷贝
        file_name = os.path.splitext(os.path.basename(file_path))[0]  # 获取文件名
        target_path = os.path.join(labels_dataset_folder, file_name + '.txt')

        with open(target_path, 'w') as f:
            for obj_cat, normalized_obj_bbox_list in zip(obj_cat_list, normalized_obj_bbox_list):
                # 构建要写入的行内容
                line = (f"{obj_cat} {normalized_obj_bbox_list[0]} {normalized_obj_bbox_list[1]} "
                        f"{normalized_obj_bbox_list[2]} {normalized_obj_bbox_list[3]}\n")

                # 将行写入文件
                f.write(line)

    elif data_version == 'v1.0-val':
        images_dataset_folder = CONF.datasets.images_val
        labels_dataset_folder = CONF.datasets.labels_val

        # jpg文件的拷贝
        os.chmod(file_path, 0o644)  # 修改原文件的只读权限，以防止多次拷贝出现报错
        shutil.copy(file_path, images_dataset_folder)
        os.chmod(file_path, 0o444)  # 恢复权限为444，表示所有用户只有读权限

        # txt文件的拷贝
        file_name = os.path.splitext(os.path.basename(file_path))[0]  # 获取文件名
        target_path = os.path.join(labels_dataset_folder, file_name + '.txt')

        with open(target_path, 'w') as f:
            for obj_cat, normalized_obj_bbox_list in zip(obj_cat_list, normalized_obj_bbox_list):
                # 构建要写入的行内容
                line = (f"{obj_cat} {normalized_obj_bbox_list[0]} {normalized_obj_bbox_list[1]} "
                        f"{normalized_obj_bbox_list[2]} {normalized_obj_bbox_list[3]}\n")

                # 将行写入文件
                f.write(line)

    elif data_version == 'v1.0-test':
        images_dataset_folder = CONF.datasets.images_test

        # jpg文件的拷贝
        os.chmod(file_path, 0o644)  # 修改原文件的只读权限，以防止多次拷贝出现报错
        shutil.copy(file_path, images_dataset_folder)
        os.chmod(file_path, 0o444)  # 恢复权限为444，表示所有用户只有读权限


if __name__ == '__main__':
    # 创建dataset路径并确保目录存在
    os.makedirs(CONF.datasets.images_train, exist_ok=True)
    os.makedirs(CONF.datasets.images_val, exist_ok=True)
    os.makedirs(CONF.datasets.labels_train, exist_ok=True)
    os.makedirs(CONF.datasets.labels_val, exist_ok=True)

    if CONF.data.version == 'v1.0-mini':
        nuim = NuImages(dataroot=CONF.PATH.DATA, version=CONF.data.version, verbose=True, lazy=True)
        print(f'{CONF.data.version}数据集中sample的数量为:{len(nuim.sample)}')
        print("================")

        for i in tqdm(range(len(nuim.sample)), desc='v1.0-mini Dataset Processing', ncols=100):
            key_camera_token = get_sample(nuim.sample[i])

            sample_data = get_samples_data(nuim.sample[i], key_camera_token)

            file_path = get_file_path(sample_data)

            object_tokens = get_object_tokens(sample_data)

            obj_cat_list, normalized_obj_bbox_list = get_object(object_tokens)

            crete_datasets(file_path, obj_cat_list, normalized_obj_bbox_list, CONF.data.version)

    elif CONF.data.version == 'v1.0':
        # v1.0-train
        nuim = NuImages(dataroot=CONF.PATH.DATA, version='v1.0-train', verbose=True, lazy=True)
        print(f'v1.0-train数据集中sample的数量为:{len(nuim.sample)}')
        print("================")

        for i in tqdm(range(len(nuim.sample)), desc='v1.0-train Dataset Processing', ncols=100):
            key_camera_token = get_sample(nuim.sample[i])

            sample_data = get_samples_data(nuim.sample[i], key_camera_token)

            file_path = get_file_path(sample_data)

            object_tokens = get_object_tokens(sample_data)

            obj_cat_list, normalized_obj_bbox_list = get_object(object_tokens)

            crete_datasets(file_path, obj_cat_list, normalized_obj_bbox_list, 'v1.0-train')

        # v1.0-val
        nuim = NuImages(dataroot=CONF.PATH.DATA, version='v1.0-val', verbose=True, lazy=True)
        print(f'v1.0-val数据集中sample的数量为:{len(nuim.sample)}')
        print("================")

        for i in tqdm(range(len(nuim.sample)), desc='v1.0-val Dataset Processing', ncols=100):
            key_camera_token = get_sample(nuim.sample[i])

            sample_data = get_samples_data(nuim.sample[i], key_camera_token)

            file_path = get_file_path(sample_data)

            object_tokens = get_object_tokens(sample_data)

            obj_cat_list, normalized_obj_bbox_list = get_object(object_tokens)

            crete_datasets(file_path, obj_cat_list, normalized_obj_bbox_list, 'v1.0-val')

        # v1.0-test
        nuim = NuImages(dataroot=CONF.PATH.DATA, version='v1.0-test', verbose=True, lazy=True)
        print(f'v1.0-test数据集中sample的数量为:{len(nuim.sample)}')
        print("================")

        for i in tqdm(range(len(nuim.sample)), desc='v1.0-test Dataset Processing', ncols=100):
            key_camera_token = get_sample(nuim.sample[i])

            sample_data = get_samples_data(nuim.sample[i], key_camera_token)

            file_path = get_file_path(sample_data)

            crete_datasets(file_path, [], [], 'v1.0-test')

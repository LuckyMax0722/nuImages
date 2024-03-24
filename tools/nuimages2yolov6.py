import logging

logging.getLogger().setLevel(logging.CRITICAL)
from pylabel import importer
from lib.config import CONF
import shutil
import os

def get_dataset(path_to_annotations, path_to_images):
    yoloclasses = ['animal', 'flat.driveable_surface', 'human.pedestrian.adult', 'human.pedestrian.child',
                   'human.pedestrian.construction_worker', 'human.pedestrian.personal_mobility',
                   'human.pedestrian.police_officer', 'human.pedestrian.stroller', 'human.pedestrian.wheelchair',
                   'movable_object.barrier', 'movable_object.debris', 'movable_object.pushable_pullable',
                   'movable_object.trafficcone', 'static_object.bicycle_rack', 'vehicle.bicycle', 'vehicle.bus.bendy',
                   'vehicle.bus.rigid', 'vehicle.car', 'vehicle.construction', 'vehicle.ego',
                   'vehicle.emergency.ambulance', 'vehicle.emergency.police', 'vehicle.motorcycle', 'vehicle.trailer',
                   'vehicle.truck']

    dataset = importer.ImportYoloV5(path=path_to_annotations, path_to_images=path_to_images, cat_names=yoloclasses,
                                    img_ext="jpg")

    return dataset


def analyze_annotations(dataset):
    print(f"Number of images: {dataset.analyze.num_images}")
    print(f"Number of classes: {dataset.analyze.num_classes}")
    print(f"Classes:{dataset.analyze.classes}")
    print(f"Class counts:\n{dataset.analyze.class_counts}")


def copy_folder(source_folder, target_folder):
    # 拷贝文件夹
    try:
        shutil.copytree(source_folder, target_folder)
        print(f"文件夹从 {source_folder} 成功拷贝到 {target_folder}")
    except shutil.Error as e:
        print(f"文件夹拷贝失败: {e}")


if __name__ == '__main__':
    copy_folder(CONF.datasets.images_train, CONF.datasets.images_train_coco)
    copy_folder(CONF.datasets.images_val, CONF.datasets.images_val_coco)
    copy_folder(CONF.datasets.labels_train, CONF.datasets.labels_train_coco)
    copy_folder(CONF.datasets.labels_val, CONF.datasets.labels_val_coco)

    if not os.path.exists(CONF.datasets.annotations_coco):
        os.makedirs(CONF.datasets.annotations_coco)

    # get training dataset json
    path_to_images = CONF.datasets.images_train_coco
    path_to_annotations = CONF.datasets.labels_train_coco
    training_dataset = get_dataset(path_to_annotations, path_to_images)

    # Analyze Annotations
    # analyze_annotations(training_dataset)

    out_path = os.path.join(CONF.datasets.annotations_coco, 'instances_train2017.json')

    training_dataset.export.ExportToCoco(output_path=out_path, cat_id_index=1)

    # get validate dataset json
    path_to_images = CONF.datasets.images_val_coco
    path_to_annotations = CONF.datasets.labels_val_coco
    validate_dataset = get_dataset(path_to_annotations, path_to_images)

    # Analyze Annotations
    # analyze_annotations(validate_dataset)

    out_path = os.path.join(CONF.datasets.annotations_coco, 'instances_val2017.json')

    validate_dataset.export.ExportToCoco(output_path=out_path, cat_id_index=1)

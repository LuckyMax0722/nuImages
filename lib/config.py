import os
from easydict import EasyDict

CONF = EasyDict()

# Main Path
CONF.PATH = EasyDict()
CONF.PATH.BASE = '/home/jiachen/nuImages'  # TODO: change this
CONF.PATH.DATA = os.path.join(CONF.PATH.BASE, 'data/nuimages')
CONF.PATH.DATASETS = os.path.join(CONF.PATH.BASE, 'datasets')
CONF.PATH.DATASETS_MINI = os.path.join(CONF.PATH.BASE, 'datasets_mini')
# Data
CONF.data = EasyDict()
CONF.data.version = 'v1.0-mini' # ['v1.0-mini', 'v1.0']


# Datasets
CONF.datasets = EasyDict()
if CONF.data.version == 'v1.0-mini':
    CONF.datasets.images_train = os.path.join(CONF.PATH.DATASETS_MINI, 'nuImages/images/train')
    CONF.datasets.images_val = os.path.join(CONF.PATH.DATASETS_MINI, 'nuImages/images/val')
    CONF.datasets.labels_train = os.path.join(CONF.PATH.DATASETS_MINI, 'nuImages/labels/train')
    CONF.datasets.labels_val = os.path.join(CONF.PATH.DATASETS_MINI, 'nuImages/labels/val')
    CONF.datasets.split_ratio = 0.8
elif CONF.data.version == 'v1.0':
    CONF.datasets.images_train = os.path.join(CONF.PATH.DATASETS, 'nuImages/images/train')
    CONF.datasets.images_val = os.path.join(CONF.PATH.DATASETS, 'nuImages/images/val')
    CONF.datasets.images_test = os.path.join(CONF.PATH.DATASETS, 'nuImages/images/test')
    CONF.datasets.labels_train = os.path.join(CONF.PATH.DATASETS, 'nuImages/labels/train')
    CONF.datasets.labels_val = os.path.join(CONF.PATH.DATASETS, 'nuImages/labels/val')


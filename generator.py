import numpy as np
import tensorflow as tf
import functions as fun


class CustomDataGenerator(tf.keras.utils.Sequence):
    def __init__(self, annotation_dir, batch_size):
        self.annotation_dir = annotation_dir
        self.batch_size = batch_size
        self.data_size = 100000

    def __getitem__(self, idx):
        batch = np.arange(idx * self.batch_size, (idx + 1) * self.batch_size)
        annotation = fun.annotation_list(self.annotation_dir, batch)
        bbox = fun.convert2int(annotation[1])/64
        images = fun.images_list(annotation[0])/255
        type(images)
        cname = fun.label_dic(annotation[0])
        targets = {
            "class_label": cname,
            "bounding_box": bbox
        }
        return images, targets

    def __len__(self):
        return self.data_size // self.batch_size

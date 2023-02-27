import tensorflow as tf
import numpy as np
import functions as fun


class CustomDataGenerator(tf.keras.utils.Sequence):
    def __init__(self, val_dir, batch_size):
        self.val_dir = val_dir
        self.batch_size = batch_size
        self.data_size = 10000

    def __getitem__(self, idx):
        batch = np.arange(idx * self.batch_size, (idx + 1) * self.batch_size)
        annotation = fun.val_list(self.val_dir, batch)
        bbox = fun.convert2int(annotation[2])/64
        images = fun.np_val_images(annotation[0])/255
        cname = fun.label_dic(annotation[1])
        targets = {
            "class_label": cname,
            "bounding_box": bbox
        }
        return images, targets

    def __len__(self):
        return self.data_size // self.batch_size

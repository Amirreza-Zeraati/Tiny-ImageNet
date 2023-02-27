import model as model_maker
import generator as data_generator
import val_generator
import tensorflow as tf
import datetime

print('Building data generator...')
data = data_generator.CustomDataGenerator("data/tiny-imagenet-200/wnids.txt", 10)
val_data = val_generator.CustomDataGenerator("data/tiny-imagenet-200/val/val_annotations.txt", 10)

print('Building model...')
model = model_maker.mobile_v((64, 64, 3))
log_dir = "reports/fit/" + datetime.datetime.now().strftime("%Y%m%d-%H%M%S")
tensorboard_callback = tf.keras.callbacks.TensorBoard(log_dir=log_dir, histogram_freq=1)

print('Start training...')
H = model.fit(
    data,
    validation_data=val_data,
    epochs=35,
    verbose=1,
    callbacks=[tensorboard_callback])

model.save('models')

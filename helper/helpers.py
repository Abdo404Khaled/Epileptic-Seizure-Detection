import gc
import tensorflow as tf
import matplotlib.pyplot as plt
from tensorflow.keras.preprocessing.image import load_img, img_to_array

def load_and_preprocess_image(image_path, target_size=(64, 64)):
    image = load_img(image_path, target_size=target_size, color_mode='rgb')
    image = img_to_array(image)
    image = image / 255.0
    return image

def scheduler(epoch, lr):
    if epoch < 5:
        return lr
    else:
        return tf.keras.backend.get_value(lr * tf.math.exp(-0.1))

class GarbageCollectorCallback(tf.keras.callbacks.Callback):
    def on_epoch_end(self, epoch, logs=None):
        gc.collect()
        
def plot_and_save(history, metric, title, filename):
    plt.figure()
    plt.plot(history.history[metric])
    plt.plot(history.history['val_' + metric])
    plt.title(title)
    plt.ylabel(metric)
    plt.xlabel('epoch')
    plt.legend(['train', 'validation'], loc='upper left')
    plt.savefig(filename)
    plt.close()
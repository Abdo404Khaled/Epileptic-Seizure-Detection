from helper.helpers import *
from tensorflow.keras.callbacks import LearningRateScheduler, EarlyStopping # type: ignore

DATA_PATH = 'data/'

lr_scheduler = LearningRateScheduler(scheduler)
early_stopping = EarlyStopping(monitor='val_loss', patience=10, restore_best_weights=True)
gc_epoch = GarbageCollectorCallback()
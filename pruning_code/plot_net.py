import models
import tensorflow.compat.v1.keras.backend as K
K.set_image_data_format('channels_first')
#from tensorflow.compat.v1.keras.utils import plot_model
from tensorflow.keras.utils import plot_model
#import tensorflow.keras.utils.vis_utils
import matplotlib.pyplot as plt


model_used='EEGNet'        #'name of model, EEGNet or DeepCNN'
nb_classes=2
channels=16
samples=256

# Build Model
if model_used == 'EEGNet':
    model = models.EEGNet(nb_classes=nb_classes, Chans=channels, Samples=samples)
elif model_used == 'DeepCNN':
    model = models.DeepConvNet(nb_classes=nb_classes, Chans=channels, Samples=samples)
else:
    raise Exception('No such model:{}'.format(model_used))

model.compile(optimizer='Adam', loss='sparse_categorical_crossentropy', metrics=['acc', ])
weights = model.get_weights()
#plot_model(model, "EEGNet_complex.png",show_shapes=True,expand_nested=True)
print(model.summary())

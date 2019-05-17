import tensorflow as tf
import PIL.ImageOps
import io
import urllib
import random
import os
import sys

import utils.logging.tensorboard_utils as tensorboard_utils
import utils.image_processing.image_utils as image_utils
import utils.configuration.settings as settings
import copy 


from tensorflow.keras.applications.vgg19 import VGG19
from tensorflow.keras.models import Model
import tensorflow.keras.backend as K
from tensorflow.keras.losses import mean_squared_error
from tensorflow.keras.callbacks import Callback, EarlyStopping, ModelCheckpoint
from tensorflow.keras import applications
from tensorflow.keras.utils import normalize

from tensorflow.keras.preprocessing.image import (
    ImageDataGenerator,
    array_to_img,
    img_to_array,
    load_img,
)

from sklearn.utils import shuffle

from skimage.color import rgb2gray

from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import (
    Conv2D,
    PReLU,
    Dense,
    Reshape,
    Flatten,
    Activation,
    ZeroPadding2D,
)
from sklearn.model_selection import train_test_split
import numpy as np
from abc import ABC, abstractmethod
from os.path import isfile

tf.enable_eager_execution()
config = tf.ConfigProto(
    allow_soft_placement=True, device_count={"GPU": 1, "CPU": 1}
)  # Laptop has 1080 with 2560 CUDA cores and 11 CPU Cores
sess = tf.Session(config=config)
tf.keras.backend.set_session(sess)

with tf.device("/gpu:0"):

    # I may use this later... 
    # class VGG_LOSS(object):
    #     def __init__(self, image_shape):
            
    #         self.image_shape = image_shape

    #     # computes VGG loss or content loss
    #     def vgg_loss(self, y_true, y_pred):
        
    #         vgg19 = VGG19(include_top=False, weights='imagenet', input_shape=self.image_shape)
    #         vgg19.trainable = False
    #         # Make trainable as False
    #         for l in vgg19.layers:
    #             l.trainable = False
    #         model = Model(inputs=vgg19.input, outputs=vgg19.get_layer('block5_conv4').output)
    #         model.trainable = False
        
    #         return K.mean(K.square(y_true - y_pred))

    class networkBase:
        
        model = Sequential()

        def getFileName(self):
            return f"saved_models/{ type(self).__name__ }.h5"

        def train(self, X_train, y_train, epochs=100):

            X_train = np.array(X_train, dtype=np.float32)
            y_train = np.array(y_train, dtype=np.float32)

            callBacks = []

            # checkpoint = ModelCheckpoint(self.getFileName(), monitor="val_acc", verbose=1, save_best_only=True, mode="max")

            # earlyStoppingCritera = EarlyStopping(
            #     monitor="val_acc",
            #     min_delta=0.01,
            #     patience=2,
            #     verbose=1,
            #     mode="auto",
            # )

            #callBacks.append(earlyStoppingCritera)
            #callBacks.append(checkpoint)

            # with tf.device("/gpu:0"):
            for iEpoch in range(epochs):
                print(f"{ type(self).__name__} Epoch {iEpoch + 1}/{epochs}")
                self.model.fit(X_train, y_train,
                validation_split=0.35,
                callbacks=callBacks)

        def predict(self, X):

            return self.model.predict(X)

        def compile(self):
            self.model.compile(optimizer="adam", loss="mean_squared_error", metrics=["accuracy"])
            
            # # load any existing models.
            # if isfile(self.getFileName()):
            #     self.model.load_weights(self.getFileName())

    class generativeNetwork(networkBase):
        def __init__(self, imageWidth, imageHeight, isColour=True):

            # supports three channels for rgb else one for greyscale
            colourChannels = 3 if isColour else 1

            # we can infer our input tensor shape from the data.
            inputShape = (imageWidth, imageHeight, colourChannels)

            self.Layers = [
                ZeroPadding2D(
                    padding=(0, 0), data_format=None, name="generativeNetwork"),
                Conv2D(5, (5, 5), padding="same"), #input_shape=inputShape),
                PReLU(alpha_initializer="zeros"),
                ZeroPadding2D(padding=(0, 0), data_format=None),
                Conv2D(3, (5, 5), padding="same", name="generative_ouput"),
                PReLU(alpha_initializer="zeros"),
            ]

            self.model = Sequential(self.Layers)
            self.compile()


            def compile(self):
                self.model.compile(optimizer="adam", loss=["mean_squared_error"], metrics=["accuracy"])
            
            #TODO: Implement this later if required.
            # def train(self, X_train, y_train, epochs=100):
            #     self.compile()
            #     #super.train(self, X_train, y_train, epochs=epochs)
       
            #     X_train = np.array(X, dtype=np.float32)
            #     y_train = np.array(y, dtype=np.float32)

            #     data_gen_args = dict(featurewise_center=True,
            #                         featurewise_std_normalization=True,
            #                         rotation_range=90,
            #                         width_shift_range=0.1,
            #                         height_shift_range=0.1,
            #                         zoom_range=0.2)

            #     X_datagen = ImageDataGenerator(**data_gen_args)
            #     y_datagen = ImageDataGenerator(**data_gen_args)

            #     datagen.fit(x_train)

            #     model.fit_generator(datagen.flow(x_train, y_train, batch_size=32),
            #                 steps_per_epoch=len(x_train) / 32, epochs=epochs)

    class discriminativeNetwork(networkBase):
        def __init__(self, imageWidth, imageHeight, isColour=True):

            # supports three channels for rgb else one for greyscale
            colourChannels = 3 if isColour else 1

            # we can infer our input tensor shape from the data.
            self.inputShape = (imageWidth, imageHeight, colourChannels)
            self.Layers = [
                ZeroPadding2D(padding=(0, 0), data_format=None, name="discriminative"),
                Conv2D(10, (3, 3), padding="same"),
                PReLU(alpha_initializer="zeros"),
                Flatten(),
                Dense(20),
                PReLU(alpha_initializer="zeros"),
                Dense(2),
                Activation("softmax"),
            ]

            self.model = Sequential(self.Layers)

            self.compile()

        def train(self, X_train, y_train, epochs=100):
            
            # adv network needs a shuffle before we start training.
            X_train, y_train = shuffle(X_train, y_train, random_state=0)

            # for layer in self.Layers:
            #     layer.trainable = True

            # self.compile()

            super().train(X_train, y_train, epochs)
     
        def compile(self):
            #self.model.compile(optimizer="adam", loss=[VGG_LOSS(self.inputShape).vgg_loss], metrics=["accuracy", "mse"])
            self.model.compile(optimizer="adam", loss=["binary_crossentropy"], metrics=["accuracy"])


    class adverserialNetwork(networkBase):
        def __init__(self, generativeNetwork, discriminativeNetwork):

            self.generativeNetwork = generativeNetwork
            self.discriminativeNetwork = copy.deepcopy(discriminativeNetwork)

            self.discriminativeNetwork.model.trainable = False
            for layer in self.discriminativeNetwork.Layers:
                layer.trainable = False

            self.Layers = (
                self.generativeNetwork.Layers + self.discriminativeNetwork.Layers
            )


            self.model = Sequential(self.Layers)

            self.discriminativeNetwork.compile()
            self.compile()


    genEpochs = 10
    disEpochs = 3
    advEpochs = 3

    imageWidth = 500
    imageHeight = 500

    X = []
    y = []
 
    genNet = generativeNetwork(imageWidth, imageHeight)
    disNet = discriminativeNetwork(imageWidth, imageHeight)
    advNet = adverserialNetwork(genNet, disNet)

    real = np.array([1, 0], dtype=np.float32)
    fake = np.array([0, 1], dtype=np.float32)

    # import utils.configuration.settings as settings
    # import utils.models.networks as networks
    X = []
    y = []


    if settings.DOWNLOAD_IMAGES:
        image_utils.download_images(settings.ALL_IMAGE_KEYWORDS)

    image_utils.remove_download_error_images('training_data')

    full_size_images = image_utils.load_images("training_data")


    for i in range(len(full_size_images)):
        image = full_size_images[i]
        imageAsArray = img_to_array(image)

        # has three colour dimensions of 500 X 500 (3 x 500 x 500 = 750000)
        image_is_colour = np.array(imageAsArray).size == 750000

        if image_is_colour:

            imageNormailized = [x / 255 for x in imageAsArray]

            thumbnail = image.resize((100, 100), PIL.Image.NEAREST)
            thumbnail = thumbnail.resize((500, 500), PIL.Image.NEAREST)
            
            # thumbnail = image.resize((100, 100), PIL.Image.LANCZOS)
            # thumbnail = thumbnail.resize((500, 500), PIL.Image.LANCZOS)

            thumbnailAsArray = img_to_array(thumbnail)
            thumbnailNormailized = [x / 255 for x in thumbnailAsArray]

            X.append(thumbnailNormailized)  # our expected input;
            y.append(imageNormailized)  # our expected output

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.33, random_state=42
    )

    random.seed = 42 # let's make sure we always get the same image. 
    rand = random.randint(0, len(X_test) - 1)

    array_to_img(X_test[rand]).save("./output/1. blury.jpg")
    array_to_img(y_test[rand]).save("./output/4. original.jpg")

 
    # genNet.load()
    # disNet.load()
    count = 0
    while True:
        count += 1
  
        dis_X = y_train.copy()
        dis_y = []
        adv_y = []
        
        # mark existing examples as real
        for i in range(len(dis_X)):
            dis_y.append(real)
            # we want to train our generator to fool the disc so we flip the target for training the adverserial network.
            adv_y.append(fake)
        
        # create an equal number of fake examples
        for i in range(len(dis_X)):
            fake_prediction = genNet.predict(np.array([X_train[i]]))[0]
            dis_X.append(fake_prediction)
            dis_y.append(fake)
            # we want to train our generator to fool the disc so we flip the target for training the adverserial network.
            adv_y.append(real)


        genNet.train(X_train, y_train, epochs=genEpochs)
        generated_image_as_array = genNet.predict(np.array([X_test[rand]]))
        array_to_img(generated_image_as_array.reshape((500, 500, 3))).save(
            "./output/2. Basic Generator.jpg"
        )

        disNet.train(dis_X, dis_y, disEpochs)
        advNet.train(dis_X, adv_y, advEpochs)

        # Let's just have a go at predicting the image.

        generated_image_as_array = genNet.predict(np.array([X_test[rand]]))

        array_to_img(generated_image_as_array.reshape((500, 500, 3))).save(
            f"./output/3. adversary.jpg"
        )

        # # save the training so that we can continue later if needed
        # genNet.save()
        # disNet.save()

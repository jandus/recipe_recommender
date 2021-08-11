from keras.optimizers import Adam
from tensorflow.python.keras.preprocessing.image import ImageDataGenerator
from tensorflow.python.keras.callbacks import LearningRateScheduler
from tensorflow.python.keras.layers import Flatten, Dense, Input, Conv2D, GlobalAvgPool2D, Activation
from tensorflow.python.keras.models import Model
from tensorflow.python.keras.preprocessing import image
from PIL import Image
import os
from tensorflow.python.keras.callbacks import ModelCheckpoint
from io import open
import json
import numpy as np
import warnings
from tensorflow.python import keras
from tensorflow.python.keras import layers
from tensorflow.python.keras.layers import Dense, Activation, Flatten, Conv2D, MaxPool2D, AvgPool2D, GlobalMaxPool2D, GlobalAvgPool2D, BatchNormalization, add, Input
#from tensorflow.python.keras.models import Model

def resnet_module(input, channel_depth, strided_pool=False ):
    residual_input = input
    stride = 1

    if(strided_pool):
        stride = 2
        residual_input = Conv2D(channel_depth, kernel_size=1, strides=stride, padding="same")(residual_input)
        residual_input = BatchNormalization()(residual_input)

    input = Conv2D(int(channel_depth/4), kernel_size=1, strides=stride, padding="same")(input)
    input = BatchNormalization()(input)
    input = Activation("relu")(input)

    input = Conv2D(int(channel_depth / 4), kernel_size=3, strides=1, padding="same")(input)
    input = BatchNormalization()(input)
    input = Activation("relu")(input)

    input = Conv2D(channel_depth, kernel_size=1, strides=1, padding="same")(input)
    input = BatchNormalization()(input)

    input = add([input, residual_input])
    input = Activation("relu")(input)

    return input



def resnet_first_block_first_module(input, channel_depth):
    residual_input = input
    stride = 1

    residual_input = Conv2D(channel_depth, kernel_size=1, strides=1, padding="same")(residual_input)
    residual_input = BatchNormalization()(residual_input)

    input = Conv2D(int(channel_depth/4), kernel_size=1, strides=stride, padding="same")(input)
    input = BatchNormalization()(input)
    input = Activation("relu")(input)

    input = Conv2D(int(channel_depth / 4), kernel_size=3, strides=stride, padding="same")(input)
    input = BatchNormalization()(input)
    input = Activation("relu")(input)

    input = Conv2D(channel_depth, kernel_size=1, strides=stride, padding="same")(input)
    input = BatchNormalization()(input)

    input = add([input, residual_input])
    input = Activation("relu")(input)

    return input


def resnet_block(input, channel_depth, num_layers, strided_pool_first = False ):
    for i in range(num_layers):
        pool = False
        if(i == 0 and strided_pool_first):
            pool = True
        input = resnet_module(input, channel_depth, strided_pool=pool)

    return input

def ResNet50(include_top=True, non_top_pooling=None, model_input=None, num_classes=1000, weights='imagenet', model_path=""):
    layers = [3,4,6,3]
    channel_depths = [256, 512, 1024, 2048]

    input_object = model_input


    output = Conv2D(64, kernel_size=7, strides=2, padding="same")(input_object)
    output = BatchNormalization()(output)
    output = Activation("relu")(output)

    output = MaxPool2D(pool_size=(3,3), strides=(2,2))(output)
    output = resnet_first_block_first_module(output, channel_depths[0])


    for i in range(4):
        channel_depth = channel_depths[i]
        num_layers = layers[i]

        strided_pool_first = True
        if(i == 0):
            strided_pool_first = False
            num_layers = num_layers - 1
        output = resnet_block(output, channel_depth=channel_depth, num_layers=num_layers, strided_pool_first=strided_pool_first)

    if(include_top):
        output = GlobalAvgPool2D(name="global_avg_pooling")(output)
        output = Dense(num_classes)(output)
        output = Activation("softmax")(output)
    else:
        if (non_top_pooling == "Average"):
            output = GlobalAvgPool2D()(output)
        elif (non_top_pooling == "Maximum"):
            output = GlobalMaxPool2D()(output)
        elif (non_top_pooling == None):
            pass

    model = Model(inputs=input_object, outputs=output)

    if(weights == "imagenet"):
        weights_path = model_path
        model.load_weights(weights_path)
    elif (weights == "trained"):
        weights_path = model_path
        model.load_weights(weights_path)

    return model

class ModelTraining:
    """
        This is the Model training class, that allows you to define a deep learning network
        from ResNet50.
        Once you instantiate this class, you must call:

        *
    """

    def __init__(self):
        self.__modelType = ""
        self.__use_pretrained_model = False
        self.__data_dir = ""
        self.__train_dir = ""
        self.__test_dir = ""
        self.__num_epochs = 10
        self.__trained_model_dir = ""
        self.__model_class_dir = ""
        self.__initial_learning_rate = 0.1 #1e-3
        self.__model_collection = []


    def setModelTypeAsResNet(self):
        """
         'setModelTypeAsResNet()' is used to set the model type to the ResNet model
                for the training instance object .
        :return:
        """
        self.__modelType = "resnet"

    def setDataDirectory(self, data_directory=""):
        """
                 'setDataDirectory()' is required to set the path to which the data/dataset to be used for
                 training is kept. The directory can have any name, but it must have 'train' and 'test'
                 sub-directory. In the 'train' and 'test' sub-directories, there must be sub-directories
                 with each having it's name corresponds to the name/label of the object whose images are
                to be kept. The structure of the 'test' and 'train' folder must be as follows:

                >> train >> class1 >> class1_train_images
                         >> class2 >> class2_train_images
                         >> class3 >> class3_train_images
                         >> class4 >> class4_train_images
                         >> class5 >> class5_train_images

                >> test >> class1 >> class1_test_images
                        >> class2 >> class2_test_images
                        >> class3 >> class3_test_images
                        >> class4 >> class4_test_images
                        >> class5 >> class5_test_images

                :return:
                """

        self.__data_dir = data_directory
        self.__train_dir = os.path.join(self.__data_dir, "food_images/train")
        self.__test_dir = os.path.join(self.__data_dir, "food_images/test")
        self.__trained_model_dir = os.path.join(self.__data_dir, "models/resnet50")
        self.__model_class_dir = os.path.join(self.__data_dir, "json")

    def lr_schedule(self, epoch):

        # Learning Rate Schedule


        lr = self.__initial_learning_rate
        total_epochs = self.__num_epochs

        check_1 = int(total_epochs * 0.9)
        check_2 = int(total_epochs * 0.8)
        check_3 = int(total_epochs * 0.6)
        check_4 = int(total_epochs * 0.4)

        if epoch > check_1:
            lr *= 1e-4
        elif epoch > check_2:
            lr *= 1e-3
        elif epoch > check_3:
            lr *= 1e-2
        elif epoch > check_4:
            lr *= 1e-1


        return lr




    def trainModel(self, num_objects, num_experiments=10, enhance_data=False, batch_size = 16, initial_learning_rate=1e-3, show_network_summary=False, training_image_size = 100):

        """
                 'trainModel()' function starts the actual training. It accepts the following values:
                 - num_objects , which is the number of classes present in the dataset that is to be used for training
                 - num_experiments , also known as epochs, it is the number of times the network will train on all the training dataset
                 - enhance_data (optional) , this is used to modify the dataset and create more instance of the training set to enhance the training result
                 - batch_size (optional) , due to memory constraints, the network trains on a batch at once, until all the training set is exhausted.
                                            The value is set to 32 by default, but can be increased or decreased depending on the meormory of the
                                            compute used for training. The batch_size is conventionally set to 16, 32, 64, 128.
                 - initial_learning_rate(optional) , this value is used to adjust the weights generated in the network. You rae advised
                                                     to keep this value as it is if you don't have deep understanding of this concept.
                 - show_network_summary(optional) , this value is used to show the structure of the network should you desire to see it.
                                                    Itis set to False by default
                 - training_image_size(optional) , this value is used to define the image size on which the model will be trained. The
                                            value is 224 by default and is kept at a minimum of 100.

                 *

                :param num_objects:
                :param num_experiments:
                :param enhance_data:
                :param batch_size:
                :param initial_learning_rate:
                :param show_network_summary:
                :param training_image_size:
                :return:
                """
        self.__num_epochs = num_experiments
        self.__initial_learning_rate = initial_learning_rate
        lr_scheduler = LearningRateScheduler(self.lr_schedule)


        num_classes = num_objects

        if(training_image_size < 100):
            warnings.warn("The specified training_image_size {} is less than 100. Hence the training_image_size will default to 100.".format(training_image_size))
            training_image_size = 100



        image_input = Input(shape=(training_image_size, training_image_size, 3))
        if (self.__modelType == "resnet"):
            model = ResNet50(weights="custom", num_classes=num_classes, model_input=image_input)

        optimizer = Adam(lr=self.__initial_learning_rate, decay=1e-4)
        model.compile(loss="categorical_crossentropy", optimizer=optimizer, metrics=["accuracy"])
        if (show_network_summary == True):
            model.summary()

        model_name = 'model_ex-{epoch:03d}_acc-{val_accuracy:03f}.h5'

        if not os.path.isdir(self.__trained_model_dir):
            os.makedirs(self.__trained_model_dir)

        if not os.path.isdir(self.__model_class_dir):
            os.makedirs(self.__model_class_dir)

        model_path = os.path.join(self.__trained_model_dir, model_name)

        checkpoint = ModelCheckpoint(filepath=model_path,
                                     monitor='val_accuracy',
                                     verbose=1,
                                     save_weights_only=True,
                                     save_freq='epoch')

        if (enhance_data == True):
            print("Using Enhanced Data Generation")

        height_shift = 0
        width_shift = 0
        if (enhance_data == True):
            height_shift = 0.1
            width_shift = 0.1

        train_datagen = ImageDataGenerator(
            rescale=1. / 255,
            horizontal_flip=enhance_data, height_shift_range=height_shift, width_shift_range=width_shift)

        test_datagen = ImageDataGenerator(
            rescale=1. / 255)

        train_generator = train_datagen.flow_from_directory(self.__train_dir, target_size=(training_image_size, training_image_size),
                                                            batch_size=batch_size,
                                                            class_mode="categorical")
        test_generator = test_datagen.flow_from_directory(self.__test_dir, target_size=(training_image_size, training_image_size),
                                                          batch_size=batch_size,
                                                          class_mode="categorical")

        class_indices = train_generator.class_indices
        class_json = {}
        for eachClass in class_indices:
            class_json[str(class_indices[eachClass])] = eachClass

        with open(os.path.join(self.__model_class_dir, "model_class.json"), "w+") as json_file:
            json.dump(class_json, json_file, indent=4, separators=(",", " : "),
                      ensure_ascii=True)
            json_file.close()
        print("JSON Mapping for the model classes saved to ", os.path.join(self.__model_class_dir, "model_class.json"))

        num_train = len(train_generator.filenames)
        num_test = len(test_generator.filenames)
        print("Number of experiments (Epochs) : ", self.__num_epochs)

        #
        model.fit_generator(train_generator, steps_per_epoch=int(num_train / batch_size), epochs=self.__num_epochs,
                            validation_data=test_generator,
                            validation_steps=int(num_test / batch_size), callbacks=[checkpoint, lr_scheduler])
                            #validation_steps=int(num_test / batch_size), callbacks=[lr_scheduler])

#create object of class
model_trainer = ModelTraining()
model_trainer.setModelTypeAsResNet()
model_trainer.setDataDirectory("../data")
model_trainer.trainModel(num_objects=77, num_experiments=5, enhance_data=True, batch_size=32, show_network_summary=True)

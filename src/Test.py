from keras.optimizers import Adam
from tensorflow.python.keras.preprocessing.image import ImageDataGenerator
from tensorflow.python.keras.callbacks import LearningRateScheduler
from tensorflow.python.keras.layers import Flatten, Dense, Input, Conv2D, GlobalAvgPool2D, Activation
from tensorflow.python.keras.models import Model
from tensorflow.python.keras.preprocessing import image
from PIL import Image
import os
import sys
from tensorflow.python.keras.callbacks import ModelCheckpoint
from io import open
import json
import numpy as np
import warnings
from tensorflow.python import keras
from tensorflow.python.keras import layers
from tensorflow.python.keras.layers import Dense, Activation, Flatten, Conv2D, MaxPool2D, AvgPool2D, GlobalMaxPool2D, GlobalAvgPool2D, BatchNormalization, add, Input

CLASS_INDEX = None

def preprocess_input(x):
    """Preprocesses a tensor encoding a batch of images.

    # Arguments
        x: input Numpy tensor, 4D.
        data_format: data format of the image tensor.

    # Returns
        Preprocessed tensor.
    """


    # 'RGB'->'BGR'
    x *= (1./255)

    return x

def decode_predictions(preds, top=5, model_json=""):

    global CLASS_INDEX

    if CLASS_INDEX is None:
        CLASS_INDEX = json.load(open(model_json))
    results = []
    for pred in preds:
        top_indices = pred.argsort()[-top:][::-1]
        for i in top_indices:
            each_result = []
            each_result.append(CLASS_INDEX[str(i)])
            each_result.append(pred[i])
            results.append(each_result)

    return results

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

class CustomImagePrediction:
    """
                This is the image prediction class for custom models trained with the 'ModelTraining' class. It provides support for 4 different models which are:
                 ResNet50, SqueezeNet, DenseNet121 and Inception V3. After instantiating this class, you can set it's properties and
                 make image predictions using it's pre-defined functions.

                 The following functions are required to be called before a prediction can be made
                 * setModelPath() , path to your custom model
                 * setJsonPath , , path to your custom model's corresponding JSON file
                 * At least of of the following and it must correspond to the model set in the setModelPath()
                  [setModelTypeAsSqueezeNet(), setModelTypeAsResNet(), setModelTypeAsDenseNet, setModelTypeAsInceptionV3]
                 * loadModel() [This must be called once only before making a prediction]

                 Once the above functions have been called, you can call the predictImage() function of the prediction instance
                 object at anytime to predict an image.
        """

    def __init__(self):
        self.__modelType = ""
        self.modelPath = ""
        self.jsonPath = ""
        self.numObjects = 13
        self.__modelLoaded = False
        self.__model_collection = []
        self.__input_image_size = 100

    def setModelPath(self, model_path):
        """
        'setModelPath()' function is required and is used to set the file path to the model adopted from the list of the
        available 4 model types. The model path must correspond to the model type set for the prediction instance object.

        :param model_path:
        :return:
        """
        self.modelPath = model_path

    def setJsonPath(self, model_json):
        """
        'setJsonPath()'

        :param model_path:
        :return:
        """
        self.jsonPath = model_json

    def setModelTypeAsResNet(self):
        """
         'setModelTypeAsResNet()' is used to set the model type to the ResNet model
                for the prediction instance object .
        :return:
        """
        self.__modelType = "resnet"

    def loadModel(self, prediction_speed="normal", num_objects=10):
        """
        'loadModel()' function is used to load the model structure into the program from the file path defined
        in the setModelPath() function. This function receives an optional value which is "prediction_speed".
        The value is used to reduce the time it takes to predict an image, down to about 50% of the normal time,
        with just slight changes or drop in prediction accuracy, depending on the nature of the image.
        * prediction_speed (optional); Acceptable values are "normal", "fast", "faster" and "fastest"

        :param prediction_speed :
        :return:
        """

        self.numObjects = num_objects

        if (prediction_speed == "normal"):
            self.__input_image_size = 100
        elif (prediction_speed == "fast"):
            self.__input_image_size = 160
        elif (prediction_speed == "faster"):
            self.__input_image_size = 120
        elif (prediction_speed == "fastest"):
            self.__input_image_size = 100

        if (self.__modelLoaded == False):

            image_input = Input(shape=(self.__input_image_size, self.__input_image_size, 3))

            if (self.__modelType == ""):
                raise ValueError("You must set a valid model type before loading the model.")

            elif (self.__modelType == "resnet"):
                #import numpy as np
                from tensorflow.python.keras.preprocessing import image
                #from resnet50 import ResNet50
                #from custom_utils import preprocess_input
                #from custom_utils import decode_predictions
                try:
                    model = ResNet50(model_path=self.modelPath, weights="trained", model_input=image_input, num_classes=self.numObjects)
                    self.__model_collection.append(model)
                    self.__modelLoaded = True
                except:
                    raise ValueError("You have specified an incorrect path to the ResNet model file.")

    def predictImage(self, image_input, result_count=1, input_type="file"):
        """
        'predictImage()' function is used to predict a given image by receiving the following arguments:
            * input_type (optional) , the type of input to be parsed. Acceptable values are "file", "array" and "stream"
            * image_input , file path/numpy array/image file stream of the image.
            * result_count (optional) , the number of predictions to be sent which must be whole numbers between
                1 and the number of classes present in the model

        This function returns 2 arrays namely 'prediction_results' and 'prediction_probabilities'. The 'prediction_results'
        contains possible objects classes arranged in descending of their percentage probabilities. The 'prediction_probabilities'
        contains the percentage probability of each object class. The position of each object class in the 'prediction_results'
        array corresponds with the positions of the percentage possibilities in the 'prediction_probabilities' array.


        :param input_type:
        :param image_input:
        :param result_count:
        :return prediction_results, prediction_probabilities:
        """
        prediction_results = []
        prediction_probabilities = []
        if (self.__modelLoaded == False):
            raise ValueError("You must call the loadModel() function before making predictions.")

        else:

            if (self.__modelType == "resnet"):

                model = self.__model_collection[0]

                #from custom_utils import preprocess_input
                #from custom_utils import decode_predictions
                if (input_type == "file"):
                    try:
                        image_to_predict = image.load_img(image_input, target_size=(
                        self.__input_image_size, self.__input_image_size))
                        image_to_predict = image.img_to_array(image_to_predict, data_format="channels_last")
                        image_to_predict = np.expand_dims(image_to_predict, axis=0)

                        image_to_predict = preprocess_input(image_to_predict)
                    except:
                        raise ValueError("You have set a path to an invalid image file.")
                elif (input_type == "array"):
                    try:
                        image_input = Image.fromarray(np.uint8(image_input))
                        image_input = image_input.resize((self.__input_image_size, self.__input_image_size))
                        image_input = np.expand_dims(image_input, axis=0)
                        image_to_predict = image_input.copy()
                        image_to_predict = np.asarray(image_to_predict, dtype=np.float64)
                        image_to_predict = preprocess_input(image_to_predict)
                    except:
                        raise ValueError("You have parsed in a wrong numpy array for the image")
                elif (input_type == "stream"):
                    try:
                        image_input = Image.open(image_input)
                        image_input = image_input.resize((self.__input_image_size, self.__input_image_size))
                        image_input = np.expand_dims(image_input, axis=0)
                        image_to_predict = image_input.copy()
                        image_to_predict = np.asarray(image_to_predict, dtype=np.float64)
                        image_to_predict = preprocess_input(image_to_predict)
                    except:
                        raise ValueError("You have parsed in a wrong stream for the image")

                prediction = model.predict(x=image_to_predict, steps=1)




                try:

                    predictiondata = decode_predictions(prediction, top=int(result_count), model_json=self.jsonPath)

                    for result in predictiondata:
                        prediction_results.append(str(result[0]))
                        prediction_probabilities.append(str(result[1] * 100))


                except:
                    raise ValueError("An error occured! Try again.")

                return prediction_results, prediction_probabilities


def main(argv):
    """
    Main function.
    """
    #create object  class
    execution_path = os.getcwd()

    prediction = CustomImagePrediction()
    prediction.setModelTypeAsResNet()
    prediction.setModelPath("../data/models/resnet50/model_ex-005_acc-0.991592.h5")
    prediction.setJsonPath("../data/json/model_class.json")
    prediction.loadModel(num_objects=77)

    predictions, probabilities = prediction.predictImage("../data/test_images/"+argv, result_count=131)

    for eachPrediction, eachProbability in zip(predictions, probabilities):
        if float(eachProbability) > 60:
            print(eachPrediction , " : " , eachProbability)


if __name__ == "__main__":
    main(sys.argv[1])

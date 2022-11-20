import tensorflow as tf
import numpy as np
import argparse
from PIL import Image
import time
import io
from numpy import asarray
from PIL import ImageOps
#import tflite_runtime.interpreter as tflite
#ref: https://www.tensorflow.org/lite/guide/get_started https://www.tensorflow.org/lite/guide/inference#load_and_run_a_model_in_python
#ref: https://github.com/lkk688/AndroidIntelligentApp/blob/main/pythonTFlite/tfliteclassify2.py

def testtfliteexport(saved_model_dir):
    converter = tf.lite.TFLiteConverter.from_saved_model(saved_model_dir)
    tflite_model = converter.convert()
    open("converted_model.tflite", "wb").write(tflite_model)

#https://www.tensorflow.org/lite/performance/model_optimization
def tflitequanexport(saved_model_dir):
    #post-training quantization quantizes weights from floating point to 8-bits of precision
    converter_int8 = tf.lite.TFLiteConverter.from_saved_model(saved_model_dir)
    converter_int8.optimizations = [tf.lite.Optimize.DEFAULT]

    #val_ds=None
    from TFClassifier.Datasetutil.TFdatasetutil import loadTFdataset
    train_ds, val_ds, class_names, imageshape = loadTFdataset('Cifar10', 'kerasdataset', '/Users/admin/Documents/GitHub/MultiModalClassifier/TFClassifier/outputs/images', 28, 28, 1)
    def representative_data_gen():
        for input_value, _ in val_ds.take(100):
            yield [input_value]
    
    converter_int8.representative_dataset = representative_data_gen
    #To require the converter to only output integer operations, one can specify:
    converter_int8.target_spec.supported_ops = [tf.lite.OpsSet.TFLITE_BUILTINS_INT8] #[tf.float16]

    tflite_model = converter_int8.convert()
    tflite_model_file = 'converted_model_quant.tflite'

    with open(tflite_model_file, "wb") as f:
        f.write(tflite_model)

#to ensure compatibility with integer only devices (such as 8-bit microcontrollers) and accelerators (such as the Coral Edge TPU), you can enforce full integer quantization for all ops including the input and output, by using the following steps:
def tflitequanintexport(saved_model_dir):
    #post-training quantization quantizes weights from floating point to 8-bits of precision
    converter_int8 = tf.lite.TFLiteConverter.from_saved_model(saved_model_dir)
    converter_int8.optimizations = [tf.lite.Optimize.DEFAULT]

    #val_ds=None
    from TFClassifier.Datasetutil.TFdatasetutil import loadTFdataset
    train_ds, val_ds, class_names, imageshape = loadTFdataset('Cifar10', 'kerasdataset', '/Users/admin/Documents/GitHub/MultiModalClassifier/TFClassifier/outputs/images', 28, 28, 1)
    def representative_data_gen():
        for input_value, _ in val_ds.take(100):
            yield [input_value]
    
    converter_int8.representative_dataset = representative_data_gen
    #To require the converter to only output integer operations, one can specify:
    converter_int8.target_spec.supported_ops = [tf.lite.OpsSet.TFLITE_BUILTINS_INT8]

    #New add compared with tflitequanexport to enforce full integer quantization for all ops including the input and output
    converter_int8.inference_input_type = tf.int8  # or tf.uint8
    converter_int8.inference_output_type = tf.int8  # or tf.uint8

    tflite_model = converter_int8.convert()
    tflite_model_file = 'converted_model_quantint.tflite'

    with open(tflite_model_file, "wb") as f:
        f.write(tflite_model)


def testtfliteinference(tflite_model_path):
    # Load the TFLite model and allocate tensors.
    interpreter = tf.lite.Interpreter(model_path=tflite_model_path)
    interpreter.allocate_tensors()

    # Get input and output tensors.
    input_details = interpreter.get_input_details()
    print("input_details",input_details)
    output_details = interpreter.get_output_details()
    print("output_details",output_details)

    # Test the model on random input data.
    input_shape = input_details[0]['shape']#[1, 180, 180, 3]

    floating_model = input_details[0]['dtype'] == np.float32

    #image_path='/home/lkk/Developer/MyRepo/MultiModalClassifier/tests/imgdata/sunflower.jpeg'
    image_path='/Users/admin/Documents/GitHub/MultiModalClassifier/semi-semitrailer-truck-tractor-highway.webp'
    img_height = input_shape[1] #180
    img_width = input_shape[2] #180
    print(img_height,img_width)

    if floating_model:
        input_data=loadimage(image_path, img_height, img_width)
    else:
        input_data=loadimageint(image_path, img_height, img_width)

    input_data = np.expand_dims(input_data, axis=3)
    print("\n\n\n\n\n\n\n\ninput_data.shape()",input_data.shape)

   
    #input_data=ImageOps.grayscale(input_data)
    tensor_index = input_details[0]['index']
    print("tttt",tensor_index)
    interpreter.set_tensor(tensor_index, input_data)

    interpreter.invoke()

    # The function `get_tensor()` returns a copy of the tensor data.
    # Use `tensor()` in order to get a pointer to the tensor.
    output_data = interpreter.get_tensor(output_details[0]['index'])
    print(output_data)
    output = np.squeeze(output_data) # or output_data[0]

    # If the model is quantized (uint8 data), then dequantize the results
    if output_details[0]['dtype'] == np.uint8:
        scale, zero_point = output_details[0]['quantization']
        output = scale * (output - zero_point)

    classindex = np.argmax(output, axis=-1)
    class_names = ['airplane', '	automobile', 'bird', 'cat', 'deer',
               'dog', 'frog', 'horse', 'ship', 'truck']
    print(class_names[classindex])

def loadimage(path, img_height, img_width):
    # load image
    # load image
    image = Image.open(path).resize((img_width, img_height))
    image=ImageOps.grayscale(image)

    image = np.array(image)
    #Convert uint8 to int8
    image = image - 127.0
    image = np.array(image, dtype=np.float32)
    print(np.min(image), np.max(image))#-128 127
    #input=image[np.newaxis, ...]
    # add N dim
    # input_data=image[np.newaxis,:, :]
    #input_data = np.expand_dims(image, axis=0)
    input_data = np.expand_dims(image, axis=0)

    print("\n\n\n\n\n\n\n\ninput_data.shape()",input_data.shape)
    return input_data

def loadimageint(path, img_height, img_width):
    # load image
    image = Image.open(path).resize((img_width, img_height))
    image=ImageOps.grayscale(image)
    print(image.shape)

    image = np.array(image)
    #Convert uint8 to int8
    image = image - 127.0
    image = np.array(image, dtype=np.int8)
    print(np.min(image), np.max(image))#-128 127
    #input=image[np.newaxis, ...]
    # add N dim
    #input_data = np.expand_dims(image, axis=0)
    input_data = np.expand_dims(image, axis=0)

    #input_data=input_data[np.newaxis,:, :]
    print("\n\n\n\n\n\n\n\ninput_data.shape()",input_data.shape)
    return input_data

if __name__ == '__main__':
    saved_model_dir = '/Users/admin/Documents/GitHub/MultiModalClassifier/TFClassifier/outputs/images'
    testtfliteexport(saved_model_dir)
    tflitequanexport(saved_model_dir)
    #tflitequanintexport(saved_model_dir)

    testtfliteinference("converted_model_quant.tflite")#"converted_model.tflite"
   # testtfliteinference("converted_model_quantint.tflite")

    

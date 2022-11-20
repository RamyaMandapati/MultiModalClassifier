import configargparse #pip install configargparse
import tensorflow as tf
import PIL
import PIL.Image
import matplotlib.pyplot as plt
import numpy as np
import time
import os
print(tf.__version__)

model = None 

parser = configargparse.ArgParser(description='myTFClassifyInference')
parser.add_argument('--data_name', type=str, default='cifar10',
                    help='data name: cifar10')
parser.add_argument('--data_type', default='kerasdataset', choices=['tfds', 'kerasdataset', 'imagefolder', 'TFrecord'],
                    help='the type of data')  # gs://cmpelkk_imagetest/*.tfrec
parser.add_argument('--data_path', type=str, default='/home/lkk/.keras/datasets/flower_photos',
                    help='path to get data')
parser.add_argument('--img_height', type=int, default=28,
                    help='resize to img height')
parser.add_argument('--img_width', type=int, default=28,
                    help='resize to img width')
# network
parser.add_argument('--model_name', default='cnnsimple1', choices=['cnnsimple1', 'cnnsimple2', 'cnnsimple3', 'cnnsimple4','mobilenetmodel1','create_simplemodelChange'],
                    help='the network')
parser.add_argument('--model_path', type=str, default='./outputs/flower_mobilenetmodel1_0630',
                    help='Model path.')
parser.add_argument('--GPU', type=bool, default=True,
                    help='use GPU')
parser.add_argument('--TPU', type=bool, default=False,
                    help='use GPU')
parser.add_argument('--MIXED_PRECISION', type=bool, default=False,
                    help='use MIXED_PRECISION')


args = parser.parse_args()

def loadsavedmodel(path):
    reconstructed_model = tf.keras.models.load_model(path)#"gs://cmpelkk_imagetest/saved_models/my_savedmodel202102")
    reconstructed_model.summary()
    return reconstructed_model

def tfgetimagearray(path, img_height, img_width):
    sunflower_url = "https://storage.googleapis.com/download.tensorflow.org/example_images/592px-Red_sunflower.jpg"
    sunflower_path = tf.keras.utils.get_file('Red_sunflower', origin=sunflower_url)

    img = tf.keras.preprocessing.image.load_img(
        sunflower_path, target_size=(img_height, img_width)
    )
    img_array = tf.keras.preprocessing.image.img_to_array(img) #(224, 224, 3)
    print('Data Type: %s' % img_array.dtype) #float32

    # normalize to the range 0-1
    img_array /= 255.0

    return img_array

def pltgetonlineimagearray(url):
    import matplotlib.pyplot as plt 
    from urllib.request import urlopen

    img = plt.imread(urlopen(url), format='JPG')
    plt.imshow(img)

def PILgetonlineimagearray(url, img_height, img_width):
    from PIL import Image
    from numpy import asarray

    from urllib.request import urlopen
    from PIL import Image,ImageOps

    image = Image.open(urlopen(url))

    # Size of the image in pixels (size of original image)
    # (This is not mandatory)
    width, height = image.size
    image=image.resize((img_width, img_height)) #(width, height) https://pillow.readthedocs.io/en/stable/reference/Image.html
    image=ImageOps.grayscale(image)
    # imgpath=requests.get(url, stream=True).raw
    # image = Image.open(imgpath)

    pixels = asarray(image) #to numpy array
    # confirm pixel range is 0-255
    print('Data Type: %s' % pixels.dtype)
    print('Min: %.3f, Max: %.3f' % (pixels.min(), pixels.max()))
    # convert from integers to floats
    pixels = pixels.astype('float32')
    # normalize to the range 0-1
    pixels /= 255.0
    # confirm the normalization
    print('Min: %.3f, Max: %.3f' % (pixels.min(), pixels.max()))

    print('Data Type: %s' % pixels.dtype) #float32
    
    return pixels

def inference(infermodel, img_np, class_names):
    img_array = tf.expand_dims(img_np, 0) # Create a batch (1, 224, 224, 3)

    predictions = infermodel.predict(img_array)#(1, 5)
    score = tf.nn.softmax(predictions[0])#Tensor: shape=(5,)
    print(score)
    print(
       "This image most likely belongs to {} with a {:.2f} percent confidence."
       .format(class_names[np.argmax(score)], 100 * np.max(score))
    )

def main():

    model=loadsavedmodel('/Users/admin/Documents/GitHub/MultiModalClassifier/TFClassifier/outputs/images')

    url='https://cdn.allbirds.com/image/upload/f_auto,q_auto,w_533,b_rgb:f5f5f5/cms/semi-semitrailer-truck-tractor-highway.webp'#truck
    img_array = PILgetonlineimagearray(url, args.img_height, args.img_width)

    pltgetonlineimagearray(url)
    #img_array = tfgetimagearray(args.data_path, args.img_height, args.img_width)

    class_names=['airplane', 'automobile', 'bird', 'cat', 'deer',
               'dog', 'frog', 'horse', 'ship', 'truck']
    inference(model, img_array, class_names)

if __name__ == '__main__':
    main()

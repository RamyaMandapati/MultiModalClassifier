
I chose Option 5 to deploy the model to TFServing
- 1. *Classifying model*
- 2. *TF Lite* 
- 3. *Deployed the model with TF Serving and using the REST API end point from it*
# Change made to the code.
Option chosen: TensorRT inference ,Uploading to TF Serving and TensorflowLite inference.
Procedure followed to setup the code:
- 1.Run the setup given in the repo in any virtual environment.
- 2.Make changes in the file to accomodate a different data set CIfar10 present in TensorFlow.Keras.
- 3.Train and generate the model, to see it's accuracy and time required to train, then save the model and convert it to TFLite model and apply inference to compare the accuracy and time.
- 4.Trained the mode, saved it and uploaded to TF Serving in  colab to execute the code present in apiserving.py file and then used the end point http://localhost:8501/v1/models/saved_model:predict to predict the image and the model predicted the data with most accurately.

## 1.TF Inference
- - Trained the Cifar10 dataset.
- - - Changed the parameters and class names to create an inference model, This is tested with a random image and the accuracy is around 75%.
<img width="1296" alt="Screen Shot 2022-11-20 at 10 31 12 AM" src="https://user-images.githubusercontent.com/101368541/202919551-0f8ee7d2-a16c-47c9-9fd1-9cc0acd92595.png">
<img width="656" alt="Screen Shot 2022-11-20 at 1 48 55 PM" src="https://user-images.githubusercontent.com/101368541/202928043-718f0405-86e5-471a-93c8-70e48afad82c.png">

- - - This model is saved and can be used to create TFLite model.

## 2.TF Lite 
- - - Lite models are used for the mobile devices and embedded devices where the model has to be more accurate with less size.
- - - Export TF lite would take the model saved from the previous step and then converts it to a lite model which is then used to make inferences.



- - - Screen shots and changes are available here
- - - Commits are as follows:


## 3.Serving with REST APIs
- - - Here I chose option 5 to upload the model to TF Serving.
### Steps followed:
- - - 1.Trained the classification model using the myTFDistributedTrainer.py, created a new model parameters in the CNNSimpleModels.py with name create_simplemodelTest2.
- - - 2.This would create an output folder inside output/fashion/1. We use this model with our API to make predictions.
- - - 3.we use the api http://localhost:8501/v1/models/saved_model:predict which will return the prediction.
- - - 5.apiserving.py has the code related to uploading the model with TF Serving. The model which is saved inside outputs/fashion/1 folder, is done in collab and the screen shots of it are below
<img width="334" alt="Screen Shot 2022-11-20 at 10 15 53 AM" src="https://user-images.githubusercontent.com/101368541/202919374-95e1fcf6-ee94-4248-94a9-6a899889f670.png">

<img width="877" alt="Screen Shot 2022-11-20 at 10 16 10 AM" src="https://user-images.githubusercontent.com/101368541/202919380-81bb07a1-38ea-4c42-ab4e-6c32fa533d38.png">


# MultiModalClassifier
This is a project repo for multi-modal deep learning classifier with popular models from Tensorflow and Pytorch. The goal of these baseline models is to provide a template to build on and can be a starting point for any new ideas, applications. If you want to learn basics of ML and DL, please refer this repo: https://github.com/lkk688/DeepDataMiningLearning.

# Package setup
Install this project in development mode
```bash
(venv38) MyRepo/MultiModalClassifier$ python setup.py develop
```
After the installation, the package "MultimodalClassifier==0.0.1" is installed in your virtual environment. You can check the import
```bash
>>> import TFClassifier
>>> import TFClassifier.Datasetutil
>>> import TFClassifier.Datasetutil.Visutil
```

If you went to uninstall the package, perform the following step
```bash
(venv38) lkk@cmpeengr276-All-Series:~/Developer/MyRepo/MultiModalClassifier$ python setup.py develop --uninstall
```

# Code organization
* [DatasetTools](./DatasetTools): common tools and code scripts for processing datasets
* [TFClassifier](./TFClassifier): Tensorflow-based classifier
  * [myTFDistributedTrainerv2.py](./TFClassifier/myTFDistributedTrainerv2.py): main training code
  * [myTFInference.py](./TFClassifier/myTFInference.py): main inference code
  * [exportTFlite.py](./TFClassifier/exportTFlite.py): convert form TF model to TFlite
* [TorchClassifier](./TorchClassifier): Pytorch-based classifier
  * [myTorchTrainer.py](./TorchClassifier/myTorchTrainer.py): Pytorch main training code
  * [myTorchEvaluator.py](./TorchClassifier/myTorchEvaluator.py): Pytorch model evaluation code 

# Tensorflow Lite
* Tensorflow lite guide [link](https://www.tensorflow.org/lite/guide)
* [exportTFlite](\TFClassifier\exportTFlite.py) file exports model to TFlite format.
  * testtfliteexport function exports the float format TFlite model
  * tflitequanexport function exports the TFlite model with post-training quantization, the model size can be reduced by
![image](https://user-images.githubusercontent.com/6676586/126202680-e2e53942-7951-418c-a461-99fd88d2c33e.png)
  * The converted quantized model won't be compatible with integer only devices (such as 8-bit microcontrollers) and accelerators (such as the Coral Edge TPU) because the input and output still remain float in order to have the same interface as the original float only model.
* To ensure compatibility with integer only devices (such as 8-bit microcontrollers) and accelerators (such as the Coral Edge TPU), we can enforce full integer quantization for all ops including the input and output, add the following code into function tflitequanintexport
```bash
converter_int8.inference_input_type = tf.int8  # or tf.uint8
converter_int8.inference_output_type = tf.int8  # or tf.uint8
```
  * The check of the floating model during inference will show false
```bash
floating_model = input_details[0]['dtype'] == np.float32
```
  * When preparing the image data for the int8 model, we need to conver the uint8 (0-255) image data to int8 (-128-127) via loadimageint function
  
# TensorRT inference
Check this [Colab](https://colab.research.google.com/drive/1aCbuLCWEuEpTVFDxA20xKPFW75FiZgK-?usp=sharing) (require SJSU google account) link to learn TensorRT inference for Tensorflow models.
Check these links for TensorRT inference for Pytorch models: 
* https://github.com/NVIDIA-AI-IOT/torch2trt
* https://learnopencv.com/how-to-convert-a-model-from-pytorch-to-tensorrt-and-speed-up-inference/
* https://developer.nvidia.com/blog/speeding-up-deep-learning-inference-using-tensorrt/

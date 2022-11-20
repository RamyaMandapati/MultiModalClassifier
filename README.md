
# Table of contents[Options selected]
- 1. *RT Inference*
- 2. *TF Lite* 
- 3. *Serving with REST APIs*
# Change made to the code.
Option chosen: TensorRT inference ,serving with REST APIs and TensorflowLite inference
Procedure followed to setup the code:
- 1.Created the virtual environment and ran the command as given in the repo.
- 2.Installed the required packages as some where not installed while setup process.
- 3.Had to create the output folder to store the models generated.
- 4.Made changes to python code to accomodate the training of new model for MNIST data.
- 5.Executed the code and generated the model and then used the model saved to convert to lite mode and madethe inference and then compared the accuracy and predictions made by the model.Below are some details regarding changes and results.
- 6.Trained a cnn model and saved so that it can be used for api serving and then uploaded that model to the drive and used colab to execute the code present in apiserving.py file and then used the end point http://localhost:8501/v1/models/saved_model:predict to predict the image and th model redicted the image class exactly. More details in below sections.
# Changes made at CNNsimplemodels.py, myTFInference.py, exportTFlite.py
## 1.RT Inference
- - Trained with Fashion MNIST model.
- - - Created a new model (inside CNNsimplemodels.py ) with different set of parameters to train the MNIST data and trained the model.
- - - Made necessary changes like the class name and other parameter for MNIST Model, and used ImageOps to convert the RGB to grey scale to make the immage array shape to (28,28,1). This creates the inference model, which is tested with the image of sneaker and the prediction came out to be good with more accuracy.
- - - The output model has been stored in output/fashion folder which will furthur be used for converting to the lite model and use for predictions.
![alt](https://github.com/vamshidhar199/MultiModalClassifier/blob/main/Screen%20Shot%202022-04-17%20at%205.43.48%20PM.png)
![alt](https://github.com/vamshidhar199/MultiModalClassifier/blob/main/Screen%20Shot%202022-04-17%20at%208.44.32%20PM.png)
## 2.TF Lite 
- - - Lite models are used for the mobile devices and embedded devices where the model has to be more accurate with less size.
- - - Export TF lite would take the model saved from the previous step and then converts it to a lite model which is then used to make inferences and thistime the converted model has predicted the sneakers as sandles whihc is close to the actual prediction.
- - -Below ae some screen shots regarding to it.
![alt](https://github.com/vamshidhar199/MultiModalClassifier/blob/main/Screen%20Shot%202022-04-18%20at%2012.11.38%20AM.png)

- - - Screen shots and changes have been pushed to my git repository.
- - - Commits are as follows:
- https://github.com/vamshidhar199/MultiModalClassifier/commit/29efad83c4414dbf8507fa7fc536b64bbe1ff6a6
- https://github.com/vamshidhar199/MultiModalClassifier/commit/91f35ed9117139dcf4f734392f1e3316ad4c61ad
- https://github.com/vamshidhar199/MultiModalClassifier/commit/2f45f5776ad7c69d76fbad5d169f2ca7b505abbc
- https://github.com/vamshidhar199/MultiModalClassifier/commit/3142548704cb40dce739d47e2e1164ca7fe38d67

## 3.Serving with REST APIs
- - - Serve a TensorFlow model with TensorFlow Serving.
### Steps followed:
- - - 1.Trained the classification model using the myTFDistributedTrainer.py, created a new model parameters in the CNNSimpleModels.py with name create_simplemodelTest2.
- - - 2.This would create an output folder inside output/fashion/1. We use this model with our API to make predictions.
- - - 3.The restfull API returns the JSON format and from there we need to extract the predictions, this repsonse is generated when we call http://localhost:8501/v1/models/saved_model:predict which will return the result in JSON format.
- - - 4.class_names[np.argmax(predictions[0])], np.argmax(predictions[0]), class_names[test_labels[0]], test_labels[0]), using this code we get the image out of. the array for predictions.
- - - 5.apiserving.py is the file whihc contains the necessary code, it has been executed in colab. The model which is saved inside outputs/fashion/1 folder, has been uploaded to my drive and executed the process of the serving by executing the code in colab the predictions are made as follows.
- - -![alt](https://github.com/vamshidhar199/MultiModalClassifier/blob/main/beforePred.png)
- - -![alt](https://github.com/vamshidhar199/MultiModalClassifier/blob/main/precitionUsingApi.png)


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

import configargparse #pip install configargparse
import tensorflow as tf
import PIL
import PIL.Image
import matplotlib.pyplot as plt
import numpy as np
import time
import os
print(tf.__version__)

from TFClassifier.Datasetutil.TFdatasetutil import loadTFdataset #loadtfds, loadkerasdataset, loadimagefolderdataset
from TFClassifier.myTFmodels.CNNsimplemodels import createCNNsimplemodel

model = None 
# import logger

parser = configargparse.ArgParser(description='myTFDistributedClassify')
parser.add_argument('--data_name', type=str, default='cifar10',
                    help='data name mnist, cifar10')
parser.add_argument('--data_type', default='kerasdataset', choices=['tfds', 'kerasdataset', 'imagefolder', 'TFrecord'],
                    help='the type of data')  # gs://cmpelkk_imagetest/*.tfrec
parser.add_argument('--data_path', type=str, default='/home/kaikai/.keras/datasets/flower_photos',
                    help='path to get data')
parser.add_argument('--save_path', type=str, default='./outputs/images',
                    help='path to save the model')
# network
parser.add_argument('--model_name', default='cnnsimple1', choices=['cnnsimple1', 'cnnsimple2','create_simplemodelChange','create_simplemodelTest1','create_simplemodelTest2'],
                    help='the network')
parser.add_argument('--arch', default='Tensorflow', choices=['Tensorflow', 'Pytorch'],
                    help='Model Name, default: Tensorflow.')
parser.add_argument('--batchsize', type=int, default=32,
                    help='batch size')
parser.add_argument('--epochs', type=int, default=12,
                    help='epochs')
parser.add_argument('--GPU', type=bool, default=True,
                    help='use GPU')
parser.add_argument('--TPU', type=bool, default=False,
                    help='use GPU')


args = parser.parse_args()





def create_model(strategy,numclasses, metricname='accuracy'):
    with strategy.scope():
        model = tf.keras.Sequential([
            tf.keras.layers.Conv2D(
                32, 3, activation='relu', input_shape=(28, 28, 1)),
            tf.keras.layers.MaxPooling2D(),
            tf.keras.layers.Flatten(),
            tf.keras.layers.Dense(64, activation='relu'),
            tf.keras.layers.Dense(numclasses)
        ])

        model.compile(loss=tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True),
                      optimizer=tf.keras.optimizers.Adam(),
                      metrics=[metricname])
        return model

# Function for decaying the learning rate.
# You can define any decay function you need.
def learningratefn(epoch):
    if epoch < 3:
        return 1e-3
    elif epoch >= 3 and epoch < 7:
        return 1e-4
    else:
        return 1e-5

# Callback for printing the LR at the end of each epoch.
class PrintLR(tf.keras.callbacks.Callback):
  def on_epoch_end(self, epoch, logs=None):
    print('\nLearning rate for epoch {} is {}'.format(epoch + 1,
                                                      model.optimizer.lr.numpy()))

def plot_history(history, metric, val_metric):
    acc = history.history[metric]
    val_acc = history.history[val_metric]

    loss = history.history['loss']
    val_loss = history.history['val_loss']

    epochs_range = range(len(acc))

    fig = plt.figure(figsize=(12, 8))
    plt.subplot(1, 2, 1)
    plt.plot(epochs_range, acc, label='Training Accuracy')
    plt.plot(epochs_range, val_acc, label='Validation Accuracy')
    plt.legend(loc='lower right')
    plt.ylim([min(plt.ylim()), 1])
    plt.grid(True)
    plt.title('Training and Validation Accuracy')

    plt.subplot(1, 2, 2)
    plt.plot(epochs_range, loss, label='Training Loss')
    plt.plot(epochs_range, val_loss, label='Validation Loss')
    plt.legend(loc='upper right')
    plt.grid(True)
    plt.title('Training and Validation Loss')
    plt.show()
    fig.savefig('traininghistory.pdf')

def main():
    print("Tensorflow Version: ", tf.__version__)
    print("Keras Version: ", tf.keras.__version__)

    if args.GPU:
        physical_devices = tf.config.list_physical_devices('GPU')
        num_gpu = len(physical_devices)
        print("Num GPUs:", num_gpu)
        # Create a MirroredStrategy object. This will handle distribution, and provides a context manager (tf.distribute.MirroredStrategy.scope) to build your model inside.
        strategy = tf.distribute.MirroredStrategy()  # for GPU or multi-GPU machines
        print("Number of accelerators: ", strategy.num_replicas_in_sync)
        BUFFER_SIZE = 10000
        BATCH_SIZE_PER_REPLICA = args.batchsize #64
        BATCH_SIZE = BATCH_SIZE_PER_REPLICA * strategy.num_replicas_in_sync


    train_ds, val_ds, class_names, imageshape = loadTFdataset(args.data_name, args.data_type)
    #train_ds, test_data, num_train_examples, num_test_examples, class_names=loadimagefolderdataset('flower')
    #train_data, test_data, num_train_examples, num_test_examples =loadkerasdataset('cifar10')
    #train_data, test_data, num_train_examples, num_test_examples = loadtfds(args.tfds_dataname)

    # train_data, test_data, num_train_examples, num_test_examples = loadtfds(
    #     args.tfds_dataname)

    #Tune for performance
    AUTOTUNE = tf.data.AUTOTUNE
    train_ds = train_ds.cache().shuffle(BUFFER_SIZE).prefetch(buffer_size=AUTOTUNE)
    val_ds = val_ds.cache().prefetch(buffer_size=AUTOTUNE)

    numclasses=len(class_names)

    global model
    metricname='accuracy'
    metrics=[metricname]
    with strategy.scope():
        model = createCNNsimplemodel(args.model_name, numclasses, imageshape, metrics)
    #model = create_model(strategy,numclasses, metricname)
    model.summary()

    # Define the checkpoint directory to store the checkpoints
    checkpoint_dir = args.save_path #'./training_checkpoints'
    # Name of the checkpoint files
    checkpoint_prefix = os.path.join(checkpoint_dir, "ckpt_{epoch}")

    callbacks = [
        tf.keras.callbacks.TensorBoard(log_dir=args.save_path),
        tf.keras.callbacks.ModelCheckpoint(filepath=checkpoint_prefix,
                                        save_weights_only=True),
        tf.keras.callbacks.LearningRateScheduler(learningratefn),
        PrintLR()
    ]

    #steps_per_epoch = num_train_examples // BATCH_SIZE  # 2936 is the length of train data
    #print("steps_per_epoch:", steps_per_epoch)
    start_time = time.time()
    #train the model 
    history = model.fit(train_ds, validation_data=val_ds, epochs=args.epochs, callbacks=callbacks)
    # history = model.fit(train_ds, validation_data=val_ds,
    #                     steps_per_epoch=steps_per_epoch, epochs=EPOCHS, callbacks=[lr_callback])

    valmetricname="val_"+metricname
    final_accuracy = history.history[valmetricname][-5:]
    print("FINAL ACCURACY MEAN-5: ", np.mean(final_accuracy))
    print("TRAINING TIME: ", time.time() - start_time, " sec")

    plot_history(history, metricname, valmetricname)

    #Export the graph and the variables to the platform-agnostic SavedModel format. After your model is saved, you can load it with or without the scope.
    model.save(args.save_path, save_format='tf')
    version = 1
    export_path = os.path.join(args.save_path, str(version))
    print('export_path = {}\n'.format(export_path))

    tf.keras.models.save_model(
            model,
            export_path,
            overwrite=True,
            include_optimizer=True,
            save_format=None,
            signatures=None,
            options=None
    )


    eval_loss, eval_acc = model.evaluate(val_ds)
    print('Eval loss: {}, Eval Accuracy: {}'.format(eval_loss, eval_acc))

if __name__ == '__main__':
    main()

### Import Statements ###
import os
import time

os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'
from tensorflow.keras import mixed_precision
mixed_precision.set_global_policy('mixed_float16')

from datetime import datetime
import numpy as np
import pandas as pd
import seaborn as sn
import tensorflow as tf
import matplotlib.pyplot as plt
from matplotlib import ticker
from sklearn.metrics import confusion_matrix
from tensorflow.keras import optimizers, losses, metrics
from tensorflow.keras.preprocessing import image_dataset_from_directory

from data import apply_augmentation
from vit_model import create_model


def complete_run(mlp_size=1024,
                 transformer_count=2,
                 linear_count=5,
                 class_weight=None,
                 optimizer=None,
                 patch_size=10,
                 num_attention=20):
    ### Helper Functions ###
    def create_dataset(image_path, IMAGE_HEIGHT, IMAGE_WIDTH, BATCH_SIZE):
        """Function to create a dataloader from an image path"""
        dataset = image_dataset_from_directory(image_path, image_size=(IMAGE_HEIGHT, IMAGE_WIDTH), batch_size=BATCH_SIZE)  # Reading the images from the path
        class_names = dataset.class_names  # Class names detected by the built-in function
        print("Current Directory:", image_path)
        print("Label names:", class_names)

        total_images_in_dataset = dataset.cardinality().numpy() * BATCH_SIZE  # Finding the total image count in the dataset
        print(f"Total number of images in dataset: {total_images_in_dataset}\n\n")

        return dataset

    def plot_and_save(training,
                      validation,
                      metric,
                      lower,
                      upper,
                      whole_number,
                      history):
        """
        Plots training vs. validation metrics (e.g., accuracy, loss)
        and saves the figure.
        """

        # Get the actual number of epochs from the history object
        real_epochs = len(history.history[training])

        factor = 100 if whole_number else 1

        plt.figure(figsize=(8, 6))

        # Plot training curve
        plt.plot(
            np.array(history.history[training]) * factor,
            label=f'Training {metric}',
            linewidth=2,
            linestyle='--',
            color='k'
        )

        # Plot validation curve
        plt.plot(
            np.array(history.history[validation]) * factor,
            label=f'Validation {metric}',
            linewidth=2,
            color='red'
        )

        plt.title(f'Training and Validation {metric}', fontsize=20)
        plt.xlabel('Epoch', fontsize=16)
        plt.ylabel(metric, fontsize=16)

        # Dynamically set the x-axis range
        plt.xlim(0, real_epochs - 1)

        step = 2
        plt.xticks(
            np.arange(0, real_epochs, step),
            np.arange(0, real_epochs, step),
            fontsize=14
        )

        plt.ylim(lower, upper)
        plt.yticks(fontsize=14)
        plt.legend(fontsize=12)

        plt.savefig(SAVE_FOLDER_PATH + f'/{SAVE_FOLDER_NAME}_{metric.lower()}_plot.png', dpi=600)
        plt.close()

    ### HyperParameters ###
    USE_HIGH_RES = True
    IMAGE_HEIGHT = 200 if USE_HIGH_RES else 100
    IMAGE_WIDTH = 600 if USE_HIGH_RES else 300

    SPLIT_SIX_CATEGORIES = False
    FOLDER_EXTENSION = "_1_6" if SPLIT_SIX_CATEGORIES else "_1_2"
    TRAINING_DATA_PATH = "../DATA/training_images" + FOLDER_EXTENSION
    TESTING_DATA_PATH = "../DATA/testing_images" + FOLDER_EXTENSION

    IMAGE_CHANNELS = 3
    NUM_CLASSES = 6 if SPLIT_SIX_CATEGORIES else 2
    CLASS_NAMES = list(range(1, NUM_CLASSES + 1))

    BATCH_SIZE = 40
    PATCH_SIZE = patch_size
    assert IMAGE_WIDTH % PATCH_SIZE == 0 and IMAGE_HEIGHT % PATCH_SIZE == 0, print("Image Width or Height is not divisible by patch size")
    NUM_PATCHES = (IMAGE_HEIGHT // PATCH_SIZE) * (IMAGE_WIDTH // PATCH_SIZE)

    EMBEDDING_DIMS = IMAGE_CHANNELS * PATCH_SIZE ** 2
    MLP_SIZE = mlp_size
    NUM_EPOCHS = 10
    TRANSFORMER_LAYER_NUMBER = transformer_count
    LINEAR_LAYER_NUMBER = linear_count
    NUM_ATTENTION_HEADS = num_attention

    LEARNING_RATE = 0.00003
    WEIGHT_DECAY = 0.0001
    DROPOUT_RATE = 0.5

    OPTIMIZER = optimizers.AdamW(learning_rate=LEARNING_RATE, weight_decay=WEIGHT_DECAY) if optimizer is None else optimizer
    LOSS_METRIC = losses.SparseCategoricalCrossentropy(from_logits=True)
    CLASS_WEIGHTS = {1: 1.0, 2: 1.0} if class_weight is None else class_weight

    MIN_DELTA = 0.01
    PATIENCE = 10
    BASELINE = 80
    START_FROM = 20

    SEED = 42

    param_log = "PARAMETERS THAT ARE USED IN THIS RUN\n\n"  # Creating a parameter log to add to final string and write to the log file

    # Adding all the parameters to the parameter log with the relevant labels 
    param_log += f"Image Shape -> ({IMAGE_HEIGHT}, {IMAGE_WIDTH}, {IMAGE_CHANNELS})\n"
    param_log += f"Batch Size -> {BATCH_SIZE}\n"
    param_log += f"Patch Size -> {PATCH_SIZE}\n"
    param_log += f"Number of Patches -> {NUM_PATCHES}\n\n"
    param_log += f"Embedding Dimensions -> {EMBEDDING_DIMS}\n"
    param_log += f"Multi-Layer Perceptron Size -> {MLP_SIZE}\n"
    param_log += f"Number Epochs -> {NUM_EPOCHS}\n"
    param_log += f"Number of Transformer Layers -> {TRANSFORMER_LAYER_NUMBER}\n"
    param_log += f"Number of Linear Layers -> {LINEAR_LAYER_NUMBER}\n"
    param_log += f"Number of Attention Heads -> {NUM_ATTENTION_HEADS}\n\n"
    param_log += f"Learning Rate -> {LEARNING_RATE}\n"
    param_log += f"Weight Decay -> {WEIGHT_DECAY}\n"
    param_log += f"Dropout Rate -> {DROPOUT_RATE}\n"
    param_log += f"Optimizer Configuration -> {OPTIMIZER.get_config()}\n"
    param_log += f"Loss Metric Configuration -> {LOSS_METRIC.get_config()}\n"
    param_log += f"Current Class Weights -> {CLASS_WEIGHTS}\n\n"
    param_log += f"Minimum Delta -> {MIN_DELTA}\n"
    param_log += f"Patience -> {PATIENCE}\n"
    param_log += f"Baseline -> {BASELINE}\n"
    param_log += f"Start From -> {START_FROM}\n\n"
    param_log += f"Random Seed -> {SEED}\n\n"
    param_log += "\n\n"

    print(param_log)

    parameter_dictionary = {}

    parameter_dictionary["TRANSFORMER_LAYER_NUMBER"] = TRANSFORMER_LAYER_NUMBER
    parameter_dictionary["EMBEDDING_DIMS"] = EMBEDDING_DIMS
    parameter_dictionary["NUM_ATTENTION_HEADS"] = NUM_ATTENTION_HEADS
    parameter_dictionary["MLP_SIZE"] = MLP_SIZE
    parameter_dictionary["NUM_CLASSES"] = NUM_CLASSES
    parameter_dictionary["DROPOUT_RATE"] = DROPOUT_RATE
    parameter_dictionary["LINEAR_LAYER_NUMBER"] = LINEAR_LAYER_NUMBER
    parameter_dictionary["NUM_PATCHES"] = NUM_PATCHES
    parameter_dictionary["PATCH_SIZE"] = PATCH_SIZE

    ### Creating a TensorFlow Dataset from the Balanced and Augmented Data ###
    apply_augmentation(SPLIT_SIX_CATEGORIES, SEED)

    train_dataset = create_dataset(TRAINING_DATA_PATH, IMAGE_HEIGHT, IMAGE_WIDTH, BATCH_SIZE)
    test_dataset = create_dataset(TESTING_DATA_PATH, IMAGE_HEIGHT, IMAGE_WIDTH, BATCH_SIZE)

    ### Creating the model using the function in the model.py ###
    vision_transformer_model = create_model(parameter_dictionary)

    ### Setting Up the Early Stopping ###
    early_stopping = tf.keras.callbacks.EarlyStopping(monitor='val_sparse_categorical_accuracy',
                                                      min_delta=MIN_DELTA,
                                                      patience=PATIENCE,
                                                      verbose=1,
                                                      restore_best_weights=True,
                                                      baseline=BASELINE,
                                                      start_from_epoch=START_FROM)

    ### Training the Model ###
    vision_transformer_model.compile(optimizer=OPTIMIZER, loss=LOSS_METRIC, metrics=[metrics.SparseCategoricalAccuracy()])

    training_start = time.time()
    history = vision_transformer_model.fit(train_dataset,
                                           validation_data=test_dataset,
                                           epochs=NUM_EPOCHS,
                                           class_weight=CLASS_WEIGHTS,
                                           callbacks=[early_stopping])
    training_end = time.time()

    training_time_seconds = int(training_end - training_start)
    hours, remainder = divmod(training_time_seconds, 3600)
    minutes, seconds = divmod(remainder, 60)
    training_time = f"{hours:02}:{minutes:02}:{seconds:02}"

    param_log += f'Training Duration (HH:MM:SS) -> {training_time}\n'

    ### Saving the Model Weights ###
    now = datetime.now()
    current_date = now.strftime("%Y%m%d_%Hh%Mm")
    current_folder_name = os.path.basename(os.getcwd())
    SAVE_FOLDER_NAME = f"{current_folder_name}_{current_date}"
    SAVE_FOLDER_PATH = f"../RESULTS/{SAVE_FOLDER_NAME}"
    os.makedirs(SAVE_FOLDER_PATH, exist_ok=True)

    vision_transformer_model.save_weights(f'{SAVE_FOLDER_PATH}/weights_{current_date}.weights.h5')

    ### Plotting Accuracy and Loss ###
    plot_and_save(training='sparse_categorical_accuracy', validation='val_sparse_categorical_accuracy', metric='Accuracy', lower=0, upper=100, whole_number=True, history=history)
    plot_and_save(training='loss', validation='val_loss', metric='Loss', lower=0, upper=5, whole_number=False, history=history)

    train_loss, train_acc = vision_transformer_model.evaluate(train_dataset)
    test_loss, test_acc = vision_transformer_model.evaluate(test_dataset)

    # Final evaluation results
    param_log += f"\nFinal Evaluation After Training\n"
    param_log += f"Training Loss: {train_loss:.2f}, Training Accuracy: {train_acc:.2f}\n"
    param_log += f"Testing Loss: {test_loss:.2f}, Testing Accuracy: {test_acc:.2f}\n"

    y_true, y_pred = [], []
    for images, labels in test_dataset:
        predictions = vision_transformer_model.predict(images)
        y_true.extend(labels.numpy())
        y_pred.extend(tf.argmax(predictions, axis=1).numpy())

    confusion_mtx = confusion_matrix(y_true, y_pred)
    confusion_mtx_percent = confusion_mtx.astype('float') / confusion_mtx.sum(axis=1)[:, np.newaxis]
    confusion_dataframe = pd.DataFrame(confusion_mtx_percent, index=CLASS_NAMES, columns=CLASS_NAMES)

    ### Creating and Saving the Confusion Matrix ###
    plt.figure(figsize=(10, 8))
    heatmap = sn.heatmap(confusion_dataframe, annot=True, cmap='crest', fmt='.2%', annot_kws={"size": 15, "weight": "bold"})  # Create the heatmap
    plt.xlabel('Predicted Label', fontsize=16, weight="bold")
    plt.ylabel('True Label', fontsize=16, weight="bold")
    plt.title('Confusion Matrix (Percentages)', fontsize=20, weight="bold")
    plt.tick_params(axis='both', which='major', labelsize=14)
    cbar = heatmap.collections[0].colorbar
    cbar.ax.tick_params(labelsize=15)
    cbar.ax.yaxis.set_major_formatter(ticker.FuncFormatter(lambda x, pos: f'{x * 100:.0f}%'))
    plt.savefig(SAVE_FOLDER_PATH + f'/{SAVE_FOLDER_NAME}_confusion_matrix.png', dpi=600)  # Save the figure

    ### Saving the log file ###
    LOG_FILE = f"{SAVE_FOLDER_PATH}/{SAVE_FOLDER_NAME}_LOGS.txt"
    with open(LOG_FILE, mode="w") as file:
        file.write(param_log)

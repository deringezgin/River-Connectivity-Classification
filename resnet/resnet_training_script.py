### Import Statements ###
import os, random, time
os.environ['TF_CPP_MIN_LOG_LEVEL']='3'
from tensorflow.keras import mixed_precision
mixed_precision.set_global_policy('mixed_float16')

import datetime as dt
from datetime import datetime
import numpy as np
import pandas as pd
import seaborn as sn
import tensorflow as tf
import matplotlib.pyplot as plt
from matplotlib import ticker
from sklearn.metrics import confusion_matrix
from tensorflow.keras import layers, models, optimizers, losses, Model, metrics, mixed_precision
from tensorflow.keras.preprocessing import image_dataset_from_directory
from data import apply_augmentation
from resnet_model import create_model


def complete_run(initial_filters=64, 
                 blocks_per_stage=[3,8,36,3],
                 linear_layer_number=1, 
                 batch_size=40, 
                 class_weight=None, 
                 optimizer=None):
    ### Helper Functions ###
    def print_label_counts(ds):
        """Function to print the image count for each label in the dataset."""
        class_counts = {}
        for images, labels_batch in ds:
            for label in labels_batch.numpy() + 1:
                class_counts[label] = class_counts.get(label, 0) + 1
    
        sorted_class_counts = dict(sorted(class_counts.items()))
        total_dataset = sum(class_counts.values())
    
        print("Class Counts:")
        for class_label, count in sorted_class_counts.items():
            print(f"Class {class_label}: {count} samples")
        print("The dataset has total of ", total_dataset, "images.")
    
    
    def print_and_append_str(string_var: str, text: str) -> str:
        """Simple Function to print a string to the screen and also add it to another input string."""
        print(text)
        string_var += text + "\n"
        return string_var
    
    
    def create_dataset(image_path, IMAGE_HEIGHT, IMAGE_WIDTH, BATCH_SIZE):
        """Function to create a dataloader from an image path"""
        dataset = image_dataset_from_directory(image_path, image_size=(IMAGE_HEIGHT, IMAGE_WIDTH), batch_size=BATCH_SIZE)
        class_names = dataset.class_names
        print("Current Directory:", image_path)
        print("Label names:", class_names)
        
        total_images_in_dataset = dataset.cardinality().numpy() * BATCH_SIZE
        print(f"Total number of images in dataset: {total_images_in_dataset}\n\n")
    
        return dataset
    
    
    def plot_and_save(training, validation, metric, lower, upper, whole_number, epochs, history):
        """Function to plot two different value sets in the same metric in a line graph"""
        factor = 100 if whole_number else 1 
        plt.figure(figsize=(8, 6))
        plt.plot(np.array(history.history[training]) * factor, label=f'Training {metric}', linewidth=3, linestyle='--', color='k')
        plt.plot(np.array(history.history[validation]) * factor, label=f'Validation {metric}', linewidth=3, color='red')
        plt.title(f'Training and Validation {metric}', fontsize=20)
        plt.xlabel('Epoch', fontsize=16)
        plt.ylabel(metric, fontsize=16)
        plt.ylim(lower, upper)
        plt.xlim(0, epochs-1)
        plt.xticks(np.arange(0, epochs, 10), np.arange(0, epochs, 10), fontsize=14)
        plt.yticks(fontsize=14)
        plt.legend(fontsize=12)
        plt.savefig(SAVE_FOLDER_PATH + f'/{SAVE_FOLDER_NAME}_{metric.lower()}_plot.png', dpi=600)
    
    
    ### HyperParameters ###
    USE_HIGH_RES = True
    FOLDER_NAME = "flow_600_200" if USE_HIGH_RES else "flow_300_100"
    DATA_PATH = "../DATA/image_data/" + FOLDER_NAME
    IMAGE_HEIGHT = 200 if USE_HIGH_RES else 100
    IMAGE_WIDTH = 600 if USE_HIGH_RES else 300
    
    SPLIT_SIX_CATEGORIES = False
    CATEGORY_COUNT = 6 if SPLIT_SIX_CATEGORIES else 2
    FOLDER_EXTENSION = "_1_6" if SPLIT_SIX_CATEGORIES else "_1_2"
    TRAINING_DATA_PATH = "../DATA/training_images" + FOLDER_EXTENSION
    TESTING_DATA_PATH = "../DATA/testing_images" + FOLDER_EXTENSION
    
    IMAGE_CHANNELS = 3
    IMAGE_SHAPE = (IMAGE_HEIGHT, IMAGE_WIDTH, IMAGE_CHANNELS)
    NUM_CLASSES = 6 if SPLIT_SIX_CATEGORIES else 2
    CLASS_NAMES = list(range(1, NUM_CLASSES + 1))
    
    BATCH_SIZE = batch_size

    FILTER_MULTIPLIER_PER_STAGE = [1, 2, 4, 8]
    STRIDES_PER_STAGE = [1, 2, 2, 2]
    KERNEL_SIZE = 3
    DROPOUT_RATE = 0.5
    
    NUM_EPOCHS = 100
    LEARNING_RATE = 0.00003
    WEIGHT_DECAY = 0.0001
    
    if optimizer is None:
        OPTIMIZER = optimizers.AdamW(learning_rate=LEARNING_RATE, weight_decay=WEIGHT_DECAY)
    else:
        OPTIMIZER = optimizer
        
    LOSS_METRIC = losses.SparseCategoricalCrossentropy(from_logits=True)
    if class_weight is None: CLASS_WEIGHTS = {1:1.0, 2:1.0}
    else: CLASS_WEIGHTS = class_weight

    MIN_DELTA = 0.01
    PATIENCE = 10
    BASELINE = 70
    START_FROM = 20
    
    SEED = 42
    
    # Parameter logging
    param_log = "PARAMETERS THAT ARE USED IN THIS RUN\n\n"
    param_log += f"Image Shape -> ({IMAGE_HEIGHT}, {IMAGE_WIDTH}, {IMAGE_CHANNELS})\n"
    param_log += f"Batch Size -> {BATCH_SIZE}\n"
    param_log += f"Initial Filters -> {initial_filters}\n"
    param_log += f"Blocks per Stage -> {blocks_per_stage}\n"
    param_log += f"Filter Multipliers per Stage -> {FILTER_MULTIPLIER_PER_STAGE}\n"
    param_log += f"Strides per Stage -> {STRIDES_PER_STAGE}\n"
    param_log += f"Kernel Size -> {KERNEL_SIZE}\n"
    param_log += f"Number of Linear Layers -> {linear_layer_number}\n\n"
    param_log += f"Number of Epochs -> {NUM_EPOCHS}\n"
    param_log += f"Learning Rate -> {LEARNING_RATE}\n"
    param_log += f"Weight Decay -> {WEIGHT_DECAY}\n"
    param_log += f"Dropout Rate -> {DROPOUT_RATE}\n"
    param_log += f"Optimizer Configuration -> {OPTIMIZER.get_config()}\n"
    param_log += f"Loss Metric Configuration -> {LOSS_METRIC.get_config()}\n"
    param_log += f"Current Class Weights -> {CLASS_WEIGHTS}\n\n"
    param_log += f"Minimum Delta -> {MIN_DELTA}\n"
    param_log += f"Patience -> {PATIENCE}\n"
    param_log == f"Baseline -> {BASELINE}\n"  # Note: This line seems like a mistake (==), keeping unchanged
    param_log += f"Start From -> {START_FROM}\n\n"
    param_log += f"Random Seed -> {SEED}\n\n\n"
    
    print(param_log)
    
    parameter_dictionary = {}
    parameter_dictionary["NUM_CLASSES"] = NUM_CLASSES
    parameter_dictionary["IMAGE_HEIGHT"] = IMAGE_HEIGHT
    parameter_dictionary["IMAGE_WIDTH"] = IMAGE_WIDTH
    parameter_dictionary["IMAGE_CHANNELS"] = IMAGE_CHANNELS
    parameter_dictionary["INITIAL_FILTERS"] = initial_filters
    parameter_dictionary["BLOCKS_PER_STAGE"] = blocks_per_stage
    parameter_dictionary["FILTER_MULTIPLIER_PER_STAGE"] = FILTER_MULTIPLIER_PER_STAGE
    parameter_dictionary["STRIDES_PER_STAGE"] = STRIDES_PER_STAGE
    parameter_dictionary["KERNEL_SIZE"] = KERNEL_SIZE
    parameter_dictionary["DROPOUT_RATE"] = DROPOUT_RATE
    parameter_dictionary["LINEAR_LAYER_NUMBER"] = linear_layer_number

    ### Creating a TensorFlow Dataset from the Balanced and Augmented Data ###
    apply_augmentation(SPLIT_SIX_CATEGORIES, SEED)
    
    train_dataset = create_dataset(TRAINING_DATA_PATH, IMAGE_HEIGHT, IMAGE_WIDTH, BATCH_SIZE)
    test_dataset = create_dataset(TESTING_DATA_PATH, IMAGE_HEIGHT, IMAGE_WIDTH, BATCH_SIZE)
    
    ### Creating the model ###
    resnet_model = create_model(parameter_dictionary)

    ### Setting Up the Early Stopping ### 
    early_stopping = tf.keras.callbacks.EarlyStopping(
        monitor='val_sparse_categorical_accuracy',
        min_delta=MIN_DELTA,
        patience=PATIENCE,
        verbose=1,
        restore_best_weights=True,
        baseline=BASELINE,
        start_from_epoch=START_FROM
    )
    
    ### Training the Model ###
    resnet_model.compile(optimizer=OPTIMIZER, loss=LOSS_METRIC, metrics=[metrics.SparseCategoricalAccuracy()])
    
    training_start = time.time()
    history = resnet_model.fit(train_dataset,
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
    SAVE_FOLDER_PATH = f"../RESULTS/RESNET/{SAVE_FOLDER_NAME}"
    os.makedirs(SAVE_FOLDER_PATH, exist_ok=True)
    
    resnet_model.save_weights(f'{SAVE_FOLDER_PATH}/weights_{current_date}.weights.h5')    
    
    ### Plotting Accuracy and Loss ###
    plot_and_save('sparse_categorical_accuracy', 'val_sparse_categorical_accuracy', "Accuracy",0, 100, True, NUM_EPOCHS, history)
    plot_and_save('loss', 'val_loss', "Loss", 0, 5, False, NUM_EPOCHS, history)
    
    train_loss, train_acc = resnet_model.evaluate(train_dataset)
    test_loss, test_acc = resnet_model.evaluate(test_dataset)
    
    # Final evaluation results
    param_log += f"\nFinal Evaluation After Training\n"
    param_log += f"Training Loss: {train_loss:.2f}, Training Accuracy: {train_acc:.2f}\n"
    param_log += f"Testing Loss: {test_loss:.2f}, Testing Accuracy: {test_acc:.2f}\n"
    
    y_true, y_pred = [], []
    for images, labels in test_dataset:
        predictions = resnet_model.predict(images)
        y_true.extend(labels.numpy())
        y_pred.extend(tf.argmax(predictions, axis=1).numpy())
    
    
    confusion_mtx = confusion_matrix(y_true, y_pred)
    confusion_mtx_percent = confusion_mtx.astype('float') / confusion_mtx.sum(axis=1)[:, np.newaxis]
    confusion_dataframe = pd.DataFrame(confusion_mtx_percent, index=CLASS_NAMES, columns=CLASS_NAMES)
    
    ### Creating and Saving the Confusion Matrix ###
    plt.figure(figsize=(10, 8))
    heatmap = sn.heatmap(confusion_dataframe, annot=True, cmap='crest', fmt='.2%', annot_kws={"size": 15, "weight":"bold"})
    plt.xlabel('Predicted Label', fontsize=16, weight="bold")
    plt.ylabel('True Label', fontsize=16, weight="bold")
    plt.title('Confusion Matrix (Percentages)', fontsize=20, weight="bold")
    plt.tick_params(axis='both', which='major', labelsize=14)
    cbar = heatmap.collections[0].colorbar
    cbar.ax.tick_params(labelsize=15)
    cbar.ax.yaxis.set_major_formatter(ticker.FuncFormatter(lambda x, pos: f'{x * 100:.0f}%'))
    plt.savefig(SAVE_FOLDER_PATH + f'/{SAVE_FOLDER_NAME}_confusion_matrix.png', dpi=600)
    
    ### Saving the log file ###
    LOG_FILE = f"{SAVE_FOLDER_PATH}/{SAVE_FOLDER_NAME}_LOGS.txt"
    with open(LOG_FILE, mode="w") as file:
        file.write(param_log)


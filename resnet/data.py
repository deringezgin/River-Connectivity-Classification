"""
Derin Gezgin

File for splitting the data into training and testin datasets
It takes care of the data augmentation and balancing the datasets
It is used by the main training script in each training run but has a fixed random seed which is also set

"""

# Import Statements

import os, random, shutil
import cv2 as cv
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt


### Helper Functions ###

def parse_site_id(fileName: str) -> int:
    """Function to parse the site-id from a file name"""
    site_block = fileName.split("_")[0]  # Retrieve the block of string that has the site-id in it
    site_id_str = site_block[1:]  # Excluding the first character (S) from the retrieved string
    assert site_id_str.isnumeric(), "Invalid site-ID!! The site ID should be an integer at the beginning of the file name coming after S\nA sample site-id is S23456."  # Being sure that the string is a valid integer
    return int(site_id_str)  # Returning the integer format


def balance_column(column, target_total, SEED):
    """Function to balance the values in a column as much as possible while maintaining the target_total"""
    random.seed(SEED)
    print("A random number for testing the seed -->", random.randint(0, 100))
    column_total, values = column[-1], column[:-1]  # Splitting the column values

    if target_total == column_total: return column  # If the column is already set, return the column

    new_values = [0] * len(values)  # Setting up a new list for the new values
    difference = target_total - sum(new_values)  # The difference between the target total and the sum of the current total
    while difference > 0:  # While we have a difference between the two values
        non_zero_indexes = [i for i, x in enumerate(values) if x != 0]
        non_zero = len(non_zero_indexes)
        target_value = difference // non_zero  # The target value if we equally split each non-zero index
        if target_value == 0 and difference != 0:  # If the difference is less than non_zero count, target_value will be 0 which is a problem. We cover this condition here
            for index in random.sample(non_zero_indexes, difference):  # Randomly sampling indexes from the non-zero indexes and appending 1 to those indexes
                new_values[index] += 1
                values[index] -= 1
        else:
            for i in range(len(values)):  # For each value in values
                value = values[i]  # The current value
                if value == 0:  # If the value is 0, jump to the next iteration
                    continue
                elif value < target_value:  # If the value is less than the target value, append what we have to the same index in the new_value list, subtract the same amount from the initial values
                    new_values[i] += value
                    values[i] -= value
                else:  # If the value is more than the target value, just append the target value and subtract the amount from the initial values
                    new_values[i] += target_value
                    values[i] -= target_value
        difference = target_total - sum(new_values)  # Update the difference to determine continue running to loop or not

    new_values.append(sum(new_values))  # Appending the total to the end
    return new_values  # Returning the new-column values as a list


def balance_and_pick_images(df, DATA_AUG_MULTIPLIER, SEED):
    """Function to Work on a DataFrame of Image Frequencies
    This function will take the dataframe, find the labels to augment and the labels to down-sample
    It'll assure that equal amount of images is taken from each site and label"""

    new_df = df.copy()  # Creating a copy dataframe
    last_row = new_df.iloc[-1].values.tolist()  # Extracting the last row as it has the totals
    min_image_count = min(last_row) * DATA_AUG_MULTIPLIER  # The min. number of images in a specific category * Data Augmentation is our lower-bound
    apply_augmentation = [index + 1 for index, value in enumerate(last_row) if value < min_image_count]  # We should apply augmentation if the image count is lower than min_count
    new_df[apply_augmentation] = new_df[apply_augmentation] * DATA_AUG_MULTIPLIER  # Viewing the augmented values

    for column in df.columns:  # For each column in the dataframe
        balanced_values = balance_column(new_df[column].tolist(), min_image_count, SEED)  # Apply the balancing function
        new_df[column] = pd.Series(balanced_values, index=new_df.index)  # Update the column
    new_df["Total"] = new_df.sum(axis=1)
    return new_df, apply_augmentation


def print_file_count(path: str):
    """Function to print the file count per each sub-folder in a large data folder.
    It also calculates the minimum file count in a subfolder and returns it"""

    print("\nCurrent File Counts in the folder", path)
    total_file_count = 0  # Variable to keep track of the total file count
    subfolder_paths = [f.path for f in os.scandir(path) if f.is_dir()]  # Storing the sub-folder paths
    subfolder_paths.sort(key=lambda x: int(''.join(filter(str.isdigit, os.path.basename(x)))))  # Sorting by label name
    counts = []  # List that stores the file counts. We need this to calculate the min count
    for subfolder in subfolder_paths:  # For each folder, calculate the file name, append it to the list and sum, and print it.
        folder_name = os.path.basename(subfolder)
        file_count = len(os.listdir(subfolder))
        counts.append(file_count)
        total_file_count += file_count
        print(f"Label --> {folder_name}: {file_count} files")

    print("We're working on total of ", total_file_count, "files.\n")


def horizontal_flip(image_path, target_dir, current_image_name):
    """Function to flip the image horizontally, and save the image into the target directory with a new name"""
    image = cv.imread(image_path)  # Reading the image
    if image is None: return  # If the image is None, return
    flipped_image = cv.flip(image, 1)  # Flip the image
    new_image_name = f"{os.path.splitext(current_image_name)[0]}_flipped{os.path.splitext(current_image_name)[1]}"  # Set the new image name
    flipped_file_path = os.path.join(target_dir, new_image_name)  # The new path for the image
    cv.imwrite(flipped_file_path, flipped_image)  # Save the image


def clahe_enhancement(image_path, target_dir, current_image_name):
    """Function to apply the clahe enhancement to an image and save the image into the target directory with a new name"""
    image = cv.imread(image_path)  # Reading the image
    if image is None: return  # If the image is None, return

    l, a, b = cv.split(cv.cvtColor(image, cv.COLOR_BGR2LAB))  # Converting to color space and splitting it

    clahe = cv.createCLAHE(clipLimit=2.0, tileGridSize=(4, 4))  # Applying the enhancement
    cl = clahe.apply(l)

    limg = cv.merge((cl, a, b))  # Merging the image

    final_image = cv.cvtColor(limg, cv.COLOR_LAB2BGR)  # Creating the final image

    # Setting up the new image name, and saving it to the correct file path
    new_image_name = f"{os.path.splitext(current_image_name)[0]}_enhanced{os.path.splitext(current_image_name)[1]}"
    enhanced_file_path = os.path.join(target_dir, new_image_name)
    cv.imwrite(enhanced_file_path, final_image)


def convert_to_dict(df):
    """Function to convert a Pandas DataFrame to a dictionary and also exclude the last column"""
    nested_dict = df.iloc[:-1].to_dict(orient='index')
    new_dict = {}
    for idx, data in nested_dict.items():
        for col, val in data.items():
            new_dict.setdefault(col, {})[idx] = val
    return new_dict


def create_new_directory(folder_name, CATEGORY_COUNT):
    """Function to check if a directory already exists, remove it and create a new one with the correct subfolder names
    The function also reorganizes the dictionary so that the keys of the outer dictionary are the column names of the DataFrame,
    and the values are dictionaries where the keys are the index values and the values are the cell values."""

    if os.path.exists(folder_name): shutil.rmtree(folder_name)
    os.makedirs(folder_name)  # Create the new folder
    for i in range(1, CATEGORY_COUNT + 1): os.makedirs(os.path.join(folder_name, str(i)))


def extract_subfolder_paths(parent_directory):
    """Function to extract the subfolder paths from a parent directory"""
    subfolder_paths = [f.path for f in os.scandir(parent_directory) if f.is_dir()]  # Getting each subfolder path from the parent directory
    subfolder_paths.sort(key=lambda x: int(os.path.basename(x)))  # Sorting them by basename
    return subfolder_paths


def apply_augmentation(SPLIT_SIX, SEED):
    random.seed(SEED)
    print("A random number for testing the seed -->", random.randint(0, 100))
    MAIN_DATA_FOLDER = "../DATA/"
    USE_HIGH_RES = True  # Switch to run the program on the higher resolution images.
    FOLDER_NAME = "flow_600_200" if USE_HIGH_RES else "flow_300_100"
    DATA_PATH = f"{MAIN_DATA_FOLDER}image_data/{FOLDER_NAME}"

    SPLIT_SIX_CATEGORIES = SPLIT_SIX
    CATEGORY_COUNT = 6 if SPLIT_SIX_CATEGORIES else 2
    FOLDER_EXTENSION = "_1_6" if SPLIT_SIX_CATEGORIES else "_1_2"
    TRAINING_FOLDER = f"{MAIN_DATA_FOLDER}training_images{FOLDER_EXTENSION}"
    TESTING_FOLDER = f"{MAIN_DATA_FOLDER}testing_images{FOLDER_EXTENSION}"
    COLUMNS = list(range(1, CATEGORY_COUNT + 1))

    # Setting up the folder we're going to save the .csv files and plots

    save_folder_name = f"{MAIN_DATA_FOLDER}site_based_data{FOLDER_EXTENSION}"  # Name of the new folder we're going to save the plots and .csv files
    folder_path = os.path.join(os.getcwd(), save_folder_name)  # Path to the new folder
    os.makedirs(folder_path, exist_ok=True)  # Create the directory if it doesn't already exist

    TESTING_SPLIT = 0.2
    DATA_AUG_MULTIPLIER = 4

    subfolder_paths = extract_subfolder_paths(DATA_PATH)

    site_label_dictionary = {}
    for subfolder in subfolder_paths:  # For each folder
        current_label = int(os.path.basename(subfolder))  # Get the integer folder name as the current label we're working on

        for file_name in os.listdir(subfolder):  # For each image
            current_site = parse_site_id(file_name)  # Get the current site name
            site_label_dictionary.setdefault(current_site, {})  # If the site doesn't exist, create an empty dictionary
            site_label_dictionary[current_site][current_label] = site_label_dictionary[current_site].get(current_label, 0) + 1  # Now increase the counter in the specific label of the current site
    if not SPLIT_SIX_CATEGORIES:
        for site in site_label_dictionary.keys():
            new_labels = {1: 0, 2: 0}

            for label, count in site_label_dictionary[site].items():
                if label in [1, 2, 3]:
                    new_labels[1] += count
                elif label in [4, 5, 6]:
                    new_labels[2] += count
            site_label_dictionary[site] = new_labels

    # Create a DataFrame from the dictionary
    df = pd.DataFrame(site_label_dictionary).T.fillna(0).astype(int)
    df = df.reindex(columns=COLUMNS)

    df.loc[len(df)] = df.sum()  # Add a "Total" row at the end
    df["Total"] = df.sum(axis=1)  # Add a Total column at the end

    # Sort the DataFrame by the values in the "Total" column, excluding the last row
    sorted_df = df.iloc[:-1].sort_values(by='Total', ascending=False)
    df = pd.concat([sorted_df, df.iloc[[-1]]])

    # Add the 80 percent row to see the training counts if we decide to have 80% Training split
    last_row_values_for_training = df.iloc[-1, 0:CATEGORY_COUNT].values
    new_row_values_for_training = np.floor((1 - TESTING_SPLIT) * last_row_values_for_training).astype(int)
    new_row_values_for_training = np.append(new_row_values_for_training, np.sum(new_row_values_for_training))  # Add total of values excluding the label
    new_row_training = pd.DataFrame([new_row_values_for_training], columns=COLUMNS + ["Total"], index=[int((1 - TESTING_SPLIT) * 100)])
    df = pd.concat([df, new_row_training])

    # Calculate the difference between the second-last and the last row and add it to another row labeled 20 to see the testing amounts we'd have if we have a 20% testing split
    second_last_row_values_for_testing = df.iloc[-2, 0:CATEGORY_COUNT + 1].values
    last_row_values_for_testing = df.iloc[-1, 0:CATEGORY_COUNT + 1].values
    difference_values_for_testing = second_last_row_values_for_testing - last_row_values_for_testing
    new_row_for_testing = pd.DataFrame([difference_values_for_testing], columns=COLUMNS + ["Total"], index=[int(TESTING_SPLIT * 100)])
    df = pd.concat([df, new_row_for_testing])

    zero_counts = (df == 0).sum()  # Counting the number of zeros in each column
    sorted_columns = zero_counts.sort_values(ascending=False).index.tolist()[:-1]  # Sorting the columns by the number of zeros in descending order

    TARGET_TESTING_AMOUNTS = df.loc[int(TESTING_SPLIT * 100)].tolist()[:-1]  # Extracting the values from the row with index 20

    formatted_dict = df.head(-3).drop(columns="Total").to_dict(orient="index")  # Convert the DataFrame to the desired dictionary format

    for site, inner_dict in formatted_dict.items(): formatted_dict[site] = [inner_dict.get(i, 0) for i in range(1, CATEGORY_COUNT + 1)]

    picked_site_ids_for_testing = []
    PICKED_TESTING_TOTALS = [0] * CATEGORY_COUNT
    temp_dict = formatted_dict.copy()

    for label in sorted_columns:
        TARGET_INDEX = label - 1
        CURRENT_TESTING_AMOUNT = TARGET_TESTING_AMOUNTS[TARGET_INDEX]
        filtered_sorted_dict = {k: v for k, v in sorted(temp_dict.items(), key=lambda item: item[1][TARGET_INDEX]) if v[TARGET_INDEX] != 0}
        picked_site_ids_filtered = filtered_sorted_dict.keys()
        remaining_sites = filtered_sorted_dict.copy()
        for SITE_ID, IMAGE_COUNTS in filtered_sorted_dict.items():
            CURRENT_IMAGE_COUNT = PICKED_TESTING_TOTALS[TARGET_INDEX]
            if CURRENT_IMAGE_COUNT + IMAGE_COUNTS[TARGET_INDEX] < CURRENT_TESTING_AMOUNT:
                picked_site_ids_for_testing.append(SITE_ID)
                for i in range(len(PICKED_TESTING_TOTALS)): PICKED_TESTING_TOTALS[i] += IMAGE_COUNTS[i]
                del remaining_sites[SITE_ID]
            else:
                break
        temp_dict = {k: v for k, v in temp_dict.items() if k not in picked_site_ids_filtered}

    testing_difference = [n1 - n2 for n1, n2 in zip(TARGET_TESTING_AMOUNTS, PICKED_TESTING_TOTALS)]

    training_sites_df = df.drop(index=picked_site_ids_for_testing).iloc[:-3]
    training_sites_df.loc[len(training_sites_df)] = training_sites_df.sum()
    training_sites_df.to_csv(save_folder_name + "/training_sites_before_aug" + FOLDER_EXTENSION + ".csv", index=True)

    training_sites_df, train_augmentation = balance_and_pick_images(training_sites_df, DATA_AUG_MULTIPLIER, SEED)
    training_sites_df.to_csv(save_folder_name + "/training_sites_after_aug" + FOLDER_EXTENSION + ".csv", index=True)

    testing_sites_df = df.loc[picked_site_ids_for_testing]
    testing_sites_df.loc[len(testing_sites_df)] = testing_sites_df.sum()
    testing_sites_df.to_csv(save_folder_name + "/testing_sites_before_aug" + FOLDER_EXTENSION + ".csv", index=True)

    testing_sites_df, test_augmentation = balance_and_pick_images(testing_sites_df, DATA_AUG_MULTIPLIER, SEED)
    testing_sites_df.to_csv(save_folder_name + "/testing_sites_after_aug" + FOLDER_EXTENSION + ".csv", index=True)

    required_image_counts = pd.concat([training_sites_df.iloc[:-1], testing_sites_df.iloc[:-1]]).sort_index()  # Merging training and testing
    available_values = df.iloc[:-3].copy().sort_index()  # Only picking the required columns from the main dataframe

    difference_df = required_image_counts - available_values  # Finding the difference between two dataframes
    difference_dictionary = difference_df.to_dict()

    # Excluding values less than or equal to 0 to make the dictionary smaller and simpler
    filtered_difference_dict = {row_key: {col_key: val for col_key, val in row_val.items() if val > 0} for row_key, row_val in difference_dictionary.items() if row_key != 'Total'}

    current_working_dir = os.getcwd()  # Get the current working directory
    AUGMENTED_FOLDER = f"{MAIN_DATA_FOLDER}augmentation_folder{FOLDER_EXTENSION}"
    destination_dir = os.path.join(current_working_dir, AUGMENTED_FOLDER)  # Destination directory within the current working directory

    if os.path.exists(destination_dir): shutil.rmtree(destination_dir)

    if SPLIT_SIX_CATEGORIES:
        shutil.copytree(DATA_PATH, destination_dir)  # Copy the directory
    else:
        # Create the new subfolders in the destination directory
        new_subfolder_1 = os.path.join(destination_dir, '1')
        new_subfolder_2 = os.path.join(destination_dir, '2')
        os.makedirs(new_subfolder_1, exist_ok=True)
        os.makedirs(new_subfolder_2, exist_ok=True)

        # Copy files from subfolders 1, 2, 3 to the new subfolder named '1'
        for i in [1, 2, 3]:
            subfolder_path = os.path.join(DATA_PATH, str(i))
            for file_name in os.listdir(subfolder_path):
                source_file = os.path.join(subfolder_path, file_name)
                destination_file = os.path.join(new_subfolder_1, file_name)
                shutil.copy2(source_file, destination_file)

        # Copy files from subfolders 4, 5, 6 to the new subfolder named '2'
        for i in [4, 5, 6]:
            subfolder_path = os.path.join(DATA_PATH, str(i))
            for file_name in os.listdir(subfolder_path):
                source_file = os.path.join(subfolder_path, file_name)
                destination_file = os.path.join(new_subfolder_2, file_name)
                shutil.copy2(source_file, destination_file)

    augmentation_functions = [horizontal_flip, clahe_enhancement]  # Augmentation functions in a list
    for label, required_images in filtered_difference_dict.items():  # For each label and sub-dictionary of image counts per site
        target_directory = f"{AUGMENTED_FOLDER}/{label}"  # The target directory we'd save our images in
        for data_augmentation in augmentation_functions:  # For each data augmentation function
            for file in os.listdir(target_directory):  # For each file in the directory
                file_site = parse_site_id(file)  # Getting the file-site
                if (file_site in required_images) and (required_images[file_site] > 0):  # If the site-id is in the augmentation-list and we still require images to augment
                    image_path = os.path.join(target_directory, file)  # Exact path of the image
                    data_augmentation(image_path, target_directory, file)  # Apply the data augmentation
                    required_images[file_site] -= 1  # Decrease the required amount as we augmented for once

    augmented_subfolder_paths = extract_subfolder_paths(AUGMENTED_FOLDER)

    site_label_paths_dictionary = {}
    for augmented_subfolder in augmented_subfolder_paths:  # For each folder
        current_label = int(os.path.basename(augmented_subfolder))  # Get the integer folder name as the current label we're working on

        for file_name in os.listdir(augmented_subfolder):  # For each image
            current_site = parse_site_id(file_name)  # Get the current site name
            site_label_paths_dictionary.setdefault(current_site, {})  # If the site doesn't exist, create an empty dictionary
            site_label_paths_dictionary[current_site].setdefault(current_label, [])  # If the label doesn't exist, create an empty list
            site_label_paths_dictionary[current_site][current_label].append(os.path.join(augmented_subfolder, file_name))  # Append the full path to the list

    training_testing_dicts = [convert_to_dict(training_sites_df), convert_to_dict(testing_sites_df)]

    create_new_directory(TRAINING_FOLDER, CATEGORY_COUNT)
    create_new_directory(TESTING_FOLDER, CATEGORY_COUNT)

    augmented_site_label_dictionary = {}

    for subfolder in augmented_subfolder_paths:
        current_label = int(os.path.basename(subfolder))  # Get the integer folder name as the current label we're working on

        for file_name in os.listdir(subfolder):
            current_site = parse_site_id(file_name)  # Get the current site name
            current_site_dict = augmented_site_label_dictionary.setdefault(current_label, {})  # Get or create inner dictionary for current label
            current_site_dict.setdefault(current_site, []).append(os.path.join(subfolder, file_name))  # Append image path to site list

    # Saving the training and testing dictionaries in separate variables
    training_dict = training_testing_dicts[0]
    testing_dict = training_testing_dicts[1]

    for i in range(1, CATEGORY_COUNT + 1): # For each label
        # For the sake of readability, save the current directories, paths and site values in a variable
        current_training_directory = TRAINING_FOLDER + "/" + str(i)
        current_testing_directory = TESTING_FOLDER + "/" + str(i)
        current_augmented_paths = augmented_site_label_dictionary[i]
        current_training_sites = training_dict[i]
        current_testing_sites = testing_dict[i]

        for site, path_list in current_augmented_paths.items():  # For each site in the
            if site in current_training_sites:  # If the current site is a training site
                required_training_images = current_training_sites[site]  # The required image count for that label
                selected_training_image_paths = random.sample(path_list, required_training_images)  # Randomly picking that specific amount of images
                for image_path in selected_training_image_paths: shutil.copy(image_path, current_training_directory)  # Copying the picked images into the new directory

            elif site in current_testing_sites:  # If the current site is a testing site to the same stuff for the testing
                required_testing_images = current_testing_sites[site]
                selected_testing_image_paths = random.sample(path_list, required_testing_images)
                for image_path in selected_testing_image_paths: shutil.copy(image_path, current_testing_directory)
    
    if os.path.exists(destination_dir): shutil.rmtree(destination_dir)
    print("\nAugmentation and split process Done!\n")
    

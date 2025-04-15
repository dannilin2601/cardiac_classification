import random
import tensorflow as tf
import numpy as np
from tqdm import tqdm
import os
import cv2
import pandas as pd  
import gc
import shutil
from tensorflow.keras.utils import to_categorical
import psutil

seed = 42
tf.random.set_seed(seed)
np.random.seed(seed)
random.seed(seed)


# Path to the directory containing video files and folders
video_folder = '../working/'

# List all subdirectories (frame folders) in the directory
subdirectories = [subdir for subdir in os.listdir(video_folder) if os.path.isdir(os.path.join(video_folder, subdir))]

# Initialize the tqdm progress bar
pbar = tqdm(total=len(subdirectories), desc='Removing folders', unit='folder')

# Loop through each subdirectory (frame folder)
for subdir in subdirectories:
    folder_path = os.path.join(video_folder, subdir)
    
    # Delete the folder and its contents
    shutil.rmtree(folder_path)
    
    # Update the progress bar
    pbar.update(1)

# Close the progress bar
pbar.close()

print('Folder removal completed.')


# Path to the directory containing the extracted frame folders
frame_folders = '../working/'

# Remove all files and subdirectories from the specified directory
for filename in os.listdir(frame_folders):
    file_path = os.path.join(frame_folders, filename)
    try:
        if os.path.isfile(file_path):
            os.unlink(file_path)
        elif os.path.isdir(file_path):
            shutil.rmtree(file_path)
    except Exception as e:
        print(f"Error deleting {file_path}: {e}")
        
print("All files and subdirectories removed from", frame_folders)


# Path to the directory containing video files
video_folder = '../input/echonet/Videos'

amount_picture = 38

# List all video files in the directory
video_files = [file for file in os.listdir(video_folder) if file.endswith('.avi')]

# Shuffle the video files list
random.shuffle(video_files)

# Calculate 70% of the total number of video files
subset_count = int(1 * len(video_files))

# Take the first 70% of the shuffled list
video_files_subset = video_files[:subset_count]


# Initialize tqdm for the overall progress
pbar = tqdm(total=len(video_files_subset), desc='Processing Videos', unit='video')

# Loop through each video file in the 70% subset
for video_file in video_files_subset:
    # ... (rest of the code remains unchanged)

    video_path = os.path.join(video_folder, video_file)
    
    # Create a folder with the same name as the video file
    folder_name = os.path.splitext(video_file)[0]
    os.makedirs(folder_name, exist_ok=True)
    
    # Open the video file
    cap = cv2.VideoCapture(video_path)

    # Check if the video file was successfully opened
    if not cap.isOpened():
        print(f'Error opening video file: {video_path}')
        continue

    # Get total number of frames in the video
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    
    # Loop through the first 100 frames in the video file
    frame_count = 1
    for frame_count in range(amount_picture):
        # Read a frame from the video file
        ret, frame = cap.read()

        # Check if the frame was successfully read
        if not ret:
            break

        # Save the frame as an image file inside the folder
        frame_name = os.path.join(folder_name, f'frame_{frame_count:04d}.jpg')  # Use zero-padding for sorting
        cv2.imwrite(frame_name, frame)

    # Release the video file
    cap.release()
    
    # Update the overall progress bar
    pbar.update(1)

# Close the overall progress bar
pbar.close()

print('Frames extraction completed for half of the video files.')

# Path to the directory containing the extracted frame folders
frame_folders = '../working/'

# List all subdirectories (frame folders) in the directory
subdirectories = [subdir for subdir in os.listdir(frame_folders) if os.path.isdir(os.path.join(frame_folders, subdir))]

# Print the total number of subdirectories created
total_subdirectories = len(subdirectories)
print(f'Total subdirectories created: {total_subdirectories}')

# Path to the directory containing the extracted frame folders
frame_folders = '../working/'

# List all subdirectories (frame folders) in the directory
subdirectories = [subdir for subdir in os.listdir(frame_folders) if os.path.isdir(os.path.join(frame_folders, subdir))]

# Initialize a list to store folder names with fewer than 100 images
folders_less_than_100 = []

# Initialize variables to keep track of the maximum and minimum image counts
max_image_count = 0
min_image_count = float('inf')  # Initialize with a large value

# Loop through each subdirectory
for subdir in subdirectories:
    # Count the number of image files in the subdirectory
    image_count = len([file for file in os.listdir(os.path.join(frame_folders, subdir)) if file.endswith('.jpg')])
    
    if image_count < amount_picture:
        folders_less_than_100.append(subdir)
        
        # Update the maximum and minimum image counts
        max_image_count = max(max_image_count, image_count)
        min_image_count = min(min_image_count, image_count)

# Print the names of folders with fewer than 100 images
#print("\nFolders with fewer than 100 images:")
#for folder in folders_less_than_100:
    #print(folder)

# Print the total count of folders with fewer than 100 images
print(f"Total folders with fewer than 38 images: {len(folders_less_than_100)}")

# Print the maximum and minimum image counts from folders with fewer than 100 images
print(f"Maximum image count: {max_image_count}")
print(f"Minimum image count: {min_image_count}")


# Path to the directory containing the extracted frame folders
frame_folders = '../working/'

# Loop through each folder in the frame_folders directory
for folder_name in os.listdir(frame_folders):
    # Check if the folder has fewer than 100 images and remove it
    if folder_name in folders_less_than_100:
        folder_path = os.path.join(frame_folders, folder_name)
        if os.path.exists(folder_path):
            shutil.rmtree(folder_path)
            #print(f"Removed folder: {folder_name}")

print('Folders removal completed.')


# Count the total number of remaining folders
remaining_folders = len([folder for folder in os.listdir(frame_folders) if os.path.isdir(os.path.join(frame_folders, folder))])

print(f'Total folders remaining: {remaining_folders}')
print('Folders removal completed.')


# Get the available RAM in bytes
available_ram = psutil.virtual_memory().available

# Convert bytes to gigabytes
available_ram_gb = available_ram / (1024 ** 3)

print(f"Available RAM: {available_ram_gb:.2f} GB")

# Path to the CSV file
csv_file_path = '../input/filelist2/FileList 2.csv'

# Read the CSV file and create a DataFrame
data_df = pd.read_csv(csv_file_path,
                   sep = ';',
                   engine = 'python')

print(data_df.columns)

# Assuming 'data_df' is your DataFrame and 'frame_folders' is the directory containing the folders
# Initialize lists to store predictor data and labels
train_predictor = []
train_labels = []

pixel = 112
normalize_factor = 1  # To normalize the image data

# Loop through each folder in the directory
for folder_name in tqdm(os.listdir(frame_folders), desc='Processing Folders', unit='folder'):
    # Attempt to retrieve class label and split type for the current folder
    class_label_result = data_df.loc[data_df['FileName'] == folder_name, 'Class'].values
    split_result = data_df.loc[data_df['FileName'] == folder_name, 'Split'].values
    
    # Check if the retrieval was successful (i.e., the arrays are not empty)
    if len(class_label_result) > 0 and len(split_result) > 0:
        class_label = class_label_result[0]
        split = split_result[0]
    else:
        # Handle the case where the folder name was not found in the DataFrame
        print(f"Folder {folder_name} not found in DataFrame. Skipping.")
        continue
    
    # Construct the folder path
    folder_path = os.path.join(frame_folders, folder_name)
    
    # Initialize a list to store image data from the folder
    folder_images = []
    
    # Loop through image files in the folder
    for image_file in os.listdir(folder_path):
        image_path = os.path.join(folder_path, image_file)
        
        # Read the image in grayscale mode and convert to float32 for precision
        image = cv2.imread(image_path, cv2.IMREAD_GRAYSCALE).astype('float32')
        
        if image is None:
            continue  # If the image file cannot be read, skip it
        
        # Resize the image to 'pixel x pixel'
        image = cv2.resize(image, (pixel, pixel))
        
        # Normalize the image data
        image /= normalize_factor
        
        # Add the image to the list
        folder_images.append(image)
    
    # Check if folder_images is not empty before stacking
    if folder_images:
        # Convert the list of images into a NumPy array
        folder_data = np.stack(folder_images, axis=0)
        
        # Add the folder data and corresponding label to the appropriate dataset
        if split == 'TRAIN':
            train_predictor.append(folder_data)
            train_labels.append(class_label)
    else:
        print(f"No images found or processed in folder {folder_name}. Skipping.")
        continue


train_predictor = np.array(train_predictor)
train_predictor = train_predictor.reshape(7442, 38, pixel, pixel, 1)
np.save('../working/train_predictor_112x112.npy', train_predictor)

del train_predictor

# Suggests to the garbage collector to release unreferenced memory
gc.collect()

print('Train Predictor Tensor Deleted !')

train_labels = to_categorical(train_labels)

np.save('../working/train_labels_112x112.npy', train_labels)


# Initialize lists to store predictor data and labels

val_predictor = []
val_labels = []
test_predictor = []
test_labels = []

pixel = 112

# Loop through each folder in the directory
for folder_name in tqdm(os.listdir(frame_folders), desc='Processing Folders', unit='folder'):
    
    # Lookup the class label and split type corresponding to the folder name from the DataFrame
    class_label_result = data_df.loc[data_df['FileName'] == folder_name, 'Class'].values
    split_result = data_df.loc[data_df['FileName'] == folder_name, 'Split'].values
    
    # Check if the query returned any result before accessing
    if len(class_label_result) > 0 and len(split_result) > 0:
        class_label = class_label_result[0]
        split = split_result[0]
    else:
        # Handle the case where the folder name does not match any row in the DataFrame
        print(f'No matching record found for folder: {folder_name}')
        continue  # Skip to the next folder_name
    
    # Construct the folder path
    folder_path = os.path.join(frame_folders, folder_name)
    
    # Initialize a list to store image data from the folder
    folder_images = []
    
    # Loop through image files in the folder
    for image_file in os.listdir(folder_path):
        image_path = os.path.join(folder_path, image_file)
        image = cv2.imread(image_path, cv2.IMREAD_GRAYSCALE).astype('float32')
        
        # Resize the image to 28 by 28 pixels
        image = cv2.resize(image, (pixel, pixel))
        
        # Normalize the image
        image = image / 1
        
        # Add the 2D image to the list, no flattening
        folder_images.append(image)
    
    # Convert the list of images into a NumPy array
    if folder_images:  # Check if folder_images is not empty
        folder_data = np.stack(folder_images, axis=0)
        
        if split == 'VAL':
            val_predictor.append(folder_data)
            val_labels.append(class_label)
        elif split == 'TEST':
            test_predictor.append(folder_data)
            test_labels.append(class_label)



val_predictor = np.array(val_predictor)
val_labels = np.array(val_labels)
test_predictor = np.array(test_predictor)
test_labels = np.array(test_labels)

val_labels = to_categorical(val_labels)
test_labels = to_categorical(test_labels)

val_predictor = val_predictor.reshape(1279, 38, pixel, pixel,1)
test_predictor = test_predictor.reshape(1275, 38, pixel, pixel,1)


# Path to the directory containing video files and folders
video_folder = '../working/'

# List all subdirectories (frame folders) in the directory
subdirectories = [subdir for subdir in os.listdir(video_folder) if os.path.isdir(os.path.join(video_folder, subdir))]

# Initialize the tqdm progress bar
pbar = tqdm(total=len(subdirectories), desc='Removing folders', unit='folder')

# Loop through each subdirectory (frame folder)
for subdir in subdirectories:
    folder_path = os.path.join(video_folder, subdir)
    
    # Delete the folder and its contents
    shutil.rmtree(folder_path)
    
    # Update the progress bar
    pbar.update(1)

# Close the progress bar
pbar.close()

print('Folder removal completed.')



np.save('../working/val_predictor_112x112.npy', val_predictor)
np.save('../working/val_labels_112x112.npy', val_labels)
np.save('../working/test_predictor_112x112.npy', test_predictor)
np.save('../working/test_labels_112x112.npy', test_labels)
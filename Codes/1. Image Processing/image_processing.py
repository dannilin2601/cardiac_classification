import cv2
import numpy as np
import os
from PIL import Image
import matplotlib.pyplot as plt
from scipy.spatial import ConvexHull
from matplotlib.path import Path

def optical_flow(data):

    optical_flow_data = np.zeros((data.shape[0], data.shape[1] - 1, data.shape[2], data.shape[3], 2))

    # Loop through each sequence
    for i in range(data.shape[0]):
        # Loop through each frame in the sequence
        for j in range(1, data.shape[1]):  # Starting from 1 to have a previous frame
            prev_frame = data[i, j - 1, :, :, 0]
            next_frame = data[i, j, :, :, 0]

            # Calculate optical flow
            flow = cv2.calcOpticalFlowFarneback(prev_frame, next_frame, None, 0.5, 3, 15, 3, 5, 1.2, 0)

            # Store the horizontal and vertical components of the flow
            optical_flow_data[i, j - 1, :, :, 0] = flow[:, :, 0]  # Horizontal flow
            optical_flow_data[i, j - 1, :, :, 1] = flow[:, :, 1]  # Vertical flow
            
    return optical_flow_data

def load_images_from_directory(directory):
    image_arrays = []
    for filename in os.listdir(directory):
        if filename.endswith(".jpg") or filename.endswith(".png"):
            filepath = os.path.join(directory, filename)
            with Image.open(filepath) as img:
                img = img.convert("L")
                img_array = np.array(img)
                image_arrays.append(img_array)
    image_arrays = np.array(image_arrays)
    image_arrays = np.expand_dims(image_arrays, axis=-1)
    image_arrays = np.expand_dims(image_arrays, axis=0)
    return image_arrays

def create_convex_hull_mask(average_magnitude, image_size):
    y_points, x_points = np.where(average_magnitude > 3.5)
    if len(x_points) < 3:
        raise ValueError("Not enough points to form a convex hull.")
    points = np.column_stack((x_points, y_points))
    hull = ConvexHull(points)
    hull_points = points[hull.vertices]
    hull_path = Path(hull_points)
    xx, yy = np.meshgrid(np.arange(image_size[1]), np.arange(image_size[0]))
    grid_points = np.c_[xx.ravel(), yy.ravel()]
    mask_condition = hull_path.contains_points(grid_points).reshape(image_size)

    return mask_condition

def apply_mask_to_frames(input_dir, mask):
    masked_frames = []
    
    # Loop through each file in the input directory
    for filename in sorted(os.listdir(input_dir)):
        if filename.endswith(".jpg") or filename.endswith(".png"):
            filepath = os.path.join(input_dir, filename)
            
            # Open the image and convert it to grayscale
            im_input = Image.open(filepath).convert('L')
            im = np.array(im_input, dtype=float)
            
            # Apply the mask
            im[~mask] = 0
            
            # Append the masked frame to the list
            masked_frames.append(im)
    
    # Convert the list to a numpy array
    masked_frames = np.array(masked_frames)

    #print(f"Masked frames shape: {masked_frames.shape}")
    
    return masked_frames

def save_masked_frames(frames, output_dir):
    masked_frames = frames

    os.makedirs(output_dir, exist_ok=True)
    for i in range(masked_frames.shape[0]):
        frame = masked_frames[i]
        img = Image.fromarray(frame.astype('uint8'))
        img.save(os.path.join(output_dir, f'frame_{i:03d}.png'))

def process_directory(input_directory, output_directory):

    image_arrays = load_images_from_directory(input_directory)
    computed_optical_flow = optical_flow(image_arrays)
    computed_optical_flow = computed_optical_flow[0]
    magnitudes = np.sqrt(computed_optical_flow[..., 0]**2 + computed_optical_flow[..., 1]**2)
    average_magnitude = np.mean(magnitudes, axis=0)

    mask_condition = create_convex_hull_mask(average_magnitude, image_arrays.shape[2:4])

    #plt.figure(figsize=(10, 8))
    #plt.imshow(mask_condition, cmap='gray', origin='upper')
    #plt.title('Generated Mask')
    #plt.show()

    masked_frames = apply_mask_to_frames(input_directory, mask_condition)

    save_masked_frames(masked_frames, output_directory)

    print(f"Saved {masked_frames.shape[0]} cropped images to {output_directory}")

def main(input_base_directory, output_base_directory):

    for folder_name in os.listdir(input_base_directory):

        if folder_name != '.DS_Store':  # Skip .DS_Store files

            input_directory = os.path.join(input_base_directory, folder_name)
            output_directory = os.path.join(output_base_directory, folder_name[-2:])

            if not os.path.exists(output_directory):
                os.makedirs(output_directory)
                
            process_directory(input_directory, output_directory)

# Example usage
input_base_directory = "/Users/dannilin/Desktop/19072024_cropping - cópia/Fotos"
output_base_directory = "/Users/dannilin/Desktop/19072024_cropping - cópia/Processed_Videos_ordered"
main(input_base_directory, output_base_directory)
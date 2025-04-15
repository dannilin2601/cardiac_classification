from .utils.resnet_model import *
from .utils.echonet_pipeline import *

import random
import tensorflow as tf
import numpy as np
from sklearn.utils.class_weight import compute_class_weight
from keras.optimizers import Adam
from keras.layers import Dense, Input, concatenate
from keras.models import Model
import os
import matplotlib.pyplot as plt

seed = 42
np.random.seed(seed)
random.seed(seed)
tf.random.set_seed(seed)

train_predictor_path = '../13082024_56x56_unnomalized/13082024_train_predictor_56x56.npy'
train_labels_path = '../13082024_56x56_unnomalized/13082024_train_labels_56x56.npy'

val_predictor_path = '../13082024_56x56_unnomalized/13082024_val_predictor_56x56.npy'
val_labels_path = '../13082024_56x56_unnomalized/13082024_val_labels_56x56.npy'

test_predictor_path = '../13082024_56x56_unnomalized/13082024_test_predictor_56x56.npy'
test_labels_path = '../13082024_56x56_unnomalized/13082024_test_labels_56x56.npy'

train_predictor = np.load(train_predictor_path, mmap_mode='r')
train_labels = np.load(train_labels_path, mmap_mode='r')
val_predictor = np.load(val_predictor_path, mmap_mode='r')
val_labels = np.load(val_labels_path, mmap_mode='r')
test_predictor = np.load(test_predictor_path, mmap_mode='r')
test_labels = np.load(test_labels_path, mmap_mode='r')


# Convert one-hot encoded labels to single-dimensional labels
y_train_flat = np.argmax(train_labels, axis=1)

# Compute class weights
class_weights = compute_class_weight('balanced', classes=np.unique(y_train_flat), y=y_train_flat)
class_weights_dict = {i: class_weights[i] for i in range(len(class_weights))}

pixel = 28
num_outputs = 2  # Define the number of outputs here

# Instantiate the models for each branch
optical_flow_model = Resnet3DBuilder.build_resnet_18(input_shape=(37, pixel, pixel, 2), num_outputs=num_outputs)
image_model = Resnet3DBuilder.build_resnet_18(input_shape=(38, pixel, pixel, 1), num_outputs=num_outputs)

# Create input layers
optical_flow_input = optical_flow_model.input
image_input = image_model.input

# Assuming you want to remove the last layer and use the penultimate layer's output
optical_flow_output = optical_flow_model.layers[-2].output
image_output = image_model.layers[-2].output

# Concatenate the outputs of the two branches
combined = concatenate([image_output, optical_flow_output])

# Define the logits layer explicitly
logits = Dense(num_outputs, activation=None, name='logits')(combined)

# Add activation to the logits for final predictions
predictions = tf.keras.layers.Activation('sigmoid' if num_outputs == 1 else 'softmax', name='predictions')(logits)

# Create the dual-input model
model = Model(inputs=[image_input, optical_flow_input], outputs=predictions)

# Verify the model summary to ensure the logits layer exists
model.summary()

model.compile(optimizer=Adam(learning_rate=1e-4),
              loss='binary_crossentropy',  # or 'binary_crossentropy' if you have two classes
              metrics=['accuracy'])

epochs = 10

augmentation_factor = 1

BATCH_SIZE = 16
train_gen = batch_generator(train_predictor, train_labels, BATCH_SIZE, shuffle=True)
val_gen = batch_generator(val_predictor, val_labels, BATCH_SIZE, shuffle=True)

# Calculate steps per epoch for training and validation
# Assuming each part has an equal number of samples
train_samples_per_part = train_predictor.shape[0]*augmentation_factor  # Total number of training samples
val_samples_per_part = val_predictor.shape[0]*augmentation_factor   # Total number of validation samples
train_steps_per_part = int(np.ceil(train_samples_per_part / BATCH_SIZE))
val_steps_per_part = int(np.ceil(val_samples_per_part / BATCH_SIZE))

# Define the directory to store the weights
checkpoint_dir = "training_weights"
os.makedirs(checkpoint_dir, exist_ok=True)  # Ensure the directory exists

# Create the custom callback instance
custom_cp_callback = CustomModelCheckpoint(
    base_dir=checkpoint_dir,  # Base directory
    save_weights_only=True,
    verbose=1
)

history = model.fit(
    train_gen,  # Training data generator
    epochs=epochs,  # Number of epochs to train for
    steps_per_epoch=train_steps_per_part,  # Number of steps per epoch
    validation_data=val_gen,  # Validation data generator
    validation_steps=val_steps_per_part,
    shuffle=True, # Number of validation steps
    callbacks=[custom_cp_callback]  # List of callbacks to apply during training
)

model.save('my_last_model.h5')

train_accuracy = history.history['accuracy']
val_accuracy = history.history['val_accuracy']
epochs = range(1, len(train_accuracy) + 1)

plt.figure(figsize=(10, 6))
plt.plot(epochs, train_accuracy, label='Training Accuracy')
plt.plot(epochs, val_accuracy, label='Validation Accuracy')
plt.xlabel('Epochs')
plt.ylabel('Accuracy')
plt.title('Training and Validation Accuracy Over Epochs')
plt.legend()
plt.grid(True)
plt.show()

# Plot step accuracies
plt.figure(figsize=(10, 6))
plt.plot(custom_cp_callback.step_accuracies, label='Step Accuracy')
plt.xlabel('Steps')
plt.ylabel('Accuracy')
plt.title('Training Accuracy Per Step')
plt.legend()
plt.grid(True)
plt.show()

plot_confusion_matrix(train_predictor, train_labels, model)
plot_confusion_matrix(val_predictor, val_labels, model)
plot_confusion_matrix(test_predictor, test_labels, model)
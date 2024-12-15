import json
import tensorflow as tf
import numpy as np

from pathlib import Path
from sklearn.model_selection import train_test_split
from keras.preprocessing.image import load_img, img_to_array

from keras.layers import Input, Conv2D, Activation, BatchNormalization, MaxPooling2D, Flatten, Dense, Dropout
from keras.models import Model
from keras.optimizers import Adam
from keras.callbacks import EarlyStopping, ModelCheckpoint, TensorBoard

import argparse


def load_dataset(dataset_root, train_split=0.8, batch_size=8, random_state=42):
    # Load paths
    depth_paths = sorted([str(path) for path in dataset_root.iterdir() if 'depth_map' in path.stem])
    segmt_paths = sorted([str(path) for path in dataset_root.iterdir() if 'seg_image' in path.stem])
    label_paths = sorted([str(path) for path in dataset_root.iterdir() if path.suffix == '.json'])

    # Get splits
    combined = list(zip(depth_paths, segmt_paths, label_paths))
    (train_data, val_data) = train_test_split(combined, test_size=1-train_split, random_state=random_state)

    # Unzip the data back into separate lists
    (train_depth_paths, train_segmt_paths, train_label_paths) = zip(*train_data)
    (val_depth_paths, val_segmt_paths, val_label_paths) = zip(*val_data)

    # Create Train Dataset from paths
    train_depth_dataset = tf.data.Dataset.from_tensor_slices(np.array(train_depth_paths))
    train_segmt_dataset = tf.data.Dataset.from_tensor_slices(np.array(train_segmt_paths))
    train_label_dataset = tf.data.Dataset.from_tensor_slices(np.array(train_label_paths))

     # Create Train Dataset from paths
    valid_depth_dataset = tf.data.Dataset.from_tensor_slices(np.array(val_depth_paths))
    valid_segmt_dataset = tf.data.Dataset.from_tensor_slices(np.array(val_segmt_paths))
    valid_label_dataset = tf.data.Dataset.from_tensor_slices(np.array(val_label_paths))
    
    # Define loading functions
    def load_and_concatenate(depth_path, segmt_path):
        print(depth_path, type(depth_path))
        depth_array = img_to_array(load_img(depth_path, target_size=(60, 80)))/255.0
        segmt_array = img_to_array(load_img(segmt_path, target_size=(60, 80)))/255.0
        return np.concatenate([depth_array, segmt_array], axis=1)

    def get_label(label_path):
        with open(label_path, 'r') as file:
            label = json.load(file)
        phi, dobs, dc = (label['phi']/360, label['dobs']/1000 if label['dobs'] <= 1000 else 1000, label['dc']/5 if label['dc'] <= 5 else 5)
        return np.array([phi, dobs, dc])
    
    # Zip the datasets together
    train_dataset = tf.data.Dataset.zip((train_depth_dataset, train_segmt_dataset, train_label_dataset))
    train_dataset = train_dataset.map(lambda x, y, z: (load_and_concatenate(x, y), get_label(z)))
    train_dataset = train_dataset.batch(batch_size)

    # Zip the datasets together
    valid_dataset = tf.data.Dataset.zip((valid_depth_dataset, valid_segmt_dataset, valid_label_dataset))
    valid_dataset = valid_dataset.map(lambda x, y, z: (load_and_concatenate(x, y), get_label(z)))
    valid_dataset = valid_dataset.batch(batch_size)
    
    return train_dataset, valid_dataset

def build_model(input_size=(60, 80, 6), num_outputs=3, channels=(8, 16, 32)):
    inputs = Input(shape=input_size)
    x = inputs
    for ch in channels:
        x = Conv2D(ch, (3, 3), padding='same')(x)
        x = Activation('relu')(x)
        x = BatchNormalization(axis=-1)(x)
        x = MaxPooling2D(pool_size=(2, 2))(x)
    x = Flatten()(x)
    x = Dense(16)(x)
    x = Activation('relu')(x)
    x = BatchNormalization(axis=-1)(x)
    x = Dropout(0.5)(x)
    x = Dense(4)(x)
    x = Activation('relu')(x)
    outputs = Dense(num_outputs, activation='linear')(x)
    model = Model(inputs=inputs, outputs=outputs)
    return model

def get_arguments():
    arguments = argparse.ArgumentParser()
    arguments.add_argument('-m', '--mode', choices=['train', 'eval'], required=True)
    arguments.add_argument('-lr', '--learning_rate', default=0.01, type=float)
    arguments.add_argument('-o', '--output_path', type=str)
    arguments.add_argument('-d', '--dataset_root', type=str)
    arguments.add_argument('-e', '--epochs', type=int)
    arguments.add_argument('-ckpt', '--checkpoint_path', type=str, default=None)
    return arguments.parse_args()

def evaluate_model(model, valid_images, valid_labels):
    # Evaluate the model on the given dataset
    loss, accuracy = model.evaluate(valid_images, valid_labels)  # Get both loss and accuracy
    print(f"Evaluation Loss: {loss}")
    print(f"Evaluation Accuracy: {accuracy}")

if __name__ == "__main__":
    arguments = get_arguments()
    # Load Dataset
    train_dataset, valid_dataset = load_dataset(Path(arguments.dataset_root))
    # Build Model
    model = build_model()
    # Build Optimizer
    optimizer = Adam(lr=arguments.learning_rate, decay=0.001)
    # Compile
    model.compile(loss='mean_absolute_percentage_error', metrics=['accuracy'], optimizer=optimizer)
    if arguments.mode == 'train':
        # Early Stopping
        early_stopping = EarlyStopping(monitor='val_loss', patience=10)  # Monitor validation loss
        # Model Checkpoint
        checkpoint_filepath = f"{arguments.output_path}/best_model.h5"  # Path to save the best model
        model_checkpoint_callback = ModelCheckpoint(
            filepath=checkpoint_filepath,
            save_weights_only=False,  # Save the entire model (not just weights)
            monitor='val_loss',  # Monitor validation loss
            mode='min',  # 'min' because you want to minimize validation loss
            save_best_only=True  # Only save the best model
        )
        # TensorBoard
        log_dir = "logs/"  # You can customize the log directory
        tensorboard_callback = TensorBoard(log_dir=log_dir, histogram_freq=1)

        # Fit the model
        model.fit_generator(
            train_dataset,
            epochs=arguments.epochs, 
            validation_data=valid_dataset,
            callbacks=[early_stopping, model_checkpoint_callback, tensorboard_callback]
        )
    elif arguments.mode == 'eval':
        if not arguments.checkpoint:
            raise ValueError('You should specify the checkpoint to evaluate')
        # Load the saved model (assuming you saved it as 'best_model.h5')
        model.load_weights(arguments.checkpoint)  
        # Evaluate the model
        evaluate_model(model, valid_images, valid_labels)
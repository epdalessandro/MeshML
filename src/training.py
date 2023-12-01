import numpy as np
import tensorflow as tf
import matplotlib.pyplot as plt
from preprocess_inputs import read_data, get_split
from process_data import numInputs as numIn, numOutputs as numOut
from models.normal import normal
from models.model import paper
# from models.biggerH3 import biggerH3
# from models.biggerH1H2 import biggerH1H2
# from models.biggerHidden import biggerHidden
import sys
from os import path

# Hyperparameters
# batchSize = 2048 #TODO: uncomment
bufferSize = 8192
patience = 5

def train_model_fit(model, checkpoint_callback, patience_callback, batchSize, train_dataset_batches, val_dataset_batches):
    history = model.fit(train_dataset_batches, batch_size=batchSize, epochs=100, validation_data=val_dataset_batches, callbacks=[checkpoint_callback, patience_callback])
    return history

def train_model(model, train_dataset, val_dataset):
    # Metrics
    train_err_metric = tf.keras.metrics.MeanSquaredError()
    val_err_metric = tf.keras.metrics.MeanSquaredError()

    count_to_patience = 0
    global_val_loss =  float("inf")
    epoch = 1
    while count_to_patience < patience:  # TODO: go until loss starts to increase, not a static number of epochs
        print(f"Start of epoch {epoch}")

        for step, (x_batch, y_batch) in enumerate(train_dataset):
            with tf.GradientTape() as tape:
                train_logits = model(x_batch, training=True)
                train_loss = model.loss(y_batch, train_logits)
        
            gradient = tape.gradient(train_loss, model.trainable_weights)
            model.optimizer.apply_gradients(zip(gradient, model.trainable_weights))

            # Update training error metric
            train_err_metric.update_state(y_batch, train_logits)
    
            # if step % 100 == 0:
            #     print(f"Training loss for one batch at step {step} = {train_loss}")
            #     print(f"{(step+1) * batchSize} samples seen so far")
        
        # Get training results for this epoch
        training_err = train_err_metric.result()
        train_err_metric.reset_states()
        print(f"Training loss for epoch {epoch} = {train_loss}")
        print(f"Training error for epoch {epoch} = {training_err}")

        # Get validation results for this epoch
        for x_val_batch, y_val_batch in val_dataset:
            val_logits = model(x_val_batch, training=False)
            val_loss = model.loss(y_val_batch, val_logits)
            val_err_metric.update_state(y_val_batch, val_logits)

        validation_err = val_err_metric.result()
        val_err_metric.reset_states()
        print(f"Validation loss for epoch {epoch} = {val_loss}")
        print(f"Validation error for epoch {epoch} = {validation_err}")

        if(global_val_loss <= val_loss): # If our loss this epoch is greater than previous loss increment patience
            count_to_patience += 1
        else:
            global_val_loss = val_loss # Set our past loss to current loss
            count_to_patience = 0
        epoch += 1 # Increment epoch count

    return model

def test_model(model, test_dataset):
    test_acc_metric = tf.keras.metrics.Accuracy()

    for x_test_batch, y_test_batch in test_dataset:
        test_logits = model(x_test_batch, training=False)
        test_loss = model.loss(y_test_batch, test_logits)
        test_acc_metric.update_state(y_test_batch, test_logits)

    test_acc = test_acc_metric.result()
    print(f"Test loss = {test_loss}")
    print(f"Test accuracy = {test_acc}")

def test_model_evaluate(model, batchSize, x_test, y_test, modelName):
    results = model.evaluate(x_test, y_test, batch_size=batchSize)
    with open(modelName + str(batchSize) + "_data.csv", "a") as fileID:
        print("_____Evaluation Metrics_____", file=fileID)
        print("Mean: ", results[0], file=fileID)
        print("MSE: ", results[1], file=fileID)
        print("MAE: ", results[2], file=fileID)
        print("MAPE: ", results[3], file=fileID)

def plot_metrics(history, identifier, modelName):
    fig, [ax1, ax2, ax3] = plt.subplots(1,3)
    fig.suptitle("Training Metrics")
    ax1.plot(history.history["loss"], label="Loss")
    ax1.plot(history.history["val_loss"], label="Val Loss")
    ax2.plot(history.history["mean_squared_error"], label="MSE")
    ax3.plot(history.history["mean_absolute_percentage_error"], label="MAPE")
    ax1.set(xlabel="Epoch", ylabel="Loss")
    ax2.set(xlabel="Epoch", ylabel="Matrix Metrix MSE")
    ax3.set(xlabel="Epoch", ylabel="Matrix Metrix MAPE")
    ax1.legend()
    with open(modelName + identifier + "_data.csv", "a") as fileID:
        print("_____Training Metrics_____", file=fileID)
        print("Loss: ", history.history["loss"], file=fileID)
        print("Val Loss: ", history.history["val_loss"], file=fileID)
        print("MSE: ", history.history["mean_squared_error"], file=fileID)
        print("MAPE: ", history.history["mean_absolute_percentage_error"], file=fileID)
    plt.savefig(identifier + modelName + ".png")

def main():
    # Get Data
    processed_df = read_data(sys.argv[1])

    train, val, test = get_split(processed_df, train_split=0.8, val_split=0.1, test_split=0.1)

    # Split up the partitios into their inputs and outputs
    x_train, y_train = train[:,0:numIn], train[:,numIn:numIn+numOut]
    x_val, y_val = val[:,0:numIn], val[:,numIn:numIn+numOut]
    x_test, y_test = test[:,0:numIn], test[:,numIn:numIn+numOut]

    # Compile the partitions into a single dataset, shuffle and then batch them
    train_dataset = tf.data.Dataset.from_tensor_slices((x_train, y_train))
    val_dataset = tf.data.Dataset.from_tensor_slices((x_val, y_val))
    test_dataset = tf.data.Dataset.from_tensor_slices((x_test, y_test))

    batchSize = int(sys.argv[2])
    # Get a uniformly random shuffle of all elements
    train_dataset_batches = train_dataset.shuffle(buffer_size=np.shape(train)[0]).batch(batchSize, drop_remainder=False) 
    # Don't shuffle validation, might want to read in a specific order for debugging
    val_dataset_batches = val_dataset.batch(batchSize, drop_remainder=False)
    test_dataset_batches = test_dataset.shuffle(buffer_size=np.shape(test)[0]).batch(batchSize, drop_remainder=False)

    models = [normal, paper]#, biggerH1H2, biggerH3, biggerHidden]
    modelNames = ["normal", "paper"]#, "biggerH1H2", "biggerH3", "biggerHidden"]

    for modelIdx in range(len(models)):

        # checkpoint_path = "../checkpoints/" + modelNames[modelIdx] + "Models/cp-{epoch:04d}.ckpt"
        checkpoint_path = "../checkpoints/" + modelNames[modelIdx] + "BestWeights.ckpt"
        checkpoint_callback = tf.keras.callbacks.ModelCheckpoint(filepath=checkpoint_path, verbose=1, 
                                                        save_best_only=True, monitor="val_mean_squared_error", 
                                                        mode="min", save_weights_only=True)
        patience_callback = tf.keras.callbacks.EarlyStopping(monitor="val_mean_squared_error", patience=10, mode="min", restore_best_weights=True, start_from_epoch=10)

        # Train the model
        history = train_model_fit(models[modelIdx], checkpoint_callback, patience_callback, batchSize, train_dataset_batches, val_dataset_batches)
        plot_metrics(history, str(batchSize), modelNames[modelIdx])

        # Evaluate Model
        test_model_evaluate(models[modelIdx], batchSize, x_test, y_test, modelNames[modelIdx])

# python3 training.py "../data/processed_clipped_data.csv" 1024/2048

if __name__ == "__main__":
    main()

# Useful Pandas Commands
# print(dataframe.head())
# dataframe.info()
# print(dataframe.isnull().sum())

# 100 training epochs, 80/10/10 train/test/val split
# Data Preprocessing Results
# 1) Preprocessing data with clipping and the processing the paper does:
    # val_mean_squared_error: 1.7176 - val_mean_absolute_error: 0.8836 - val_mean_absolute_percentage_error: 72.3456
# 2) Preprocessing data with clipping and the processing the paper does + normalized to a standard distribution after:
    # val_mean_squared_error: 0.9061 - val_mean_absolute_error: 0.6249 - val_mean_absolute_percentage_error: 118.4602
# 3) Preprocessing data with outlier removal:
    # val_loss: 72067017196175360.0000 - val_mean_squared_error: 72067017196175360.0000 - val_mean_absolute_error: 99657736.0000 - val_mean_absolute_percentage_error: 134645696.0000
# 4) Preprocessing data with outlier removal and processing the paper does:
    # val_loss: 1.2013 - val_mean_squared_error: 1.2013 - val_mean_absolute_error: 0.7745 - val_mean_absolute_percentage_error: 76.1387
# 5) Preprocessing data with outlier removal and processing the paper does + normalized to a standard distribution after:
    # val_loss: 1.0760 - val_mean_squared_error: 1.0760 - val_mean_absolute_error: 0.7504 - val_mean_absolute_percentage_error: 100.0000

# Regularization Results
# 0) No regularizer: Same as data preprocessing (5)
# 1) L1 regularizer: 
    # val_loss: 1.1055 - val_mean_squared_error: 1.0332 - val_mean_absolute_error: 0.7372 - val_mean_absolute_percentage_error: 160.5657
# 2) L2 regularizer
    # val_loss: 1.0868 - val_mean_squared_error: 1.0066 - val_mean_absolute_error: 0.7189 - val_mean_absolute_percentage_error: 192.5771

# Architecture Results
    # Starting with 100 epochs of training, 80/10/10 train/test/val split, (5) data preprocessing and (2) regularization
    # Number of hidden layers
        # 5x100x3: same as regularization (2)
        # 5x100x100x3:
            # val_loss: 1.0958 - val_mean_squared_error: 1.0047 - val_mean_absolute_error: 0.7174 - val_mean_absolute_percentage_error: 204.9996
        # 5x100x100x100x3:
            # val_loss: 1.1030 - val_mean_squared_error: 0.9953 - val_mean_absolute_error: 0.7108 - val_mean_absolute_percentage_error: 206.2811
    # Batch Norm
        # 5x100x3:
            # val_loss: 0.8670 - val_mean_squared_error: 0.8541 - val_mean_absolute_error: 0.6272 - val_mean_absolute_percentage_error: 127.9274
        # 5x100x100x3:
            # val_loss: 2.9919 - val_mean_squared_error: 2.9768 - val_mean_absolute_error: 0.7271 - val_mean_absolute_percentage_error: 162.0858
        # 5x100x100x100x3
            # val_loss: 7.9600 - val_mean_squared_error: 7.9443 - val_mean_absolute_error: 0.6918 - val_mean_absolute_percentage_error: 200.4576

# 5x100x3 model with batch norm, an l2 regularizer, and preprocessing data with outlier removal and paper ranging
    # val_loss: 1.3105 - val_mean_squared_error: 1.2062 - val_mean_absolute_error: 0.7383 - val_mean_absolute_percentage_error: 78.1498
# 5x100x3 model with batch norm, an l2 regularizer, and preprocessing data with outlier removal and paper ranging + normalized after:
    # val_loss: 1.0876 - val_mean_squared_error: 1.0067 - val_mean_absolute_error: 0.7180 - val_mean_absolute_percentage_error: 198.4075

# Increasing learning rate to 0.01 yields --> learning rate = 0.001
    # val_mean_squared_error: 1.8347 - val_mean_absolute_error: 0.9762 - val_mean_absolute_percentage_error: 109.4312

# No batch norm, learning rate 0.001, 5x100x3, l2, (4) data --> batch norm is good
    # val_loss: 3.3664 - val_mean_squared_error: 2.6632 - val_mean_absolute_error: 1.0835 - val_mean_absolute_percentage_error: 75.2888
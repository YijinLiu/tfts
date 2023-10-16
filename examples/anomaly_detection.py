from absl import app
from absl import flags
from matplotlib import pyplot as plt
import numpy as np
import pandas as pd
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers


FLAGS = flags.FLAGS
flags.DEFINE_string('training_csv', 'data/art_daily_small_noise.csv', '')
flags.DEFINE_string('testing_csv', 'data/art_daily_jumpsup.csv', '')
flags.DEFINE_string('index_col', 'timestamp', '')
flags.DEFINE_integer('time_steps', 288, '')
flags.DEFINE_float('learning_rate', 0.001, '')
flags.DEFINE_integer('num_epochs', 50, '')
flags.DEFINE_integer('batch_size', 128, '')
flags.DEFINE_float('validation_split', 0.1, '')
flags.DEFINE_string('loss_img_file', 'loss.png', '')
flags.DEFINE_string('loss_hist_img_file', 'loss_hist.png', '')
flags.DEFINE_string('anomalous_img_file', 'anomalous.png', '')
flags.DEFINE_boolean('force_cpu', False, '')
flags.DEFINE_boolean('debug', False, '')
flags.DEFINE_boolean('debug_plot', False, '')


def create_sequences(values: np.ndarray, time_steps: int) -> np.ndarray:
    result = []
    for i in range(len(values) - time_steps + 1):
        result.append(values[i : (i + time_steps)])
    return np.stack(result)


def detect_anomaly():
    training_df = pd.read_csv(FLAGS.training_csv, parse_dates=True, index_col=FLAGS.index_col)
    testing_df = pd.read_csv(FLAGS.testing_csv, parse_dates=True, index_col=FLAGS.index_col)
    print(f'{len(training_df)} training samples and {len(testing_df)} testing examples.')

    if FLAGS.debug:
        training_df.head()
        testing_df.head()

    if FLAGS.debug_plot:
        fig, ax = plt.subplots()
        training_df.plot(legend=False, ax=ax)
        plt.show()
        fig, ax = plt.subplots()
        testing_df.plot(legend=False, ax=ax)
        plt.show()

    training_mean = training_df.mean()
    training_std = training_df.std()
    normalized_training_df = (training_df - training_mean) / training_std

    x_training = create_sequences(normalized_training_df.values, FLAGS.time_steps)
    input_shape = x_training.shape
    print('Training input shape: ', input_shape)

    model = keras.Sequential([
        layers.Input(shape=(input_shape[1], input_shape[2])),
        layers.Conv1D(
            filters=32, kernel_size=7, padding='same', strides=2, activation='relu'),
        layers.Dropout(rate=0.2),
        layers.Conv1D(
            filters=16, kernel_size=7, padding='same', strides=2, activation='relu'),
        layers.Conv1DTranspose(
            filters=16, kernel_size=7, padding='same', strides=2, activation='relu'),
        layers.Dropout(rate=0.2),
        layers.Conv1DTranspose(
            filters=32, kernel_size=7, padding='same', strides=2, activation='relu'),
        layers.Conv1DTranspose(
            filters=1, kernel_size=7, padding='same'),
    ])
    model.compile(optimizer=keras.optimizers.Adam(learning_rate=FLAGS.learning_rate), loss='mse')

    if FLAGS.debug:
        model.summary()

    history = model.fit(
        x_training,
        x_training,
        epochs=FLAGS.num_epochs,
        batch_size=FLAGS.batch_size,
        validation_split=FLAGS.validation_split,
        callbacks=[
            keras.callbacks.EarlyStopping(monitor='val_loss', patience=5, mode='min')
        ],
    )

    if FLAGS.loss_img_file:
        plt.clf()
        plt.plot(history.history['loss'], label='Training Loss')
        plt.plot(history.history['val_loss'], label='Validation Loss')
        plt.legend()
        plt.savefig(FLAGS.loss_img_file)
        print(f'Saved {FLAGS.loss_img_file}!')

    x_training_pred = model.predict(x_training)
    training_mae_loss = np.mean(np.abs(x_training_pred - x_training), axis=1)

    if FLAGS.loss_hist_img_file:
        plt.clf()
        plt.hist(training_mae_loss, bins=50)
        plt.xlabel("Training MAE loss")
        plt.ylabel("No of samples")
        plt.savefig(FLAGS.loss_hist_img_file)
        print(f'Saved {FLAGS.loss_hist_img_file}!')

    threshold = np.max(training_mae_loss)
    print("Reconstruction error threshold: ", threshold)

    normalized_testing_df = (testing_df - training_mean) / training_std
    if FLAGS.debug_plot:
        fig, ax = plt.subplots()
        normalized_testing_df.plot(legend=False, ax=ax)
        plt.show()

    x_testing = create_sequences(normalized_testing_df.values, FLAGS.time_steps)
    print("Testing input shape: ", x_testing.shape)

    x_testing_pred = model.predict(x_testing)
    testing_mae_loss = np.mean(np.abs(x_testing_pred - x_testing), axis=1)
    testing_mae_loss = testing_mae_loss.reshape((-1))
    anomalies = testing_mae_loss > threshold
    anomalous_data_indices = []
    for data_idx in range(FLAGS.time_steps - 1, len(testing_mae_loss)):
        if np.all(anomalies[data_idx - FLAGS.time_steps + 1 : data_idx]):
            anomalous_data_indices.append(data_idx)

    anomalous_testing_df = testing_df.iloc[anomalous_data_indices]
    fig, ax = plt.subplots()
    testing_df.plot(legend=False, ax=ax)
    anomalous_testing_df.plot(legend=False, ax=ax, color='r')
    if FLAGS.anomalous_img_file:
        plt.savefig(FLAGS.anomalous_img_file)
        print(f'Saved {FLAGS.anomalous_img_file}!')
    else:
        plt.show()


def main(argv):
    if FLAGS.force_cpu:
        with tf.device('/cpu:0'):
            detect_anomaly()
    else:
        gpus = tf.config.experimental.list_physical_devices('GPU')
        if gpus:
            try:
                for gpu in gpus:
                    tf.config.experimental.set_memory_growth(gpu, True)
            except RuntimeError as e:
                print(e)
        detect_anomaly()


app.run(main)

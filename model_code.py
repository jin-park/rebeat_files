import pandas as pd
from pathlib import Path
import tensorflow as tf
import datetime
from math import floor

path = Path("./data/2024-04-26 19:26:28.042459.csv")
df = pd.read_csv(path)


start, end = start, end = 0, len(df)
df.loc[start:end, "force"] = df.loc[start:end]["force"].map(lambda x: x / 12000)
df.loc[start:end, "accel_z"] = df.loc[start:end]["accel_z"].map(lambda x: x / 20)
df.loc[start:end, "disp_z"] = df.loc[start:end]["disp_z"].map(lambda x: x / 7)


train_x = df.loc[start:start + floor((end-start)*0.8)][["accel_z", "force"]]
train_y = df.loc[start:start + floor((end-start)*0.8)]["disp_z"]

valid_x = df.loc[start + floor((end-start)*0.1):start + floor((end-start)*0.1)*2][["accel_z", "force"]]
valid_y = df.loc[start + floor((end-start)*0.1):start + floor((end-start)*0.1)*2]["disp_z"]

test_x = df.loc[start + floor((end-start)*0.1)*2:end][["accel_z", "force"]]
test_y = df.loc[start + floor((end-start)*0.1)*2:end]["disp_z"]


seq_length = 16
tf.random.set_seed(0)  # extra code – ensures reproducibility
train_ds = tf.keras.utils.timeseries_dataset_from_array(
    train_x.to_numpy(),
    targets=train_y[seq_length:],
    sequence_length=seq_length,
    batch_size=32,
    shuffle=True,
    seed=0
)

valid_ds = tf.keras.utils.timeseries_dataset_from_array(
    valid_x.to_numpy(),
    targets=valid_y[seq_length:],
    sequence_length=seq_length,
    batch_size=32
)

test_ds = tf.keras.utils.timeseries_dataset_from_array(
    test_x.to_numpy(),
    targets=test_y[seq_length:],
    sequence_length=seq_length,
    batch_size=32
)


tf.random.set_seed(0)
lstm_model = tf.keras.Sequential([
    tf.keras.layers.LSTM(32, input_shape=[10, 2]),
    tf.keras.layers.Dense(1)
])

# bilstm_model = tf.keras.Sequential([
#     tf.keras.layers.Bidirectional(tf.keras.layers.LSTM(128, return_sequences=True, input_shape=[16, 2])),
#     tf.keras.layers.Bidirectional(tf.keras.layers.LSTM(32)),
#     tf.keras.layers.Dense(1)
# ])

def fit_and_evaluate(model, train_set, valid_set, learning_rate, epochs=500):
    early_stopping_cb = tf.keras.callbacks.EarlyStopping(
        monitor="val_mae", patience=50, restore_best_weights=True)
    opt = tf.keras.optimizers.SGD(learning_rate=learning_rate, momentum=0.9)
    model.compile(loss=tf.keras.losses.Huber(), optimizer=opt, metrics=["mae"])
    model.fit(train_set, validation_data=valid_set, epochs=epochs,
                    callbacks=[early_stopping_cb])
    valid_loss, valid_mae = model.evaluate(valid_set)
    return valid_mae * 7

error = fit_and_evaluate(rnn_2layer_model, train_ds, valid_ds, learning_rate=0.02)

export_dir = "./saved_model/" + str(datetime.datetime.now()) + ".tflite"

converter = tf.lite.TFLiteConverter.from_keras_model(lstm_model)
converter.optimizations = [tf.lite.Optimize.DEFAULT]
converter.experimental_new_converter=True
converter.target_spec.supported_ops = [tf.lite.OpsSet.TFLITE_BUILTINS,tf.lite.OpsSet.SELECT_TF_OPS]
tflite_model = converter.convert()

tflite_model_file = Path(export_dir)
tflite_model_file.write_bytes(tflite_model)
# +
import keras_tuner

from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import BatchNormalization
from tensorflow.keras.layers import Conv2D, Conv3D
from tensorflow.keras.layers import MaxPooling2D, MaxPooling3D, GlobalAveragePooling3D
from tensorflow.keras.layers import Activation
from tensorflow.keras.layers import Dropout
from tensorflow.keras.layers import Dense
from tensorflow.keras.layers import Flatten
from tensorflow.keras.layers import Input
from tensorflow.keras.models import Model
from tensorflow.keras.layers import concatenate
from tensorflow.keras.optimizers import Adam


# -

def create_mlp(hp, dim):
    model = Sequential()
    model.add(Dense(128, input_dim=dim, activation="relu", name='input_volume'))
    model.add(Dropout(0.35))
    model.add(Dense(32, activation="relu"))
    model.add(Dropout(0.15))
    model.add(Dense(256, activation="relu"))
    model.add(Dense(128, activation="relu"))
    return model

def create_cnn(hp, width, height, depth, filters=([32, 64, 128, 256])):
    inputShape = (height, width, depth, 1)
    chanDim = -1
    inputs = Input(shape=inputShape, name='input_pet')
    inputs_l = Input(shape=inputShape, name='input_left')
    inputs_r = Input(shape=inputShape, name='input_right')
    for (i, f) in enumerate(filters):
        if i == 0:
            l = inputs_l
            r = inputs_r
            r = Conv2D(32, kernel_size=3, padding="same", activation='relu')(r)
            r = BatchNormalization(axis=chanDim)(r)
            r = MaxPooling2D(pool_size=2)(r)
            l = Conv2D(32, kernel_size=3, padding="same", activation='relu')(l)
            l = BatchNormalization(axis=chanDim)(l)
            l = MaxPooling2D(pool_size=2)(l)
            x = concatenate([l, r])
        x = Conv3D(f, kernel_size=3, padding="same")(x)
        x = Activation("relu")(x)
        x = BatchNormalization()(x)
        x = MaxPooling3D(pool_size=2)(x)
    x = Flatten()(x)
    x = Dense(128)(x)
    x = Activation("relu")(x)
    if hp.Boolean("BatchNormalization_1"):
        x = BatchNormalization(axis=chanDim)(x)
    if hp.Boolean("dropout"):
        x = Dropout(0.40)(x)
    x = Dense(hp.Int("fc_2", min_value=32, max_value=1024, step=128), 
                               activity_regularizer=tf.keras.regularizers.L2(0.01), 
                               bias_regularizer = tf.keras.regularizers.L2(0.01))(x)
    x = Activation("relu")(x)
    model = Model(inputs, x)
    return model

def create_model(hp):
    mlp = create_mlp(hp, arr_scaled.shape[1])
    cnn = create_cnn(hp, 160, 160, 96)

    combinedInput = concatenate([mlp.output, cnn.output])

    x = Dense(hp.Int("fc_conc_1", min_value=32, max_value=1024, step=256), activation="relu",
                               activity_regularizer=tf.keras.regularizers.L2(0.01), 
                               bias_regularizer = tf.keras.regularizers.L2(0.01))(combinedInput)
    
    x = Dense(hp.Int("fc_conc_1", min_value=32, max_value=1024, step=256), activation="relu",
                               activity_regularizer=tf.keras.regularizers.L2(0.01), 
                               bias_regularizer = tf.keras.regularizers.L2(0.01))(x)
    x = Dense(3, activation="softmax")(x)

    model = Model(inputs=[mlp.input, cnn.input], outputs=x)
    
    learning_rate = hp.Float("lr", min_value=1e-5, max_value=1e-2, sampling="log")
    model.compile(
        optimizer=Adam(learning_rate=learning_rate),
        loss="categorical_crossentropy",
        metrics=["categorical_accuracy"]
    )
    
    return model


def main():
    tuner = keras_tuner.RandomSearch(
        hypermodel=create_model,
        objective="val_categorical_accuracy",
        max_trials=10,
        executions_per_trial=5,
        overwrite=True,
        directory="./hp_search",
        project_name="hp_search_multi-modal",
    )

    tuner.search(x=train_dataset, epochs=5, validation_data=test_dataset)
if __name__ == "__main__":
    main()

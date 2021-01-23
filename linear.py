import  tensorflow as tf
from tensorflow.keras import  Model, Sequential
from tensorflow.python.keras.layers import RandomFlip, RandomRotation, Dense, Flatten
from tensorflow.keras.optimizers import SGD


def create_base_model(add_custom_layers_func) -> Model:
    m = Sequential()
    m.add(RandomFlip("horizontal_and_vertical"))
    m.add(RandomRotation(0.2))

    add_custom_layers_func(m)

    m.add(Flatten())
    m.add(Dense(252, activation=tf.keras.activations.softmax))
    m.compile(
        optimizer=SGD(),
        loss=tf.keras.losses.categorical_crossentropy,
        metrics=[tf.keras.metrics.categorical_accuracy]
    )

    return m

def linearMod(Senq):
    pass


if __name__ == '__main__':
    m = create_base_model(linearMod)
    print(m)
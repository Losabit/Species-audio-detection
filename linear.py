import  tensorflow as tf
from tensorflow.keras import  Model, Sequential
from tensorflow.python.keras.layers import RandomFlip, RandomRotation, Dense, Flatten
from tensorflow.keras.optimizers import SGD
from dataset_param import  *

def create_base_model(add_custom_layers_func) -> Model:
    m = Sequential()
    m.add(RandomFlip("horizontal_and_vertical"))
    m.add(RandomRotation(0.2))

    add_custom_layers_func(m)

    m.add(Flatten())
    m.add(Dense(len(len_classes), activation=tf.keras.activations.softmax))
    m.compile(
        optimizer=SGD(),
        loss=tf.keras.losses.categorical_crossentropy,
        metrics=[tf.keras.metrics.categorical_accuracy]
    )

    return m

def linearMod(Senq):
    pass

def train_models(m:Model,x,y):
    m.fit(
        x,
        y,
        epochs=epch
    )

if __name__ == '__main__':
    m = create_base_model(linearMod)
    print(m)
    #train_models(m,)

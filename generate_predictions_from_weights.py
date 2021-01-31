from main_efficient_net import *

if __name__ == '__main__':
    if tf.test.gpu_device_name():
        print('Default GPU Device:{}'.format(tf.test.gpu_device_name()))
    else:
        print("Please install GPU version of TF")

    print(F"Creating model...")
    model = create_efficient_net_models()

    input_imgen = tf.keras.preprocessing.image.ImageDataGenerator(rescale=1.0 / 255)

    validation_generator = input_imgen.flow_from_directory(directory=DATASET_VAL_DIRECTORY,
                                                           target_size=(
                                                               IMAGE_WIDTH,
                                                               IMAGE_HEIGHT),
                                                           batch_size=batch_size,  # To verify maybe error
                                                           class_mode="categorical",
                                                           shuffle=True)

    print(F"Evaluation du model...")
    model.evaluate(validation_generator,
                   steps=validation_generator.samples // batch_size)

    print(F"Chargement des poids...")
    model.load_weights(WEIGHT_FILE_NAME)

    print(F"Evaluation du model...")
    validation_generator._set_index_array()  # Reinit du iterator
    model.evaluate(validation_generator,
                   steps=validation_generator.samples // batch_size)

    print(F"Sauvegarde des prédictions a partir du modele...")
    predict_and_save_in_submission(model, higher_than, 0.4)

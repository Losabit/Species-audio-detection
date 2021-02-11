import tensorflow as tf
import matplotlib.pyplot as plt
from dataset_param import *
from main import create_base_model, add_convnet, add_mlp_layers

SR = 48000
TRAIN_TFREC = os.path.join(PROJECT_PATH, 'dataset', 'rfcx-species-audio-detection', 'tfrecords', 'train')
TEST_TFREC = os.path.join(PROJECT_PATH, 'dataset', 'rfcx-species-audio-detection', 'tfrecords', 'test')
# tf.compat.v1.enable_eager_execution()

feature_description = {
    'recording_id': tf.io.FixedLenFeature([], tf.string, default_value=''),
    'audio_wav': tf.io.FixedLenFeature([], tf.string, default_value=''),
    'label_info': tf.io.FixedLenFeature([], tf.string, default_value=''),
}

feature_test_description = {
    'recording_id': tf.io.FixedLenFeature([], tf.string, default_value=''),
    'audio_wav': tf.io.FixedLenFeature([], tf.string, default_value='')
}

parse_dtype = {
    'audio_wav': tf.float32,
    'recording_id': tf.string,
    'species_id': tf.int32,
    'songtype_id': tf.int32,
    't_min': tf.float32,
    'f_min': tf.float32,
    't_max': tf.float32,
    'f_max': tf.float32,
    'is_tp': tf.int32
}


def parse_function(example_proto):
    sample = tf.io.parse_single_example(example_proto, feature_description)
    wav, _ = tf.audio.decode_wav(sample['audio_wav'], desired_channels=1) # mono
    label_info = tf.strings.split(sample['label_info'], sep='"')[1]
    labels = tf.strings.split(label_info, sep=';')

    @tf.function
    def cut_audio(label):
        items = tf.strings.split(label, sep=',')

        spid = tf.squeeze(tf.strings.to_number(items[0], tf.int32))
        soid = tf.squeeze(tf.strings.to_number(items[1], tf.int32))
        tmin = tf.squeeze(tf.strings.to_number(items[2]))
        fmin = tf.squeeze(tf.strings.to_number(items[3]))
        tmax = tf.squeeze(tf.strings.to_number(items[4]))
        fmax = tf.squeeze(tf.strings.to_number(items[5]))
        tp = tf.squeeze(tf.strings.to_number(items[6], tf.int32))

        tmax_s = tmax * tf.cast(SR, tf.float32)
        tmin_s = tmin * tf.cast(SR, tf.float32)
        cut_s = tf.cast(DURATION_CUT * SR, tf.float32)
        all_s = tf.cast(60 * SR, tf.float32)
        tsize_s = tmax_s - tmin_s
        cut_min = tf.cast(
            tf.maximum(0.0,
                       tf.minimum(tmin_s - (cut_s - tsize_s) / 2,
                                  tf.minimum(tmax_s + (cut_s - tsize_s) / 2, all_s) - cut_s)
                       ), tf.int32
        )
        cut_max = cut_min + DURATION_CUT * SR
        _sample = {
            'audio_wav': tf.reshape(wav[cut_min:cut_max], [DURATION_CUT * SR]),
            'recording_id': sample['recording_id'],
            'species_id': spid,
            'songtype_id': soid,
            't_min': tmin - tf.cast(cut_min, tf.float32) / tf.cast(SR, tf.float32),
            'f_min': fmin,
            't_max': tmax - tf.cast(cut_min, tf.float32) / tf.cast(SR, tf.float32),
            'f_max': fmax,
            'is_tp': tp
        }
        return _sample

    samples = tf.map_fn(cut_audio, labels, dtype=parse_dtype)
    return samples


def test_parse_function(example_proto):
    sample = tf.io.parse_single_example(example_proto, feature_test_description)
    wav, _ = tf.audio.decode_wav(sample['audio_wav'], desired_channels=1) # mono
    sample['audio_wav'] = tf.reshape(wav, [60 * SR])
    return sample

@tf.function
def filter_tp(x):
    return x['is_tp'] == 1


def wav_to_spec(x):
    mel_power = 2

    stfts = tf.signal.stft(x["audio_wav"], frame_length=2048, frame_step=512, fft_length=2048) # frame_length=255, frame_step=128
    spectrograms = tf.abs(stfts) ** mel_power

    # Warp the linear scale spectrograms into the mel-scale.
    num_spectrogram_bins = stfts.shape[-1]
    lower_edge_hertz, upper_edge_hertz, num_mel_bins = 40.0, 24000.0, IMAGE_WIDTH

    linear_to_mel_weight_matrix = tf.signal.linear_to_mel_weight_matrix(
        num_mel_bins, num_spectrogram_bins, SR, lower_edge_hertz,
        upper_edge_hertz)
    mel_spectrograms = tf.tensordot(
        spectrograms, linear_to_mel_weight_matrix, 1)
    mel_spectrograms.set_shape(spectrograms.shape[:-1].concatenate(
        linear_to_mel_weight_matrix.shape[-1:]))

    # Compute a stabilized log to get log-magnitude mel-scale spectrograms.
    log_mel_spectrograms = tf.math.log(mel_spectrograms + 1e-6)
    x["audio_wav"] = tf.transpose(log_mel_spectrograms)# (num_mel_bins, frames)
    return x


def create_annot_test(sample):
    return sample["audio_wav"]


@tf.function
def create_annot(x):
    targ = tf.one_hot(x["species_id"], len_classes, on_value=x["is_tp"], off_value=0)
    return x['audio_wav'], tf.cast(targ, tf.float32)
    '''
    return {
        'input': x["audio_wav"],
        'target': tf.cast(targ, tf.float32)
    }
    '''


def load_dataset(files, have_label=True):
    autotune = tf.data.experimental.AUTOTUNE
    if have_label:
        return tf.data.TFRecordDataset(files, num_parallel_reads=autotune)\
            .map(parse_function, num_parallel_calls=autotune).unbatch() \
            .filter(filter_tp)\
            .map(wav_to_spec, num_parallel_calls=autotune)\
            .map(create_annot, num_parallel_calls=autotune)
            # .shuffle(2048)
    else:
        return tf.data.TFRecordDataset(files, num_parallel_reads=autotune)\
            .map(test_parse_function, num_parallel_calls=autotune)\
            .map(wav_to_spec, num_parallel_calls=autotune)\
            .map(create_annot_test, num_parallel_calls=autotune)# .shuffle(2048)


if __name__ == '__main__':
    tfrecs = sorted(tf.io.gfile.glob(TRAIN_TFREC + '/*.tfrec'))
    split_ind = int((1 - validation_split) * len(tfrecs))
    train_tfrec, val_tfrec = tfrecs[:split_ind], tfrecs[split_ind:]
    test_tfrec = tf.io.gfile.glob(TEST_TFREC + "/*.tfrec")

    train_dataset = load_dataset(train_tfrec)
    val_dataset = load_dataset(val_tfrec)
    test_dataset = load_dataset(test_tfrec, False)
    print(train_dataset)
    print(test_dataset)

    def show_batch(image_batch, label_batch):
        plt.imshow(image_batch / 255.0)
        plt.title("Classe : " + str(label_batch))
        plt.axis("off")
        plt.show()

    '''
    image_batch, label_batch = next(iter(train_dataset))
    test_image_batch = next(iter(test_dataset))

    show_batch(image_batch.numpy(), label_batch.numpy())
    show_batch(test_image_batch.numpy(), label_batch.numpy())
    '''
    config = tf.compat.v1.ConfigProto()
    config.gpu_options.allow_growth = True
    sess = tf.compat.v1.Session(config=config)

    model = create_base_model(add_mlp_layers)
    log = model.fit(
        train_dataset,
        validation_data=val_dataset,
        epochs=3
    )
    '''
    print("Evaluation du dataset : ")
    print("[Train] => ")
    model.evaluate(create_dataset_iterator(DATASET_TRAIN_DIRECTORY, train_size),
                   steps=train_size // batch_size)
    
    print("[Validation] => ")
    model.evaluate(create_dataset_iterator(DATASET_VAL_DIRECTORY, val_size),
                   steps=val_size // batch_size)
    
    print("Sauvegarde des prédictions sur le jeu de test : ")
    predict_and_save_in_submission(model, higher_than, 0.4)
    '''
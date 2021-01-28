import numpy as np
import soundfile as sf


class SedDataset:
    def __init__(self, df, period=10, stride=5, audio_transform=None, data_path="train", mode="train"):

        self.period = period
        self.stride = stride
        self.audio_transform = audio_transform
        self.data_path = data_path
        self.mode = mode

        self.df = df.groupby("recording_id").agg(lambda x: list(x)).reset_index()

    def __len__(self):
        return len(self.df)

    def __getitem__(self, idx):
        record = self.df.iloc[idx]

        y, sr = sf.read(f"{self.data_path}/{record['recording_id']}.flac")

        if self.mode != "test":
            #y, label = crop_or_pad(y, sr, period=self.period, record=record, mode=self.mode)

            if self.audio_transform:
                y = self.audio_transform(samples=y, sample_rate=sr)
        else:
            y_ = []
            i = 0
            effective_length = self.period * sr
            stride = self.stride * sr
            y = np.stack([y[i:i + effective_length].astype(np.float32) for i in
                          range(0, 60 * sr + stride - effective_length, stride)])
            label = np.zeros(24, dtype='f')
            if self.mode == "valid":
                for i in record['species_id']:
                    label[i] = 1

        return {
            "image": y,
            "target": label,
            "id": record['recording_id']
        }
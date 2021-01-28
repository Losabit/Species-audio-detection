import os, random
import numpy as np
import torch
import torch.nn as nn


def init_layer(layer):
    nn.init.xavier_uniform_(layer.weight)

    if hasattr(layer, "bias"):
        if layer.bias is not None:
            layer.bias.data.fill_(0.)


def init_bn(bn):
    bn.bias.data.fill_(0.)
    bn.weight.data.fill_(1.0)


def do_mixup(x, mixup_lambda):
    out = (x[0 :: 2].transpose(0, -1) * mixup_lambda[0 :: 2] + \
        x[1 :: 2].transpose(0, -1) * mixup_lambda[1 :: 2]).transpose(0, -1)
    return out


def interpolate(x: torch.Tensor, ratio: int):
    """Interpolate data in time domain. This is used to compensate the
    resolution reduction in downsampling of a CNN.

    Args:
      x: (batch_size, time_steps, classes_num)
      ratio: int, ratio to interpolate
    Returns:
      upsampled: (batch_size, time_steps * ratio, classes_num)
    """
    (batch_size, time_steps, classes_num) = x.shape
    upsampled = x[:, :, None, :].repeat(1, 1, ratio, 1)
    upsampled = upsampled.reshape(batch_size, time_steps * ratio, classes_num)
    return upsampled


def pad_framewise_output(framewise_output: torch.Tensor, frames_num: int):
    """Pad framewise_output to the same length as input frames. The pad value
    is the same as the value of the last frame.
    Args:
      framewise_output: (batch_size, frames_num, classes_num)
      frames_num: int, number of frames to pad
    Outputs:
      output: (batch_size, frames_num, classes_num)
    """
    pad = framewise_output[:, -1:, :].repeat(
        1, frames_num - framewise_output.shape[1], 1)
    """tensor for padding"""

    output = torch.cat((framewise_output, pad), dim=1)
    """(batch_size, frames_num, classes_num)"""

    return output


def seed_everything(seed):
    random.seed(seed)
    os.environ['PYTHONHASHSEED'] = str(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.backends.cudnn.deterministic = True


def create_train_folds(path, folds, seed):
    from sklearn.model_selection import StratifiedKFold
    import pandas as pd
    inpath = os.path.join(os.getcwd(), 'dataset', 'rfcx-species-audio-detection')
    metadata_inpath = os.path.join(inpath, 'train_tp.csv')

    train = pd.read_csv(metadata_inpath).sort_values("recording_id")
    ss = None

    train_gby = train.groupby("recording_id")[["species_id"]].first().reset_index()
    train_gby = train_gby.sample(frac=1, random_state=seed).reset_index(drop=True)
    train_gby.loc[:, 'kfold'] = -1

    x = train_gby["recording_id"].values
    y = train_gby["species_id"].values

    kfold = StratifiedKFold(n_splits=folds)
    for fold, (t_idx, v_idx) in enumerate(kfold.split(x, y)):
        train_gby.loc[v_idx, "kfold"] = fold

    train = train.merge(train_gby[['recording_id', 'kfold']], on="recording_id", how="left")
    print(train.kfold.value_counts())
    train.to_csv(path, index=False)
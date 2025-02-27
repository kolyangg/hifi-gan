from __future__ import absolute_import, division, print_function, unicode_literals

import glob
import os
import argparse
import json
import torch
import numpy as np
from scipy.io.wavfile import write
from env import AttrDict
from meldataset import mel_spectrogram, MAX_WAV_VALUE, load_wav
from models import Generator

import torch.nn.functional as F

h = None
device = None

def load_checkpoint(filepath, device):
    assert os.path.isfile(filepath)
    print("Loading '{}'".format(filepath))
    checkpoint_dict = torch.load(filepath, map_location=device)
    print("Complete.")
    return checkpoint_dict


def pad_wav_for_inference(wav, hop_size, upsample_rates, win_size, center, original_len=None):
    """
    1) Zero-pad 'wav' so its length is a multiple of (hop_size * product_of_upsample_rates).
    2) If 'center=True', optionally add half the window size so the final STFT window is fully captured.
    3) Optionally keep track of original_len if you want to trim after generation.
    """
    total_upsample = np.prod(upsample_rates)  # e.g. [8,8,4] => 256
    multiple = hop_size * total_upsample      # e.g. 256*256 => 65536

    # 1) Pad to multiple
    remainder = wav.shape[0] % multiple
    pad_size = 0
    if remainder != 0:
        pad_size += (multiple - remainder)

    # 2) If using center=True in STFT, add half window so last frame isn't truncated
    if center:
        pad_size += (win_size // 2)

    # Apply the final pad
    if pad_size > 0:
        wav = F.pad(wav, (0, pad_size))
    return wav


def get_mel(x):
    """
    We call mel_spectrogram with center=True (or whatever matches training).
    Change 'center=True/False' to whatever your model was trained with.
    """
    return mel_spectrogram(
        x,
        h.n_fft, h.num_mels, h.sampling_rate,
        h.hop_size, h.win_size,
        h.fmin, h.fmax,
        center=True  # or False if that matches training
    )


def scan_checkpoint(cp_dir, prefix):
    pattern = os.path.join(cp_dir, prefix + '*')
    cp_list = glob.glob(pattern)
    if len(cp_list) == 0:
        return ''
    return sorted(cp_list)[-1]


def inference(a):
    generator = Generator(h).to(device)

    state_dict_g = load_checkpoint(a.checkpoint_file, device)
    generator.load_state_dict(state_dict_g['generator'])

    filelist = os.listdir(a.input_wavs_dir)
    os.makedirs(a.output_dir, exist_ok=True)

    generator.eval()
    generator.remove_weight_norm()

    with torch.no_grad():
        for filname in filelist:
            if not filname.lower().endswith('.wav'):
                continue  # skip non-wav

            # 1) Load wav
            wav, sr = load_wav(os.path.join(a.input_wavs_dir, filname))
            if sr != h.sampling_rate:
                raise ValueError(
                    f"File {filname} has SR={sr}, but config SR={h.sampling_rate}. "
                    "Please resample your audio to match."
                )

            original_length = wav.shape[0]  # keep for optional post-trim

            # 2) Normalize
            wav = wav / MAX_WAV_VALUE
            wav = torch.FloatTensor(wav).to(device)

            # 3) Pad the raw waveform for leftover frames
            wav = pad_wav_for_inference(
                wav,
                hop_size=h.hop_size,
                upsample_rates=h.upsample_rates,
                win_size=h.win_size,
                center=True,    # or False if your training used that
                original_len=original_length
            )

            # 4) Compute mel
            x = get_mel(wav.unsqueeze(0))

            # 5) Run generator
            y_g_hat = generator(x)
            audio = y_g_hat.squeeze()

            # 6) Convert to int16
            audio = audio * MAX_WAV_VALUE
            audio = audio.cpu().numpy().astype('int16')

            # 7) (Optional) Trim or pad back to exact original length
            #    Only do this if you absolutely need them to align sample-by-sample:
            #    If there's "new content" from padding, you might lose it by trimming.
            #    So do this carefully:
            if a.trim_output:
                if audio.shape[0] > original_length:
                    audio = audio[:original_length]
                else:
                    # If it's shorter for some reason, pad to match
                    diff = original_length - audio.shape[0]
                    audio = np.pad(audio, (0, diff), mode='constant')

            # 8) Write to disk
            out_name = os.path.splitext(filname)[0] + '_generated.wav'
            output_file = os.path.join(a.output_dir, out_name)
            write(output_file, h.sampling_rate, audio)
            print("Wrote:", output_file)


def main():
    print('Initializing Inference Process..')

    parser = argparse.ArgumentParser()
    parser.add_argument('--input_wavs_dir', default='test_files')
    parser.add_argument('--output_dir', default='generated_files')
    parser.add_argument('--checkpoint_file', required=True)
    parser.add_argument('--trim_output', action='store_true', default=True,
                        help="Trim/pad final output to match exact original #samples.")
    a = parser.parse_args()

    config_file = os.path.join(os.path.split(a.checkpoint_file)[0], 'config.json')
    with open(config_file) as f:
        data = f.read()

    global h
    json_config = json.loads(data)
    h = AttrDict(json_config)

    torch.manual_seed(h.seed)
    global device
    if torch.cuda.is_available():
        torch.cuda.manual_seed(h.seed)
        device = torch.device('cuda')
    else:
        device = torch.device('cpu')

    inference(a)


if __name__ == '__main__':
    main()

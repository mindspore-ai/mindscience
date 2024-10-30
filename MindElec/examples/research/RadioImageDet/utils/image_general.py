# Copyright 2024 Huawei Technologies Co., Ltd
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
# http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
# ===========================================================================
"""image data general"""
from PIL import Image
import numpy as np
from scipy.signal import stft
import matplotlib.pyplot as plt
from signal_general import SignalGeneral

def show_iq_stft(signal_i, signal_q, fs=1024, nperseg=1024):
    '''
    :param signal_i: I component
    :param signal_q: Q component
    :param fs: sampling frequency
    :param nperseg:
    :return: Numpy image matrix of float type
    '''
    sig = np.power(np.add(np.power(signal_i, 2), np.power(signal_q, 2)), 0.5)
    # sig = sig_i + 1j * sig_q
    f, t, spectrum = stft(sig, fs, nperseg=nperseg)
    plt.pcolormesh(t, f, np.abs(spectrum), shading='gouraud')
    plt.title('STFT Magnitude')
    plt.ylabel('Frequency [Hz]')
    plt.xlabel('Time [sec]')
    plt.show()

    print(f.shape)
    print(t.shape)
    print(spectrum.shape)

    save_image = np.abs(spectrum[:512, :512]).astype(np.float32)
    save_image = np.rot90(save_image, 1)
    plt.imshow(save_image, cmap='gray')
    plt.show()

    print(save_image.shape)

    return save_image


if __name__ == '__main__':
    # Tele2G_DL_CDMA
    # temp = SignalGeneral('CDMA', '../audio/audio3.mp3', carrier_freq_range=(870e6, 885e6), segment_duration=6)
    # Mobi2G_DL_GSM900
    # temp = SignalGeneral('GSM', '../audio/audio3.mp3', carrier_freq_range=(935e6, 955e6), segment_duration=6)
    # Uni2G_DL_GSM900
    # temp = SignalGeneral('GSM', '../audio/audio3.mp3', carrier_freq_range=(955e6, 960e6), segment_duration=6)
    # Mobi2G_DL_GSM1800
    # temp = SignalGeneral('GSM', '../audio/audio3.mp3', carrier_freq_range=(1805e6, 1820e6), segment_duration=6)
    # Uni4G_DL_FDD1800
    # temp = SignalGeneral('FDD', '../audio/audio3.mp3', carrier_freq_range=(1840e6, 1860e6), segment_duration=6)
    # Tele4G_DL_FDD1800
    # temp = SignalGeneral('FDD', '../audio/audio3.mp3', carrier_freq_range=(1860e6, 1875e6), segment_duration=6)
    # Mobi3G_TDSCDMA
    # temp = SignalGeneral('TDSCDMA', '../audio/audio3.mp3', carrier_freq_range=(2010e6, 2025e6), segment_duration=6)
    # Uni3G_DL_WCDMA
    # temp = SignalGeneral('WCDMA', '../audio/audio3.mp3', carrier_freq_range=(2130e6, 2140e6), segment_duration=6)
    # Tele4G_DL_FDD2100
    temp = SignalGeneral('FDD', '../audio/audio3.mp3', carrier_freq_range=(2110e6, 2130e6), segment_duration=6)
    # Uni4G_DL_FDD2100
    # temp = SignalGeneral('FDD', '../audio/audio3.mp3', carrier_freq_range=(2140e6, 2155e6), segment_duration=6)
    # Mobi4G_TDLTE_10M
    # temp = SignalGeneral('TDLTE_10M', '../audio/audio3.mp3', carrier_freq_range=(1905e6, 1915e6), segment_duration=6)
    # Mobi4G_TDLTE_20M
    # temp = SignalGeneral('TDLTE_20M', '../audio/audio3.mp3', carrier_freq_range=(2575e6, 2635e6), segment_duration=6)

    signal = iter(temp)
    _, _ = next(signal)
    _, _ = next(signal)
    mod_i, mod_q = next(signal)
    img = show_iq_stft(mod_i, mod_i, fs=8 * 10e9, nperseg=1024)

    image = Image.fromarray(img)
    img_array = np.array(image).astype(np.float32)
    min_val = np.min(img_array)
    max_val = np.max(img_array)
    scaled = ((img_array - min_val) / (max_val - min_val) * 255).astype(np.uint8)
    image = Image.fromarray(scaled)
    image = image.convert('RGB')
    image.save('Tele4G_DL_FDD2100.jpg') # your save path

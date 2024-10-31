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
"""signal data general"""
import numpy as np
import matplotlib.pyplot as plt
from pydub import AudioSegment


class SignalGeneral:
    """Signal data general class"""
    def __init__(self, signal_type, audio_file,
                 carrier_freq_range=(870e6, 885e6), segment_duration=5):
        self.signal_type = signal_type

        audio = AudioSegment.from_mp3(audio_file)
        samples = np.array(audio.get_array_of_samples()).astype(np.float32)

        if audio.channels == 2:
            self.i_signal = samples[0::2]
            self.q_signal = samples[1::2]
        else:
            self.i_signal = samples
            self.q_signal = np.zeros(len(self.i_signal))

        if np.max(np.abs(self.i_signal)) != 0:
            self.i_signal /= np.max(np.abs(self.i_signal))
        if np.max(np.abs(self.q_signal)) != 0:
            self.q_signal /= np.max(np.abs(self.q_signal))

        self.fs = audio.frame_rate
        total_duration = len(self.i_signal) / self.fs
        self.total_segments = int(np.ceil(total_duration / segment_duration))
        self.segment_samples = int(segment_duration * self.fs)

        f_center = (carrier_freq_range[0] + carrier_freq_range[1]) / 2
        self.f_up = f_center + 5e6
        self.f_down = f_center - 5e6

    def __iter__(self):
        self.segment = 0
        return self

    def __next__(self):
        if self.segment < self.total_segments:
            start_idx = self.segment * self.segment_samples
            end_idx = min(start_idx + self.segment_samples, len(self.i_signal))

            i_segment = self.i_signal[start_idx:end_idx]
            q_segment = self.q_signal[start_idx:end_idx]

            t = np.arange(0, len(i_segment) / self.fs, 1 / self.fs)
            carrier_i_up = np.cos(2 * np.pi * self.f_up * t)
            carrier_q_up = np.sin(2 * np.pi * self.f_up * t)
            carrier_i_down = np.cos(2 * np.pi * self.f_down * t)
            carrier_q_down = np.sin(2 * np.pi * self.f_down * t)

            i_segment = i_segment[:len(carrier_i_up)]
            q_segment = q_segment[:len(carrier_q_up)]

            if self.signal_type == 'FDD':
                modulated_i_up = i_segment * carrier_i_up
                modulated_q_up = q_segment * carrier_q_up

                modulated_i_down = i_segment * carrier_i_down
                modulated_q_down = q_segment * carrier_q_down

                modulated_i = modulated_i_up + modulated_i_down
                modulated_q = modulated_q_up + modulated_q_down

            elif self.signal_type == 'CDMA':
                chip_rate = 1e5
                code_length = 32
                code = np.random.choice([-1, 1], size=code_length)

                def cdma_modulate(sig, code, chip_rate, fs):
                    chip_duration = max(1, int(fs / chip_rate))
                    if not sig.any():
                        return np.array([]), np.array([])
                    modulated_signal = np.zeros(len(sig) * chip_duration)
                    for i, _ in enumerate(sig):
                        start_idx = i * chip_duration
                        end_idx = start_idx + chip_duration
                        if end_idx <= len(modulated_signal):
                            modulated_signal[start_idx:end_idx] = np.tile(
                                code if sig[i] > 0 else -code,
                                int(np.ceil(chip_duration / len(code))))[:chip_duration]
                    return modulated_signal

                if i_segment.any():
                    cdma_i = cdma_modulate(i_segment, code, chip_rate, self.fs)
                    cdma_q = cdma_modulate(q_segment, code, chip_rate, self.fs)
                else:
                    cdma_i = np.array([])
                    cdma_q = np.array([])

                if cdma_i.any() and carrier_i_up.any():
                    modulated_i = cdma_i[:len(
                        carrier_i_up)] * carrier_i_up
                    modulated_q = cdma_q[:len(carrier_q_up)] * carrier_q_up
                else:
                    modulated_i = np.zeros(len(carrier_i_up))
                    modulated_q = np.zeros(len(carrier_q_up))

            elif self.signal_type == 'GSM':
                def gmsk_modulate(sig):
                    h = 0.5
                    phase = np.cumsum(sig) * np.pi * h
                    i_mod = np.cos(phase)
                    q_mod = np.sin(phase)
                    return i_mod, q_mod

                i_modulated, q_modulated = gmsk_modulate(i_segment)
                modulated_i = i_modulated * carrier_i_up
                modulated_q = q_modulated * carrier_q_up

            elif self.signal_type == 'TDLTE_10M':
                modulated_i = self.ofdm_modulate(i_segment, 128, t) * carrier_i_up
                modulated_q = self.ofdm_modulate(q_segment, 128, t) * carrier_q_up

            elif self.signal_type == 'TDLTE_20M':
                modulated_i = self.ofdm_modulate(i_segment, 256, t) * carrier_i_up
                modulated_q = self.ofdm_modulate(q_segment, 256, t) * carrier_q_up

            elif self.signal_type == 'TDSCDMA':
                def tdscdma_modulate(sig):
                    spread_signal = np.zeros(len(sig))
                    for i, _ in enumerate(sig):
                        spread_signal[i] = sig[i] * \
                                           np.random.choice([-1, 1])
                    return spread_signal

                modulated_i = tdscdma_modulate(i_segment) * carrier_i_up
                modulated_q = tdscdma_modulate(q_segment) * carrier_q_up

            elif self.signal_type == 'WCDMA':
                def wcdma_modulate(sig):
                    qpsk_symbols = np.zeros_like(sig)
                    for i, _ in enumerate(sig):
                        if sig[i] >= 0:
                            qpsk_symbols[i] = 1
                        else:
                            qpsk_symbols[i] = -1
                    return qpsk_symbols

                modulated_i = wcdma_modulate(i_segment) * carrier_i_up
                modulated_q = wcdma_modulate(q_segment) * carrier_q_up

            else:
                raise ValueError("Unsupported signal type!")

            # 绘制信号
            plt.figure(figsize=(12, 6))
            plt.subplot(2, 1, 1)
            plt.plot(t[0:1000], modulated_i[0:1000])
            plt.title(f'Modulated I Component ({self.signal_type})')
            plt.xlabel('Time (s)')
            plt.ylabel('Amplitude')
            plt.grid()

            plt.subplot(2, 1, 2)
            plt.plot(t[0:1000], modulated_q[0:1000])
            plt.title(f'Modulated Q Component ({self.signal_type})')
            plt.xlabel('Time (s)')
            plt.ylabel('Amplitude')
            plt.grid()

            plt.tight_layout()
            plt.show()

            self.segment += 1
            return modulated_i, modulated_q
        raise StopIteration

    def ofdm_modulate(self, sig, num_subcarriers, t):
        subcarrier_spacing = 15000
        modulated_signal = np.zeros_like(sig)
        for i in range(num_subcarriers):
            subcarrier = np.cos(
                2 * np.pi * (i * subcarrier_spacing) * t[:len(sig)])
            modulated_signal += sig * subcarrier
        return modulated_signal


if __name__ == '__main__':
    tele2g_dl_cdma = SignalGeneral(
        'CDMA',
        'audio/audio3.mp3',
        carrier_freq_range=(
            870e6,
            885e6),
        segment_duration=5)
    signal = iter(tele2g_dl_cdma)

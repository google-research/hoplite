# coding=utf-8
# Copyright 2024 The Perch Authors.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

"""Handcrafted features for linear models."""

import dataclasses
from typing import Self

from hoplite.zoo import zoo_interface
import librosa
from ml_collections import config_dict
import numpy as np


@dataclasses.dataclass
class HandcraftedFeaturesModel(zoo_interface.EmbeddingModel):
  """Wrapper for simple feature extraction."""

  window_size_s: float
  hop_size_s: float
  melspec_config: config_dict.ConfigDict
  aggregation: str = 'beans'

  @classmethod
  def from_config(
      cls, config: config_dict.ConfigDict
  ) -> 'HandcraftedFeaturesModel':
    return cls(**config)

  @classmethod
  def beans_baseline(
      cls, sample_rate=32000, frame_rate=100
  ) -> 'HandcraftedFeaturesModel':
    stride = sample_rate // frame_rate
    mel_config = config_dict.ConfigDict({
        'sample_rate': sample_rate,
        'features': 160,
        'stride': stride,
        'kernel_size': 2 * stride,
        'freq_range': (60.0, sample_rate / 2.0),
        'power': 2.0,
    })
    features_config = config_dict.ConfigDict({
        'compute_mfccs': True,
        'aggregation': 'beans',
    })
    config = config_dict.ConfigDict({
        'sample_rate': sample_rate,
        'melspec_config': mel_config,
        'features_config': features_config,
        'window_size_s': 1.0,
        'hop_size_s': 1.0,
    })
    # pylint: disable=unexpected-keyword-arg
    return HandcraftedFeaturesModel.from_config(config)

  def melspec(self, audio_array: np.ndarray) -> np.ndarray:
    framed_audio = self.frame_audio(
        audio_array, self.window_size_s, self.hop_size_s
    )
    specs = []
    for frame in framed_audio:
      specs.append(
          librosa.feature.melspectrogram(
              y=frame,
              sr=self.sample_rate,
              hop_length=self.melspec_config.stride,
              win_length=self.melspec_config.kernel_size,
              center=True,
              n_mels=self.melspec_config.features,
              power=self.melspec_config.power,
          )
      )
    return np.stack(specs, axis=0)

  def embed(self, audio_array: np.ndarray) -> zoo_interface.InferenceOutputs:
    # Melspecs will have shape [melspec_channels, frames]
    melspecs = self.melspec(audio_array)
    if self.aggregation == 'beans':
      features = np.concatenate(
          [
              melspecs.mean(axis=-1),
              melspecs.std(axis=-1),
              melspecs.min(axis=-1),
              melspecs.max(axis=-1),
          ],
          axis=-2,
      )
    else:
      raise ValueError(f'unrecognized aggregation: {self.aggregation}')
    # Add a trivial channels dimension.
    features = features[:, np.newaxis, :]
    return zoo_interface.InferenceOutputs(features, None, None)

  def batch_embed(
      self, audio_batch: np.ndarray
  ) -> zoo_interface.InferenceOutputs:
    return zoo_interface.batch_embed_from_embed_fn(self.embed, audio_batch)

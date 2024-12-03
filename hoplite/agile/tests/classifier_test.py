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

"""Tests for linear classifier implementation."""

import os
import tempfile

from hoplite.agile import classifier
from ml_collections import config_dict
import numpy as np

from absl.testing import absltest


class ClassifierTest(absltest.TestCase):

  def setUp(self):
    super().setUp()
    # `self.create_tempdir()` raises an UnparsedFlagAccessError, which is why
    # we use `tempdir` directly.
    self.tempdir = tempfile.mkdtemp()

  def _make_linear_classifier(self, embedding_dim, classes):
    np.random.seed(1234)
    beta = np.float32(np.random.normal(size=(embedding_dim, len(classes))))
    beta_bias = np.float32(np.random.normal(size=(len(classes),)))
    embedding_model_config = config_dict.ConfigDict({
        'model_name': 'nelson',
    })
    return classifier.LinearClassifier(
        beta, beta_bias, classes, embedding_model_config
    )

  def test_call_linear_classifier(self):
    embedding_dim = 8
    classes = ('a', 'b', 'c')
    classy = self._make_linear_classifier(embedding_dim, classes)

    batch_embeddings = np.random.normal(size=(10, embedding_dim))
    predictions = classy(batch_embeddings)
    self.assertEqual(predictions.shape, (10, len(classes)))

    single_embedding = np.random.normal(size=(embedding_dim,))
    predictions = classy(single_embedding)
    self.assertEqual(predictions.shape, (len(classes),))

  def test_save_load_linear_classifier(self):
    embedding_dim = 8
    classes = ('a', 'b', 'c')
    classy = self._make_linear_classifier(embedding_dim, classes)
    classy_path = os.path.join(self.tempdir, 'classifier.json')
    classy.save(classy_path)
    classy_loaded = classifier.LinearClassifier.load(classy_path)
    np.testing.assert_allclose(classy_loaded.beta, classy.beta)
    np.testing.assert_allclose(classy_loaded.beta_bias, classy.beta_bias)
    self.assertSequenceEqual(classy_loaded.classes, classy.classes)
    self.assertEqual(classy_loaded.embedding_model_config.model_name, 'nelson')


if __name__ == '__main__':
  absltest.main()

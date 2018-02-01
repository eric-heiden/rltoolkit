# Copyright 2017 The dm_control Authors.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#    http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or  implied.
# See the License for the specific language governing permissions and
# limitations under the License.
# ============================================================================

"""Integration tests for rendering."""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import os
import unittest

# Internal dependencies.
from absl.testing import absltest
from absl.testing import parameterized
from dm_control import render
from dm_control.mujoco.testing import decorators
from dm_control.mujoco.testing import image_utils
from six.moves import zip  # pylint: disable=redefined-builtin


DEBUG_IMAGE_DIR = os.environ.get('TEST_UNDECLARED_OUTPUTS_DIR',
                                 absltest.get_default_test_tmpdir())

NUM_THREADS = 4
CALLS_PER_THREAD = 1


@unittest.skipIf(render.DISABLED, render.DISABLED_MESSAGE)
class RenderTest(parameterized.TestCase):

  @parameterized.named_parameters(image_utils.SEQUENCES.items())
  @image_utils.save_images_on_failure(output_dir=DEBUG_IMAGE_DIR)
  @decorators.run_threaded(num_threads=NUM_THREADS,
                           calls_per_thread=CALLS_PER_THREAD)
  def test_render(self, sequence):
    for expected, actual in zip(sequence.iter_load(), sequence.iter_render()):
      image_utils.assert_images_close(expected, actual)

  @decorators.run_threaded(num_threads=NUM_THREADS,
                           calls_per_thread=CALLS_PER_THREAD)
  @image_utils.save_images_on_failure(output_dir=DEBUG_IMAGE_DIR)
  def test_render_multiple_physics_per_thread(self):
    cartpole = image_utils.cartpole
    humanoid = image_utils.humanoid
    cartpole_frames = []
    humanoid_frames = []
    for cartpole_frame, humanoid_frame in zip(cartpole.iter_render(),
                                              humanoid.iter_render()):
      cartpole_frames.append(cartpole_frame)
      humanoid_frames.append(humanoid_frame)

    for expected, actual in zip(cartpole.iter_load(), cartpole_frames):
      image_utils.assert_images_close(expected, actual)

    for expected, actual in zip(humanoid.iter_load(), humanoid_frames):
      image_utils.assert_images_close(expected, actual)


if __name__ == '__main__':
  absltest.main()

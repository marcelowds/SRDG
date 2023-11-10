# coding=utf-8
# Copyright 2020 The Google Research Authors.
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

# Lint as: python3
"""Training NCSN++ on Church with VE SDE."""

from configs.default_ve_configs import get_default_configs


def get_config():
  config = get_default_configs()
  # training
  training = config.training
  training.sde = 'vesde'
  training.continuous = True

  # sampling
  sampling = config.sampling
  sampling.method = 'pc'
  sampling.predictor = 'reverse_diffusion'
  sampling.corrector = 'langevin'

  # data
  data = config.data
  data.dataset = 'CelebAHQ'
  data.image_size = 128
  data.upscaling = 16

  ## tf records quis campi 90 images sample 
  data.tfrecords_path = 'tfrecords/sample_quis.tfrecords'
  
  ## classifiers path
  classifier = config.classifier
  # GENDER #
  classifier.ckpt_path_gender = 'classifier_ckpt/ckpt_gender'
  # GLASSES #
  classifier.ckpt_path_glasses = 'classifier_ckpt/ckpt_glasses'
  # BEARD #
  classifier.ckpt_path_beard = 'classifier_ckpt/ckpt_beard'

  ##    
  ## ATTRIBUTES QUIS-CAMPI ##
  attributes = config.attributes
  attributes.quis_gender = [1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,0,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,0,0,1,1,1,0,1,1,1,0,1,1,1,1,0,1,1,0,1,1,0,0,1,0,1,1,1,0,1,0,0,0,1,1,0,0,0,0,1,0,0,0,1,0,1,1,1,1,1,0,1,0,1]
  attributes.quis_glasses_1 = [0,1,1,1,1,0,0,0,1,0,1,0,1,0,0,1,0,0,0,1,1,0,1,0,0,1,1,1,0,1,1,1,1,1,0,1,0,0,1,0,0,0,0,0,1,0,0,1,0,0,0,0,1,1,0,1,0,0,1,0,1,0,0,0,1,1,0,1,1,0,0,0,1,0,1,1,1,1,1,1,1,0,0,0,0,1,0,1,1,0]
  attributes.quis_beard = [1,0,0,1,0,1,1,1,0,1,0,1,0,1,1,0,1,1,1,0,1,0,1,0,0,0,0,0,1,0,1,0,0,1,0,0,0,1,1,0,0,1,1,1,1,0,1,0,1,1,1,1,0,0,1,0,1,1,1,0,1,0,0,1,1,0,1,1,1,0,1,1,1,1,1,1,1,1,1,0,1,0,0,1,0,0,1,1,1,1]


  attributes.quis_gender = [1,1,1,1,1,1,0,1]
  attributes.quis_glasses_1 = [1,1,1,1,1,1,1,1]
  attributes.quis_beard = [0,0,0,0,0,0,1,0]




  # model
  model = config.model
  model.name = 'ncsnpp'
  model.sigma_max = 348
  model.scale_by_sigma = True
  model.ema_rate = 0.999
  model.normalization = 'GroupNorm'
  model.nonlinearity = 'swish'
  model.nf = 128
  model.ch_mult = (1, 1, 2, 2, 2, 2, 2)
  model.num_res_blocks = 2
  model.attn_resolutions = (16,)
  model.resamp_with_conv = True
  model.conditional = True
  model.fir = True
  model.fir_kernel = [1, 3, 3, 1]
  model.skip_rescale = True
  model.resblock_type = 'biggan'
  #model.progressive = 'output_skip'
  model.progressive = 'none'
  model.progressive_input = 'input_skip'
  model.progressive_combine = 'sum'
  model.attention_type = 'ddpm'
  model.init_scale = 0.
  model.fourier_scale = 16
  model.conv_size = 3

  return config

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

# pylint: skip-file
"""Training and evaluation for score-based generative models. """

import gc
import io
import os
import time
from typing import Any
import matplotlib.pyplot as plt
import flax
import flax.jax_utils as flax_utils
import jax
import jax.numpy as jnp
import numpy as np
import tensorflow as tf
import torch
import tensorflow_gan as tfgan
import logging
import functools
from flax.metrics import tensorboard
from flax.training import checkpoints#, train_state
# Keep the import below for registering all model definitions
from models import ddpm, ncsnv2, ncsnpp
import losses
import sampling
import utils
from models import utils as mutils
import datasets
import evaluation
import likelihood
import sde_lib
from absl import flags
import numpy as onp
from torchvision.utils import make_grid, save_image
#from models import wideresnet_noise_conditional
import controllable_generation
from aux import manipule
import flax.core.frozen_dict

from PIL import Image
import PIL

FLAGS = flags.FLAGS


def train(config, workdir):
    """Runs the training pipeline.

    Args:
      config: Configuration to use.
      workdir: Working directory for checkpoints and TF summaries. If this
        contains checkpoint training will be resumed from the latest checkpoint.
    """

    # Create directories for experimental logs
    sample_dir = os.path.join(workdir, "samples")
    tf.io.gfile.makedirs(sample_dir)

    rng = jax.random.PRNGKey(config.seed)
    tb_dir = os.path.join(workdir, "tensorboard")
    tf.io.gfile.makedirs(tb_dir)
    if jax.host_id() == 0:
      writer = tensorboard.SummaryWriter(tb_dir)

    # Initialize model.
    rng, step_rng = jax.random.split(rng)
    score_model, init_model_state, initial_params = mutils.init_model(step_rng, config)
    optimizer = losses.get_optimizer(config).create(initial_params)
    state = mutils.State(step=0, optimizer=optimizer, lr=config.optim.lr,
                         model_state=init_model_state,
                         ema_rate=config.model.ema_rate,
                         params_ema=initial_params,
                         rng=rng)  # pytype: disable=wrong-keyword-args

    # Create checkpoints directory
    checkpoint_dir = os.path.join(workdir, "checkpoints")
    # Intermediate checkpoints to resume training after pre-emption in cloud environments
    checkpoint_meta_dir = os.path.join(workdir, "checkpoints-meta")
    tf.io.gfile.makedirs(checkpoint_dir)
    tf.io.gfile.makedirs(checkpoint_meta_dir)
    # Resume training when intermediate checkpoints are detected
    state = checkpoints.restore_checkpoint(checkpoint_meta_dir, state)
    # `state.step` is JAX integer on the GPU/TPU devices
    initial_step = int(state.step)
    rng = state.rng

    # Build data iterators
    train_ds, eval_ds, _ = datasets.get_dataset(config,
                                                additional_dim=config.training.n_jitted_steps,
                                                uniform_dequantization=config.data.uniform_dequantization)
    train_iter = iter(train_ds)  # pytype: disable=wrong-arg-types
    eval_iter = iter(eval_ds)  # pytype: disable=wrong-arg-types
    # Create data normalizer and its inverse
    scaler = datasets.get_data_scaler(config)
    inverse_scaler = datasets.get_data_inverse_scaler(config)

    # Setup SDEs
    if config.training.sde.lower() == 'vesde':
      sde = sde_lib.VESDE(sigma_min=config.model.sigma_min, sigma_max=config.model.sigma_max, N=config.model.num_scales)
      sampling_eps = 1e-5
    else:
      raise NotImplementedError(f"SDE {config.training.sde} not implemented.")

    # Build one-step training and evaluation functions
    optimize_fn = losses.optimization_manager(config)
    continuous = config.training.continuous
    reduce_mean = config.training.reduce_mean
    likelihood_weighting = config.training.likelihood_weighting
    train_step_fn = losses.get_step_fn(sde, score_model, train=True, optimize_fn=optimize_fn,
                                       reduce_mean=reduce_mean, continuous=continuous,
                                       likelihood_weighting=likelihood_weighting)
    # Pmap (and jit-compile) multiple training steps together for faster running
    p_train_step = jax.pmap(functools.partial(jax.lax.scan, train_step_fn), axis_name='batch', donate_argnums=1)
    eval_step_fn = losses.get_step_fn(sde, score_model, train=False, optimize_fn=optimize_fn,
                                      reduce_mean=reduce_mean, continuous=continuous,
                                      likelihood_weighting=likelihood_weighting)
    # Pmap (and jit-compile) multiple evaluation steps together for faster running
    p_eval_step = jax.pmap(functools.partial(jax.lax.scan, eval_step_fn), axis_name='batch', donate_argnums=1)

    # Building sampling functions
    if config.training.snapshot_sampling:
      sampling_shape = (config.training.batch_size // jax.local_device_count(), config.data.image_size,
                        config.data.image_size, config.data.num_channels)
      sampling_fn = sampling.get_sampling_fn(config, sde, score_model, sampling_shape, inverse_scaler, sampling_eps)

    # Replicate the training state to run on multiple devices
    pstate = flax_utils.replicate(state)
    num_train_steps = config.training.n_iters

    # In case there are multiple hosts (e.g., TPU pods), only log to host 0
    if jax.host_id() == 0:
      logging.info("Starting training loop at step %d." % (initial_step,))
    rng = jax.random.fold_in(rng, jax.host_id())

    # JIT multiple training steps together for faster training
    n_jitted_steps = config.training.n_jitted_steps
    # Must be divisible by the number of steps jitted together
    assert config.training.log_freq % n_jitted_steps == 0 and \
           config.training.snapshot_freq_for_preemption % n_jitted_steps == 0 and \
           config.training.eval_freq % n_jitted_steps == 0 and \
           config.training.snapshot_freq % n_jitted_steps == 0, "Missing logs or checkpoints!"

    for step in range(initial_step, num_train_steps + 1, config.training.n_jitted_steps):
      # Convert data to JAX arrays and normalize them. Use ._numpy() to avoid copy.
      batch = jax.tree_map(lambda x: scaler(x._numpy()), next(train_iter))  # pylint: disable=protected-access
      rng, *next_rng = jax.random.split(rng, num=jax.local_device_count() + 1)
      next_rng = jnp.asarray(next_rng)
      # Execute one training step
      #print("->t1")
      (_, pstate), ploss = p_train_step((next_rng, pstate), batch)
      #print("->t2")
      loss = flax.jax_utils.unreplicate(ploss).mean()
      # Log to console, file and tensorboard on host 0
      if jax.host_id() == 0 and step % config.training.log_freq == 0:
        logging.info("step: %d, training_loss: %.5e" % (step, loss))
        writer.scalar("training_loss", loss, step)

      # Save a temporary checkpoint to resume training after pre-emption periodically
      if step != 0 and step % config.training.snapshot_freq_for_preemption == 0 and jax.host_id() == 0:
        saved_state = flax_utils.unreplicate(pstate)
        saved_state = saved_state.replace(rng=rng)
        checkpoints.save_checkpoint(checkpoint_meta_dir, saved_state,
                                    step=step // config.training.snapshot_freq_for_preemption,
                                    keep=1)

      # Report the loss on an evaluation dataset periodically
      if step % config.training.eval_freq == 0:
        eval_batch = jax.tree_map(lambda x: scaler(x._numpy()), next(eval_iter))  # pylint: disable=protected-access
        rng, *next_rng = jax.random.split(rng, num=jax.local_device_count() + 1)
        next_rng = jnp.asarray(next_rng)
        (_, _), peval_loss = p_eval_step((next_rng, pstate), eval_batch)
        eval_loss = flax.jax_utils.unreplicate(peval_loss).mean()
        if jax.host_id() == 0:
          logging.info("step: %d, eval_loss: %.5e" % (step, eval_loss))
          writer.scalar("eval_loss", eval_loss, step)

      # Save a checkpoint periodically and generate samples if needed
      if step != 0 and step % config.training.snapshot_freq == 0 or step == num_train_steps:
        # Save the checkpoint.
        #print('---->>>>entre checkpoint')
        if jax.host_id() == 0:
          saved_state = flax_utils.unreplicate(pstate)
          saved_state = saved_state.replace(rng=rng)
          checkpoints.save_checkpoint(checkpoint_dir, saved_state,
                                      step=step // config.training.snapshot_freq,
                                      keep=np.inf)

        # Generate and save samples
        if config.training.snapshot_sampling:
          rng, *sample_rng = jax.random.split(rng, jax.local_device_count() + 1)
          sample_rng = jnp.asarray(sample_rng)
          sample, n = sampling_fn(sample_rng, pstate)
          this_sample_dir = os.path.join(
            sample_dir, "iter_{}_host_{}".format(step, jax.host_id()))
          tf.io.gfile.makedirs(this_sample_dir)
          image_grid = sample.reshape((-1, *sample.shape[2:]))
          nrow = int(np.sqrt(image_grid.shape[0]))
          sample = np.clip(sample * 255, 0, 255).astype(np.uint8)
          with tf.io.gfile.GFile(
              os.path.join(this_sample_dir, "sample.np"), "wb") as fout:
            np.save(fout, sample)

          with tf.io.gfile.GFile(
              os.path.join(this_sample_dir, "sample.png"), "wb") as fout:
            utils.save_image(image_grid, fout, nrow=nrow, padding=2)


def evaluate(config,
             workdir,
             eval_folder="eval"):
  """Evaluate trained models.

  Args:
    config: Configuration to use.
    workdir: Working directory for checkpoints.
    eval_folder: The subfolder for storing evaluation results. Default to
      "eval".
  """

  # Create directory to eval_folder
  eval_dir = os.path.join(workdir, 'checkpoints')
  tf.io.gfile.makedirs(eval_dir)

  rng = jax.random.PRNGKey(config.seed + 1)

  # Build data pipeline
  sample_ds, __, _ = datasets.get_dataset(config,
                                              additional_dim=None,
                                              uniform_dequantization=config.data.uniform_dequantization,
                                              evaluation=True)
  #print(f'--->>>sample_ds shape{sample_ds}')
  sample_iter = iter(sample_ds)

  # Create data normalizer and its inverse
  scaler = datasets.get_data_scaler(config)
  inverse_scaler = datasets.get_data_inverse_scaler(config)


  # Initialize model
  rng, model_rng = jax.random.split(rng)
  score_model, init_model_state, initial_params = mutils.init_model(model_rng, config)
  optimizer = losses.get_optimizer(config).create(initial_params)
  state = mutils.State(step=0, optimizer=optimizer, lr=config.optim.lr,
                       model_state=init_model_state,
                       ema_rate=config.model.ema_rate,
                       params_ema=initial_params,
                       rng=rng)  # pytype: disable=wrong-keyword-args


  checkpoint_dir = os.path.join(workdir, "checkpoints-meta")


  # Setup SDEs
  if config.training.sde.lower() == 'vesde':
    sde = sde_lib.VESDE(sigma_min=config.model.sigma_min, sigma_max=config.model.sigma_max, N=config.model.num_scales)
    sampling_eps = 1e-5
  else:
    raise NotImplementedError(f"SDE {config.training.sde} not implemented.")

  sampling_shape = (config.eval.batch_size // jax.local_device_count(),
                      config.data.image_size, config.data.image_size,
                      config.data.num_channels)
  
  ## CLASSIFIERS PATH
  # GENDER #
  ckpt_path_gender = config.classifier.ckpt_path_gender 
  # GLASSES # 
  ckpt_path_glasses = config.classifier.ckpt_path_glasses
  # BEARD #
  ckpt_path_beard = config.classifier.ckpt_path_beard

  batch_size = config.eval.batch_size
  shape = (batch_size, 128, 128, 3)
  random_seed = 0
  rng = jax.random.PRNGKey(random_seed)
  rng, step_rng = jax.random.split(rng)
  rng, *step_rng = jax.random.split(rng, jax.local_device_count() + 1)
  predictor = sampling.get_predictor(config.sampling.predictor.lower())
  corrector = sampling.get_corrector(config.sampling.corrector.lower())
  snr = 0.16
  n_steps = 2
  probability_flow = False

  classifier_gender, classifier_params_gender, batch_stats_gender = mutils.create_my_classifier(rng, batch_size, ckpt_path_gender)
  classifier_glasses, classifier_params_glasses, batch_stats_glasses = mutils.create_my_classifier(rng, batch_size, ckpt_path_glasses)
  classifier_beard, classifier_params_beard, batch_stats_beard = mutils.create_my_classifier(rng, batch_size, ckpt_path_beard)


  pc_conditional_sampler = controllable_generation.get_pc_conditional_sampler(
    sde, score_model,
    classifier_gender, classifier_params_gender, batch_stats_gender,
    classifier_glasses, classifier_params_glasses, batch_stats_glasses,
    classifier_beard, classifier_params_beard, batch_stats_beard,
    shape, predictor, corrector, inverse_scaler,
    snr, n_steps=n_steps,
    probability_flow=probability_flow,
    continuous=config.training.continuous)
  rng, *step_rng = jax.random.split(rng, jax.local_device_count() + 1)
  step_rng = jnp.asarray(step_rng)
  state = checkpoints.restore_checkpoint(checkpoint_dir, state)
  pstate = flax.jax_utils.replicate(state)

  quis_gender = config.attributes.quis_gender
  quis_glasses_1 = config.attributes.quis_glasses_1
  quis_nobeard = config.attributes.quis_beard


  begin_sampling_round = 0
  num_sampling_rounds = 8
  print("Initializing the sampling")
  for r in range(begin_sampling_round, num_sampling_rounds):
    batch = next(sample_iter)
    img = batch['image']._numpy()
    img = scaler(img)
    low_res = img[:,:,:,:,3:6]
    original_hr = img[:, :, :, :, 0:3]
    low = jnp.asarray(low_res)

    st = time.time()
    label_gender = quis_gender[r]
    label_glasses = quis_glasses_1[r]
    label_beard = quis_nobeard[r]
    
    labels_gender = jnp.ones((jax.local_device_count(), shape[0]), dtype=jnp.int32) * label_gender
    labels_glasses = jnp.ones((jax.local_device_count(), shape[0]), dtype=jnp.int32) * label_glasses
    labels_beard = jnp.ones((jax.local_device_count(), shape[0]), dtype=jnp.int32) * label_beard
    

    samples = pc_conditional_sampler(step_rng, pstate, low, labels_gender, labels_glasses, labels_beard)

    et = time.time()

    elapsed_time = et - st

    print('Elapsed time:', elapsed_time, 'seconds')


    for i in range(batch_size):
      index = r*batch_size+i+1
      manipule.save_samples(inverse_scaler(low_res[:,i,:,:,:]),index,'-lr',workdir)
      manipule.save_samples(samples[:,i,:,:,:],index,'-srdg',workdir)
      manipule.save_samples(inverse_scaler(original_hr[:,i,:,:,:]),index,'-hr',workdir)


    print(f"Figure {r+1} saved")



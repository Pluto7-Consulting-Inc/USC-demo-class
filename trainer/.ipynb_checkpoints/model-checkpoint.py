#!/usr/bin/env python

# Copyright 2017 Google Inc. All Rights Reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#      http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

import tensorflow as tf
import shutil

tf.logging.set_verbosity(tf.logging.INFO)

# Initializing
TIMESERIES_COL = 'rawdata'
SEQ_LEN = 90
DEFAULTS = None
N_INPUTS = None

# In each sequence, [1-60] are features, and [60-90] are the label
N_OUTPUTS = 30

def init(hparams):
  global SEQ_LEN, DEFAULTS, N_INPUTS, kernel_1, kernel_2, filter_1, filter_2
  SEQ_LEN =  hparams['sequence_length']
  DEFAULTS = [[0.0] for x in range(0, SEQ_LEN)]
  N_INPUTS = SEQ_LEN - N_OUTPUTS
  kernel_1 = hparams['kernel_size_1']
  kernel_2 = hparams['kernel_size_2']
  filter_1 = hparams['filter_1']
  filter_2 = hparams['filter_2']

# read data and convert to needed format
def read_dataset(filename, mode, batch_size):  
  def _input_fn():
    # could be a path to one file or a file pattern.
    input_file_names = tf.train.match_filenames_once(filename)
    filename_queue = tf.train.string_input_producer(
        input_file_names, num_epochs=None, shuffle=True)

    reader = tf.TextLineReader()
    _, value = reader.read_up_to(filename_queue, num_records=batch_size)

    value_column = tf.expand_dims(value, -1)
    
    # all_data is a list of tensors
    all_data = tf.decode_csv(value_column, record_defaults=DEFAULTS)  
    inputs = all_data[:len(all_data)-N_OUTPUTS]
    label = all_data[len(all_data)-N_OUTPUTS : ]
    
    # from list of tensors to tensor with one more dimension
    inputs = tf.concat(inputs, axis=1)
    label = tf.concat(label, axis=1)

    # returning features, label
    return {TIMESERIES_COL: inputs}, label
  return _input_fn

def cnn_model(features, mode, params):
  # flatten with new shape = (?, 60, 1)
  X = tf.reshape(features[TIMESERIES_COL], [-1, N_INPUTS, 1])

  c1 = tf.layers.Conv1D(filters=filter_1,
                        kernel_size=kernel_1, padding='same', 
                        activation=tf.nn.relu)(X)
  p1 = tf.layers.MaxPooling1D(pool_size=2, strides=2)(c1) #(?, 30, 30)

  c2 = tf.layers.Conv1D(filters=filter_2,
                        kernel_size=kernel_2, padding='same',
                        activation=tf.nn.relu)(p1)
  p2 = tf.layers.MaxPooling1D(pool_size=2, strides=2)(c2) #(?, 15, 30)
  
  c2flat = tf.layers.Flatten()(p2)
  predictions = tf.layers.Dense(60)(c2flat)

  return predictions

#specifies what the caller of predict() method has to provide
def serving_input_fn():
  feature_placeholders = {
    TIMESERIES_COL: tf.placeholder(tf.float32, [None, N_INPUTS])
  }
  
  features = {
    key: tf.expand_dims(tensor, -1)
    for key, tensor in feature_placeholders.items()
  }
  features[TIMESERIES_COL] = tf.squeeze(features[TIMESERIES_COL], axis=[2])

  return tf.estimator.export.ServingInputReceiver(features, feature_placeholders)

def compute_errors(features, labels, predictions):
  if predictions.shape[1] == 1:
    loss = tf.losses.mean_squared_error(labels, predictions)
    rmse = tf.metrics.root_mean_squared_error(labels, predictions)
    return loss, rmse
  else:
    # create full labels tensor of shape [?, 90]
    labelsN = tf.concat([features[TIMESERIES_COL], labels], axis=1)
  
    # slice out last 30 elements from labelsN to have shape [?, 60]
    labelsN = labelsN[:, 30:]
  
    # compute loss & rmse metrics
    loss = tf.losses.mean_squared_error(labelsN, predictions)
    rmse = tf.metrics.root_mean_squared_error(labelsN, predictions)        
    return loss, rmse

# create the inference model
def sequence_regressor(features, labels, mode, params):
    
  predictions = cnn_model(features, mode, params)

  # loss function
  loss = None
  train_op = None
  eval_metric_ops = None

  if mode == tf.estimator.ModeKeys.TRAIN or mode == tf.estimator.ModeKeys.EVAL:
    loss, rmse = compute_errors(features, labels, predictions)
    
    if mode == tf.estimator.ModeKeys.TRAIN: 
      # this is for batch normalization 
      update_ops = tf.get_collection(tf.GraphKeys.UPDATE_OPS)
      with tf.control_dependencies(update_ops):
        # set up training operation
        train_op = tf.contrib.layers.optimize_loss(
                     loss,
                     tf.train.get_global_step(),
                     learning_rate=params['learning_rate'],
                     optimizer="Adam"
                   )

    # metric used for evaluation
    eval_metric_ops = {"rmse": rmse}

  # create predictions
  predictions_dict = {"predicted": predictions}

  # return EstimatorSpec
  return tf.estimator.EstimatorSpec(
      mode=mode,
      predictions=predictions_dict,
      loss=loss,
      train_op=train_op,
      eval_metric_ops=eval_metric_ops,
      export_outputs={'predictions': tf.estimator.export.PredictOutput(predictions_dict)}
  )

def train_and_evaluate(output_dir, hparams):

  # used to wrap the model_fn and returns ops necessary to perform training, evaluation, or predictions
  estimator = tf.estimator.Estimator(
                model_fn = sequence_regressor,                  
                params = hparams,
                config = tf.estimator.RunConfig(
                           save_checkpoints_secs = hparams['min_eval_frequency']
                         ),
                model_dir = output_dir
              )
  
  train_spec = tf.estimator.TrainSpec(
                 input_fn = read_dataset(
                   filename = hparams['train_data_paths'],
                   mode = tf.estimator.ModeKeys.TRAIN,
                   batch_size = hparams['train_batch_size']
                 ),
                 max_steps = hparams['train_steps']
               )

  exporter = tf.estimator.LatestExporter('exporter', serving_input_fn)

  #eval_spec consists of computing metrics to judge the performance of the trained model.
  eval_spec = tf.estimator.EvalSpec(
                input_fn = read_dataset(
                  filename = hparams['eval_data_paths'],
                  mode = tf.estimator.ModeKeys.EVAL,
                  batch_size = 1000
                ),
                exporters = exporter,
                start_delay_secs = 60,
                throttle_secs = 120
  )

  tf.estimator.train_and_evaluate(estimator, train_spec, eval_spec)

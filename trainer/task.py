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

import traceback
import argparse
import json
import os
import tensorflow as tf

import model

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
  # Input Arguments
    parser.add_argument(
      '--train_data_paths',
      help='GCS or local path to training data',
      required=True
  )
    parser.add_argument(
      '--eval_data_paths',
      help='GCS or local path to evaluation data',
      required=True
  )
    parser.add_argument(
      '--train_batch_size',
      help='Batch size for training steps',
      type=int,
      default=32
  )
    parser.add_argument(
      '--learning_rate',
      help='Initial learning rate for training',
      type=float,
      default=0.005
  )
    parser.add_argument(
      '--train_steps',
      help='Steps to run the training job for. A step is one batch-size',
      type=int,
      default=500
  )
    parser.add_argument(
      '--sequence_length',
      help='This model works with fixed length sequences of 90. 60 are inputs, last 30 is output',
      type=int,
      default=90
  )
    parser.add_argument(
      '--kernel_size_1',
      help='This model works well with fixed kernel size 14',
      type=int,
      default=14
  )

    parser.add_argument(
      '--kernel_size_2',
      help='This model works well with fixed kernel size 14',
      type=int,
      default=7
  )
    
    parser.add_argument(
      '--filter_1',
      help='This model works well with fixed kernel size 400',
      type=int,
      default=400
  )
    parser.add_argument(
      '--filter_2',
      help='This model works well with fixed kernel size 400',
      type=int,
      default=400
  )
    parser.add_argument(
      '--output_dir',
      help='GCS location to write checkpoints and export models',
      required=True
  )
    parser.add_argument(
      '--job-dir',
      help='this model ignores this field, but it is required by gcloud',
      default='junk'
  )
    parser.add_argument(
      '--eval_delay_secs',
      help='How long to wait before running first evaluation',
      default=10,
      type=int
  )
    parser.add_argument(
      '--min_eval_frequency',
      help='Minimum number of training steps between evaluations',
      default=60,
      type=int
  )

    args = parser.parse_args()
    hparams = args.__dict__
  
  # unused args provided by service
    job_dir = hparams.pop('job_dir')
    hparams.pop('job-dir', None)

    output_dir = hparams.pop('output_dir')

  # This code can be removed if you are not using hyperparameter tuning
    output_dir = os.path.join(
      output_dir,
      json.loads(
          os.environ.get('TF_CONFIG', '{}')
      ).get('task', {}).get('trial', '')
  )

  # calculate train_steps if not provided
    if hparams['train_steps'] < 1:
       # 1,000 steps at batch_size of 100
        hparams['train_steps'] = (1000 * 100) // hparams['train_batch_size']
        print ("Training for {} steps".format(hparams['train_steps']))
  
    model.init(hparams)
  
  # Run the training job
    model.train_and_evaluate(output_dir, hparams)

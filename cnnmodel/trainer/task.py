import traceback
import argparse
import json
import os
import tensorflow as tf

import model

def compute_errors(features, labels, predictions):
  if predictions.shape[1] == 1:
    loss = tf.losses.mean_squared_error(labels, predictions)
    rmse = tf.metrics.root_mean_squared_error(labels, predictions)
    return loss, rmse
  else:
    # create full labels tensor of shape [?, 90]
    labelsN = tf.concat([features[model.TIMESERIES_COL], labels], axis=1)
  
    # slice out last 30 elements from labelsN to have shape [?, 60]
    labelsN = labelsN[:, 30:]
  
    # compute loss & rmse metrics
    loss = tf.losses.mean_squared_error(labelsN, predictions)
    rmse = tf.metrics.root_mean_squared_error(labelsN, predictions)        
    return loss, rmse

# create the inference model
def sequence_regressor(features, labels, mode, params):
    
  predictions = model.cnn_model(features, mode, params)

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
  save_freq = max(1, min(100, hparams['train_steps']/100))

  # used to wrap the model_fn and returns ops necessary to perform training, evaluation, or predictions
  estimator = tf.estimator.Estimator(
                model_fn = sequence_regressor,                  
                params = hparams,
                config = tf.estimator.RunConfig(
                           save_checkpoints_steps=save_freq,
                           save_checkpoints_secs =None
                         ),
                model_dir = output_dir
              )
  
  train_spec = tf.estimator.TrainSpec(
                 input_fn = model.read_dataset(
                   filename=hparams['train_data_paths'],
                   mode = tf.estimator.ModeKeys.TRAIN,
                   batch_size=hparams['train_batch_size']
                 ),
                 max_steps = hparams['train_steps']
               )

  exporter = tf.estimator.LatestExporter('Servo', model.serving_input_fn)

  eval_freq = max(1, min(120, hparams['train_steps']/5))

  #eval_spec consists of computing metrics to judge the performance of the trained model.
  eval_spec = tf.estimator.EvalSpec(
                input_fn = model.read_dataset(
                  filename=hparams['eval_data_paths'],
                  mode = tf.estimator.ModeKeys.EVAL,
                  batch_size=1000
                ),
                steps = 1,
                exporters = exporter,
                start_delay_secs = eval_freq,
                throttle_secs = eval_freq
              )

  tf.estimator.train_and_evaluate(estimator, train_spec, eval_spec)

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
      default=326
  )
  parser.add_argument(
      '--learning_rate',
      help='Initial learning rate for training',
      type=float,
      default=0.0047
  )
  parser.add_argument(
      '--train_steps',
      help='Steps to run the training job for. A step is one batch-size',
      type=int,
      default=0
  )
  parser.add_argument(
      '--sequence_length',
      help='This model works with fixed length sequences of 90. 60 are inputs, last 30 is output',
      type=int,
      default=90
  )
  parser.add_argument(
      '--kernel_size',
      help='This model works well  with fixed kernel size 15 ',
      type=int,
      default=15
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
  train_and_evaluate(output_dir, hparams)

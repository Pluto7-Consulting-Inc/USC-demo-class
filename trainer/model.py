import tensorflow as tf
import shutil

tf.logging.set_verbosity(tf.logging.INFO)

# Initializing
TIMESERIES_COL = 'rawdata'
SEQ_LEN = None
DEFAULTS = None
N_INPUTS = None

# In each sequence, [1-60] are features, and [60-30] is label
N_OUTPUTS = 30

def init(hparams):
  global SEQ_LEN, DEFAULTS, N_INPUTS, kernel
  SEQ_LEN =  hparams['sequence_length']
  DEFAULTS = [[0.0] for x in range(0, SEQ_LEN)]
  N_INPUTS = SEQ_LEN - N_OUTPUTS
  kernel = hparams['kernel_size']

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

  c1 = tf.layers.conv1d(X, filters=N_INPUTS//2,
                          kernel_size=kernel, strides=1, 
                          padding='same', activation=tf.nn.relu) 
  p1 = tf.layers.max_pooling1d(c1, pool_size=2, strides=2) #(?, 30, 30)

  c2 = tf.layers.conv1d(p1, filters=N_INPUTS//2,
                          kernel_size=kernel, strides=1, 
                          padding='same', activation=tf.nn.relu)
  p2 = tf.layers.max_pooling1d(c2, pool_size=2, strides=2) #(?, 15, 30)
    
  outlen = (N_INPUTS//4) * (N_INPUTS//2)
  c2flat = tf.reshape(p2, [-1, outlen])
  predictions = tf.layers.dense(c2flat, 60, activation=None)

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

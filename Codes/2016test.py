
# coding: utf-8

# In[1]:


"""
A class to read the XML doc with each MRI scan. Each scan is included with an XML with various information.
"""

# parser
import xml.etree.ElementTree as ET
import os


class XMLReader:
    # Instantiates
    def __init__(self, path):
        # Safety check to make sure the file path is valid
        try:
            self.tree = ET.parse(path)
            self.root = self.tree.getroot()
        except IOError as ioerr:
            print("Failed to parse\n")
            print(ioerr)
            print("\n")

    # Returns 0 for NC, 1 for MCI, 2 for AD
    def subject_status(self):
        for i in self.root.iter("researchGroup"):
            if i.text == "CN":
                return 0
            elif i.text == "MCI":
                return 1
            elif i.text == "AD":
                return 2

    # Returns patient number
    def subject_identifier(self):
        for i in self.root.iter("subjectIdentifier"):
            return i.text

    # checks to see if the current XML doc is for an MRI
    def is_mri(self):
        for i in self.root.iter("modality"):
            if i.text == "MRI":
                return True

    # finds the image ID
    def getderiveduid(self):
        for i in self.root.iter("imageUID"):
            return i.text

    # finds a path to the respective scan from current directory
    def path_to_scan(self, origin):
        # first folder, patient number
        id = self.subject_identifier()
        path = "/" + id + "/"
        # second folder, scan label
        for i in self.root.iter("processedDataLabel"):
            label = i.text.split(";")
            break
        firstItem = True
        for i in label:
            if firstItem == True:
                path = path + i.replace(" ", "")
                firstItem = False
                continue
            path = path + "__" + i.strip().replace(" ", "_")
        
        '''
        # third folder scan date
        for i in self.root.iter("dateAcquired"):
            split = i.text.split(" ")
            lhs = split[0]
            rhs = split[1]
            break
        path = path + "/" + lhs
        rhsplit = rhs.split(":")
        for i in rhsplit:
            path = path + "_" + i
        '''
        # third folder scan date
        item3 = os.listdir(origin + path)
        loc = []
        for i in item3:
            if 'DS' not in i:
                loc.append(i)
        path = path + "/" + loc[0]
        
        # fourth folder, series number
        for i in self.root.iter("seriesIdentifier"):
            sid = i.text
            break
        path = path + "/" + "S" + sid
        
        
        # finally finds the scan, checks to see if its a .nii file
        items = os.listdir(origin + path)
        scans = 0
        scan = []
        for i in items:
            curr = i.split(".")
            if curr[len(curr) - 1] == "nii":
                scan.append(i)
                scans += 1
        if scans > 1:
            imageid = self.getderiveduid()
            for i in scan:
                parsed = i.replace(".nii", "").split("_")
                for x in parsed:
                    if x == imageid:
                        return path + "/" + i
        return origin + path + "/" + scan[0]


# In[2]:


import os
from collections import deque as Q

import tensorflow as tf
import numpy as np

from tensorflow.python.platform import gfile
import nibabel as nib


# Process images of this size.
# image size of 256 x 256 x 166
# If one alters this, then the entire model
# architecture will change and any model would need to be retrained.
IMAGE_SIZE = 256  # W and L (x and y)
IMAGE_SLICE = 166  # depth / slice (z)

# Global constants describing the ADNI data set.
NUM_CLASSES = 3
NUM_EXAMPLES_PER_EPOCH_FOR_TRAIN = 9
NUM_EXAMPLES_PER_EPOCH_FOR_EVAL = 1


# In[3]:


def _generate_image_and_label_batch(image, label, min_queue_examples,
                                    batch_size):
    """Construct a queued batch of images and labels.
  Args:
    image: 3-D Tensor of [width, height, slice] of type.float32.
    label: 1-D Tensor of type.int32
    min_queue_examples: int32, minimum number of samples to retain
      in the queue that provides of batches of examples.
    batch_size: Number of images per batch.
  Returns:
    images: Images. 4D tensor of [batch_size, width, height, slice] size.
    labels: Labels. 1D tensor of [batch_size] size.
  """
    # Create a queue that shuffles the examples, and then
    # read 'batch_size' images + labels from the example queue.
    num_preprocess_threads = 3
    images, label_batch = tf.train.shuffle_batch(
        [image, label],
        batch_size=batch_size,
        num_threads=num_preprocess_threads,
        capacity=min_queue_examples + 3 * batch_size,
        min_after_dequeue=min_queue_examples)

    # Display the training images in the visualizer.
    tf.summary.image('images', images)

    return images, tf.reshape(label_batch, [batch_size])


# In[4]:


def read_ADNI_image(filename_queue):
    """Reads NiFTi data file, next in the queue.
  Args:
    filename_queue: A queue of strings with the filenames to read from.
  Returns:
    An object representing a single example, with the following fields:
      height: number of rows in the result (256)
      width: number of columns in the result (256)
      slice: number of slices (3d mri) (170)
      key: file name
      label: an int32 Tensor with the label 0, 1, or 2.
      float32image: a [height, width, depth] float32 Tensor with the image data
      nim: nii image reader/writer
      rawImage: Numpy array
  """

    class mri(object):
        pass

    result = mri()
    ff = filename_queue.popleft().split('||') # FIFO so popleft
    im = ff[0]
    result.nim = nib.load(im)
    result.data = result.nim.get_data().astype('float32')
    #result.rawImage = result.nim.data # numpy array
    # padding to make sure the numpy array is the right shape
    dims = result.nim.shape
    if dims[0] != 256 or dims[1] != 256 or dims[2] != 166:
        print('fails')
    result.float32image = tf.convert_to_tensor(result.data)
    # Dimensions of the images in the ADNI dataset.
    result.height = 256
    result.width = 256
    result.slice = 166  # brain 'slice' of an MRI
    # file name for reference later
    result.filename = result.nim.get_filename()
    result.label = int(ff[1])

    return result


# In[5]:


def img_inputs(eval_data, data_dir, batch_size):
    """Construct input
  Args:
    eval_data: bool, indicating if one should use the train or eval data set.
    data_dir: Path to data directory.
    batch_size: Number of images per batch.
  Returns:
    images: Images. 4D tensor of [batch_size, width, height, slice] size.
    labels: Labels. 1D tensor of [batch_size] size.
  """
    if not eval_data:
        all = os.listdir(data_dir)
        xmls = []
        for x in all:
            temp = x.split(".")
            if temp[len(temp)-1] == "xml":
                xmls.append(x)
        ims = []
        for f in xmls:
            r = XMLReader(data_dir+'/'+f)
            p = r.path_to_scan(data_dir)
            la = r.subject_status()
            if not gfile.Exists(p):
                raise ValueError('Failed to find file: ' + f)
            else:
                ims.append(p+'||'+ str(la))
        num_examples_per_epoch = NUM_EXAMPLES_PER_EPOCH_FOR_TRAIN
    else:
        all = os.listdir(data_dir)
        xmls = []
        for x in all:
            temp = x.split(".")
            if temp[len(temp)-1] == "xml":
                xmls.append(x)
        ims = []
        for f in xmls:
            r = XMLReader(data_dir+'/'+f)
            p = r.path_to_scan(data_dir)
            la = r.subject_status()
            if not gfile.Exists(p):
                raise ValueError('Failed to find file: ' + f)
            else:
                ims.append(p+'||'+str(la))
        num_examples_per_epoch = NUM_EXAMPLES_PER_EPOCH_FOR_EVAL

    # Create a queue that produces the filenames to read.
    filename_queue = Q(iterable=ims, maxlen=len(ims))
    
    # Read examples from files in the filename queue.
    im = read_ADNI_image(filename_queue)

    # Ensure that the random shuffling has good mixing properties.
    min_fraction_of_examples_in_queue = 0.4
    min_queue_examples = int(num_examples_per_epoch *
                             min_fraction_of_examples_in_queue)

    # Generate a batch of images and labels by building up a queue of examples.
    return _generate_image_and_label_batch(im.float32image, im.label, min_queue_examples, batch_size)
    


# In[6]:


"""Builds the AD network.
Summary of available functions:
 # Compute input images and labels for training. If you would like to run
 # evaluations, use input() instead.
 inputs, labels = inputs()
 # Compute inference on the model inputs to make a prediction.
 predictions = inference(inputs)
 # Compute the total loss of the prediction with respect to the labels.
 loss = loss(predictions, labels)
 # Create a graph to run one step of training with respect to the loss.
 train_op = train(loss, global_step)
"""
import re

import tensorflow as tf

FLAGS = tf.app.flags.FLAGS

# Basic model parameters.
#tf.app.flags.DEFINE_string('f', '', 'kernel')
tf.app.flags.DEFINE_integer('batch_size', 3,
                            """Number of images to process in a batch.""")
tf.app.flags.DEFINE_string('data_dir', './ADNI',
                           """Path to the ADNI data directory.""")

# Global constants describing the ADNI data set.
#IMAGE_SIZE = ADNI_input.IMAGE_SIZE
#NUM_CLASSES = ADNI_input.NUM_CLASSES
#NUM_EXAMPLES_PER_EPOCH_FOR_TRAIN = ADNI_input.NUM_EXAMPLES_PER_EPOCH_FOR_TRAIN
#NUM_EXAMPLES_PER_EPOCH_FOR_EVAL = ADNI_input.NUM_EXAMPLES_PER_EPOCH_FOR_EVAL


# Constants describing the training process.
MOVING_AVERAGE_DECAY = 0.9999     # The decay to use for the moving average.
NUM_EPOCHS_PER_DECAY = 350.0      # Epochs after which learning rate decays.
LEARNING_RATE_DECAY_FACTOR = 0.1  # Learning rate decay factor.
INITIAL_LEARNING_RATE = 0.1       # Initial learning rate.

# If a model is trained with multiple GPU's prefix all Op names with tower_name
# to differentiate the operations. Note that this prefix is removed from the
# names of the summaries when visualizing a model.
TOWER_NAME = 'tower'


# In[20]:


def _variable_on_cpu(name, shape, initializer):
    """Helper to create a Variable stored on CPU memory.
    Args:
    name: name of the variable
    shape: list of ints
    initializer: initializer for Variable
    Returns:
    Variable Tensor
    """
    with tf.device('/cpu:0'):
        var = tf.get_variable(name, shape, initializer=initializer)
    return var


def _variable_with_weight_decay(name, shape, stddev, wd):
    """Helper to create an initialized Variable with weight decay.
    Note that the Variable is initialized with a truncated normal distribution.
    A weight decay is added only if one is specified.
    Args:
    name: name of the variable
    shape: list of ints
    stddev: standard deviation of a truncated Gaussian
    wd: add L2Loss weight decay multiplied by this float. If None, weight
        decay is not added for this Variable.
    Returns:
    Variable Tensor
    """
    var = _variable_on_cpu(name, shape,
                         tf.truncated_normal_initializer(stddev=stddev))
    if wd:
        weight_decay = tf.multiply(tf.nn.l2_loss(var), wd, name='weight_loss__raw_')
        tf.add_to_collection('losses', weight_decay)
    return var


# In[8]:


def inputs(eval_data):
    """Construct input for AD evaluation and training.
    Args:
    eval_data: bool, indicating if one should use the train or eval data set.
    Returns:
    images: Images. 4D tensor of [batch_size, height, width, slice] size.
    labels: Labels. 1D tensor of [batch_size] size.
    """
    if eval_data is True:
        data_dir = "./ADNI/test_data"
    else:
        data_dir = "./ADNI/train_data"
    return img_inputs(eval_data=eval_data, data_dir=data_dir,batch_size=FLAGS.batch_size)


# In[9]:


def inference(images):
    """Build the CNN model.
    Args:
    images: Images returned from  inputs().
    Returns:
    Logits.
    """
  # conv1
    with tf.variable_scope('conv1') as scope:
        kernel = _variable_with_weight_decay('weights', shape=[5, 5, 166, 64],
                                             stddev=1e-4, wd=0.0)
        conv = tf.nn.conv2d(images, kernel, [1, 1, 1, 1], padding='SAME')
        biases = _variable_on_cpu('biases', [64], tf.constant_initializer(0.0))
        bias = tf.nn.bias_add(conv, biases)
        conv1 = tf.nn.relu(bias, name=scope.name)

  # pool1
    pool1 = tf.nn.max_pool(conv1, ksize=[1, 3, 3, 1], strides=[1, 2, 2, 1],
                         padding='SAME', name='pool1')
  # norm1
    norm1 = tf.nn.lrn(pool1, 4, bias=1.0, alpha=0.001 / 9.0, beta=0.75, # input / (bias + alpha * sqr_sum ** beta)
                    name='norm1')

  # conv2
    with tf.variable_scope('conv2') as scope:
        kernel = _variable_with_weight_decay('weights', shape=[5, 5, 64, 64],
                                             stddev=1e-4, wd=0.0)
        conv = tf.nn.conv2d(norm1, kernel, [1, 1, 1, 1], padding='SAME')
        biases = _variable_on_cpu('biases', [64], tf.constant_initializer(0.1))
        bias = tf.nn.bias_add(conv, biases)
        conv2 = tf.nn.relu(bias, name=scope.name)

  # norm2
    norm2 = tf.nn.lrn(conv2, 4, bias=1.0, alpha=0.001 / 9.0, beta=0.75,
                    name='norm2')
  # pool2
    pool2 = tf.nn.max_pool(norm2, ksize=[1, 3, 3, 1],
                         strides=[1, 2, 2, 1], padding='SAME', name='pool2')

  # local3
    with tf.variable_scope('local3') as scope:
    # Move everything into depth so we can perform a single matrix multiply.
        dim = 1
        for d in pool2.get_shape()[1:].as_list():
            dim *= d
        reshape = tf.reshape(pool2, [FLAGS.batch_size, dim])

        weights = _variable_with_weight_decay('weights', shape=[dim, 384], stddev=0.04, wd=0.004)
        biases = _variable_on_cpu('biases', [384], tf.constant_initializer(0.1))
        local3 = tf.nn.relu(tf.matmul(reshape, weights) + biases, name=scope.name)

  # local4
    with tf.variable_scope('local4') as scope:
        weights = _variable_with_weight_decay('weights', shape=[384, 192], stddev=0.04, wd=0.004)
        biases = _variable_on_cpu('biases', [192], tf.constant_initializer(0.1))
        local4 = tf.nn.relu(tf.matmul(local3, weights) + biases, name=scope.name)

  # softmax, i.e. softmax(WX + b)
    with tf.variable_scope('softmax_linear') as scope:
        weights = _variable_with_weight_decay('weights', [192, NUM_CLASSES],
                                              stddev=1/192.0, wd=0.0)
        biases = _variable_on_cpu('biases', [NUM_CLASSES],
                                  tf.constant_initializer(0.0))
        softmax_linear = tf.add(tf.matmul(local4, weights), biases, name=scope.name)

    return softmax_linear


# In[10]:


def loss(logits, labels):
    """Add L2Loss to all the trainable variables.
    Add summary for for "Loss" and "Loss/avg".
    Args:
    logits: Logits from inference().
    labels: Labels from distorted_inputs or inputs(). 1-D tensor
            of shape [batch_size]
    Returns:
    Loss tensor of type float.
    """
    # Reshape the labels into a dense Tensor of
    # shape [batch_size, NUM_CLASSES].
    sparse_labels = tf.reshape(labels, [FLAGS.batch_size, 1])
    indices = tf.reshape(tf.range(FLAGS.batch_size), [FLAGS.batch_size, 1])
    concated = tf.concat([indices, sparse_labels],1)
    dense_labels = tf.sparse_to_dense(concated,
                                    [FLAGS.batch_size, NUM_CLASSES],
                                    1.0, 0.0)

  # Calculate the average cross entropy loss across the batch.
    cross_entropy = tf.nn.softmax_cross_entropy_with_logits(logits = logits, labels = dense_labels, name='cross_entropy_per_example')
    cross_entropy_mean = tf.reduce_mean(cross_entropy, name='cross_entropy')
    tf.add_to_collection('losses', cross_entropy_mean)

    # The total loss is defined as the cross entropy loss plus all of the weight
    # decay terms (L2 loss).
    return tf.add_n(tf.get_collection('losses'), name='total_loss')


# In[11]:


def _add_loss_summaries(total_loss):
    """Add summaries for losses in the CNN.
    Generates moving average for all losses and associated summaries for
    visualizing the performance of the network.
    Args:
    total_loss: Total loss from loss().
    Returns:
    loss_averages_op: op for generating moving averages of losses.
    """
    # Compute the moving average of all individual losses and the total loss.
    loss_averages = tf.train.ExponentialMovingAverage(0.9, name='avg')
    losses = tf.get_collection('losses')
    loss_averages_op = loss_averages.apply(losses + [total_loss])

  # Attach a scalar summary to all individual losses and the total loss; do the
  # same for the averaged version of the losses.
    for l in losses + [total_loss]:
        # Name each loss as '(raw)' and name the moving average version of the loss
        # as the original loss name.
        tf.summary.scalar(l.op.name +' (raw)', l)
        tf.summary.scalar(l.op.name, loss_averages.average(l))
    return loss_averages_op


# In[12]:


def CNN_train(total_loss, global_step):
    """Train CNN model.
    Create an optimizer and apply to all trainable variables. Add moving
    average for all trainable variables.
    Args:
    total_loss: Total loss from loss().
    global_step: Integer Variable counting the number of training steps
      processed.
    Returns:
    train_op: op for training.
    """
    # Variables that affect learning rate.
    num_batches_per_epoch = NUM_EXAMPLES_PER_EPOCH_FOR_TRAIN / FLAGS.batch_size
    decay_steps = int(num_batches_per_epoch * NUM_EPOCHS_PER_DECAY)

    # Decay the learning rate exponentially based on the number of steps.
    lr = tf.train.exponential_decay(INITIAL_LEARNING_RATE,
                                  global_step,
                                  decay_steps,
                                  LEARNING_RATE_DECAY_FACTOR,
                                  staircase=True)
    tf.summary.scalar('learning_rate', lr)

    # Generate moving averages of all losses and associated summaries.
    loss_averages_op = _add_loss_summaries(total_loss)

    # Compute gradients.
    with tf.control_dependencies([loss_averages_op]):
        opt = tf.train.GradientDescentOptimizer(lr)
        grads = opt.compute_gradients(total_loss)

  # Apply gradients.
    apply_gradient_op = opt.apply_gradients(grads, global_step=global_step)

    # Add histograms for trainable variables.
    for var in tf.trainable_variables():
        tf.summary.histogram(var.op.name, var)

  # Add histograms for gradients.
    for grad, var in grads:
        if grad is not None:
            tf.summary.histogram(var.op.name + '/gradients', grad)

  # Track the moving averages of all trainable variables.
    variable_averages = tf.train.ExponentialMovingAverage(
      MOVING_AVERAGE_DECAY, global_step)
    variables_averages_op = variable_averages.apply(tf.trainable_variables())

    with tf.control_dependencies([apply_gradient_op, variables_averages_op]):
        train_op = tf.no_op(name='train')

    return train_op


# In[13]:


from six.moves import xrange
import tensorflow as tf

FLAGS = tf.app.flags.FLAGS

tf.app.flags.DEFINE_string('train_dir', './ADNI/train',
                           """Directory where to write event logs """
                           """and checkpoint.""")
tf.app.flags.DEFINE_integer('max_steps', 1,
                            """Number of batches to run.""")
tf.app.flags.DEFINE_boolean('log_device_placement', False,
                            """Whether to log device placement.""")


# In[14]:


def train():
    """Train the CNN for a number of steps."""
    with tf.Graph().as_default():
        global_step = tf.Variable(0, trainable=False)

        # Get images and labels for ADNI.
        images, labels = inputs(False)

        # Build a Graph that computes the logits predictions from the
        # inference model.
        logits = inference(images)

        # Calculate loss.
        loss1 = loss(logits, labels)

        # Build a Graph that trains the model with one batch of examples and
        # updates the model parameters.
        train_op = CNN_train(loss1, global_step)

        # Create a saver.
        saver = tf.train.Saver(tf.all_variables())

        # Build the summary operation based on the TF collection of Summaries.
        summary_op = tf.summary.merge_all()

        # Build an initialization operation to run below.
        init = tf.global_variables_initializer()

        # Start running operations on the Graph.
        sess = tf.Session(config=tf.ConfigProto(
            log_device_placement=FLAGS.log_device_placement))
        sess.run(init)

        # Start the queue runners.
        tf.train.start_queue_runners(sess=sess)

        summary_writer = tf.summary.FileWriter(FLAGS.train_dir,
                                                graph_def=sess.graph_def)

    for step in xrange(FLAGS.max_steps):
    # Save the model checkpoint periodically.
        if step % 1000 == 0 or (step + 1) == FLAGS.max_steps:
            checkpoint_path = os.path.join(FLAGS.train_dir, 'model.ckpt')
            saver.save(sess, checkpoint_path, global_step=step)


# In[21]:


train()


# # Evaluation

# In[22]:


"""Evaluation for CNN."""

from datetime import datetime
import math
import time

import numpy as np
import tensorflow as tf

FLAGS = tf.app.flags.FLAGS

tf.app.flags.DEFINE_string('eval_dir', './ADNI/test',
                           """Directory where to write event logs.""")
tf.app.flags.DEFINE_string('eval_data', 'test',
                           """Either 'test' or 'train_eval'.""")
tf.app.flags.DEFINE_string('checkpoint_dir', './ADNI/train',
                           """Directory where to read model checkpoints.""")
tf.app.flags.DEFINE_integer('eval_interval_secs', 1,
                            """How often to run the eval.""")
tf.app.flags.DEFINE_integer('num_examples', 8,
                            """Number of examples to run.""")
tf.app.flags.DEFINE_boolean('run_once', False,
                         """Whether to run eval only once.""")


# In[23]:


def eval_once(saver, summary_writer, top_k_op, summary_op):
    """Run Eval once.
    Args:
    saver: Saver.
    summary_writer: Summary writer.
    top_k_op: Top K op.
    summary_op: Summary op.
    """
    with tf.Session() as sess:
        ckpt = tf.train.get_checkpoint_state(FLAGS.checkpoint_dir)
        if ckpt and ckpt.model_checkpoint_path:
            # Restores from checkpoint
            saver.restore(sess, ckpt.model_checkpoint_path)
            # Assuming model_checkpoint_path looks something like:
            #   /my-favorite-path/ADNI/model.ckpt-0,
            global_step = ckpt.model_checkpoint_path.split('/')[-1].split('-')[-1]
        else:
            print('No checkpoint file found')
            return

    # Start the queue runners.
        coord = tf.train.Coordinator()
        try:
            threads = []
            for qr in tf.get_collection(tf.GraphKeys.QUEUE_RUNNERS):
                threads.extend(qr.create_threads(sess, coord=coord, daemon=True,
                                                 start=True))

            num_iter = int(math.ceil(FLAGS.num_examples / FLAGS.batch_size))
            true_count = 0  # Counts the number of correct predictions.
            total_sample_count = num_iter * FLAGS.batch_size
            step = 0
            while step < num_iter and not coord.should_stop():
                predictions = sess.run([top_k_op])
                true_count += np.sum(predictions)
                step += 1

            # Compute precision @ 1.
            precision = true_count / total_sample_count
            print('%s: precision @ 1 = %.3f' % (datetime.now(), precision))

        except Exception as e:
            coord.request_stop(e)

        coord.request_stop()
        coord.join(threads, stop_grace_period_secs=10)


# In[24]:


def evaluate():
    """Eval CNN for a number of steps."""
    with tf.Graph().as_default():
    # Get images and labels for ADNI.
        eval_data = FLAGS.eval_data == 'test'
        images, labels = inputs(eval_data=eval_data)

        # Build a Graph that computes the logits predictions from the
        # inference model.
        logits = inference(images)

        # Calculate predictions.
        top_k_op = tf.nn.in_top_k(logits, labels, 1)

        # Restore the moving average version of the learned variables for eval.
        variable_averages = tf.train.ExponentialMovingAverage(
            MOVING_AVERAGE_DECAY)
        variables_to_restore = variable_averages.variables_to_restore()
        saver = tf.train.Saver(variables_to_restore)

        # Build the summary operation based on the TF collection of Summaries.
        summary_op = tf.summary.merge_all()

        graph_def = tf.get_default_graph().as_graph_def()
        summary_writer = tf.summary.FileWriter(FLAGS.eval_dir,
                                                graph_def=graph_def)

        while True:
            eval_once(saver, summary_writer, top_k_op, summary_op)
            if FLAGS.run_once:
                break
            time.sleep(FLAGS.eval_interval_secs)


# In[25]:


evaluate()


import argparse
import sys

import tensorflow as tf
from tensorflow.contrib.framework.python.ops.variables import get_or_create_global_step
from tensorflow.python.platform import tf_logging as logging
import inception_preprocessing
from inception_resnet_v2 import inception_resnet_v2, inception_resnet_v2_arg_scope
import os
import time
slim = tf.contrib.slim

FLAGS = None

image_size = 299
hidden1_size = 64

num_classes = 10
gray_scale = True

file_pattern = 'Hela_%s_*.tfrecord'
file_pattern_for_counting='Hela'

items_to_descriptions = {
    'image': 'A 3-channel RGB coloured flower image that is either tulips, sunflowers, roses, dandelion, or daisy.',
    'label': 'A label that is as such -- 0:daisy, 1:dandelion, 2:roses, 3:sunflowers, 4:tulips'
}


#================= TRAINING INFORMATION ==================
#State the number of epochs to train
num_epochs = 5

#State your batch size
batch_size = 8

#Learning rate information and configuration (Up to you to experiment)
initial_learning_rate = 0.0002
learning_rate_decay_factor = 0.7
num_epochs_before_decay = 2

#============== DATASET LOADING ======================

def label_to_name(labels_file):
    labels = open(labels_file, 'r')
    labels_to_name = {}
    for line in labels:
        label, string_name = line.split(':')
        string_name = string_name[:-1]  # Remove newline
        labels_to_name[int(label)] = string_name
    return labels_to_name


#We now create a function that creates a Dataset class which will give us many TFRecord files to feed in the examples into a queue in parallel.
def get_split(split_name, dataset_dir, labels_to_name, file_pattern=file_pattern, file_pattern_for_counting=file_pattern_for_counting):
    '''
    Obtains the split - training or validation - to create a Dataset class for feeding the examples into a queue later on. This function will
    set up the decoder and dataset information all into one Dataset class so that you can avoid the brute work later on.
    Your file_pattern is very important in locating the files later.

    INPUTS:
    - split_name(str): 'train' or 'validation'. Used to get the correct data split of tfrecord files
    - dataset_dir(str): the dataset directory where the tfrecord files are located
    - file_pattern(str): the file name structure of the tfrecord files in order to get the correct data
    - file_pattern_for_counting(str): the string name to identify your tfrecord files for counting

    OUTPUTS:
    - dataset (Dataset): A Dataset class object where we can read its various components for easier batch creation later.
    '''

    #First check whether the split_name is train or validation
    if split_name not in ['train', 'validation']:
        raise ValueError('The split_name %s is not recognized. Please input either train or validation as the split_name' % (split_name))

    #Create the full path for a general file_pattern to locate the tfrecord_files
    file_pattern_path = os.path.join(dataset_dir, file_pattern % (split_name))

    #Count the total number of examples in all of these shard
    num_samples = 0
    file_pattern_for_counting = file_pattern_for_counting + '_' + split_name
    tfrecords_to_count = [os.path.join(dataset_dir, file) for file in os.listdir(dataset_dir) if file.startswith(file_pattern_for_counting)]
    for tfrecord_file in tfrecords_to_count:
        for record in tf.python_io.tf_record_iterator(tfrecord_file):
            num_samples += 1

    #Create a reader, which must be a TFRecord reader in this case
    reader = tf.TFRecordReader

    #Create the keys_to_features dictionary for the decoder
    keys_to_features = {
      'image/encoded': tf.FixedLenFeature((), tf.string, default_value=''),
      'image/format': tf.FixedLenFeature((), tf.string, default_value='jpg'),
      'image/class/label': tf.FixedLenFeature(
          [], tf.int64, default_value=tf.zeros([], dtype=tf.int64)),
    }

    #Create the items_to_handlers dictionary for the decoder.
    items_to_handlers = {
    'image': slim.tfexample_decoder.Image(),
    'label': slim.tfexample_decoder.Tensor('image/class/label'),
    }

    #Start to create the decoder
    decoder = slim.tfexample_decoder.TFExampleDecoder(keys_to_features, items_to_handlers)

    #Create the labels_to_name file
    labels_to_name_dict = labels_to_name

    #Actually create the dataset
    dataset = slim.dataset.Dataset(
        data_sources = file_pattern_path,
        decoder = decoder,
        reader = reader,
        num_readers = 4,
        num_samples = num_samples,
        num_classes = num_classes,
        labels_to_name = labels_to_name_dict,
        items_to_descriptions = items_to_descriptions)

    return dataset

def load_batch(dataset, batch_size, height=image_size, width=image_size, is_training=True):
    '''
    Loads a batch for training.

    INPUTS:
    - dataset(Dataset): a Dataset class object that is created from the get_split function
    - batch_size(int): determines how big of a batch to train
    - height(int): the height of the image to resize to during preprocessing
    - width(int): the width of the image to resize to during preprocessing
    - is_training(bool): to determine whether to perform a training or evaluation preprocessing

    OUTPUTS:
    - images(Tensor): a Tensor of the shape (batch_size, height, width, channels) that contain one batch of images
    - labels(Tensor): the batch's labels with the shape (batch_size,) (requires one_hot_encoding).

    '''
    #First create the data_provider object
    data_provider = slim.dataset_data_provider.DatasetDataProvider(
        dataset,
        common_queue_capacity = 24 + 3 * batch_size,
        common_queue_min = 24)

    #Obtain the raw image using the get method
    raw_image, label = data_provider.get(['image', 'label'])

    #Perform the correct preprocessing for this image depending if it is training or evaluating
    image = inception_preprocessing.preprocess_image(raw_image, height, width, is_training)

    #As for the raw images, we just do a simple reshape to batch it up
    raw_image = tf.expand_dims(raw_image, 0)
    raw_image = tf.image.resize_nearest_neighbor(raw_image, [height, width])
    raw_image = tf.squeeze(raw_image)

    #Batch up the image by enqueing the tensors internally in a FIFO queue and dequeueing many elements with tf.train.batch.
    images, raw_images, labels = tf.train.batch(
        [image, raw_image, label],
        batch_size = batch_size,
        num_threads = 4,
        capacity = 4 * batch_size,
        allow_smaller_final_batch = True)

    return images, raw_images, labels

def get_pretrained_net(train_data, train_images):
    # Create the model inference
    with slim.arg_scope(inception_resnet_v2_arg_scope()):
        _, end_points = inception_resnet_v2(train_images, num_classes=train_data.num_classes, is_training=True)

    # Define the scopes that you want to exclude for restoration
    if gray_scale:
        exclude = ['InceptionResnetV2/Logits', 'InceptionResnetV2/AuxLogits', 'InceptionResnetV2/Conv2d_1a_3x3']
    else:
        exclude = ['InceptionResnetV2/Logits', 'InceptionResnetV2/AuxLogits']

    variables_to_restore = slim.get_variables_to_restore(exclude=exclude)
    # print('variables_to_restore: ', variables_to_restore)
    return exclude, end_points, variables_to_restore

def add_custom_layers(end_points, train_data):
    # Define new layers (see InceptionResnet implementation to know why)
    pre_logits_flatten = end_points['PreLogitsFlatten']
    hidden1 = slim.fully_connected(pre_logits_flatten, hidden1_size, activation_fn=None,
                                   scope='AddedClassifier/Hidden1')
    hidden1 = slim.dropout(hidden1, 1.0, is_training=True,
                           scope='AddedClassifier/Hidden1/Dropout')

    custom_logits = slim.fully_connected(hidden1, train_data.num_classes, activation_fn=None,
                                         scope='AddedClassifier/Logits')
    end_points['Logits'] = custom_logits
    end_points['Predictions'] = tf.nn.softmax(custom_logits, name='Predictions')

    return end_points, custom_logits

def get_batch_data(data_name, is_training, labels_to_name):
    data = get_split(data_name, FLAGS.dataset_dir, file_pattern=file_pattern, labels_to_name=labels_to_name)
    images, _, labels = load_batch(data, batch_size=batch_size, is_training=is_training)
    return data, images, labels

def get_unrestored_variables(exclude):
    unrestored_variables = []
    for scope in exclude:
        # not_restored_vars.extend([var for var in tf.get_collection(tf.GraphKeys.GLOBAL_VARIABLES, scope=scope)])
        unrestored_variables = tf.contrib.framework.get_variables(scope)
        unrestored_vars_init = tf.variables_initializer(unrestored_variables)
    # print("unrestored variables: ", unrestored_variables)
    return unrestored_variables, unrestored_vars_init

def get_trainable_variables(scope_name):
    return [var for var in tf.get_collection(tf.GraphKeys.GLOBAL_VARIABLES, scope=scope_name)]

def main(_):
    #Create the log directory here. Must be done here otherwise import will activate this unneededly.
    if not os.path.exists(FLAGS.log_dir):
        os.mkdir(FLAGS.log_dir)

    labels_to_name=label_to_name(FLAGS.labels_file)

    #======================= TRAINING PROCESS =========================
    #Now we start to construct the graph and build our model
    graph = tf.Graph()
    with graph.as_default():

        tf.logging.set_verbosity(tf.logging.INFO) #Set the verbosity to INFO level

        #First create the dataset and load one batch
        train_data = get_split('train', FLAGS.dataset_dir, file_pattern=file_pattern, labels_to_name=labels_to_name)
        train_images, _, train_labels = load_batch(train_data, batch_size=batch_size, is_training=True)
        # train_data, train_images, train_labels = get_batch_data('train', True, labels_to_name=labels_to_name)
        val_data, val_images, val_labels = get_batch_data('validation', False, labels_to_name=labels_to_name)

        #Know the number steps to take before decaying the learning rate and batches per epoch
        num_batches_per_epoch = int(train_data.num_samples / batch_size)
        num_steps_per_epoch = num_batches_per_epoch #Because one step is one batch processed=
        print("num_batches_per_epoch: ", num_batches_per_epoch, " | num_steps_per_epoch: ", num_steps_per_epoch)
        decay_steps = int(num_epochs_before_decay * num_steps_per_epoch)

        exclude, end_points, variables_to_restore = get_pretrained_net(train_data, train_images)
        end_points, custom_logits = add_custom_layers(end_points, train_data)

        trainable_variables = get_trainable_variables("AddedClassifier")
        unrestored_variables, unrestored_variables_init = get_unrestored_variables(exclude)

        trainable_variables.extend(unrestored_variables)
        # print("trainable variables: ", trainable_variables)

        trainable_variables_init = tf.variables_initializer(trainable_variables)

        # Perform one-hot-encoding of the labels (Try one-hot-encoding within the load_batch function!)
        train_one_hot_labels = slim.one_hot_encoding(train_labels, train_data.num_classes)

        #Performs the equivalent to tf.nn.sparse_softmax_cross_entropy_with_logits but enhanced with checks
        train_loss = tf.losses.softmax_cross_entropy(onehot_labels = train_one_hot_labels, logits = custom_logits)
        train_total_loss = tf.losses.get_total_loss()    #obtain the regularization losses as well

        # Create the global step for monitoring the learning_rate and training.
        global_step = get_or_create_global_step()
        global_step_init = tf.variables_initializer([global_step])

        #Define your exponentially decaying learning rate
        lr = tf.train.exponential_decay(
            learning_rate = initial_learning_rate,
            global_step = global_step,
            decay_steps = decay_steps,
            decay_rate = learning_rate_decay_factor,
            staircase = True)

        #Now we can define the optimizer that takes on the learning rate
        optimizer = tf.train.AdamOptimizer(learning_rate = lr)

        #Create the train_op.
        # train_op = slim.learning.create_train_op(train_total_loss, optimizer)
        train_op = optimizer.minimize(train_total_loss, var_list=trainable_variables)

        #State the metrics that you want to predict. We get a predictions that is not one_hot_encoded.
        predictions = tf.argmax(end_points['Predictions'], 1)
        probabilities = end_points['Predictions']

        train_accuracy, train_accuracy_update = tf.contrib.metrics.streaming_accuracy(predictions, train_labels)
        train_metrics_op = tf.group(train_accuracy_update, probabilities)

        # validation_accuracy, validation_accuracy_update = tf.contrib.metrics.streaming_accuracy(predictions, validation_labels)
        # validation_metrics_op = tf.group(validation_accuracy_update)

        #Now finally create all the summaries you need to monitor and group them into one summary op.
        tf.summary.scalar('losses/Total_Loss', train_total_loss)
        tf.summary.scalar('accuracy', train_accuracy)
        tf.summary.scalar('learning_rate', lr)
        my_summary_op = tf.summary.merge_all()

        #Now we need to create a training step function that runs both the train_op, metrics_op and updates the global_step concurrently.
        def train_step(sess, train_op, global_step_op):
            '''
            Simply runs a session for the three arguments provided and gives a logging on the time elapsed for each global step
            '''
            #Check the time for each sess run

            start_time = time.time()
            total_loss, global_step_count, train_accuracy_val = sess.run([train_op, global_step, train_metrics_op])
            time_elapsed = time.time() - start_time

            #Run the logging to print some results
            logging.info('global step %s: loss: train acc: %.4f (%.2f sec/step) %.4f', global_step_count, total_loss,
                         time_elapsed, train_accuracy_val)

            return total_loss, global_step_count

        saver = tf.train.Saver(variables_to_restore)
        def restore_fn():
            return tf.contrib.framework.assign_from_checkpoint_fn(FLAGS.checkpoint_file, variables_to_restore)
        def init_fn(sess):
            return saver.restore(sess, FLAGS.checkpoint_file)
        # Run the managed session
        config = tf.ConfigProto()
        config.gpu_options.allow_growth = True
        # sess = tf.Session(config=config, graph=graph)
        sv = tf.train.Supervisor(logdir=FLAGS.log_dir, summary_op=None, init_fn=init_fn, graph=graph)

    # with tf.Session(config=config, graph=graph)as sess:
    with sv.managed_session() as sess:
        # with sv.managed_session() as sess:
        # init_global = tf.global_variables_initializer()
        # sess.run(init_global)
        # init_local = tf.local_variables_initializer()
        # sess.run(init_local)
        # # restore_fn(sess)
        # restore_fn = restore_fn()
        # restore_fn(sess)
        # sess.run(global_step_init)
        sess.run(trainable_variables_init)

        for step in range(num_steps_per_epoch * num_epochs):
            # At the start of every epoch, show the vital information:
            print("step", step)
            if step % num_batches_per_epoch == 0:
                logging.info('Epoch %s/%s', step / num_batches_per_epoch + 1, num_epochs)
                learning_rate_value, train_accuracy_value = sess.run([lr, train_accuracy])
                logging.info('Current Learning Rate: %s', learning_rate_value)
                logging.info('Current Streaming Accuracy: %s', train_accuracy_value)
                logging.info('Epoch: %s/%s', step / num_batches_per_epoch + 1, num_epochs)
                # logging.info('Current Validation Accuracy: %.4f', sess.run(validation_accuracy))


                # optionally, print your logits and predictions for a sanity check that things are going fine.
                # logits_value, probabilities_value, predictions_value, labels_value = sess.run(
                #     [custom_logits, probabilities, predictions, train_labels])
                # print (train_labels)
                # labels_value = sess.run([train_labels])
                # print ('labels: ', labels_value)
                logits_value, probabilities_value, predictions_value, labels_value = sess.run(
                    [custom_logits, probabilities, predictions, train_labels])
                print('logits: \n', logits_value)
                print('Probabilities: \n', probabilities_value)
                print('predictions: \n', predictions_value)
                print('Labels:\n:', labels_value)

            # Log to summaries every 50 step.
            if step % 50 == 0:
                train_loss, _ = train_step(sess, train_op, sv.global_step)
                summaries = sess.run(my_summary_op)
                # sv.summary_computed(sess, summaries)

            if step % 100 == 0:
                train_loss, _ = train_step(sess, train_op, sv.global_step)
                logging.info('Current train Accuracy: %s', sess.run(train_accuracy))
                # accuracy_value = validation_step(sess, validation_metrics_op, sv.global_step)

                # If not, simply run the training step
            else:
                # loss, _ = train_step(sess, train_op, sv.global_step)
                loss, _ = train_step(sess, train_op, sv.global_step)

        # We log the final training loss and accuracy
        logging.info('Final Loss: %s', train_loss)
        logging.info('Final Accuracy: %s', sess.run(train_accuracy))

        # Once all the training has been done, save the log files and checkpoint model
        logging.info('Finished training! Saving model to disk now.')
        # saver.save(sess, "./flowers_model.ckpt")
        # sv.saver.save(sess, sv.save_path, global_step=sv.global_step)

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument(
        '--dataset_dir',
        help='GCS or local paths to evaluation data',
        # nargs='+',
        required=True
    )

    parser.add_argument(
        '--log_dir',
        # nargs='+',
        required=True
    )

    parser.add_argument(
        '--checkpoint_file',
        # nargs='+',
        required=True
    )

    parser.add_argument(
        '--labels_file',
        # nargs="+",
        required=True
    )

    FLAGS, unparsed = parser.parse_known_args()
    print(FLAGS)
    tf.app.run(main=main, argv=[sys.argv[0]] + unparsed)


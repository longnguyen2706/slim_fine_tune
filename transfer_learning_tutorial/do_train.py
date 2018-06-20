import argparse
import sys

import tensorflow as tf
from tensorflow.python.platform import tf_logging as logging
import load_data
import net
slim = tf.contrib.slim


FLAGS = None

def get_batch(data_dir, batch_size, is_train):
    filenames, labels = load_data.list_images(data_dir)
    num_classes = len(set(labels))

    dataset = tf.data.Dataset.from_tensor_slices((filenames, labels))
    dataset = dataset.map(load_data.parse_function)
    if is_train:
        dataset = dataset.map(load_data.train_preprocess)
    else:
        dataset = dataset.map(load_data.eval_preprocess)

    dataset = dataset.shuffle(buffer_size=10000)  # don't forget to shuffle
    batched_dataset = dataset.batch(batch_size)

    return num_classes, batched_dataset

def check_accuracy(sess, correct_prediction, is_training, dataset_init_op):
    """
    Check the accuracy of the model on either train or val (depending on dataset_init_op).
    """
    # Initialize the correct dataset
    sess.run(dataset_init_op)
    num_correct, num_samples = 0, 0
    while True:
        try:
            correct_pred = sess.run(correct_prediction, {is_training: False})
            num_correct += correct_pred.sum()
            num_samples += correct_pred.shape[0]
        except tf.errors.OutOfRangeError:
            break

    # Return the fraction of datapoints that were correctly classified
    acc = float(num_correct) / num_samples
    return acc
def main(_):

    graph = tf.Graph()
    with graph.as_default():
        tf.logging.set_verbosity(tf.logging.INFO) #Set the verbosity to INFO level'

        num_classes, batched_train_dataset = get_batch(FLAGS.train_dir, FLAGS.train_batch, True)
        _, batched_eval_dataset = get_batch(FLAGS.eval_dir, FLAGS.eval_batch, False)


        # Now we define an iterator that can operator on either dataset.
        # The iterator can be reinitialized by calling:
        #     - sess.run(train_init_op) for 1 epoch on the training set
        #     - sess.run(val_init_op)   for 1 epoch on the valiation set
        # Once this is done, we don't need to feed any value for images and labels
        # as they are automatically pulled out from the iterator queues.

        # A reinitializable iterator is defined by its structure. We could use the
        # `output_types` and `output_shapes` properties of either `train_dataset`
        # or `validation_dataset` here, because they are compatible.
        print ("batch datashape", batched_train_dataset.output_shapes)
        iterator = tf.data.Iterator.from_structure(batched_train_dataset.output_types,
                                                   batched_train_dataset.output_shapes)
        images, labels = iterator.get_next()

        train_init_op = iterator.make_initializer(batched_train_dataset)
        eval_init_op = iterator.make_initializer(batched_eval_dataset)

        is_training = tf.placeholder(tf.bool)
        exclude, end_points, variables_to_restore = net.get_pretrained_net(images, num_classes, FLAGS.gray_scale, is_training)
        end_points, custom_logits = net.add_custom_layers(end_points, num_classes, FLAGS.hidden1_size)
        trainable_variables = net.get_trainable_variables("AddedClassifier")
        unrestored_variables, unrestored_variables_init = net.get_unrestored_variables(exclude)
        trainable_variables.extend(unrestored_variables)
        # print("trainable variables: ", trainable_variables)
        trainable_variables_init = tf.variables_initializer(trainable_variables)

        init_fn = tf.contrib.framework.assign_from_checkpoint_fn(FLAGS.checkpoint_file, variables_to_restore)

        tf.losses.sparse_softmax_cross_entropy(labels=labels, logits=custom_logits)
        loss = tf.losses.get_total_loss()


        # optimizer
        finetune_optimizer = tf.train.GradientDescentOptimizer(FLAGS.finetune_learning_rate)
        finetune_op = finetune_optimizer.minimize(loss, var_list= trainable_variables)

        full_optimizer = tf.train.GradientDescentOptimizer(FLAGS.full_learning_rate)
        full_op = full_optimizer.minimize(loss)

        # eval metric
        predictions = tf.to_int32(tf.argmax(end_points['Predictions'], 1))
        probabilities = end_points['Predictions']
        correct_prediction = tf.equal(predictions, labels)
        accuracy = tf.reduce_mean(tf.cast(correct_prediction, tf.float32))

        tf.get_default_graph().finalize()

    with tf.Session(graph=graph) as sess:
        init_fn (sess)
        sess.run(trainable_variables_init)

        # Update only the last layer for a few epochs.
        for epoch in range(2):
            # Run an epoch over the training data.
            print('Starting epoch %d / %d' % (epoch + 1, 2))
            # Here we initialize the iterator with the training set.
            # This means that we can go through an entire epoch until the iterator becomes empty.
            sess.run(train_init_op)
            while True:
                try:
                    sess.run([finetune_op], {is_training: True})
                    current_loss = sess.run(loss, {is_training: False})
                    print("loss: ", current_loss)
                    print('accuracy: ', sess.run(accuracy, {is_training: False}))
                except tf.errors.OutOfRangeError:
                    break

            # Check accuracy on the train and val sets every epoch.
            train_acc = check_accuracy(sess, correct_prediction, is_training, train_init_op)
            val_acc = check_accuracy(sess, correct_prediction, is_training, eval_init_op)
            print('Train accuracy: %f' % train_acc)
            print('Val accuracy: %f\n' % val_acc)



if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument(
        '--train_dir',
        # nargs='+',
        default='/home/long/Desktop/processed_image/train'
    )

    parser.add_argument(
        '--eval_dir',
        # nargs="+",
        default='/home/long/Desktop/processed_image/eval'
    )

    parser.add_argument(
        '--log_dir',

    )

    parser.add_argument(
        '--checkpoint_file',
        default='/mnt/6B7855B538947C4E/pretrained_model/inception-resnet/inception_resnet_v2_2016_08_30.ckpt'
    )

    parser.add_argument(
        '--train_batch',
        # nargs='+',
        default=8
    )
    parser.add_argument(
        '--eval_batch',
        # nargs='+',
        default=16
    )
    parser.add_argument(
        '--gray_scale',
        # nargs='+',
        default=True
    )

    parser.add_argument(
        '--hidden1_size',
        # nargs='+',
        default=50
    )


    parser.add_argument(
        '--finetune_learning_rate',
        # nargs='+',
        default=0.1
    )

    parser.add_argument(
        '--full_learning_rate',
        # nargs='+',
        default=0.01
    )
    FLAGS, unparsed = parser.parse_known_args()
    print(FLAGS)
    tf.app.run(main=main, argv=[sys.argv[0]] + unparsed)


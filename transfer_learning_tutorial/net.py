import tensorflow as tf
from inception_resnet_v2 import inception_resnet_v2, inception_resnet_v2_arg_scope
slim = tf.contrib.slim

def get_pretrained_net(images, num_classes, gray_scale, is_training):
    # Create the model inference
    with slim.arg_scope(inception_resnet_v2_arg_scope()):
        _, end_points = inception_resnet_v2(images, num_classes=num_classes, is_training=is_training)

    # Define the scopes that you want to exclude for restoration
    if gray_scale:
        exclude = ['InceptionResnetV2/Logits', 'InceptionResnetV2/AuxLogits', 'InceptionResnetV2/Conv2d_1a_3x3']
    else:
        exclude = ['InceptionResnetV2/Logits', 'InceptionResnetV2/AuxLogits']

    variables_to_restore = slim.get_variables_to_restore(exclude=exclude)
    # print('variables_to_restore: ', variables_to_restore)
    return exclude, end_points, variables_to_restore

def add_custom_layers(end_points, num_classes, hidden1_size):
    # Define new layers (see InceptionResnet implementation to know why)
    pre_logits_flatten = end_points['PreLogitsFlatten']
    hidden1 = slim.fully_connected(pre_logits_flatten, hidden1_size, activation_fn=None,
                                   scope='AddedClassifier/Hidden1')
    hidden1 = slim.dropout(hidden1, 1.0, is_training=True,
                           scope='AddedClassifier/Hidden1/Dropout')

    custom_logits = slim.fully_connected(hidden1, num_classes, activation_fn=None,
                                         scope='AddedClassifier/Logits')
    end_points['Logits'] = custom_logits
    end_points['Predictions'] = tf.nn.softmax(custom_logits, name='Predictions')

    return end_points, custom_logits

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
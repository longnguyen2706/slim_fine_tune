import tensorflow as tf
import os
slim = tf.contrib.slim
from transfer_learning_tutorial import inception_preprocessing

num_classes = 10
gray_scale = True

IMAGE_SIZE = 299

# return filenames and associated labels in list format
# the dir shoudld be as follow:
# dir/
#    class1/
#    class2/

def list_images(directory):
    labels = os.listdir(directory)
    labels.sort()

    files_and_labels = []
    for label in labels:
        for f in os.listdir(os.path.join(directory, label)):
            files_and_labels.append((os.path.join(directory, label, f), label))

    filenames, labels = zip(*files_and_labels)
    filenames = list(filenames)
    labels = list(labels)
    unique_labels = list(set(labels))

    label_to_int = {}
    for i, label in enumerate(unique_labels):
        label_to_int[label] = i


    labels = [label_to_int[l] for l in labels]
    return filenames, labels

# read image from filename and return tf32 array
def parse_function(filename, label):
    image_string = tf.read_file(filename)
    image_decoded = tf.image.decode_jpeg(image_string, channels=3)
    image = tf.cast(image_decoded, tf.float32)

    height, width = tf.shape(image)[0], tf.shape(image)[1]
    height = tf.to_float(height)
    width = tf.to_float(width)

    # scale = tf.cond(tf.greater(height, width),
    #                 lambda: IMAGE_SIZE/ width,
    #                 lambda: IMAGE_SIZE / height)
    scale = 1 # non resize
    new_height = tf.to_int32(height * scale)
    new_width = tf.to_int32(width * scale)

    resized_image = tf.image.resize_images(image, [new_height, new_width])  # (2)
    return resized_image, label

# preprocess image
def train_preprocess (image, label):
    preprocess_image = inception_preprocessing.preprocess_image(image, IMAGE_SIZE, IMAGE_SIZE, True)
    return preprocess_image, label

def eval_preprocess (image, label):
    preprocess_image = inception_preprocessing.preprocess_image(image, IMAGE_SIZE, IMAGE_SIZE, False)
    return preprocess_image, label

def main():
    train_filenames, train_labels = list_images('/home/long/Desktop/processed_image/train')
    train_dataset = tf.data.Dataset.from_tensor_slices((train_filenames, train_labels))
    train_dataset = train_dataset.map(parse_function)
    train_dataset = train_dataset.map(train_preprocess)
    train_dataset = train_dataset.shuffle(buffer_size=10000)  # don't forget to shuffle
    batched_train_dataset = train_dataset.batch(8)
    print (batched_train_dataset.output_shapes, batched_train_dataset.output_types)
    iterator = tf.data.Iterator.from_structure(batched_train_dataset.output_types,
                                                       batched_train_dataset.output_shapes)
    # iterator = batched_train_dataset.make_one_shot_iterator()
    images, labels = iterator.get_next()
    # next_element = iterator.get_next()

    train_init_op = iterator.make_initializer(batched_train_dataset)

    with tf.Session() as sess:
        sess.run(train_init_op)
        for i in range(1):
            images, labels = sess.run([images, labels])
            print(images)
            print(images.shape)
            print(labels)
if __name__ == "__main__":
    main()



    #
    # def label_to_name(labels_file):
    #     labels = open(labels_file, 'r')
    #     labels_to_name = {}
    #     for line in labels:
    #         label, string_name = line.split(':')
    #         string_name = string_name[:-1]  # Remove newline
    #         labels_to_name[int(label)] = string_name
    #     return labels_to_name
    #
    # def get_data(split_name, dataset_dir, labels_to_name, file_pattern=file_pattern, file_pattern_for_counting=file_pattern_for_counting):
    #     # filenames = ["/var/data/file1.tfrecord", "/var/data/file2.tfrecord"]
    #     tfrecords_to_count = [os.path.join(dataset_dir, file) for file in os.listdir(dataset_dir) if
    #                           file.startswith(file_pattern_for_counting)]
    #     dataset = tf.data.TFRecordDataset(tfrecords_to_count)
    #     return dataset
    #     # file_pattern_path = os.path.join(dataset_dir, file_pattern % (split_name))

    # def _parse_function(example_proto):
    #     features = {"image": tf.FixedLenFeature((), tf.string, default_value=""),
    #                 "label": tf.FixedLenFeature((), tf.int32, default_value=0),
    #                 'image/class/label': tf.FixedLenFeature(
    #                     [], tf.int64, default_value=tf.zeros([], dtype=tf.int64))
    #                 }
    #     parsed_features = tf.parse_single_example(example_proto, features)
    #     return parsed_features["image"], parsed_features["label"]

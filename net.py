import tensorflow as tf

# loading data
NUM_EPOCHS = 10
BATCH_SIZE = 128


def decode_csv(line):
    items = tf.string_split(tf.string_strip([line]), delimiter=",").values
    print(type(items))
    features = [tf.string_to_number(items[i], tf.float32) for i in [2, 3, 5, 6, 7]]
    labels = [tf.string_to_number(items[i], tf.float32) for i in [1, 4]]
    return features, labels


default_column_value = [[0.0] for _ in range(8)]
filenames = tf.constant(["./dataset_train.txt"])
dataset = tf.data.TextLineDataset(filenames).skip(1)
dataset = dataset.map(decode_csv)
dataset = dataset.repeat(NUM_EPOCHS)
dataset = dataset.shuffle(buffer_size=1000).batch(BATCH_SIZE)
iterator = dataset.make_initializable_iterator()
# batch_features, batch_labels = iterator.get_next()
print(type(iterator))
print(dir(iterator))


# X_data = next_element[[1, 3, 5, 6, 7]]
# Y_data = next_element[[2, 4]]
# print(len(next_element))

# draw data flow graph
INPUT_DIMENSION = 5
OUTPUT_DIMENSION = 2


X = tf.placeholder(tf.float32, [None, INPUT_DIMENSION], name='X')

W = tf.Variable(tf.zeros([INPUT_DIMENSION, OUTPUT_DIMENSION]))
b = tf.Variable(tf.zeros([OUTPUT_DIMENSION]))

net_y = tf.matmul(X, W) + b

expected_y = tf.placeholder(tf.float32, [None, OUTPUT_DIMENSION], name='Y')

loss = tf.reduce_mean(tf.square(expected_y - net_y, name='loss'))


# train
LEARNING_RATE = 0.2
optimizer = tf.train.GradientDescentOptimizer(learning_rate=LEARNING_RATE)
train_step = optimizer.minimize(loss)

with tf.Session() as sess:
    # sess.run(train_step, feed_dict={X:X_data, Y:Y_data})
    tf.global_variables_initializer().run()
    tf.local_variables_initializer().run()
    sess.run(iterator.initializer)
    # sess.run(iterator.initializer)
    # sess.run(train_step, feed_dict={x: X_data, expected_y: Y_data})
    for step in range(1, BATCH_SIZE+1):
        batch_features, batch_labels = iterator.get_next()
        print(sess.run(train_step,
              feed_dict={X: batch_features, expected_y: batch_labels}))

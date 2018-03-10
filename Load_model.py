import numpy as np
import tensorflow as tf
from tensorflow.contrib import rnn

main_words = ['h', 'i', 'e', 'l', 'o']
x_data = ['e', 'e', 'e', 'e', 'e', 'e']
y_data = ['i', 'h', 'e', 'l', 'l', 'o']

matrix1 = []
for i in x_data:
    matrix = [0] * len(main_words)

    if i in main_words:
        matrix[main_words.index(i)] = 1

    matrix1.append(matrix)
data_x = np.reshape(matrix1, [1, -1, 5])

data_y = [main_words.index(i) for i in y_data if i in main_words]
data_y = np.reshape(data_y, [1, -1])
print(data_y)
print(data_x)

X_data = tf.placeholder(tf.float32, [None, 6, 5])
Y_data = tf.placeholder(tf.int32, [None, 6])
cell = rnn.BasicLSTMCell(num_units=5, state_is_tuple=True)
initial_cell = cell.zero_state(1, dtype=tf.float32)
output, states = tf.nn.dynamic_rnn(cell, X_data, initial_state=initial_cell, dtype=tf.float32)
x_for_fc = tf.reshape(output, [-1, 5])
output = tf.contrib.layers.fully_connected(inputs=x_for_fc, num_outputs=5, activation_fn=None)
output = tf.reshape(output, [1, 6, 5])
weights = tf.ones([1, 6])
loss = tf.contrib.seq2seq.sequence_loss(logits=output, targets=Y_data, weights=weights)
loss_cal = tf.reduce_mean(loss)
mina = tf.train.AdamOptimizer(0.1).minimize(loss_cal)
pred = tf.argmax(output, axis=2)
saver = tf.train.Saver()

with tf.Session() as tt:
    tt.run(tf.global_variables_initializer())

    saver.restore(tt, '/Users/exepaul/Desktop/tensor/jklop.ckpt')
    wow = tt.run(pred, feed_dict={X_data: data_x})
    print([main_words[i] for i in np.squeeze(wow)])

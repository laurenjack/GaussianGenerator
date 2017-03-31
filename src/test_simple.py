import numpy as np
import tensorflow as tf

test_array = np.array([[1.0 + float(i) for i in xrange(10)],
                      [6.0 + float(i) for i in xrange(10)],
                      [11.0 + float(i) for i in xrange(10)]])

# test_array = test_array.transpose()
#
# indices = tf.placeholder(tf.int32, [2], "indices")
# test_tensor = tf.Variable(test_array, "Test Array")
# #selected_tensors = tf.gather(tensors_to_select, indices)
# batches = tf.slice(test_tensor, [0, 0], [4, 3])
#
# sess = tf.Session()
# sess.run(tf.initialize_all_variables())
# actual_selected = sess.run(batches, feed_dict={indices:[0, 2]})
# print actual_selected

q = tf.FIFOQueue(3, tf.float64)
for i in xrange(5):
    c = float(i) + 1.0
    enq = q.enqueue(tf.constant(c * test_array))

deq = q.dequeue()

sess = tf.Session()
print deq.run(session=sess)
print deq.run(session=sess)
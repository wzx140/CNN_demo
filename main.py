import tensorflow as tf
from cnn import Cnn
import config
import util

x_train_orig, y_train_orig, x_test_orig, y_test_orig, classes = util.load_data_set()
x_train = util.pre_treat(x_train_orig)
x_test = util.pre_treat(x_test_orig)
y_train = util.pre_treat(y_train_orig, is_x=False, class_num=len(classes))
y_test = util.pre_treat(y_test_orig, is_x=False, class_num=len(classes))

cnn = Cnn(config.conv_layers, config.fc_layers, config.filters, config.learning_rate, config.beta1, config.beta2)

(m, n_H0, n_W0, n_C0) = x_train.shape
n_y = y_train.shape[1]

# construction calculation graph
cnn.initialize(n_H0, n_W0, n_C0, n_y)
cnn.forward()
cost = cnn.cost()
optimizer = cnn.get_optimizer(cost)
predict, accuracy = cnn.predict()

init = tf.global_variables_initializer()

with tf.Session() as sess:
    sess.run(init)

    for i in range(1, config.num_epochs + 1):
        num_mini_batches = int(m / config.mini_batch_size)
        # seed += 1
        mini_batches = util.random_mini_batches(x_train, y_train, config.mini_batch_size)

        cost_value = 0
        for mini_batch in mini_batches:
            (mini_batch_x, mini_batch_y) = mini_batch
            _, temp_cost = sess.run([optimizer, cost], feed_dict={cnn.x: mini_batch_x, cnn.y: mini_batch_y})
            cost_value += temp_cost
        cost_value /= num_mini_batches

        train_accuracy = sess.run(accuracy, feed_dict={cnn.x: x_train, cnn.y: y_train})
        test_accuracy = sess.run(accuracy, feed_dict={cnn.x: x_test, cnn.y: y_test})

        if config.print_cost and (i == 1 or i % 5 == 0):
            print('Iteration %d' % i)
            print('Cost: %f' % cost_value)
            print('Train accuracy: %f' % train_accuracy)
            print('Test accuracy: %f\n' % test_accuracy)

    # predict your image
    X = util.load_pic(config.img_path)
    if X is not None:
        predict_y = sess.run(predict, feed_dict={cnn.x: X})
        print('The output: ' + str(predict_y))

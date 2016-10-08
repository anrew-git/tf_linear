import numpy as np
import tensorflow as tf
import matplotlib.pyplot as plt

# Prepre input data for regression. X from 1 to 100 with step 0.1
# Y = X+ 10*cos(X/5) 
X_gen = np.arange(100, step=.1)
Y_gen = X_gen + 10 * np.cos(X_gen/5)

#Number of samples. 100/0.1 = 1000
n_samples = 1000

#Batch size
batch_size = 100

#Steps number
steps_number = 10000

# Tensorflow is sensitive to shapes, so reshaping without data change
# It were  (n_samples,), now should be (n_samples, 1)

X_gen = np.reshape(X_gen, (n_samples,1))
Y_gen = np.reshape(Y_gen, (n_samples,1))

# Preparing  placeholders

X = tf.placeholder(tf.float32, shape=(batch_size, 1))
Y = tf.placeholder(tf.float32, shape=(batch_size, 1))

# Define variables to be learned

with tf.variable_scope("linear-regression"):
    k = tf.get_variable("weights", (1, 1),
         initializer=tf.random_normal_initializer())
    b = tf.get_variable("bias", (1,),
         initializer=tf.constant_initializer(0.0))

y_predicted = tf.matmul(X, k) + b

loss = tf.reduce_sum((Y - y_predicted)**2)

# Sample code to  solve this problem
opt_operation = tf.train.AdamOptimizer().minimize(loss)


display_step = 1000

with tf.Session() as sess:
# Initialize Variables in graph

    sess.run(tf.initialize_all_variables())

    # Optimization loop for steps_number steps
    for i in range(steps_number):
        # Select random minibatch
        indices = np.random.choice(n_samples, batch_size)       
        X_batch, y_batch = X_gen[indices], Y_gen[indices]
        # Do optimization step
        sess.run([opt_operation, loss], feed_dict={X: X_batch, Y: y_batch})        
        # Display logs per epoch step
        if (i+1) % display_step == 0:
            c = sess.run(loss, feed_dict={X: X_batch, Y: y_batch})
            print("Epoch:", '%04d' % (i+1), "cost=", "{:.9f}".format(c),"k=", sess.run(k), "b=", sess.run(b))
# Graphic display
    plt.plot(X_gen, Y_gen, 'ro', label='Original data')   
    plt.plot(X_gen, sess.run(k) * X_gen + sess.run(b), label='Fitted line')
    plt.legend()
    plt.show()


    sess.close()



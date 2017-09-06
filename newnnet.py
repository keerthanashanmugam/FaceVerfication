
import tensorflow as tf
import pandas as pd
import numpy as np
from scipy.misc import imread
from sklearn.metrics import accuracy_score

tou = tf.constant(3.0, dtype=tf.float32)
mu = 100
lamda = tf.constant(0.01, dtype=tf.float32)
beta = tf.constant(0.01, dtype=tf.float32)
splitsize = 5401
seed = 128

input_num_units = 14160
dummy = 7080
hidden_num_units = 1000
output_num_units = 100

epochs = 10
batch_size = 6000

#keerthana

#data = pd.read_csv('../Matlab/data.csv',sep = "\t")
f = open ( '../Matlab/trail.csv' , 'r')
data = np.array( [ map(float,line.split('\t')) for line in f ])


train_x = data[:,:-1]
train_y = data[:,-1:]



train_x, val_x = train_x[:splitsize], train_x[splitsize:]
train_y, val_y = train_y[:splitsize], train_y[splitsize:]

#train_x = np.resize(train_x,(splitsize,dummy,dummy))

rng = np.random.RandomState(seed)


# define placeholders
x = tf.placeholder(tf.float32, [None, input_num_units])
y = tf.placeholder(tf.float32, [None, 1])

hidPlace_W = tf.placeholder(tf.float32, [dummy,hidden_num_units])
outPlace_W = tf.placeholder(tf.float32, [hidden_num_units,output_num_units])
hidPlace_B = tf.placeholder(tf.float32, [hidden_num_units])
outPlace_B = tf.placeholder(tf.float32, [output_num_units])

# set remaining variables

weights = {
    'hidden': tf.Variable(hidPlace_W),
    'output': tf.Variable(outPlace_W)
}

biases = {
    'hidden': tf.Variable(hidPlace_B),
    'output': tf.Variable(outPlace_B)
}


def ge(param):
	print param*beta
	x = tf.log(1+ tf.exp(beta*param)) / beta
	return x

def cost_function(out,l):
	

	# temp = tf.subtract(tou , out)

	s = (0.5) * tf.losses.hinge_loss(out ,l)
	# s = (0.5) * tf.reduce_sum(tf.map_fn(ge , temp ) )
	# for i in weights.keys():
	# 	s += (lamda/2)* tf.reduce_sum(tf.pow(weights[i],2))
	# for i in biases.keys():
	# 	s += (lamda/2)* tf.reduce_sum(tf.pow(biases[i],2))
	return s

def reg():
    s = 0
    for i in weights.keys():
      s += (lamda/2)* tf.reduce_sum(tf.pow(weights[i],2))
    for i in biases.keys():
      s += (lamda/2)* tf.reduce_sum(tf.pow(biases[i],2))  
    return s

print 'making layers now'

### define weights and biases of the neural network (refer this article if you don't understand the terminologies)


hidden_layer1 = tf.add(tf.matmul(x[:,:7080], weights['hidden']), biases['hidden'])
hidden_layer1 = tf.nn.tanh(hidden_layer1)                                                                                                                                                                           

output_layer1 = tf.add(tf.matmul(hidden_layer1, weights['output']) , biases['output'])

hidden_layer2 = tf.add(tf.matmul(x[:,7080:], weights['hidden']), biases['hidden'])
hidden_layer2 = tf.nn.tanh(hidden_layer2)

output_layer2 = tf.add(tf.matmul(hidden_layer2, weights['output']) , biases['output'])

output_layer = tf.reduce_sum(tf.pow(tf.subtract(output_layer1,output_layer2),2),1,keep_dims=True)

#cost = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(logits = output_layer,labels = y))

print 'made layers now cost function'

cost = cost_function(output_layer,y)

optimizer = tf.train.GradientDescentOptimizer(learning_rate= 0.1).minimize(cost)

init = tf.initialize_all_variables()

print 'seeesion is going to start....'

embHid_W = np.random.normal(0.0,0.1, (dummy, hidden_num_units) )
embHid_B = np.random.normal(0.0,0.1, (hidden_num_units))
embOut_W = np.random.normal(0.0,0.1, (hidden_num_units, output_num_units))
embOut_B = np.random.normal(0.0,0.1, (output_num_units))



with tf.Session() as sess:
    # create initialized variables

    # print sess.run(tf.shape(output_layer))
    # print sess.run(tf.shape(train_y))

    sess.run(init,feed_dict = {hidPlace_W: embHid_W , hidPlace_B: embHid_B , outPlace_W: embOut_W , outPlace_B : embOut_B})
    
    ### for each epoch, do:
    ###   for each batch, do:
    ###     create pre-processed batch
    ###     run optimizer by feeding batch
    ###     find cost and reiterate to minimize
    
    for epoch in range(epochs):
        avg_cost = 0
        total_batch = int(data.shape[0]/batch_size)
        for i in range(total_batch):
            #batch_x, batch_y =   batch_creator(batch_size, train_x.shape[0], 'train')
            
            batch_x = train_x
            batch_y = train_y
            # print sess.run(tf.shape(batch_x))
            # print sess.run(tf.shape(batch_y))
            # print sess.run(tf.shape(output_layer))
            # print sess.run(tf.shape(batch_y))
            _, c = sess.run([optimizer, cost], feed_dict = {x: batch_x, y: batch_y})
            
            avg_cost += c / total_batch
        #regu = reg()
        print "regularisation: ", sess.run(regu)
        print "Epoch:", (epoch+1), "cost =", "{:.5f}".format(avg_cost) , total_batch
    
    print "\nTraining complete!"
    pred = tf.equal(tf.less(output_layer,tou,name = None),val_y)

    accuracy = tf.reduce_mean(tf.cast(pred, "float"))
    print "Validation Accuracy:", accuracy.eval({x: val_x, y: val_y})
#    find predictions on val set
#    pred_temp = tf.equal(tf.argmax(output_layer, 1), tf.argmax(y, 1))
#    accuracy = tf.reduce_mean(tf.cast(pred_temp, "float"))
#    print "Validation Accuracy:", accuracy.eval({x: val_x.reshape(-1, input_num_units), y: dense_to_one_hot(val_y)})
    
#    predict = tf.argmax(output_layer, 1)
#    pred = predict.eval({x: test_x.reshape(-1, input_num_units)})

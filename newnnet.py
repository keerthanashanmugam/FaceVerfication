#FaceVerification/newnnet.py

import tensorflow as tf
import pandas as pd
import numpy as np
from scipy.misc import imread
from sklearn.metrics import accuracy_score

tou = tf.constant(3.0, dtype=tf.float64)
mu = 0.001
lamda = tf.constant(0.01, dtype=tf.float64)
beta = tf.constant(1, dtype=tf.float64)
splitsize = 5401
seed = 128

input_num_units = 14160
dummy = 7080
hidden_num_units = 1000
output_num_units = 100

epochs = 10
batch_size = 6000



#data = pd.read_csv('../Matlab/data.csv',sep = "\t")
f = open ( '../../Matlab/trail.csv' , 'r')
data = np.array( [ map(float,line.split('\t')) for line in f ])


train_x = data[:,:-1]
train_y = data[:,-1:]



train_x, val_x = train_x[:splitsize], train_x[splitsize:]
train_y, val_y = train_y[:splitsize], train_y[splitsize:]

#train_x = np.resize(train_x,(splitsize,dummy,dummy))

rng = np.random.RandomState(seed)


# define placeholders
x = tf.placeholder(tf.float64, [None, input_num_units])
y = tf.placeholder(tf.float64, [None, 1])

hidPlace_W = tf.placeholder(tf.float64, [dummy,hidden_num_units])
outPlace_W = tf.placeholder(tf.float64, [hidden_num_units,output_num_units])
hidPlace_B = tf.placeholder(tf.float64, [hidden_num_units])
outPlace_B = tf.placeholder(tf.float64, [output_num_units])

# set remaining variables

weights = {
    'hidden': tf.Variable(hidPlace_W),
    'output': tf.Variable(outPlace_W)
}

biases = {
    'hidden': tf.Variable(hidPlace_B),
    'output': tf.Variable(outPlace_B)
}

mean_h = - np.sqrt(6)/np.sqrt(dummy + hidden_num_units)
stddev_h = np.sqrt(6)/np.sqrt(dummy + hidden_num_units)

mean_o = - np.sqrt(6)/np.sqrt(hidden_num_units + output_num_units)
stddev_o = np.sqrt(6)/np.sqrt(hidden_num_units + output_num_units)

# weights = {
#     'hidden': tf.Variable(tf.zeros( [dummy,hidden_num_units])),
#     'output': tf.Variable(tf.zeros([hidden_num_units,output_num_units]))
# }


# biases = {
#     'hidden': tf.Variable(tf.zeros([hidden_num_units])),
#     'output': tf.Variable(tf.zeros([output_num_units]))
# }

# weights = {
#     'hidden': tf.Variable(tf.random_normal( [dummy,hidden_num_units], mean =mean_h,stddev = stddev_h)),
#     'output': tf.Variable(tf.random_normal([hidden_num_units,output_num_units], mean = mean_o,stddev = stddev_o ))
# }

# biases = {
#     'hidden': tf.Variable(tf.random_normal([hidden_num_units], mean =mean_h,stddev = stddev_h)),
#     'output': tf.Variable(tf.random_normal([output_num_units], mean = mean_o,stddev = stddev_o))
# }

def ge(param):
	print param*beta
	x = tf.log(1+ tf.exp(beta*param)) / beta
	return x

def cost_function(out,l):
	

    temp =  1 - tf.multiply(l,tf.subtract(tou , out))

    # s = tf.reduce_sum(out)

    # s = tf.nn.relu(temp)
    s = (0.5) * tf.losses.hinge_loss(temp ,l)
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


hidden_layer1_ = tf.add(tf.matmul(x[:,:7080], weights['hidden']), biases['hidden'])
hidden_layer1 = tf.nn.tanh(hidden_layer1_)                                                                                                                                                                           

output_layer1_ = tf.add(tf.matmul(hidden_layer1, weights['output']) , biases['output'])
output_layer1 = tf.nn.tanh(output_layer1_)

hidden_layer2_ = tf.add(tf.matmul(x[:,7080:], weights['hidden']), biases['hidden'])
hidden_layer2 = tf.nn.tanh(hidden_layer2_)

output_layer2_ = tf.add(tf.matmul(hidden_layer2, weights['output']) , biases['output'])
output_layer2 = tf.nn.tanh(output_layer2_)
output_layer = tf.reduce_sum(tf.pow(tf.subtract(output_layer1,output_layer2),2),1,keep_dims=True)

#cost = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(logits = output_layer,labels = y))

print 'made layers now cost function'

# cost = (0.5) * tf.losses.hinge_loss( tf.subtract(tou , output_layer),y)
# cost = tf.reduce_mean(tf.nn.relu(1- tf.multiply(tf.subtract(tou , output_layer) ,y) ))
cost = cost_function(output_layer,y)

#cost = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(logits = output_layer, labels = y))

opt = tf.train.GradientDescentOptimizer(learning_rate= 0.001)
grads = opt.compute_gradients(cost,var_list = [weights,biases])


optimizer = tf.train.GradientDescentOptimizer(learning_rate= mu).minimize(cost,var_list = [weights,biases])

# ################################ TESTING PURPOSE
# grads = opt.compute_gradients(cost,var_list = [weights,biases])
                    



# ################################


init = tf.initialize_all_variables()

print 'seeesion is going to start....'

# embHid_W = np.random.normal(mean_h,stddev_h, (dummy, hidden_num_units) )
# embHid_B = np.zeros(hidden_num_units,dtype = np.float64)
# embOut_W = np.random.normal(mean_o,stddev_o, (hidden_num_units, output_num_units))
# embOut_B = np.zeros(output_num_units , dtype = np.float64)


embHid_W = np.random.uniform(mean_h,stddev_h, (dummy, hidden_num_units) )
embHid_B = np.zeros(hidden_num_units,dtype = np.float64)
embOut_W = np.random.uniform(mean_o,stddev_o, (hidden_num_units, output_num_units))
embOut_B = np.zeros(output_num_units , dtype = np.float64)




# embHid_W = np.random.normal(0.0,1, (dummy, hidden_num_units) )
# embHid_B = np.random.normal(0.0,1, (hidden_num_units))
# embOut_W = np.random.normal(0.0,1, (hidden_num_units, output_num_units))
# embOut_B = np.random.normal(0.0,1, (output_num_units))


# #########################################################################################
# sess = tf.Session()
    
# sess.run(init,feed_dict = {hidPlace_W: embHid_W , hidPlace_B: embHid_B , outPlace_W: embOut_W , outPlace_B : embOut_B, x:train_x , y:train_y})

# grad_vals = sess.run([grad[0] for grad in grads])
# #########################################################################################
with tf.Session() as sess:

    # create initialized variables

    # print sess.run(tf.shape(output_layer))
    # print sess.run(tf.shape(train_y))

    sess.run(init,feed_dict = {hidPlace_W: embHid_W , hidPlace_B: embHid_B , outPlace_W: embOut_W , outPlace_B : embOut_B})

    # sess.run(init,feed_dict = {hidPlace_W: embHid_W , hidPlace_B: embHid_B , outPlace_W: embOut_W , outPlace_B : embOut_B, x:train_x , y:train_y})


    ### for each epoch, do:
    ###   for each batch, do:
    ###     create pre-processed batch
    ###     run optimizer by feeding batch
    ###     find cost and reiterate to minimize


    print "\n weights" , sess.run(weights['hidden'])
    print "\n weights" , sess.run(biases['hidden'])

    cost1 = 1000
    cost2 = 0
    count = 0

    # sess.run(grads,feed_dict = {x: train_x, y: train_y})
    # grad_vals1 = sess.run([grad[0] for grad in grads],feed_dict = {x: train_x, y: train_y})

    while( (count < 5 or (cost1 - cost2 > 0.001 )) and count<25 ):
        avg_cost = 0
        total_batch = int(data.shape[0]/batch_size)
        for i in range(total_batch):
            #batch_x, batch_y =   batch_creator(batch_size, train_x.shape[0], 'train')
            
            batch_x = train_x
            batch_y = train_y

            _, c = sess.run([optimizer, cost], feed_dict = {x: batch_x, y: batch_y})
            
            avg_cost += c / total_batch
        #regu = reg()
        print "regularisation: ", sess.run(reg())
        print "Epoch:", (count+1), "cost =", "{:.5f}".format(avg_cost) , total_batch
        cost1 = cost2
        cost2 = avg_cost
        count += 1


    sess.run(grads,feed_dict = {x: train_x, y: train_y})
    grad_vals = sess.run([grad[0] for grad in grads],feed_dict = {x: train_x, y: train_y})
    print "\nTraining complete!"

    # tem = tf.reduce_mean(output_layer)
    
    output_bool = tf.greater(output_layer,tou)
    # pred = tf.equal(output_bool,train_y)

    tr_pred = tf.equal( tf.less(train_y,0) , output_bool )
    val_pred = tf.equal( tf.less(val_y,0)  , output_bool )


    # eval_output_bool = output_bool.eval({x: train_x, y: train_y})
    # eval_tr_pred = tr_pred.eval({x: train_x, y: train_y})

    # print "\n weights" , sess.run(weights['hidden'])
    # print "\n weights" , sess.run(biases['hidden'])

    tr_accuracy = tf.reduce_mean(tf.cast(tr_pred, "float"))
    print "train Accuracy:", tr_accuracy.eval({x: train_x, y: train_y})

    val_accuracy = tf.reduce_mean(tf.cast(val_pred, "float"))
    print "validation Accuracy:", val_accuracy.eval({x: val_x, y: val_y})

    # print "pred:", eval_pred
    # print "\noutputs: ", output_layer1_.eval({x: val_x,y:val_y})
    # print "Validation Accuracy:", accuracy.eval({x: val_x, y: val_y})
    #    find predictions on val set
    #    pred_temp = tf.equal(tf.argmax(output_layer, 1), tf.argmax(y, 1))
    #    accuracy = tf.reduce_mean(tf.cast(pred_temp, "float"))
    #    print "Validation Accuracy:", accuracy.eval({x: val_x.reshape(-1, input_num_units), y: dense_to_one_hot(val_y)})
        
    #    predict = tf.argmax(output_layer, 1)
    #    pred = predict.eval({x: test_x.reshape(-1, input_num_units)})

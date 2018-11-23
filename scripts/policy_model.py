import tensorflow as tf 
import numpy as np 

class PolicyNetwork(object):

	def __init__(self, hparams, sess):
		self.s = sess


	def agent(self, hparams, sess):
		#build the graph
		self._input = tf.placeholder(tf.float32,
            shape=[None, hparams['input_size']])

        hidden1 = tf.contrib.layers.fully_connected(
            inputs=self._input,
            num_outputs=hparams['hidden_size'],
            activation_fn=tf.nn.relu,
            weights_initializer=tf.random_normal)

        logits = tf.contrib.layers.fully_connected(
            inputs=hidden1,
            num_outputs=hparams['num_actions'],
            activation_fn=None)

        self.sample = tf.reshape(tf.multinomial(logits,1), [])

        #log probs
        log_prob = tf.log(tf.nn.softmax(logits))

        #training graph
        self.acts = tf.placeholder(tf.int32)
        self.advantages = tf.placeholder(tf.float32)

        # get lob probs of actions from the episode
        indices = tf.range(0, tf.shape(log_prob)[0]) * tf.shape(log_prob)[1] + self.acts
        act_prob = tf.gather(tf.reshape(log_prob, [-1]), indices)

        #surrogate loss
        loss = -tf.reduce_sum(tf.mul(act_prob, self.advantages))

        #update
        optimizer = tf.train.RMSPropOptimizer(hparams['learning_rate'])


    def act(self,state):
    	#get an action through sampling
    	return self.s.run(self.train)

    def train_step(self, obs, acts, advantages):
    	batch_feed = {self.input: obs, self.acts: acts}






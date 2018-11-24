import tensorflow as tf 
import numpy as np 

class PolicyNetwork(object):

	def __init__(self, hparams, sess):
		self.s = sess

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
    	batch_feed = {self.input: obs, self.acts: acts, self.advantages: advantages}
    	self.s.run(self.train, feed_dict = batch_feed)

    def policy_rollout(env,agent):
    	#Run an episode

    	observation, reward, done = ____, 0, False
    	obs, acts, rews = [], [], []

        while not done:
            obs.append(observation)

            action = agent.act(observation)
            reward = #reward function
            done = #get wheter done

            acts.append(action)
            rews.append(reward)

        return obs, acts, rews


    def process_reward(rews):
        #Rewards -> Advantages for one episode
        return [len(rews)] * len(rews)

    def main():

        #hyper params
        hparams = {
            'input_size': #shape of the game,
            'hidden_size': 36,
            'num_actions': 3,
            'learning_rate': 0.1}

        #enviorment params
        eparams = {
            'num_batches': 40,
            'ep_per_batch': 10
        }


        with tf.Graph().as_default(), tf.Session() as sess:

            agent = PolicyNetwork(hparams, sess)

            sess.run(tf.initialize_all_variables())

            for batch in range(eparams['num_batches']):

                print '=====\nBATCH {}\n===='.format(batch)
                
                b_obs, b_acts, b_rews = [], [], []       

                for episode in range(eparams['ep_per_batch'])         

                    obs, acts, rews = policy_rollout()#still have to decide what to do

                    print 'Episode steps: {}'.format(len(obs))

                    b_obs.extend(obs)
                    b_acts.extend(acts)

                    advantages = process_reward(rews)
                    b_rews.extend(advantages)

                #update policy
                #normalize rewards; don't divide by 0
                b_rews = (b_rews - np.mean(b_rews)) / (np.std(b_rews) + 1e-10)

                agent.train_step(b_obs, b_acts, b_rews)

if __name__ == '__main__':
    main()








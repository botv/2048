import numpy as np 
import cPickle as pickle

class PolicyNetwork(object):

	def __init__(self):

		self.H1 = 200
		self.H2 = 100
		self.batch_size = 10
		self.learning_rate = 0.01
		self.gamma = 0.99
		self.decay_rate = 0.99
		self.resume = False
		self.render = False
		self.D = #Figure out what reward is later
		self.model = {}
  		self.model['W1'] = np.random.randn(H1,D) / np.sqrt(D) # "Xavier" initialization
  		self.model['W2'] = np.random.randn(H2,D) / np.sqrt(D) # "Xavier" initialization
  		self.model['W3'] = np.random.randn(H) / np.sqrt(H)

  		self.grad_buffer = { k : np.zeros_like(v) for k,v in model.iteritems() } # update buffers that add up gradients over a batch
		self.rmsprop_cache = { k : np.zeros_like(v) for k,v in model.iteritems() } # rmsprop memory

	def sigmoid(self, x):
		return 1.0 / (1.0 + np.exp(-x)) # sigmoid "squashing" function to interval [0,1]

	def discount_rewards(self, r):
		""" take 1D float array of rewards and compute discounted reward """
		discounted_r = np.zeros_like(r)
   	    running_add = 0
		for t in reversed(range(0, r.size)):
			if r[t] != 0: running_add = 0 # reset the sum, since this was a game boundary (pong specific!)
				running_add = running_add * gamma + r[t]
		    	discounted_r[t] = running_add
		return discounted_r

	def feed_forward(self, x):
		H1 = np.dot(model['W1'], x)
  		H1[H1<0] = 0 # ReLU nonlinearity
  		H2 = np.dot(model['W2'], H1)
  		H2[H2<0] = 0 #ReLu nonlinearity
  		logp = np.dot(model['W3'], H2)
		p = sigmoid(logp)
		hidden = {'H1': H1, 'H2': H2}
  		return p, hidden # return probability of taking action 2, and hidden state

  	def policy_backward(self, eph, epx, epdlogp):
  		""" backward pass. (eph is dict of intermediate hidden states) """
  		dW3 = np.dot(eph2.T, epdlogp).ravel()
  		dH2 = np.outer(epdlogp, model['W3'])
  		dH2[eph2 <= 0] = 0 # backprop relu
  		dW2 = np.dot(dH2.T, eph1)
  		dH1 = np.outer(dH2, model['W2'])
  		dH1[eph1 <= 0] = 0 # backprop relu
  		dW1 = np.dot(dH1.T, epx)
  		return {'W1':dW1, 'W2':dW2, 'W3': dW3}

  	def second_policy_backward(self, eph1, eph2, epx, epdlogp):
  		""" backward pass. (eph is dict of intermediate hidden states) """
  		dW3 = np.dot(epdlogp.T, eph2).ravel()
  		dH2 = np.dot(epdlogp, model['W3'])
  		dH2[eph2 <= 0] = 0 # backprop relu
  		dW2 = np.dot(dH2.T, eph1)
  		dH1 = np.dot(dH2, model['W2'])
  		dH1[eph1 <= 0] = 0 # backprop relu
  		dW1 = np.dot(dH1.T, epx)
  		return {'W1':dW1, 'W2':dW2, 'W3': dW3}

  	def runn(self):

  		xs,hs1,hs2,dlogps,drs = [],[],[],[],[]
		running_reward = None
		reward_sum = 0
		episode_number = 0
		while True:
	  		
	  		#Train the model
	  		x = #get state

	  		#Feed forward through the network and sample an action
	  		aprob, h = feed_forward(x)
	  		action = #set action here

	  		
	  		#Recording intermidiate states
	  		xs.append(x)
	  		hs1.append(h.get('H1'))
	  		hs2.append(h.get('H2'))
	  		dlogsps.append(action - aprob)

	  		#Step the enviorment
	  		#Code to perform action

	  		reward_sum += reward

  			drs.append(reward) # record reward (has to be done after we call step() to get reward for previous action)

	  		if done:
	  			episode_number += 1

	  			#Stack hidden, inputs, actions
	  			# stack together all inputs, hidden states, action gradients, and rewards for this episode
			    epx = np.vstack(xs)
			    eph1 = np.vstack(hs1)
			    eph2 = np.vstack(hs2)
			    epdlogp = np.vstack(dlogps)
			    epr = np.vstack(drs)
			    xs,hs1,hs2,dlogps,drs = [],[],[],[],[] # reset array memory

			    #compute the discounted reward backwards through time
			    discounted_epr = discount_rewards(epr)
			    # standardize the rewards to be unit normal (helps control the gradient estimator variance)
			    discounted_epr -= np.mean(discounted_epr)
			    discounted_epr /= np.std(discounted_epr)

			    epdlogp *= discounted_epr # modulate the gradient with advantage (PG magic happens right here.)
    			grad = policy_backward(eph1, eph2, epx, epdlogp)
    			for k in model: grad_buffer[k] += grad[k] # accumulate grad over batch

    			# perform rmsprop parameter update every batch_size episodes
			    if episode_number % batch_size == 0:
			      for k,v in model.iteritems():
			        g = grad_buffer[k] # gradient
			        rmsprop_cache[k] = decay_rate * rmsprop_cache[k] + (1 - decay_rate) * g**2
			        model[k] += learning_rate * g / (np.sqrt(rmsprop_cache[k]) + 1e-5)
			        grad_buffer[k] = np.zeros_like(v) # reset batch gradient buffer

			  	# boring book-keeping
			    running_reward = reward_sum if running_reward is None else running_reward * 0.99 + reward_sum * 0.01
			    print 'resetting env. episode reward total was %f. running mean: %f' % (reward_sum, running_reward)
			    if episode_number % 100 == 0: pickle.dump(model, open('save.p', 'wb'))
			    reward_sum = 0
			    observation =  # code to reset a game









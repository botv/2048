import numpy as np
import sys
import tensorflow as tf
import tensorflow.keras.layers as kl
import tensorflow.keras.losses as kls
import tensorflow.keras.optimizers as ko
import logging
from copy import deepcopy
import game

env = game.Game()

class ProbabilityDistribution(tf.keras.Model):
    def call(self, logits):
        # sample a random categorical action from given logits
        logits = logits.numpy()
        possibleActions = env.getPossible()
        logits = [logits[i] for i in possible_actions]
        logitSum = sum(logits)
        logits = [prob / logitSum for prob in logits]
        return tf.squeeze(tf.random.categorical(logits, 1), axis=-1)

class Model(tf.keras.Model):
    def __init__(self, num_actions=4):
        super().__init__('mlp_policy')
        # no tf.get_variable(), just simple Keras API
        self.conv1 = kl.Conv2D(128,2,padding='same',activation='relu')
        self.conv2 = kl.Conv2D(128,1,padding='same',activation='relu')
        self.conv3 = kl.Conv2D(128,2,padding='same',activation='relu')
        self.conv4 = kl.Conv2D(128,1,padding='same',activation='relu')
        self.flatten1 = kl.Flatten()
        self.flatten2 = kl.Flatten()
        self.hidden1 = kl.Dense(256, activation='relu')
        self.hidden2 = kl.Dense(256, activation='relu')
        self.value = kl.Dense(1, name='value')
        # logits are unnormalized log probabilities
        self.logits = kl.Dense(num_actions, name='policy_logits')
        self.dist = ProbabilityDistribution(env)

    def call(self, x):
        # inputs is a numpy array, convert to Tensor
        # separate hidden layers from the same input tensor
        hl = self.conv1(x)
        hl = self.conv2(hl)
        hl = self.flatten1(hl)
        hidden_logs = self.hidden1(hl)
        # hv = self.conv3(x)
        hv = self.conv4(x)
        hv = self.flatten2(hv)
        hidden_vals = self.hidden2(hv)
        return self.logits(hidden_logs), self.value(hidden_vals)

    def action_value(self, obs):
        # executes call() under the hood
        logits, value = self.predict(obs)
        action = self.dist.predict(logits)
        # a simpler option, will become clear later why we don't use it
        # action = tf.random.categorical(logits, 1)
        return np.squeeze(action, axis=-1), np.squeeze(value, axis=-1)


class A2CAgent:
    def __init__(self, model):
        self.model = model
        # hyperparameters for loss terms
        self.params = {'value': 0.5, 'entropy': 0.0001, 'gamma': 0.99}
        self.model = model
        self.model.compile(
            optimizer=ko.RMSprop(lr=0.0007),
            # define separate losses for policy logits and value estimate
            loss=[self._logits_loss, self._value_loss]
        )

    def train(self, env, batch_sz=32, updates=9500):
        # storage helpers for a single batch of data
        # training loop: collect samples, send to optimizer, repeat updates times
        ep_rews = [0.0]
        next_obs, rec_obs = env.reset()
        for update in range(updates):
            actions = np.empty((batch_sz,), dtype=np.int32)
            rewards, dones, values = np.empty((3, batch_sz))
            observations = np.empty((batch_sz, 4,4,16))
            for step in range(batch_sz):
                prev_obs = deepcopy(next_obs)
                observations[step] = prev_obs
                logits, value = self.model.predict(next_obs[None, :])
                actions[step], values[step] = self._prune_actions(logits, env), np.squeeze(value, axis=-1)
                next_obs, rec_obs, rewards[step], dones[step] = env.step(actions[step])
                ep_rews[-1] += rewards[step]
                if dones[step]:
                    ep_rews.append(0.0)
                    next_obs, rec_obs = env.reset()
                    print("Episode: %03d, Reward: %03d, Update: %d" % (len(ep_rews)-1, ep_rews[-2], update))
            _, next_value = self.model.predict(next_obs[None, :])
            returns, advs = self._returns_advantages(rewards, dones, values, next_value)
            # a trick to input actions and advantages through same API
            acts_and_advs = np.concatenate([actions[:, None], advs[:, None]], axis=-1)
            # performs a full training step on the collected batch
            # note: no need to mess around with gradients, Keras API handles it
            losses = self.model.train_on_batch(observations, [acts_and_advs, returns])
            tf.keras.backend.clear_session()
        return ep_rews

    def _prune_actions(self, logits, env):
        possibleActions = env.getPossible()
        logits = logits[0]
        logits = [logits[i] for i in possibleActions]
        logitSum = sum(logits)
        logits = [prob / logitSum for prob in logits]
        logits = np.reshape(logits, (1,len(logits)))
        return tf.squeeze(tf.random.categorical(logits, 1), axis=-1)

    def _returns_advantages(self, rewards, dones, values, next_value):
        # next_value is the bootstrap value estimate of a future state (the critic)
        next_value = next_value.reshape(1)
        returns = np.append(np.zeros_like(rewards), next_value, axis=-1)
        # returns are calculated as discounted sum of future rewards
        for t in reversed(range(rewards.shape[0])):
            returns[t] = rewards[t] + self.params['gamma'] * returns[t+1] * (1-dones[t])
        returns = returns[:-1]
        # advantages are returns - baseline, value estimates in our case
        advantages = returns - values
        return returns, advantages

    def _load_saved(self, env):
        print('called')
        self.train(env, updates=1)
        print('trained')
        self.model.load_weights('Saved/v1_a2c.h5')

    def test(self, env):
        obs, done, ep_reward = env.reset(), False, 0
        while not done:
            logits, value = self.model.predict(obs[None, :])
            action = self._prune_actions(logits, env)
            obs, reward, done = env.step(action)
            env.score = 0
            ep_reward += reward
        for row in env.board:
            print(row)
        return ep_reward

    def _value_loss(self, returns, value):
    # value loss is typically MSE between value estimates and returns
        return self.params['value']*kls.mean_squared_error(returns, value)

    def _logits_loss(self, acts_and_advs, logits):
        # a trick to input actions and advantages through same API
        actions, advantages = tf.split(acts_and_advs, 2, axis=-1)
        # sparse categorical CE loss obj that supports sample_weight arg on call()
        # from_logits argument ensures transformation into normalized probabilities
        weighted_sparse_ce = kls.SparseCategoricalCrossentropy(from_logits=True)
        # policy loss is defined by policy gradients, weighted by advantages
        # note: we only calculate the loss on the actions we've actually taken
        actions = tf.cast(actions, tf.int32)
        policy_loss = weighted_sparse_ce(actions, logits, sample_weight=advantages)
        # entropy loss can be calculated via CE over itself
        entropy_loss = kls.categorical_crossentropy(logits, logits, from_logits=True)
        # here signs are flipped because optimizer minimizes
        return policy_loss - self.params['entropy']*entropy_loss

model = Model(num_actions=env.action_space)
agent = A2CAgent(model)
# agent._load_saved(env)
rewards_history = agent.train(env)
print(agent.test(env))
agent.model.save_weights('Saved/v3_a2c.h5')

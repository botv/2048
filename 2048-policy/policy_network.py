import tensorflow as tf
from tensorflow.keras import Model
from tensorflow.keras.layers import Dense
import numpy as np
import game
import time

class Network(tf.keras.Model):

    def __init__(self, dense_0, dense_1, dense_2):
        super(Network, self).__init__()

        self.dense_0 = tf.keras.layers.Dense(dense_0, input_dim = 16, activation=tf.nn.relu)
        self.dense_1 = Dense(dense_1, activation=tf.nn.relu)
        self.dense_2 = Dense(dense_2, activation=tf.nn.softmax)

    def call(self, x, training=True):
        x = self.dense_0(x)
        x = self.dense_1(x)
        x = self.dense_2(x)
        return x


network = Network(32, 64, 4)

network = tf.keras.Sequential()
network.add(tf.keras.layers.Dense(16, input_dim=16,  activation='relu'))
network.add(tf.keras.layers.Dense(64, activation='relu'))
network.add(tf.keras.layers.Dense(16, activation='relu'))
network.add(tf.keras.layers.Dense(4, activation = "softmax"))
network.build()
optimizer = tf.keras.optimizers.Adam(learning_rate = 0.001)
compute_loss = tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True)


rollOuts = 10000
updateOccurence = 100
g = game.Game()

optimizer = tf.keras.optimizers.Adam()
loss_object = tf.keras.losses.SparseCategoricalCrossentropy(from_logits = True)

scores = []

def discountRewards(reward, gamma = 0.7):
    discounted_reward = np.zeros_like(reward)
    running_add = 0
    for t in reversed(range(0, reward.size)):
        running_add = running_add * gamma + reward[t]
        discounted_reward[t] = running_add
    return discounted_reward


@tf.function
def getTrainable():
    return network.trainable_variables

def getAction(action_dist, possible_actions):
    action_dist = [action_dist[i] for i in possible_actions]
    action_sum = sum(action_dist)
    action_prob = [prob / action_sum for prob in action_dist]
    action = np.random.choice(action_dist,p=action_prob)
    return action


def train():
    gradBuffer = network.trainable_variables
    for ix,grad in enumerate(gradBuffer):
      gradBuffer[ix] = grad * 0


    for e in range(rollOuts):
        g.reset()
        rollout_mem = []
        rollout_score = 0
        done = False

        while not done:
            state = g.getState()
            prevScore = g.score
            possibleActions = g.getPossible()
            with tf.GradientTape() as tape:
                logits = network(state)
                action_dist = logits.numpy()
                action = getAction(action_dist[0], possibleActions)
                action = np.argmax(action_dist == action)
                loss = loss_object([action],logits)
            state, done = g.step(action)
            # g.getHighest()
            grads = tape.gradient(loss,network.trainable_variables)
            reward = g.score - prevScore
            if done: reward-=10
            rollout_mem.append([grads, reward])
        scores.append(g.score)
        rollout_mem = np.array(rollout_mem)
        rollout_mem[:,1] = discountRewards(rollout_mem[:,1])
        for grads, reward in rollout_mem:
            for ix, grad in enumerate(grads):
                gradBuffer[ix] += grad * reward

        if e % updateOccurence == 0:
            optimizer.apply_gradients(zip(gradBuffer, network.trainable_variables))
            for i, grad in enumerate(gradBuffer):
                gradBuffer[i] = grad * 0

        if e % 100 == 0:
            print(f"Episode: {e}, Score: {np.mean(scores[-100:])}")
            for row in g.board:
                print(row)


train()
network.save_weights('./checkpoints/v1')

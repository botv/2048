import tensorflow as tf
from tensorflow.keras import Model
from tensorflow.keras.layers import Dense
import numpy as np
import game

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


rollOuts = 10000
updateOccurence = 100
g = game.Game()

optimizer = tf.keras.optimizers.Adam()
loss_object = tf.keras.losses.SparseCategoricalCrossentropy(from_logits = True)

scores = []

def discountRewards(reward, gamma = 0.8):
    discounted_reward = np.zeros_like(reward)
    running_add = 0
    for t in reversed(range(0, reward.size)):
        running_add = running_add * gamma + reward[t]
        discounted_reward[t] = running_add
    return discounted_reward


@tf.function
def getTrainable():
    return network.trainable_variables


def train():
    gradBuffer = getTrainable()
    for ix,grad in enumerate(gradBuffer):
      gradBuffer[ix] = grad * 0

    for e in range(rollOuts):
        g.reset()


        rollout_mem = []
        rollout_score = 0
        done = False

        while not done:
            state = g.getState()

            with tf.GradientTape() as tape:
                logits = network(state)
                action_dist = logits.numpy()
                action = np.random.choice(action_dist[0],p=action_dist[0])
                action = np.argmax(action_dist == action)
                loss = loss_object([action],logits)
            grads = tape.gradient(loss,getTrainable())
            state, reward, done = g.step(action)
            rollout_score += reward
            if done: reward-=10
            rollout_mem.append([grads, reward])

        scores.append(rollout_score)
        rollout_mem = np.array(rollout_mem)
        rollout_mem[:,1] = discountRewards(rollout_mem[:,1])

        for grads, reward in rollout_mem:
            for ix, grad in enumerate(grads):
                print(len(gradBuffer), 'grad')
                gradBuffer[ix] += grad * reward

        if e % updateOccurence == 0:
            optimizer.apply_gradients(zip(gradBuffer, getTrainable()))
            for i, grad in enumerate(gradBuffer):
                gradBuffer[i] = grad * 0

        if e % 100 == 0:
            print(f"Episode: {e}, Score: {np.mean(scores[-100:])}")

train()

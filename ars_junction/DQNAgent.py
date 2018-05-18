import numpy as np
import random as rn
import tensorflow as tf
import os
# Setting the seeds to get reproducible results
# https://keras.io/getting-started/faq/#how-can-i-obtain-reproducible-results-using-keras-during-development
os.environ['PYTHONHASHSEED'] = '0'
os.environ["PATH"] += os.pathsep + 'C:/Program Files (x86)/Graphviz2.38/bin/'
np.random.seed(42)
rn.seed(12345)
session_conf = tf.ConfigProto(intra_op_parallelism_threads=1, inter_op_parallelism_threads=1)
tf.set_random_seed(1234)
sess = tf.Session(graph=tf.get_default_graph(), config=session_conf)
from keras import backend as keras
keras.set_session(sess)

import gym
from collections import deque
from keras.models import Sequential
from keras.layers import Dense
from keras.optimizers import Adam
from keras.utils import plot_model
from keras import initializers
import createRoute

import environment
import matplotlib.pyplot as plt

""""
File is based on the tutorial of 
@url{https://keon.io/deep-q-learning/}
"""

# constant values
TRAIN_EPISODES  = 10000 #10000 is standaard
TEST_EPISODES   = 100    #100 is standaard
MEMORY_SIZE     = 2000
BATCH_SIZE      = 32
MAX_STEPS       = 400


class DQNAgent:
    def __init__(self, epsilon):
        self.state_size = state_size
        self.action_size = action_size
        self.memory = deque(maxlen=MEMORY_SIZE)
        self.gamma = 0.8  # discount rate
        self.epsilon = epsilon  # exploration rate
        self.epsilon_min = 0.01
        self.epsilon_decay = 0.999
        self.learning_rate = 0.001
        self.model = self._build_model()

    # Building neural Net for Deep-Q learning Model
    def _build_model(self):
        # set kernel_initializers: https://stackoverflow.com/questions/45230448/how-to-get-reproducible-result-when-running-keras-with-tensorflow-backend
        model = Sequential()
        model.add(Dense(self.state_size, input_dim=self.state_size,
                        activation='relu',
                        kernel_initializer=initializers.glorot_normal(seed=1337),
                        bias_initializer=initializers.Constant(value=0)))
        model.add(Dense(200,
                        activation='relu',
                        kernel_initializer=initializers.glorot_normal(seed=1337),
                        bias_initializer=initializers.Constant(value=0)))
        model.add(Dense(200,
                        activation='relu',
                        kernel_initializer=initializers.glorot_normal(seed=1337),
                        bias_initializer=initializers.Constant(value=0)))
        model.add(Dense(self.action_size,
                        activation='linear',
                        kernel_initializer=initializers.glorot_normal(seed=1337),
                        bias_initializer=initializers.Constant(value=0)))
        model.compile(loss='mse',
                      optimizer=Adam(lr=self.learning_rate))
        return model

    def remember(self, episode, state, action, reward, next_state, done):
        self.memory.append((episode, state, action, reward, next_state, done))

    def act(self, state, use_epsilon=True):
        if np.random.rand() <= self.epsilon and use_epsilon:
            return rn.randrange(self.action_size)
        act_values = self.model.predict(state)  # voorspel de state
        return np.argmax(act_values[0])  # returns action

    def replay(self, batch_size):
        minibatch = rn.sample(self.memory, batch_size)
        for _, state, action, reward, next_state, done in minibatch:
            target = reward
            if not done:
                target = (reward + self.gamma * np.amax(self.model.predict(next_state)[0])) # voorspel de volgende state
            target_f = self.model.predict(state) # voorspel de state
            target_f[0][action] = target
            self.model.fit(state, target_f, epochs=1, verbose=0) #train the neural network
        if self.epsilon > self.epsilon_min:
            self.epsilon *= self.epsilon_decay

    def load(self, name):
        self.model.load_weights(name)

    def save(self, name):
        self.model.save_weights(name)


def trainOrTest(batch_size, episodes, training):
    for e in range(episodes):

        # reset the env for a new episode
        state = env.reset()
        #createRoute.generate_random_routefile()
        state = np.reshape(state, [1, state_size])

        # Step through the episode until MAX_STEPS is reached
        for _ in range(MAX_STEPS):
            action = agent.act(state, use_epsilon=training)
            next_state, reward, done, _ = env.step(action)
            next_state = np.reshape(next_state, [1, state_size])
            agent.remember(e, state, action, reward, next_state, done)
            state = next_state
            if done:
                break

        # save function
        if training:
            agent.save("save/cartpole-ddqn.h5")

        # print statistics of this episode
        total_reward = sum([x[3] for x in agent.memory if x[0] == e])
        print("episode: {:d}/{:d}, total reward: {:.2f}, e: {:.2}"
              .format(e+1, episodes, total_reward, agent.epsilon))

        # Start experience replay if the agent.memory > batch_size
        if len(agent.memory) > batch_size and training:
            agent.replay(batch_size)


def plotResults():
    env.reset()
    np.save('data', env.result)
    leg = []
    for i, episode in enumerate(env.result):
        plt.plot(episode)
        leg.append('episode {}'.format(i+1))

    plt.legend(leg, loc='upper left')
    plt.xlabel('Time (0.1s/step)')
    plt.ylabel('Speed (m/s)')
    plt.show()


if __name__ == "__main__":
    env = gym.make('SumoEnv-v0')
    state_size = env.observation_space.shape[0]
    action_size = env.action_space.n

    trained = False
    if not trained:
        agent = DQNAgent(1.0)
        env.log = False
        env.test = False
        env.start(gui=False)
        trainOrTest(BATCH_SIZE, episodes=TRAIN_EPISODES, training=True)
        env.close()
    else:
        agent = DQNAgent(0.01)
        agent.load("save/cartpole-ddqn.h5")
    env.log = False
    env.test = True
    env.start(gui=True)
    trainOrTest(BATCH_SIZE, episodes=TEST_EPISODES, training=False)

    plotResults()

    agent.save('model')
    plot_model(agent.model, show_shapes=True)

    env.close()

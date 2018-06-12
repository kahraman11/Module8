import pdb

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
import constants
from constants import *

import environment
import matplotlib.pyplot as plt

""""
File is based on the tutorial of 
@url{https://keon.io/deep-q-learning/}
"""
# constant values
TRAIN_EPISODES  = 100 #10000 is standaard
TEST_EPISODES   = 100    #100 is standaard
MEMORY_SIZE     = 2000
BATCH_SIZE      = 100     #32 is standaard
MAX_STEPS       = 400   #400 is standaard


class DQNAgent:
    def __init__(self, epsilon, name):
        self.state_size = state_size
        self.action_size = action_size
        self.memory = deque(maxlen=MEMORY_SIZE)
        self.gamma = 0.8  # discount rate
        self.epsilon = epsilon  # exploration rate
        self.epsilon_min = 0.01
        self.epsilon_decay = 0.999
        self.learning_rate = 0.001
        self.model = self._build_model()
        self.name = name

    # Building neural Net for Deep-Q learning Model
    def _build_model(self):
        # set kernel_initializers: https://stackoverflow.com/questions/45230448/how-to-get-reproducible-result-when-running-keras-with-tensorflow-backend
        model = Sequential()
        model.add(Dense(self.state_size, input_dim=self.state_size,
                        activation='relu',
                        kernel_initializer=initializers.glorot_normal(seed=1337),
                        bias_initializer=initializers.Constant(value=0)))
        model.add(Dense(79,
                        activation='relu',
                        kernel_initializer=initializers.glorot_normal(seed=1337),
                        bias_initializer=initializers.Constant(value=0)))
        model.add(Dense(79,
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
        # print('replay', self.name, ' batch_size:', len(minibatch))

        # print(self.name)
        print("gotReward: ", gotReward)
        # print(DRIVEN)
        # print("collisions ", collisions)
        # print(minibatch)
        # print(self.name, self.memory[-1])

        for e, state, action, reward, next_state, done in minibatch:    # self.memory:, minibatch
            if self.name in gotReward:
                # print("!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!")
                reward = 1

            # if self.memory[-1][3] == -1:
            #     print("@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@")
            #     reward = -1

            for crash in collisions:
                if crash[0] == self.name:
                    # print("@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@")
                    reward = -1

            # print(self.name, reward)

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


def trainOrTest(batch_size, episodes, training):    # episodes = 10000
    for e in range(episodes):
        # reset the env for a new episode
        state = env.reset()
        # createRoute.generate_random_routefile()
        state = np.reshape(state, [1, state_size])

        # print("CARS: ", CARS)
        counter = 0
        new_agents = []
        hasDriven = []
        speeds = []

        for car in CARS:
            new_agents.append(agents[counter])
            counter += 1
            # print("New Agents aantal: ", len(new_agents))

        states = []

        # Step through the episode until MAX_STEPS is reached
        for step in range(MAX_STEPS):
                actions = []
                for car in CARS:
                    done = False
                    for agent in new_agents:
                        if car == agent.name:

                            # print("##############################states: ", states)
                            for status in states:
                                if status[0] == agent.name:
                                    state = status[1]
                                    break

                            action = agent.act(state, use_epsilon=training)
                            actions.append((agent.name, action))
                            env.step(action, car)
                            # print("Car: ", car, ", Agent: ", agent.name)
                            # next_state, reward, done, _ = env.step(action, car)
                            # next_state = np.reshape(next_state, [1, state_size])
                            # agent.remember(e, state, action, reward, next_state, done)
                            # state = next_state

                env.simulationStep()

            # while len(CARS) > 0:
                for agent in new_agents:
                    # print(car)
                    # print(CARS)
                    for car in list(CARS):
                        if car == agent.name:
                            next_state, reward, done, _ = env.secondStep(car)
                            next_state = np.reshape(next_state, [1, state_size])

                            if reward == - 10:
                                print(DRIVEN)
                            if car in DRIVEN:
                                if agent not in hasDriven:
                                    hasDriven.append(agent)
                                for (name, actie) in actions:
                                    if name == agent.name:
                                        action = actie
                                        break

                                agent.remember(e, state, action, reward, next_state, done)
                                speeds.append((car, state[0][0]))
                                state = next_state
                                states.append((car, state))
                                # print(hasDriven)
                                break

        if not training:
            for car in hasDriven:
                counter = 1
                value = 0
                # print("Speeds: ", speeds)
                for double in speeds:
                    # print(double)
                    if double[0] == car.name:
                        if double[1] > 0:
                            value += double[1]
                            counter += 1
                avgSpeeds.append((car.name, value/counter))

        i = 0
        if training:
            for agent in agents:
                agent.save("save/cartpole-ddqn" + str(i) + ".h5")
                i += 1

        # save function
        # if training:
        #    agent.save("save/cartpole-ddqn.h5")

        # print statistics of this episode
        for car in DRIVEN:
            for agent in hasDriven:
                if car == agent.name:
                    total_reward = sum([x[3] for x in agent.memory if x[0] == e])
                    # print(agent.name, agent.memory[-10:-1])
                    # if agent.memory[-1][3] == -1:
                    #     print("episode: {:d}/{:d}, total reward: {:.2f}, e: {:.2} agent: {:s}"
                    #           .format(e + 1, episodes, -1, agent.epsilon, agent.name))
                    # elif total_reward == -1:
                    #     print("episode: {:d}/{:d}, total reward: {:.2f}, e: {:.2} agent: {:s}"
                    #       .format(e+1, episodes, 0, agent.epsilon, agent.name))
                    # else:
                    #     print("episode: {:d}/{:d}, total reward: {:.2f}, e: {:.2} agent: {:s}"
                    #           .format(e + 1, episodes, total_reward, agent.epsilon, agent.name))
                    print("episode: {:d}/{:d}, total reward: {:.2f}, e: {:.2} agent: {:s}"
                                     .format(e + 1, episodes, total_reward, agent.epsilon, agent.name))

        # Start experience replay if the agent.memory > batch_size
        for agent in hasDriven:
            if len(agent.memory) > batch_size: # and training:
                agent.replay(batch_size)


def plotResults():
    env.reset()
    np.save('data', env.result)
    leg = []
    test = agents

    # print(test.pop(0).memory[0][1][0][0])

    # print(env.result)
    print("avgSpeeds: ", avgSpeeds)
    for agent in agents:
        leg.append(agent.name)
        result = []
        for double in avgSpeeds:
            if agent.name == double[0]:
                result.append(double[1])
        plt.plot(result)

    # for i, episode in enumerate(env.result):
    #     for _ in agents:
    #         try:
    #             plt.plot(test.pop(0).memory[0][1][0][0])
    #             leg.append("car {}".format(i+1))
    #         except:
    #             pass

    # print(env.result)

    # for i, episode in enumerate(env.result):
    #     plt.plot(episode)
    #     leg.append('episode {}'.format(i+1))

    plt.legend(leg, loc='upper left')
    plt.xlabel('Episodes')
    plt.ylabel('Avergae Speed (m/s)')
    plt.show()


if __name__ == "__main__":
    env = gym.make('SumoEnv-v0')
    state_size = env.observation_space.shape[0]
    action_size = env.action_space.n

    agents = []

    trained = True
    if not trained:
        env.log = False
        env.test = False
        env.start(gui=False)
        # print("CARS: ", CARS)
        for x in range(0, 13):
            agents.append(DQNAgent(1.0, "AUTO" + str(x)))
        # print("Agents: ", agents)

        trainOrTest(BATCH_SIZE, episodes=TRAIN_EPISODES, training=True)
        env.close()
    else:
        for x in range(0, 13):
            agent = DQNAgent(0.01, "AUTO" + str(x))
            agent.load("save/cartpole-ddqn" + str(x) + ".h5")
            agents.append(agent)
    env.log = False
    env.test = True
    env.start(gui=True)
    trainOrTest(BATCH_SIZE, episodes=TEST_EPISODES, training=False)

    plotResults()

    # agent.save('model')
    # plot_model(agent.model, show_shapes=True)

    env.close()

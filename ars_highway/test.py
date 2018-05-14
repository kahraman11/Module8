
import gym
from collections import deque
from keras.models import Sequential
from keras.layers import Dense
from keras.optimizers import Adam
from keras.utils import plot_model
from keras import initializers


import environment
import matplotlib.pyplot as plt
from agents.DQNAgent import *

env = gym.make('SumoEnv-v0')
state_size = env.observation_space.shape[0]
action_size = env.action_space.n
agent = DQNAgent(state_size, action_size)

env.log = False
env.test = False
env.start(gui=False)
trainOrTest(env, agent, state_size, BATCH_SIZE, EPISODES, training=True)
env.close()

env.log = True
env.test = True
env.start(gui=True)
trainOrTest(env, agent, state_size, BATCH_SIZE, episodes=5, training=False)

plotResults()

agent.save('model')
plot_model(env, agent.model, show_shapes=True)

env.close()
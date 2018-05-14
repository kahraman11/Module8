import numpy as np
import matplotlib.pyplot as plt


def plotResults(data):
    leg = []
    for i, episode in enumerate(data):
        plt.plot(episode)
        leg.append('episode {}'.format(i + 1))

    plt.legend(leg, loc='upper left')
    plt.xlabel('Time (0.1s/step)')
    plt.ylabel('Speed (m/s)')
    plt.show()


episodes = [5, 10, 13, 18, 26]

data = np.load("agents/result.npy").tolist()
to_plot = []
for i, row in enumerate(data):
    if i + 1 in episodes:
        to_plot.append(row)

plotResults(to_plot)

import numpy as np
import matplotlib.pyplot as plt
from multiprocessing import Pool, cpu_count
from tqdm import tqdm
from scipy.stats import norm
import os


factors = [2,12]
diff = [1,2]

experimental_config = "plots"

BBSE = "SE_with_BrakeBooster"

SE = "SE"

LUCB = "LUCB1"

BBLUCB = "LUCB1_with_BrakeBooster"

exp1 = [SE, BBSE, 4,2 ]

exp2 = [SE, BBSE, 1,2 ]

exp3 = [LUCB, BBLUCB, 1, 3 ]

#exp3 = [LUCB, BBLUCB, 1]


experiments = [exp2]
n_cpu = 10

n_trials = 1000

n_gap = int(n_trials/n_cpu)
current_dir = os.path.dirname(__file__)

colors = ['b','r','g']

prefix = current_dir + '/logs/plots/'

for exp in experiments:
    for ind in range(exp[3]):
        if (ind == 1):
            folder = current_dir + '/logs/' + exp[ind] + '/'
            sub_gap = exp[2]
            histogram_vector = np.zeros(n_trials)
            for i in range(n_cpu):
                start_ind = int(i * n_gap)
                end_ind = int((i + 1) * n_gap)
                file_name = str(i) + str(sub_gap) + '12' + '.npy'

                completeName = os.path.join(folder, file_name)

                with open(completeName, 'rb') as f:
                    temp = np.load(f)
                    histogram_vector[start_ind:end_ind] = temp[0:n_gap]

            plt.xticks(fontsize=13)
            plt.yticks(fontsize=13)

            plt.xlabel('Stopping time', fontsize=13)
            plt.ylabel('Number of Trials', fontsize=13)
            # plt.title('Histogram of stopping times')
            # plt.legend(fontsize=15)

            counts, bins, _ = plt.hist(histogram_vector, bins=1000, color=colors[ind], alpha=0.5, edgecolor=colors[ind],
                                       lw=3,
                                       label=exp[ind])
            # Fit a Gaussian curve

            # mu, sigma = norm.fit(histogram_vector)
            # print(mu, sigma)
            # Plot the Gaussian curve
            # x = np.linspace(bins[0], bins[-1], 100)
            # gaussian_curve = norm.pdf(x, mu, sigma)
            # gaussian_curve = gaussian_curve * counts.max() / gaussian_curve.max()
            # plt.plot(x, gaussian_curve, 'r-', linewidth=2)

            histogram_vector.sort()
            X = np.arange(n_trials)
            plt.plot(histogram_vector, X, color=colors[ind], lw=3, label=exp[ind] + "-CDF", alpha=0.5)
            legend = plt.legend(fontsize=15, loc = 'center right')
            legend.get_frame().set_alpha(None)
            legend.get_frame().set_facecolor((0, 0, 0, 0.01))
        else:
            folder = current_dir + '/logs/' + exp[ind] + '/'
            sub_gap = exp[2]
            histogram_vector = np.zeros(n_trials)
            for i in range(n_cpu):
                start_ind = int(i * n_gap)
                end_ind = int((i + 1) * n_gap)
                if (exp[ind] == BBLUCB):
                    file_name = str(i) + str(sub_gap) + "2" + '.npy'
                elif (exp[ind] == LUCB):
                    file_name = str(i) + '.npy'
                else:
                    file_name = str(i) + str(sub_gap) + '.npy'

                completeName = os.path.join(folder, file_name)

                with open(completeName, 'rb') as f:
                    temp = np.load(f)
                    histogram_vector[start_ind:end_ind] = temp[0:n_gap]

            plt.xticks(fontsize=13)
            plt.yticks(fontsize=13)

            plt.xlabel('Stopping time', fontsize=13)
            plt.ylabel('Number of Trials', fontsize=13)
            # plt.title('Histogram of stopping times')
            # plt.legend(fontsize=15)

            counts, bins, _ = plt.hist(histogram_vector, bins=1000, color=colors[ind], alpha=0.5, edgecolor=colors[ind],
                                       lw=3,
                                       label=exp[ind])
            # Fit a Gaussian curve

            # mu, sigma = norm.fit(histogram_vector)
            # print(mu, sigma)
            # Plot the Gaussian curve
            # x = np.linspace(bins[0], bins[-1], 100)
            # gaussian_curve = norm.pdf(x, mu, sigma)
            # gaussian_curve = gaussian_curve * counts.max() / gaussian_curve.max()
            # plt.plot(x, gaussian_curve, 'r-', linewidth=2)

            histogram_vector.sort()
            X = np.arange(n_trials)
            plt.plot(histogram_vector, X, color=colors[ind], lw=3, label=exp[ind] + "-CDF", alpha=0.5)
            legend  = plt.legend(fontsize=15)
            legend.get_frame().set_alpha(None)
            legend.get_frame().set_facecolor((0, 0, 0, 0.01))


    file_name = experimental_config + exp1[0] + str(sub_gap) + '.pdf'
    completeName = os.path.join(prefix, file_name)
    plt.savefig(completeName, format='pdf')
    plt.show()
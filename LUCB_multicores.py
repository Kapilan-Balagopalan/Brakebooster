import numpy as np
import matplotlib.pyplot as plt
from multiprocessing import Pool, cpu_count
from tqdm import tqdm
from scipy.stats import norm
import os


n_cpu = 10
n_trials = 1000
histogram_vector = np.zeros(n_trials)

no_non_stops = 0
cum_non_stops = 0
no_non_stops2 = 0
cum_non_stops2 = 0

variance = 1
mu_best = 1
mu_sub =0.9

sub_gap = 1

n_trials = 1000
delta = 0.05


#np.random.seed(100)
big_trials = 1
K = 4
eps_list = {0.00}
color_list = ['blue','g','r']

esp_count = 0


experiment_config = "LUCB1"
#experiment_config = "SE_LUCB_CB"

current_dir = os.path.dirname(__file__)
prefix = current_dir + '/logs/' + experiment_config + '/'

def LUCB1(n_gap,seed_in):
    #return
    no_non_stops = 0
    cum_non_stops = 0
    no_non_stops2 = 0
    cum_non_stops2 = 0
    np.random.seed(seed_in)
    histogram_vector_local = np.zeros(n_gap+1)
    for i in tqdm(range(n_gap)):
        switch = True
        Arms = [0, 1, 2, 3]
        Means = [mu_best, mu_sub, mu_sub, mu_sub]
        t = 1
        samples = np.zeros(K)
        ucb = np.zeros(K)
        lcb = np.zeros(K)
        samples_count = np.zeros(K)
        for j in range(K):
            samples[j] = np.random.normal(Means[j], variance, 1)
            samples_count[j] = samples_count[j] + 1
            t = t + 1
        while (True):
            for j in Arms:
                ucb[j] = samples[j] + np.sqrt(
                    np.log(1.25 * K * (np.sum(samples_count) ** 4) / delta) / (2 * samples_count[j]))
                lcb[j] = samples[j] - np.sqrt(
                    np.log(1.25 * K * (np.sum(samples_count) ** 4) / delta) / (2 * samples_count[j]))

            h_arm = np.argmax(samples)
            ucb[h_arm] =  float('-inf')
            l_arm = np.argmax(ucb)

            if (lcb[h_arm] > ucb[l_arm] ):
                break

            samples[h_arm] = (samples[h_arm] * samples_count[h_arm] + np.random.normal(Means[h_arm], variance, 1))
            samples[l_arm] = (samples[l_arm] * samples_count[l_arm] + np.random.normal(Means[l_arm], variance, 1))
            samples_count[h_arm] = samples_count[h_arm] + 1
            samples_count[l_arm] = samples_count[l_arm] + 1
            samples[h_arm] = samples[h_arm] / samples_count[h_arm]
            samples[l_arm] = samples[l_arm] / samples_count[l_arm]

            t = t + 2

        # print(t-1)
        histogram_vector_local[i] = t - 1
        #histogram_vector_local[-1] = no_non_stops
        script_name = os.path.basename(__file__)
        file_name = str(seed_in) + str(sub_gap) +  '.npy'

        completeName = os.path.join(prefix, file_name)

        with open(completeName, 'wb') as f:

            np.save(f, histogram_vector_local)

    # print(histogram_vector)
    # plt.hist(histogram_vector, bins=100, color=color_list[esp_count], alpha=0.3, edgecolor=color_list[esp_count], label='eps = ' + str(eps), lw=3)


def t():
    np.random.seed(15751)
    n_gap = int(n_trials/n_cpu)
    pool = Pool(processes=n_cpu)

    no_stops = 0

    for i in range(n_cpu):
        pool.apply_async(LUCB1, args=(n_gap,i))

    pool.close()
    pool.join()

    script_name = os.path.basename(__file__)
    for i in range(n_cpu):
        start_ind = int(i * n_gap)
        end_ind = int((i + 1) * n_gap)
        file_name = str(i) + str(sub_gap) +  '.npy'

        completeName = os.path.join(prefix, file_name)

        with open(completeName, 'rb') as f:
            temp = np.load(f)
            histogram_vector[start_ind:end_ind] = temp[0:n_gap]
            no_stops = no_stops + temp[-1]

    plt.xticks(fontsize=13)
    plt.yticks(fontsize=13)

    plt.xlabel('Stopping time', fontsize=13)
    plt.ylabel('Number of Trials', fontsize=13)
    # plt.title('Histogram of stopping times')
    # plt.legend(fontsize=15)


    counts, bins, _ =  plt.hist(histogram_vector, bins=1000, color=color_list[esp_count], alpha=0.5, edgecolor=color_list[esp_count], lw=3, label= experiment_config)
    # Fit a Gaussian curve

    mu, sigma = norm.fit(histogram_vector)
    # print(mu, sigma)
    # Plot the Gaussian curve
    x = np.linspace(bins[0], bins[-1], 100)
    gaussian_curve = norm.pdf(x, mu, sigma)
    gaussian_curve = gaussian_curve * counts.max() / gaussian_curve.max()
    #plt.plot(x, gaussian_curve, 'r-', linewidth=2)

    plt.legend(fontsize=15)
    file_name = experiment_config  + str(sub_gap) + '.png'
    completeName = os.path.join(prefix, file_name)
    plt.savefig(completeName, format='png')
    file_name = experiment_config + str(sub_gap) + '.pdf'
    completeName = os.path.join(prefix, file_name)
    plt.savefig(completeName, format='pdf')
    plt.show()
    print(no_stops)

    histogram_vector.sort()
    alpha_quantile_size = 10
    width = n_trials / alpha_quantile_size
    for i in range(alpha_quantile_size):
        print(histogram_vector[int(width * (i + 1)) - 1])



if __name__ == '__main__':
    t()




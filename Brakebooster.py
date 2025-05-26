import numpy as np
import matplotlib.pyplot as plt
from multiprocessing import Pool, cpu_count
from tqdm import tqdm
from scipy.stats import norm
import os


n_cpu = 10
n_trials = 1000
histogram_vector = np.zeros(n_trials)
eps_list = {0.00}
color_list = ['blue','g','r']
esp_count = 0
eps = 0.00
K = 4
variance = 1
mu_best = 1
mu_sub = 0.9
wfactor = 12
sub_gap = 1

delta_0 = 0.05


#L1 = np.ceil((np.log(1 + 2/delta))/np.log(1/(4*delta_0*np.exp(1))))
L1 = 1
T1 = 16384

experiment_config = "SE_with_BrakeBooster"
#experiment_config = "SE_with_BrakeBooster"
current_dir = os.path.dirname(__file__)
prefix = current_dir + '/logs/' + experiment_config + '/'


def budgetIdentification(algo_name, L, T, delta_0):
    if(algo_name == "SE_with_BrakeBooster"):
        return SE_base_algo(L, T, delta_0)
    elif(algo_name == "SE_MOD_with_BrakeBooster"):
        return SE_modified_base_algo(L, T, delta_0)
    else:
        print("Not implemented")

def SE_base_algo(L, T, delta):
    votes = np.zeros(K)
    unterminated_count = 0
    for l in range(L):
        Arms = [0, 1, 2, 3]
        Means = [mu_best, mu_sub, mu_sub, mu_sub]
        t = 1
        samples = np.zeros(K)
        samples_count = np.zeros(K)
        for j in range(K):
            samples[j] = np.random.normal(Means[j], variance, 1)
            samples_count[j] = samples_count[j] + 1
            t = t + 1
        while (len(Arms) > 1):
            for j in Arms:
                samples[j] = (samples[j] * (t-1) + np.random.normal(Means[j], variance, 1)) / t
            for j in Arms:
                # print(np.max(samples) - samples[j])
                # print( np.sqrt(2* np.log(3.3*(t**2)/delta)/t))
                # print("end")

                temp =  2*np.sqrt(np.log(K*(t ** 2) / delta) / t)
                if (np.max(samples) - samples[j] >= temp):
                    # Means.remove(Means[j])
                    samples[j] = float('-inf')
                    Arms.remove(j)
            t = t + len(Arms)
            if(t > T):
                break

        if (len(Arms) == 1):
            votes[Arms[0]] = votes[Arms[0]] + 1
        else:
            unterminated_count = unterminated_count  + 1

    if (unterminated_count > L/2):
        return -1
    else:
        return np.argmax(votes)


def SE_modified_base_algo(L, T, delta):
    votes = np.zeros(K)
    unterminated_count = 0
    for l in range(L):
        arm_idxes = [idx for idx in range(K)]
        #arm_idxes = np.arange(K)
        Means =  [mu_best] + [mu_sub]*(K-1)
        t = 1
        arm_muhats = np.zeros(K)
        arm_nsamples = np.zeros(K)
        arm_sum_rewards = np.zeros(K)
        for t in range(int(T)):
            # select arm
            _nsamples = arm_nsamples[arm_idxes]
            _idx = np.argmin(_nsamples)
            a_idx_to_pull = arm_idxes[_idx]

            # sample and get reward
            _sample = np.random.normal(Means[a_idx_to_pull], variance)
            arm_sum_rewards[a_idx_to_pull] += _sample
            arm_nsamples[a_idx_to_pull] += 1

            if (t < K):
                continue

            # compute and remove the arms
            arm_muhats[arm_idxes] = arm_sum_rewards[arm_idxes] / arm_nsamples[arm_idxes]
            thres = np.sqrt(2 * np.log(3.3 * K * (t ** 4) / delta) / arm_nsamples)
            max_muhat = np.max(arm_muhats)

            for idx in arm_idxes:
                if (max_muhat - arm_muhats[idx] >= thres[idx]):
                    # Means.remove(Means[j])
                    arm_muhats[idx] = float('-inf')
                    #index = np.argwhere(arm_idxes == idx)
                    arm_idxes.remove(idx)
                    # break

            if len(arm_idxes) <= 1:
                b_stopped = True
                break

        if (len(arm_idxes)== 1):
            votes[arm_idxes[0]] = votes[arm_idxes[0]] + 1
        else:
            unterminated_count = unterminated_count  + 1

    if (unterminated_count > L/2):
        return -1
    else:
        return np.argmax(votes)


def brake_booster(n_gap,seed_in):
    #return
    np.random.seed(seed_in)
    histogram_vector_local = np.zeros(n_gap)
    arm = -1
    factor = 1.2
    for i in tqdm(range(n_gap)):
        r = 0
        while (True):
            r = r + 1
            for j in range(r):
                c = j + 1
                L = int(np.ceil((factor ** (r - c)) * L1))
                T = np.ceil((factor ** (c - 1)) * T1)


                arm = budgetIdentification(experiment_config, L, T, delta_0)
                histogram_vector_local[i] = histogram_vector_local[i] + L * T
                if (arm != -1):
                    break

            if (arm != -1):
                break

    script_name = os.path.basename(__file__)
    file_name = str(seed_in) + str(sub_gap) + str(wfactor) + '.npy'

    completeName = os.path.join(prefix, file_name)

    with open(completeName, 'wb') as f:

        np.save(f, histogram_vector_local)









def t():
    #arm = budgetIdentification(experiment_config, L1, T1, delta_0)
    np.random.seed(15751)
    n_gap = int(n_trials/n_cpu)
    pool = Pool(processes=n_cpu)
    #brake_booster(n_gap,1)
    for i in range(n_cpu):
        pool.apply_async(brake_booster, args=(n_gap,i))

    pool.close()
    pool.join()

    script_name = os.path.basename(__file__)
    for i in range(n_cpu):
        start_ind = int(i * n_gap)
        end_ind = int((i + 1) * n_gap)
        file_name = str(i) + str(sub_gap) + str(wfactor) +  '.npy'

        completeName = os.path.join(prefix, file_name)

        with open(completeName, 'rb') as f:
            histogram_vector[start_ind:end_ind] = np.load(f)

    counts, bins, _ = plt.hist(histogram_vector, bins=100, color=color_list[esp_count], alpha=0.5, edgecolor=color_list[esp_count],lw=3, label= experiment_config)
    # Fit a Gaussian curve

    mu, sigma = norm.fit(histogram_vector)
    #print(mu, sigma)
    # Plot the Gaussian curve
    x = np.linspace(bins[0], bins[-1], 100)
    gaussian_curve = norm.pdf(x, mu, sigma)
    gaussian_curve = gaussian_curve * counts.max()/gaussian_curve.max()
    #plt.plot(x, gaussian_curve, 'r-', linewidth=2)

    #plt.ticklabel_format(style='sci', axis='x', scilimits=(0, 0))
    plt.xticks(fontsize=13)
    plt.yticks(fontsize=13)

    plt.xlabel('Stopping time', fontsize=13)
    plt.ylabel('Number of Trials', fontsize=13)
    # plt.title('Histogram of stopping times')
    plt.legend(fontsize=15)
    file_name = experiment_config + str(sub_gap) + str(wfactor)  + '.png'
    completeName = os.path.join(prefix, file_name)
    plt.savefig(completeName, format='png')
    file_name = experiment_config + str(sub_gap) + str(wfactor)  + '.pdf'
    completeName = os.path.join(prefix, file_name)
    plt.savefig(completeName, format='pdf')
    plt.show()

    histogram_vector.sort()
    alpha_quantile_size = 10
    width = n_trials/alpha_quantile_size
    for i in range(alpha_quantile_size):
        print(histogram_vector[int(width*(i+1)) - 1])





if __name__ == '__main__':
    t()




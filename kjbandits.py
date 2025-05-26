from kjunutils3_v2 import *
import test_lb as lb

import logging
logging.basicConfig(level=logging.WARN)
# def dGaussian(p,q):
#     sigma=1.0
#     return (p-q)**2/(2*sigma**2);
# 
# def dBernoulli(p,q):
#     eps = 1e-16
#     res=0
#     if (p!=q):
#         if (p<=0):
#             p = eps
#         if (p>=1):
#             p = 1-eps
#         res=(p*log(p/q) + (1-p)*log((1-p)/(1-q))) 
#     return res
# 
# div = dBernoulli
# 
class BanditEnv:
    def __init__(self):
        pass

class Bernoulli(BanditEnv):
    def __init__(self, mu,seed=None):
        self.K = len(mu)
        self.mu = mu
        self.seed_ary = gen_seeds2(self.K, seed=None)
        self.generator = [ra.RandomState(s) for s in self.seed_ary]
        self.div = lb.dBernoulli

    def get_reward(self, i_arm):
        return float(self.generator[i_arm].rand() < self.mu[i_arm])

    def get_div(self):
        return self.div


class Gaussian(BanditEnv):
    def __init__(self, mu, sig_sq, seed=1919):
        self.K = len(mu)
        self.mu = mu
        self.sig_sq = sig_sq
        self.seed_ary = gen_seeds2(self.K, seed)
        self.generator = [ra.RandomState(s) for s in self.seed_ary]
        self.div = lambda p,q: lb.dGaussian(p,q,self.sig_sq)

    def get_reward(self, i_arm):
        z = self.generator[i_arm].randn()  
        return self.mu[i_arm] + np.sqrt(self.sig_sq)*z

    def get_div(self):
        return self.div

class GaussianRigged(BanditEnv):
    """
    rig_schedule: K-dim list with each value being (n_times_to_rig, shift)
    """
    def __init__(self, mu, sig_sq, rig_schedule, seed=None):
        self.K = len(mu)
        self.mu = mu
        self.sig_sq = sig_sq
        self.seed_ary = gen_seeds2(self.K, seed=None)
        self.generator = [ra.RandomState(s) for s in self.seed_ary]
        self.div = lambda p,q: lb.dGaussian(p,q,self.sig_sq)
        self.n_pulls = np.zeros(self.K)
        self.rig_schedule = rig_schedule # dictionary of arm index to how many times

    def get_reward(self, i_arm):
        if (self.n_pulls[i_arm] <= self.rig_schedule[i_arm][0] - 1):
            noise = self.rig_schedule[i_arm][1]
        else:
            noise = np.sqrt(self.sig_sq)*self.generator[i_arm].randn()  

        self.n_pulls[i_arm] += 1
        return self.mu[i_arm] + noise


    def get_div(self):
        return self.div


class BanditAlg:
    def __init__(self):
        pass

    def print_stats(self):
        pass

class Chernoff(BanditAlg):
    def __init__(self, K, div):
        assert(K == 2)
        self.n_pulls = np.zeros(K)
        self.sum_rewards = np.zeros(K)
        self.t = 0
        self.K = K
        self.div = div
        pass

    def next_arm(self):
        if any(self.n_pulls == 0):
            return np.argmin(self.n_pulls)

        theta = self.sum_rewards / self.n_pulls
        tiltheta = self.sum_rewards.sum() / self.n_pulls.sum()

        test = self.div(theta[0], tiltheta) < self.div(theta[1], tiltheta)
#        ipdb.set_trace()
        return int(test) # choose whichever with higher divergence.

    def update(self, i_arm, reward):
        self.sum_rewards[i_arm] += reward
        self.n_pulls[i_arm] += 1
        self.t += 1

    def get_best_arm(self):
        if (self.t <= self.K):
            return -1
        me = self.sum_rewards / self.n_pulls
        max_reward = me.max()
        return ra.choice(np.where(me == max_reward)[0])

class Uniform(BanditAlg):
    def __init__(self, K):
        self.n_pulls = np.zeros(K)
        self.sum_rewards = np.zeros(K)
        self.t = 0
        self.K = K
        pass

    def next_arm(self):
        return np.argmin(self.n_pulls)

    def update(self, i_arm, reward):
        self.sum_rewards[i_arm] += reward
        self.n_pulls[i_arm] += 1
        self.t += 1

    def get_best_arm(self):
        if (self.t <= self.K):
            return -1
        me = self.sum_rewards / self.n_pulls
        max_reward = me.max()
        return ra.choice(np.where(me == max_reward)[0])

class Lucb(BanditAlg):
    """ based on lucb++ """
    def __init__(self, K, sig_sq, delta):
        self.n_pulls = np.zeros(K)
        self.sum_rewards = np.zeros(K)
        self.t = 0
        self.K = K
        self.sig_sq = sig_sq
        self.delta = delta
        self.success_yes = False
        self.success_t = -1
        self.success_arm = -1
        self.arm_queue = -1
        pass

    @staticmethod
    def dev(n_pulls, delta, sig_sq, t):
        # numer = np.log(np.log(2*n_pulls)) + 0.72*np.log(5.2/delta)
        # return np.sqrt(sig_sq)*1.7*np.sqrt(numer/n_pulls)
        logterm = np.log(1.25*t**4/delta)
        return np.sqrt(2*sig_sq*logterm/n_pulls)

    def next_arm(self):
        logging.debug(f"\n-> next_arm()")
        logging.debug(f"self.t = {self.t}")
        if any(self.n_pulls == 0):
            return np.argmin(self.n_pulls)

        if self.success_yes:
            choice = self.success_arm
        elif self.arm_queue != -1:
            logging.debug(f"queue")
            choice = self.arm_queue
            self.arm_queue = -1
        else:
            logging.debug(f"select arms")
            hatmu = self.sum_rewards / self.n_pulls
            maxval = np.max(hatmu)

            sidx = np.argsort(hatmu)[::-1]
            i_top = sidx[0]
            i_bot = sidx[1:]

            # dev_top = self.dev(self.n_pulls, self.delta/2/(self.K-1), self.sig_sq)
            # dev_bot = self.dev(self.n_pulls, self.delta/2           , self.sig_sq)
            dev_top = self.dev(self.n_pulls, self.delta/self.K, self.sig_sq, self.t)
            dev_bot = self.dev(self.n_pulls, self.delta/self.K, self.sig_sq, self.t)

            #- smallest LCB from the top
            h_t = i_top
            bar_LCB = hatmu[i_top] - dev_top[i_top]

            #- highest UCB from the bottom
            v = hatmu[i_bot] + dev_bot[i_bot]
            maxv = v.max()
            idx = ra.choice(np.where(v == maxv)[0])
            ell_t = i_bot[idx]
            bar_UCB = maxv

            logging.debug(f"sum_rewards = {self.sum_rewards}")
            logging.debug(f"n_pulls = {self.n_pulls}")
            logging.debug(f"hatmu = {hatmu}")
            logging.debug(f"dev_top = {dev_top}")
            logging.debug(f"h_t = {h_t}")
            logging.debug(f"ell_t = {ell_t}")
            logging.debug(f"bar_LCB = {bar_LCB:0.4f}")
            logging.debug(f"bar_UCB = {bar_UCB:0.4f}")

#            ipdb.set_trace()
            if (bar_LCB > bar_UCB):
                self.success_yes = True
                self.success_t = self.t
                self.success_arm = h_t
                choice = self.success_arm
            else:
                choice = h_t
                self.arm_queue = ell_t

        return choice

    def get_empirical_means(self):
        return self.sum_rewards / self.n_pulls
    
    def update(self, i_arm, reward):
        self.n_pulls[i_arm] += 1
        self.sum_rewards[i_arm] += reward
        self.t += 1

    def get_best_empirical_mean(self):
        if (self.t <= self.K):
            return np.nan
        me = self.sum_rewards / self.n_pulls
        return me.max()

    def get_best_arm(self):
        if (self.t <= self.K):
            return -1
        me = self.sum_rewards / self.n_pulls
        max_reward = me.max()
        return ra.choice(np.where(me == max_reward)[0])

    def print_stats(self):
        print('success_t = %5d, success_arm = %3d'% (self.success_t, self.success_arm))


class Lucb2(BanditAlg):
    """ based on lucb++ """
    def __init__(self, K, sig_sq, delta):
        self.n_pulls = np.zeros(K)
        self.sum_rewards = np.zeros(K)
        self.t = 0
        self.K = K
        self.sig_sq = sig_sq
        self.delta = delta
        self.success_yes = False
        self.success_t = -1
        self.success_arm = -1
        self.arm_queue = -1
        pass

    @staticmethod
    def dev(n_pulls, delta, sig_sq, t):
        # numer = np.log(np.log(2*n_pulls)) + 0.72*np.log(5.2/delta)
        # return np.sqrt(sig_sq)*1.7*np.sqrt(numer/n_pulls)
        logterm = np.log(1.25*np.log(2*n_pulls)/delta)
        return np.sqrt(2*sig_sq*logterm/n_pulls)

    def next_arm(self):
        logging.debug(f"\n-> next_arm()")
        logging.debug(f"self.t = {self.t}")
        if any(self.n_pulls == 0):
            return np.argmin(self.n_pulls)

        if self.success_yes:
            choice = self.success_arm
        elif self.arm_queue != -1:
            logging.debug(f"queue")
            choice = self.arm_queue
            self.arm_queue = -1
        else:
            logging.debug(f"select arms")
            hatmu = self.sum_rewards / self.n_pulls
            maxval = np.max(hatmu)

            sidx = np.argsort(hatmu)[::-1]
            i_top = sidx[0]
            i_bot = sidx[1:]

            # dev_top = self.dev(self.n_pulls, self.delta/2/(self.K-1), self.sig_sq)
            # dev_bot = self.dev(self.n_pulls, self.delta/2           , self.sig_sq)
            dev_top = self.dev(self.n_pulls, self.delta/(self.K-1), self.sig_sq, self.t)
            dev_bot = self.dev(self.n_pulls, self.delta, self.sig_sq, self.t)

            #- smallest LCB from the top
            h_t = i_top
            bar_LCB = hatmu[i_top] - dev_top[i_top]

            #- highest UCB from the bottom
            v = hatmu[i_bot] + dev_bot[i_bot]
            maxv = v.max()
            idx = ra.choice(np.where(v == maxv)[0])
            ell_t = i_bot[idx]
            bar_UCB = maxv

            logging.debug(f"sum_rewards = {self.sum_rewards}")
            logging.debug(f"n_pulls = {self.n_pulls}")
            logging.debug(f"hatmu = {hatmu}")
            logging.debug(f"dev_top = {dev_top}")
            logging.debug(f"h_t = {h_t}")
            logging.debug(f"ell_t = {ell_t}")
            logging.debug(f"bar_LCB = {bar_LCB:0.4f}")
            logging.debug(f"bar_UCB = {bar_UCB:0.4f}")

#            ipdb.set_trace()
            if (bar_LCB > bar_UCB):
                self.success_yes = True
                self.success_t = self.t
                self.success_arm = h_t
                choice = self.success_arm
            else:
                choice = h_t
                self.arm_queue = ell_t

        return choice

    def get_empirical_means(self):
        return self.sum_rewards / self.n_pulls
    
    def update(self, i_arm, reward):
        self.n_pulls[i_arm] += 1
        self.sum_rewards[i_arm] += reward
        self.t += 1

    def get_best_empirical_mean(self):
        if (self.t <= self.K):
            return np.nan
        me = self.sum_rewards / self.n_pulls
        return me.max()

    def get_best_arm(self):
        if (self.t <= self.K):
            return -1
        me = self.sum_rewards / self.n_pulls
        max_reward = me.max()
        return ra.choice(np.where(me == max_reward)[0])

    def print_stats(self):
        print('success_t = %5d, success_arm = %3d'% (self.success_t, self.success_arm))


class Kaufmann(BanditAlg):
    """
    track and stop.
    """
    def __init__(self, K, div, forced=True):
        self.n_pulls = np.zeros(K)
        self.sum_rewards = np.zeros(K)
        self.t = 0
        self.K = K
        self.div = div
        self.forced = forced
        self.forced_t_list = []
        pass

    def next_arm(self):
        if any(self.n_pulls == 0):
            return np.argmin(self.n_pulls)

        hatmu = self.sum_rewards / self.n_pulls
        maxval = np.max(hatmu)

        # forced sampling
        if self.forced and any(self.n_pulls < np.sqrt(self.t+1) - self.K/2):
            minpull = self.n_pulls.min()
            choice = ra.choice(np.where(self.n_pulls == minpull)[0]) 
            self.forced_t_list.append(self.t)
        else:
            # use optimal weights based on hatmu
            [_,w] = lb.OptimalWeights(hatmu, self.div)

            my_diff = self.n_pulls - w*self.t 
            minval = np.min(my_diff)
            minIdx = np.where(my_diff == minval)[0]
            if (len(minIdx) >= 2):
                print('ties! %s' % minIdx)
            
            choice = ra.choice(minIdx)
            print('t=%5d, %s, choice = %d' % (self.t,w,choice))

        return choice

    def update(self, i_arm, reward):
        self.sum_rewards[i_arm] += reward
        self.n_pulls[i_arm] += 1
        self.t += 1

    def get_best_arm(self):
        if (self.t <= self.K):
            return -1
        me = self.sum_rewards / self.n_pulls
        max_reward = me.max()
        return ra.choice(np.where(me == max_reward)[0])

    def print_stats(self):
        print('len(forced_t_list) = %s'% len(self.forced_t_list))
        print('forced_t_list = %s'% self.forced_t_list)

class ChernoffJun(BanditAlg):
    """
    our new algorithm:
    """
    def __init__(self, K, div):
        self.n_pulls = np.zeros(K)
        self.sum_rewards = np.zeros(K)
        self.t = 0
        self.K = K
        self.div = div
        pass

    def update(self, i_arm, reward):
        self.sum_rewards[i_arm] += reward
        self.n_pulls[i_arm] += 1
        self.t += 1


    def get_best_empirical_mean(self):
        if (self.t <= self.K):
            return np.nan
        me = self.sum_rewards / self.n_pulls
        return me.max()

    def get_best_arm(self):
        if (self.t <= self.K):
            return -1
        me = self.sum_rewards / self.n_pulls
        max_reward = me.max()
        return ra.choice(np.where(me == max_reward)[0])

    def next_arm(self):
        if any(self.n_pulls == 0):
            return np.argmin(self.n_pulls)

        hatmu = self.sum_rewards / self.n_pulls
        maxval = np.max(hatmu)

        i_best = ra.choice(np.where(hatmu == maxval)[0]) # break ties uniformly at random
        ratio = (self.n_pulls / self.n_pulls[i_best])
        pull_distr = self.n_pulls / sum(self.n_pulls)

        term = nans(self.K)
        term2 = nans(self.K)
        for j in range(self.K):
            if (j == i_best): 
                continue

            p = 1 / (1 + ratio[j])
            aux = p * hatmu[i_best] + (1-p) * hatmu[j]
            numer = self.div(hatmu[i_best],aux)
            denom = self.div(hatmu[j], aux)
            if (denom == 0):
                term[j] = np.infty
            else:
                term[j] = numer/denom
            term2[j] = (pull_distr[i_best] + pull_distr[j])*(p*numer + (1-p)*denom)

        v = np.nansum(term)
        if (v >= 1.0):
            choice = i_best
        else:
            minval = np.nanmin(term2)
            minIdx = np.where(term2 == minval)[0]
            if (len(minIdx) >= 2):
                print('ties! %s' % minIdx)

            choice = ra.choice(minIdx)

#         if (self.t % 1 == 0):
#             print('t=%5d, term2 = %s, choice = %s' % (self.t,term2,choice))

        return choice

class ChernoffJun_v2(BanditAlg):
    """
    interesting, missing forced exploration before!
    ChernoffJun with probability threshold beta to choose the best arm
    """
    def __init__(self, K, div):
        self.n_pulls = np.zeros(K)
        self.sum_rewards = np.zeros(K)
        self.t = 0
        self.K = K
        self.div = div
        pass

    def update(self, i_arm, reward):
        self.sum_rewards[i_arm] += reward
        self.n_pulls[i_arm] += 1
        self.t += 1

    def get_best_empirical_mean(self):
        if (self.t <= self.K):
            return np.nan
        me = self.sum_rewards / self.n_pulls
        return me.max()

    def get_best_arm(self):
        if (self.t <= self.K):
            return -1
        me = self.sum_rewards / self.n_pulls
        max_reward = me.max()
        return ra.choice(np.where(me == max_reward)[0])

    def next_arm(self):
        if any(self.n_pulls == 0):
            return np.argmin(self.n_pulls)

        if any(self.n_pulls < np.sqrt(self.t+1) - self.K/2):    # Yao added forced exploration
            return np.argmin(self.n_pulls)

        hatmu = self.sum_rewards / self.n_pulls
        maxval = np.max(hatmu)

        # with probability beta, choose the best arm
        if np.random.rand() < 0:
            return np.argmax(hatmu)
        else:
            i_best = ra.choice(np.where(hatmu == maxval)[0]) # break ties uniformly at random
            ratio = (self.n_pulls / self.n_pulls[i_best])
            pull_distr = self.n_pulls / sum(self.n_pulls)

            term = nans(self.K)
            term2 = nans(self.K)
            for j in range(self.K):
                if (j == i_best): 
                    continue

                p = 1 / (1 + ratio[j])
                aux = p * hatmu[i_best] + (1-p) * hatmu[j]
                numer = self.div(hatmu[i_best],aux)
                denom = self.div(hatmu[j], aux)
                if (denom == 0):
                    term[j] = np.infty
                else:
                    term[j] = numer/denom
                term2[j] = (pull_distr[i_best] + pull_distr[j])*(p*numer + (1-p)*denom)

            v = np.nansum(term)
            if (v >= 1.0):
                choice = i_best
            else:
                minval = np.nanmin(term2)
                minIdx = np.where(term2 == minval)[0]
                choice = ra.choice(minIdx)

            return choice


class ChernoffJun_tmp(BanditAlg):
    """
    a cheated version so it works with no noise.. just for the sake of understanding the behavior.
    """
    def __init__(self, K, div):
        self.n_pulls = np.zeros(K)
        self.sum_rewards = np.zeros(K)
        self.t = 0
        self.K = K
        self.div = div
        pass

    def update(self, i_arm, reward):
        self.sum_rewards[i_arm] += reward
        self.n_pulls[i_arm] += 1
        self.t += 1

    def get_best_arm(self):
        if (self.t <= self.K):
            return -1
        me = self.sum_rewards / self.n_pulls
        max_reward = me.max()
        return ra.choice(np.where(me == max_reward)[0])

    def next_arm(self):
        if any(self.n_pulls == 0):
            return np.argmin(self.n_pulls)

        hatmu = self.sum_rewards / self.n_pulls
        maxval = np.max(hatmu)

        i_best = ra.choice(np.where(hatmu == maxval)[0]) # break ties uniformly at random
        ratio = (self.n_pulls / self.n_pulls[i_best])
        hat_w = self.n_pulls / sum(self.n_pulls)

        term = nans(self.K)
        term2 = nans(self.K)
        for j in range(self.K):
            if (j == i_best): 
                continue

            p = 1 / (1 + ratio[j])
            aux = p * hatmu[i_best] + (1-p) * hatmu[j]
            numer = (hatmu[i_best]-aux)**2   ## in case sig_sq = 0
            denom = (hatmu[j] - aux)**2
            if (denom == 0):
                term[j] = np.infty
            else:
                term[j] = numer/denom
            term2[j] = (hat_w[i_best] + hat_w[j])*(p*numer + (1-p)*denom)

        v = np.nansum(term)
        if (v >= 1.0):
            choice = i_best
        else:
            minval = np.nanmin(term2)
            minIdx = np.where(term2 == minval)[0]
            if (len(minIdx) >= 2):
                print('ties! %s' % minIdx)

            choice = ra.choice(minIdx)

        if (self.t % 1 == 0):
#            print('t=%5d, term2 = %s, choice = %s' % (self.t,term2,choice))
            term2_a = [x * self.t for x in term2]
            N_1_sq = self.n_pulls[0]**2
            sum_other_sq = sum(self.n_pulls[1:]**2)
            F_mu = sum_other_sq / N_1_sq
#            print('t=%5d, N_1^2=%8d, sum_other_sq=%8d, F_mu=%-8g, min(term2_a)/t = %-10g, choice = %s' % (self.t, N_1_sq, sum_other_sq, F_mu, np.nanmin(term2_a)/self.n_pulls.sum(), choice))
            print('t=%5d, F_mu=%-8g, min(term2_a)/t = %-10g, |hat_w-alpha| = %-30s, choice = %s' % (self.t, F_mu, np.nanmin(term2_a)/self.n_pulls.sum(), np.abs(hat_w - self.alpha), choice))
#            print('t=%5d, F_mu=%-8g, |hat_w - alpha| = %s, choice = %s' % (self.t, F_mu, np.abs(hat_w - self.alpha), choice))
#            print('|hat_w - alpha| = %s' % np.abs(hat_w - alpha))
#            ipdb.set_trace()

        return choice

class Ucb(BanditAlg):
    """
    our new algorithm:
    """
    def __init__(self, K, R):
        self.n_pulls = np.zeros(K)
        self.sum_rewards = np.zeros(K)
        self.hat_mu = np.zeros(K)
        self.ucb = np.zeros(K)
        self.t = 0
        self.K = K
        self.R = R
        pass

    def update(self, i_arm, reward):
        self.sum_rewards[i_arm] += reward
        self.n_pulls[i_arm] += 1
        self.t += 1

    def get_best_empirical_mean(self):
        if (self.t <= self.K):
            return np.nan
        me = self.sum_rewards / self.n_pulls
        return me.max()

    def get_best_arm(self):
        if (self.t <= self.K):
            return -1
        me = self.sum_rewards / self.n_pulls
        max_reward = me.max()
        return ra.choice(np.where(me == max_reward)[0])

    def next_arm(self):
        if any(self.n_pulls == 0):
            return np.argmin(self.n_pulls)

        self.hat_mu[:] = self.sum_rewards / self.n_pulls
        maxval = np.max(self.hat_mu)

        tt = self.t+1
        one_over_delta = 1 + tt*(np.log(tt))**2
        self.ucb[:] = self.hat_mu + np.sqrt( (2*self.R**2)/self.n_pulls * np.log(one_over_delta) )
        choice = argmax_tiebreak(self.ucb, ra)

        return choice

class SequentialHalving:
    """
    FIXME: wait, the implementation is weird... why are we taking avg_reward in update()??
    """
    def __init__(self, K, B, reuse=False, seed=19):
        """
        B: n in the bandit book; the sampling budget.
        K: k in the bandit book; the size of the arm set
        """
        self.K = K
        self.B = B
        self.reuse = reuse
        self.seed = seed

        self.cur_best_arm = -1
        self.n_pulls = np.zeros(K,dtype=int)
        #self.avg_rewards = np.zeros(K)
        self.sum_rewards = np.zeros(K)
        self.my_rng = np.random.RandomState(seed)
        self.t = 1

        #- precompute the arm pull schedule
        self.T_ary, self.A_ary = SequentialHalving.calc_schedule(K,B)
        self.halving_times = np.cumsum(self.T_ary * self.A_ary[:-1])
        self.halving_times[-1] = self.B # this is cheating, but will ensure we use all the budget.
        self.L = len(self.T_ary)
        self.ell = 1   # iteration count
        self.surviving = np.arange(K)
        pass

    @staticmethod
    def calc_minimum_B(K):
        L = int(np.ceil(np.log2(K)))
        return L * 2**L

    @staticmethod
    def calc_schedule(K, B):
        """
        In : SequentialHalving.calc_schedule(10,800)                                  
        Out: (array([ 20,  40,  66, 100]), array([10,  5,  3,  2, 1]))
        """
        A = K
        L = int(np.ceil(np.log2(K)))
        assert (B >= SequentialHalving.calc_minimum_B(K)), "insufficient budget B"
        T_ary = np.zeros(L,dtype=int)
        A_ary = np.zeros(L+1,dtype=int)
        for ell in range(L):
            T_ary[ell] = int(np.floor(B/(L*A))) 
            A_ary[ell] = A
            A = int(np.ceil(A/2))
        assert A == 1, "implementation error"
        A_ary[-1] = 1
        return T_ary, A_ary

    def next_arm(self):
        my_n_pulls = self.n_pulls[self.surviving]
        my_idx = np.argmin(my_n_pulls)
        idx = self.surviving[my_idx]
        return idx

    def update(self, idx, reward):
        """ if self.ell == 1+self.L, we are just receiving extra arm pull... we could use those samples somehow, but here we just ignore them and those samples will not affect calc_best_arm()
        """
        self.n_pulls[idx] += 1
        self.sum_rewards[idx] += reward
        if (self.ell <= self.L): 
            #- at this point, self.t is equal to sum(self.n_pulls)
            #assert self.t == self.n_pulls.sum(), "implementation error" 

            if (self.t == self.halving_times[self.ell-1]):
                me = self.sum_rewards[self.surviving] / self.n_pulls[self.surviving]
                A = self.A_ary[self.ell] 

                self.cur_best_arm = self.surviving[np.argmax(me)] # FIXME: not the optimal thing to do.

                my_chosen = choose_topk_fair(me, A, randstream=self.my_rng)
                assert (len(my_chosen) == A)  # just in case..
                self.surviving = self.surviving[my_chosen] # index translation
                
                ipdb.set_trace()
                self.ell += 1
                if (self.ell <= self.L and self.reuse == False):
                    self.n_pulls[self.surviving] = 0
                    self.sum_rewards[self.surviving] = 0.0
        self.t += 1

    def get_best_arm(self):
        if (self.t <= self.K):
            return -1
        if np.any(self.n_pulls == 0):
            return self.cur_best_arm

        me = self.sum_rewards[self.surviving] / self.n_pulls[self.surviving]
        max_reward = me.max()
        return self.surviving[self.my_rng.choice(np.where(me == max_reward)[0])]

    def get_best_empirical_mean(self):
        if (self.t <= self.K):
            return np.nan
        me = self.sum_rewards[self.surviving] / self.n_pulls[self.surviving]
        return me.max()



class FCDoublingSequentialHalving:
    def __init__(self, K, reuse=True, factor=2.0, divisor=2.0, seed=19):
        """
        K      : the number of arms.
        factor : the budget of i-th SH will be `factor` times that of (i-1)-th SH. 
                 E.g., factor=2 means doubling.
        divisor: n_arms(stage l) = n_arms(stage l-1) / divisor. 
                 this also means that per_arm_sample_size(stage l) = divisor * per_arm_sample_size(stage l)
                 E.g., divisor=2 is the standard SH. 
        """
        self.K = K
        self.reuse = reuse
        assert self.reuse == True
        self.seed = seed
        self.factor = factor
        self.divisor = divisor

        self.cur_best_arm = -1
        self.n_pulls = np.zeros(K,dtype=int)
        #self.avg_rewards = np.zeros(K)
        self.sum_rewards = np.zeros(K)
        self.my_rng = np.random.RandomState(seed)
        self.t = 1

        #self.L = int(np.ceil(np.log2(K)))
        self.base_n_pulls = 1.0
        
        #- precompute the arm pull schedule
        self.T_ary, self.A_ary = \
            self.__class__.calc_schedule_2(K, self.divisor, self.base_n_pulls)

        self.i_doubling = 1
        # self.B_doubling = self.__class__.calc_minimum_budget(K, self.divisor)
        self.ell = 1   # iteration count
        self.surviving = np.arange(K)
        self.L = len(self.T_ary)

        pass

    # @staticmethod
    # def calc_minimum_B(K):
    #     L = int(np.ceil(np.log2(K)))
    #     return L * 2**L

    @staticmethod
    def calc_minimum_budget(K, divisor):
        T_ary = []
        A_ary = [] #np.zeros(L+1,dtype=int)
        while True:
            T_ary.append( 1.0/A ) #int(np.ceil(B/(L*A))) 
            A_ary.append( A )
            prev_A = A
            A = int(np.ceil(A/divisor))
            if (A == prev_A):
                A -= 1
            if (A == 1):
                A_ary.append(1)
                break
        T_ary = np.array(T_ary)
        A_ary = np.array(A_ary)

        L = len(T_ary)
        T_ary = np.array(T_ary)
        T_ary /= L
        
        T_ary /= T_ary[0] # scale it so that T_ary[0] = 1

        return T_ary @ A_ary[:-1]
#         n_surviving = K
#         L = int(np.ceil(np.log(K) / np.log(divisor)))
#         B = 0.0
#         pull_per_arm = 1
#         for l in range(L): # L, L-1, ..., 2
#             B += n_surviving * pull_per_arm
# 
#             prev_n_surviving = n_surviving
#             n_surviving = np.ceil(n_surviving / divisor)
#             if (n_surviving == prev_n_surviving):
#                 n_surviving -= 1
#             pull_per_arm *= n_surviving / prev_n_surviving # FIXME
#         return B
#                A = K
        #L = int(np.ceil(np.log(K) / np.log(divisor)))
        #assert (B >= FCDoublingSequentialHalving.calc_minimum_budget(K, divisor)), "insufficient budget B"

    @staticmethod
    def calc_schedule(K, B, divisor):
        """
        In : SequentialHalving.calc_schedule(10, 800, 2)                                  
        Out: (array([ 20,  40,  66, 100]), array([10,  5,  3,  2, 1]))
        """
        # B, L = FCDoublingSequentialHalving.calc_minimum_budget(K, divisor)
        A = K
        #L = int(np.ceil(np.log(K) / np.log(divisor)))
        #assert (B >= FCDoublingSequentialHalving.calc_minimum_budget(K, divisor)), "insufficient budget B"
        T_ary = []
        A_ary = [] #np.zeros(L+1,dtype=int)
        while True:
#        for ell in range(L):
            T_ary.append( B/A ) #int(np.ceil(B/(L*A))) 
            A_ary.append( A )
            prev_A = A
            A = int(np.ceil(A/divisor))
            if (A == prev_A):
                A -= 1
            if (A == 1):
                A_ary.append(1)
                break
        T_ary = np.array(T_ary)
        A_ary = np.array(A_ary)

        L = len(T_ary)
        T_ary = np.array(T_ary)
        T_ary /= L
        T_ary = np.ceil(T_ary)
            
        return T_ary, A_ary

    @staticmethod
    def calc_schedule_2(K, divisor, base_n_pulls):
        """
        In : FCDoublingSequentialHalving.calc_schedule_2(16,2,10)
        Out: (array([10., 20., 40., 80.]), array([16,  8,  4,  2,  1]))
        """
        # B, L = FCDoublingSequentialHalving.calc_minimum_budget(K, divisor)
        A = K
        #L = int(np.ceil(np.log(K) / np.log(divisor)))
        #assert (B >= FCDoublingSequentialHalving.calc_minimum_budget(K, divisor)), "insufficient budget B"
        T_ary = []
        A_ary = [] #np.zeros(L+1,dtype=int)
        while True:
#        for ell in range(L):
            T_ary.append( 1.0/A ) #int(np.ceil(B/(L*A))) 
            A_ary.append( A )
            prev_A = A
            A = int(np.ceil(A/divisor))
            if (A == prev_A):
                A -= 1
            if (A == 1):
                A_ary.append(1)
                break
        T_ary = np.array(T_ary)
        A_ary = np.array(A_ary)
        
        L = len(T_ary)
        T_ary = np.array(T_ary)
        T_ary /= L
        T_ary *= base_n_pulls/T_ary[0]  # make T_ary[0] == base_n_pulls

        T_ary = np.ceil(T_ary)
        T_ary[0] = np.ceil(base_n_pulls) # just in case ceil did not work correctly.
            
        #- print(T_ary, A_ary)
        return T_ary, A_ary

    def next_arm(self):
        my_n_pulls = self.n_pulls[self.surviving]
        my_idx = np.argmin(my_n_pulls)
        idx = self.surviving[my_idx]
        assert self.n_pulls[idx] <= self.T_ary[self.ell-1]
        return idx

    def update(self, idx, reward):
        """ if self.ell == 1+self.L, we are just receiving extra arm pull... we could use those samples somehow, but here we just ignore them and those samples will not affect calc_best_arm()
        """
        self.n_pulls[idx] += 1
        self.sum_rewards[idx] += reward

        #- perform elimination if needed (but I wouldn't think this loop will run more than one iteration)
        while (self.ell <= self.L and np.all(self.n_pulls[self.surviving] >= self.T_ary[self.ell-1])):
            me = self.sum_rewards[self.surviving] / self.n_pulls[self.surviving]
            n_half = self.A_ary[self.ell]

            self.cur_best_arm = self.surviving[np.argmax(me)] # FIXME: not the optimal thing to do.

            my_chosen = choose_topk_fair(me, n_half, randstream=self.my_rng)
            assert (len(my_chosen) == n_half)  # just in case..
            self.surviving = self.surviving[my_chosen] # index translation
            self.ell += 1

             #- if we reached the end, double the budget and reset!
            if (self.ell > self.L):
                self.i_doubling += 1
                #self.B_doubling *= self.divisor
                self.ell = 1
                self.surviving = np.arange(self.K)
                
                self.base_n_pulls *= self.factor
                self.T_ary, self.A_ary = FCDoublingSequentialHalving.calc_schedule_2(self.K, self.divisor, self.base_n_pulls)
            
        self.t += 1

    def get_best_arm(self):
        if np.any(self.n_pulls == 0):
            return self.cur_best_arm

        me = self.sum_rewards[self.surviving] / self.n_pulls[self.surviving]
        max_reward = me.max()
        return self.surviving[self.my_rng.choice(np.where(me == max_reward)[0])]

    def get_best_empirical_mean(self):
        if (self.t <= self.K):
            return np.nan
        me = self.sum_rewards[self.surviving] / self.n_pulls[self.surviving]
        return me.max()
    
    def get_empirical_means(self):
        return self.sum_rewards / self.n_pulls


class FCTsTci:
    def __init__(self, K, beta=0.5, sigma_sq=1.0, seed=19):
        """
        K: k in the bandit book; the size of the arm set
        """
        self.K = K
        self.seed = seed
        self.sigma_sq = sigma_sq
        self.beta = beta

        #self.cur_best_arm = -1
        self.n_pulls = np.zeros(K,dtype=int)
        self.sum_rewards = np.zeros(K)
        self.my_rng = np.random.RandomState(seed)
        self.t = 1

    def next_arm(self):
        if self.t <= self.K:
            return self.n_pulls.argmin()

        hatmu = self.sum_rewards / self.n_pulls
        if self.my_rng.rand() < self.beta:
            obj = hatmu + np.sqrt(self.sigma_sq / self.n_pulls) * self.my_rng.randn(self.K)
            obj = -obj
        else: 
            i_best = hatmu.argmax()
            obj = W_n_gaussian_all(i_best, hatmu, self.n_pulls, self.sigma_sq) + np.log(self.n_pulls)
            obj[i_best] = np.inf

        idx = np.argmin(obj)
        return idx

    def update(self, idx, reward):
        """ if self.ell == 1+self.L, we are just receiving extra arm pull... we could use those samples somehow, but here we just ignore them and those samples will not affect calc_best_arm()
        """
        self.n_pulls[idx] += 1
        self.sum_rewards[idx] += reward
        self.t += 1

    def get_best_arm(self):
        if np.any(self.n_pulls == 0):
            return -1

        hatmu = self.sum_rewards / self.n_pulls
        max_hatmu = hatmu.max()
        return np.where(hatmu == max_hatmu)[0]

    # def get_best_empirical_mean(self):
    #     if (self.t <= self.K):
    #         return np.nan
    #     me = self.sum_rewards[self.surviving] / self.n_pulls[self.surviving]
    #     return me.max()
    
    def get_empirical_means(self):
        return self.sum_rewards / self.n_pulls

################################################################################
# generic functions
################################################################################

def choose_topk_fair(ary, k, randstream=ra):
    """
    choose top k large members from ary, but break ties uniformly at random
    """
    sidx = np.argsort(ary)[::-1]
    sary = ary[sidx]
    
    threshold = (sary)[k-1]
    ties = np.where(sary == threshold)[0]
    n_ties = len(ties)
    nonties = np.where(sary > threshold)[0]
    n_nonties = len(nonties)
    if n_ties + n_nonties == k:
        chosen = sidx[:k]
    else:
        broken_ties = randstream.choice(sidx[n_nonties:n_nonties+n_ties], k-n_nonties, replace=False)
        chosen = np.concatenate( (sidx[nonties], broken_ties))
    return chosen

def argmax_tiebreak(ary, randstream=ra):
    maxidx = ary.argmax()
    eq = ary[maxidx] == ary
    if (eq.sum() != 1):
        idx = np.where(eq)[0]
        chosen = randstream.randint(len(idx))
        maxidx = idx[chosen]
    return maxidx

def kl_gaussian(mu1, mu2, sigma_sq = 1.0):
    return (mu1 - mu2)**2/(2*sigma_sq)

def W_n_gaussian(hmu1, hmu2, N1, N2, sigma_sq):
    """
    track and stop paper has a short cut for computing these.
    """
    p = N1 / (N1 + N2)
    x = p * hmu1 + (1-p) * hmu2
    return N1 * kl_gaussian(hmu1, x, sigma_sq = sigma_sq) + N2 * kl_gaussian(hmu2, x, sigma_sq=sigma_sq)

def c_n_delta(n, delta, K):
    """
    k: number of arms
    """
    return np.log(1 / delta) + 2 * np.log(1 + n / 2) + 2 + np.log(K - 1)

def W_n_gaussian_all(i_anchor, hatmu, n_pulls, sigma_sq):
    K = len(hatmu)
    ary = np.zeros(K)
    for j in range(K):
        ary[j] = W_n_gaussian(hatmu[i_anchor], hatmu[j], n_pulls[i_anchor], n_pulls[j], sigma_sq)

    return ary
            
def calc_min_W_n(hatmus, n_pulls, delta, sigma_sq = 1.0):
    K = len(hatmus)
    best_idx = np.argmax(hatmus)
    best_hatmu = hatmus[best_idx]
    best_n_pulls = n_pulls[best_idx]

    obj = [W_n_gaussian(best_hatmu, hatmus[i], best_n_pulls, n_pulls[i], sigma_sq = sigma_sq) for i in range(K) if i != best_idx]
    min_W_n = np.min(obj)
    return min_W_n
    
def run_bandit_pe(algo, env, delta, max_iter, sigma_sq = 1.0):
    table = KjTable() 
    b_stopped = False
    for t in range(max_iter):
        i_t = algo.next_arm()
        y_t = env.get_reward(i_t)
        logging.info(f"\n->t = {t}")
        # logging.info(f"i_doubling = {algo.i_doubling}")
        logging.info(f"i_t = {i_t}")
        logging.info(f"y_t = {y_t:0.4f}")
        algo.update(i_t, y_t)
        logging.info(f"sum_rewards = {algo.sum_rewards}")
        logging.info(f"n_pulls = {algo.n_pulls}")
        
        if (t < env.K):
            continue

        hatmus = algo.get_empirical_means()
        min_W_n = calc_min_W_n(hatmus, algo.n_pulls, delta, sigma_sq)
        # table.update('i_t', t, i_t)
        # table.update('min_W_n', t, min_W_n)

        if (min_W_n > c_n_delta(t, delta = delta, K = env.K)):
            b_stopped = True
            break
        
        # hatmus = algo.sum_rewards / algo.n_pulls
        # logterms = np.log(1.25*t**4/delta)
        # bonuses = np.sqrt(logterms/algo.n_pulls/2)
        # # logterms = np.log(6*algo.K*np.log2(algo.K)*algo.i_doubling**2/delta)
        # # bonuses = np.sqrt(2*logterms/algo.n_pulls)

        # sidx = np.argsort(hatmus)[::-1]
        # i_top = sidx[0]
        # i_bot = sidx[1:]

        # h_t = i_top
        # bar_LCB = hatmus[i_top] - bonuses[i_top]

        # #- highest UCB from the bottom
        # v = hatmus[i_bot] + bonuses[i_bot]
        # maxv = v.max()
        # idx = ra.choice(np.where(v == maxv)[0])
        # ell_t = i_bot[idx]
        # bar_UCB = maxv

        # if (bar_LCB > bar_UCB):
        #     b_stopped = True
        #     break
        

    # if (b_stopped == False):
    #     table.update('did_not_stop', 0, True)
    # table.update('i_best', 0, algo.get_best_arm())
    # table.update('tau', 0, t+1)
    # table.update('n_pulls', 0, algo.n_pulls.tolist())
    
    return t+1, b_stopped


def run_bandit_lucb(algo, env, delta, max_iter, sigma_sq = 1.0):
    table = KjTable() 
    b_stopped = False
    for t in range(max_iter):
        i_t = algo.next_arm()
        y_t = env.get_reward(i_t)
        logging.info(f"i_t = {i_t}")
        logging.info(f"y_t = {y_t:0.4f}")
        algo.update(i_t, y_t)
        logging.info(f"sum_rewards = {algo.sum_rewards}")
        logging.info(f"n_pulls = {algo.n_pulls}")
        
        if (t < env.K):
            continue

        # if algo.success_yes:
        #     b_stopped = True
        #     break
        
        hatmus = algo.get_empirical_means()
        min_W_n = calc_min_W_n(hatmus, algo.n_pulls, delta, sigma_sq)
        logging.debug(f"min_W_n = {min_W_n}")
        # table.update('i_t', t, i_t)
        # table.update('min_W_n', t, min_W_n)

        if (min_W_n > c_n_delta(t, delta = delta/(t**2), K = env.K)):
            b_stopped = True
            break

    # if (b_stopped == False):
    #     table.update('did_not_stop', 0, True)
        
    # table.update('i_best', 0, algo.get_best_arm())
    # table.update('tau', 0, t+1)
    # table.update('n_pulls', 0, algo.n_pulls.tolist())
    
    return t+1, b_stopped

def algo_factory(algo_name, env, delta, T):
    if (algo_name == 'uniform'):
        return Uniform(env.K)
    elif (algo_name == 'chernoff'):
        return Chernoff(env.K, env.div)
    elif (algo_name == 'chernoffjun'):
        return ChernoffJun(env.K, env.div)
    elif (algo_name == 'chernoffjun2'):
        return ChernoffJun_v2(env.K, env.div)
    elif (algo_name == 'kaufmannnaive'):
        return Kaufmann(env.K, env.div, forced=False)
    elif (algo_name == 'kaufmann'):
        return Kaufmann(env.K, env.div, forced=True)
    elif (algo_name == 'lucb'):
        return Lucb(env.K, env.sig_sq, delta)
    elif (algo_name == 'ucb'):
        return Ucb(env.K, sqrt(env.sig_sq))
    elif (algo_name == 'sh'):
        return SequentialHalving(env.K, T, reuse=False)
    elif (algo_name == 'sh-reuse'):
        return SequentialHalving(env.K, T, reuse=True)
#    def __init__(self, K, B, seed=19):
    
    else:
        raise ValueError()
    pass


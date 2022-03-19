import numpy as np
import scipy.stats
import matplotlib.pyplot as plt

def load_csv(folder, truncate):
    # load csv into a np array
    data = np.genfromtxt(folder, delimiter=',', dtype=np.int)
    
    # remove first column
    if truncate:
        data = data[:, 1:]

    print("this is data", data)

    return data

# check distribution
def check_distribution(data):
    data_length = len(data)

    # get mean of dataset (values of image, light/dark)
    mean_value = np.mean(data, axis=1)

    # measure kernel density estimation
    kde = scipy.stats.gaussian_kde(mean_value)

    # create normal distributed probablity
    normal_distribution = scipy.stats.norm(loc=np.mean(mean_value), scale=np.std(mean_value))

    # plot
    x = np.linspace(min(mean_value), max(mean_value), num=data_length)
    plt.plot(x, kde(x), label='Data')

    # probability density function
    plt.plot(x, normal_distribution.pdf(x), label='Normal distribution')
    plt.legend()

    plt.show()

    plt.savefig("normal_distribution_good")

# check cumulative_distribution
def check_cumulative_distribution(data):
    plt.figure()

    data_length = len(data)
    mean_value = np.mean(data, axis=1)

    # plot estimated CDF
    plt.plot(sorted(mean_value), np.linspace(0, 1, len(mean_value)), label="Data")

    # create normal distributed probablity
    normal_distribution = scipy.stats.norm(loc=np.mean(mean_value), scale=np.std(mean_value))

    x = np.linspace(min(mean_value), max(mean_value), num=data_length)
    plt.plot(x, normal_distribution.cdf(x), label="Normal distribution")
    
    # single pixel
    # data[0]

    plt.legend()
    plt.show()
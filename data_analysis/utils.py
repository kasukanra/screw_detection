import numpy as np
import scipy.stats
import matplotlib.pyplot as plt
from sklearn.mixture import GaussianMixture

def load_csv(folder, truncate):
    # load csv into a np array
    data = np.genfromtxt(folder, delimiter=',', dtype=np.int)
    
    # remove first column
    if truncate:
        data = data[:, 1:]

    print("this is data", data)

    return data

# check normal distribution
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
    
    # single pixel comparison, didn't use
    # data[0]

    plt.legend()
    plt.show()

def fit_mixture(data, n_components = 1):
    data = np.array(data).reshape(-1,1)
    model = GaussianMixture(n_components=n_components)
    model.fit(data)

    return model

# check gaussian_mixture
def check_gaussian_mixture(data):
    plt.figure(figsize=(16,5))
    plt.subplot(121)

    data_length = len(data)
    mean_value = np.mean(data, axis=1)
    
    # plot estimated CDF against mixture
    plt.plot(sorted(mean_value), np.linspace(0, 1, len(mean_value)), label="Estimated CDF")


    # estimate model CDF from 1000 datapoints, and truncate
    model = fit_mixture(mean_value, n_components = 2)

    n_samples = 10000
    samples, _ = model.sample(n_samples)
    samples = sorted(samples.reshape(-1))
    samples = [s for s in samples if (0 <= s <= 255)]

    # plot the CDF of gaussian mixture
    x = np.linspace(0, 1, len(samples))
    plt.plot(samples, x, label="Gaussian mixture CDF")

    plt.title("CDF of good compared to Gaussian mixture PDF")
    plt.legend()


    # plot normal distribution against mixture
    plt.subplot(122)

     # Estimate the distribution of the data and mixture
    kde_data = scipy.stats.gaussian_kde(mean_value, 0.1)
    kde_mixture = scipy.stats.gaussian_kde(samples)

    # Plot the two distributions
    x = np.linspace(min(mean_value), max(mean_value), data_length)
    plt.plot(x,kde_data(x),label='PDF Data')
    plt.plot(x,kde_mixture(x),label='Gaussian mixture PDF')

    plt.title(f"PDF of good compared to Gaussian mixture PDF")
    plt.legend()

    plt.show()
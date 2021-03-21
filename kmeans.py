import math
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from random import sample

# set to 3000 to ensure all rows of data can be displayed
pd.options.display.max_rows = 3000


# This function reads a csv file using pandas, prints out data and returns 2 numpy arrays and a list of countries
def csv_reader(filename):
    dataset = pd.read_csv(filename)
    print(dataset)
    countries = dataset[dataset.columns[0]].values
    values_array = dataset[[dataset.columns[1], dataset.columns[2]]].values
    return values_array, countries


# This function calculates distance between each data point and each centroid.
# It adds all the values to a distances_list and returns it
def calc_distance(centroids, data_array):
    distances_list = []
    for centroid in centroids:
        for item in data_array:
            distances_list.append(math.sqrt((centroid[0] - item[0]) ** 2 + (centroid[1] - item[1]) ** 2))
    return distances_list


# Assigning data values such as birthrates, life expectancies and list of countries
# from a csv file to a varible to be used later
data_values_list = csv_reader('dataBoth.csv')

# Prompting the user to enter the number of clusters and number of iterations
print()
k = int(input('Enter the number of clusters you want: '))
iterations = int(input('Enter the number of iterations that the algorithm must run: '))

# converting the ndarray to a list for sampling
data_array = np.ndarray.tolist(data_values_list[0])

# Choosing the number of random centroids from data based of user input of clusters
centroids = sample(data_array, k)
print('The centroids: ', centroids)

# Scatter plot of the data before clustering
plt.scatter(data_values_list[0][0:, 0], data_values_list[0][0:, 1], s=20)
plt.xlabel('Birthrate per 1000 population')
plt.ylabel('Life Expectancy')
plt.title('Initial plot with random centroids without clustering')
centroids_list = np.reshape(centroids, (k, 2))
plt.plot(centroids_list[0:, 0], centroids_list[0:, 1], c='r', marker="o", markersize=7, linestyle=None, linewidth=0)
plt.show()


# **********************************************************************************************************************


# This function assigns each data point to the closest cluster, those data points comes from a
# calc_distance function.
def assignments_of_mean_centroids_to_cluster(x_in=data_values_list, centroids_in=centroids, n_user=k):

    distances_list2 = np.reshape(calc_distance(centroids_in, x_in[0]), (len(centroids_in), len(x_in[0])))
    datapoint_centroids = []
    distances_min = []

    # zip() function maps similar index of multiple containers
    for value in zip(*distances_list2):
        distances_min.append(min(value))
        datapoint_centroids.append(np.argmin(value)+1)
    # Create clusters dictionary and add number of clusters according to user input
    clusters = {}

    for no_user in range(0, n_user):
        clusters[no_user+1] = []

    # Allocate each data point to it's closest cluster
    for d_point, cent in zip(x_in[0], datapoint_centroids):
        clusters[cent].append(d_point)

    # This rewrites the centroids with the new means
    for i, cluster in enumerate(clusters):
        reshaped = np.reshape(clusters[cluster], (len(clusters[cluster]), 2))
        centroids[i][0] = sum(reshaped[0:, 0])/len(reshaped[0:, 0])
        centroids[i][1] = sum(reshaped[0:, 1])/len(reshaped[0:, 1])

    print()    
    print('Centroids for this iteration are:' + str(centroids))

    # returns the list with cluster allocations in order of the countries
    # and the clusters
    return datapoint_centroids, clusters


# ************************************************ MAIN LOOP ***********************************************************


iteration = 1
while iteration <= iterations:
    iteration += 1

    print('\n')
    print('Iteration: ', (iteration-1))
    # Assigning a function to a value
    assigning = assignments_of_mean_centroids_to_cluster()
    # Create the dataframe for vizualisation
    cluster_data = pd.DataFrame({'Birth Rate': data_values_list[0][0:, 0],
                                 'Life Expectancy': data_values_list[0][0:, 1],
                                 'label': assigning[0],
                                 'Country': data_values_list[1]})

    # Create the dataframe and grouping, then print out inferences
    group_by_cluster = cluster_data[['Country', 'Birth Rate', 'Life Expectancy', 'label']].groupby('label')
    clustersCount = group_by_cluster.count()

    print()
    print('List of countries in one cluster: \n', clustersCount)
    print('\nList of countries in each cluster: \n', list(group_by_cluster))
    print('List of averages: \n', str(cluster_data.groupby(['label']).mean()))

    # Set the variable mean that holds the clusters dict
    mean = assigning[1]

    # dictionary to hold the distances between data points and clusters
    means = {}
    # The loop here will create the amount of clusters based on user input.
    for num_clusters in range(0, k):
        means[num_clusters + 1] = []

    # Calculating the squared distances between each data point and its cluster mean
    for index, data in enumerate(mean):
        array_of_data = np.array(mean[data])
        array_of_data = np.reshape(array_of_data, (len(array_of_data), 2))

        # calculation of the cluster mean of each variable
        birth_rate = sum(array_of_data[0:, 0]) / len(array_of_data[0:, 0])
        life_expectancy = sum(array_of_data[0:, 1]) / len(array_of_data[0:, 1])

        # This for loop appends the squared distance of between each data point to the means dictionary
        # in it's cluster and the cluster mean.
        for data_point in array_of_data:
            distance = math.sqrt((birth_rate-data_point[0]) ** 2 + (life_expectancy - data_point[1]) ** 2)
            means[index+1].append(distance)

    # list to hold all the sums of the means in clusters.
    total_distance = []
    for index2, summed in enumerate(means):
        total_distance.append(sum(means[index2 + 1]))

    # printing the summed distance
    print('Summed distance of all clusters: ', sum(total_distance))

    # centroids2 is reshaped for plotting
    centroids2 = np.reshape(centroids, (k, 2))

    # plotting the data with clustering
    plt.xlabel('Birthrate per 1000 population')
    plt.ylabel('Life Expectancy')

    # group of colors to be used for clusters
    color = ['blue', 'green', 'cyan', 'red', 'yellow', 'magenta']
    plt.scatter(data_values_list[0][0:, 0], data_values_list[0][0:, 1], s=20, c=assigning[0])
    plt.plot(centroids2[0:, 0], centroids2[0:, 1], c='r', marker="o", markersize=7, linestyle=None, linewidth=0)
    plt.title('Iteration: ' + str(iteration-1) +
              '\nTotaled Distances of all clusters: ' + str(round(sum(total_distance), 2)))
    plt.show()

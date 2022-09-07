import math
import numpy as np
import operator

# Euclidean Distance
def euclideanDistance(instance1, instance2, length):
    distance = 0
    for x in range(length):
        distance += pow((instance1[x] - instance2[x]), 2)
    return math.sqrt(distance)

# Manhattan Distance v2
def manhattanDistance(instance1, instance2, length):
    distance = 0
    for x in range(length):
        distance += float(abs(instance1[x] - instance2[x]))
    return distance

# KNN
def getKNeighbors(trainingSet, testInstance, k, Distance):
    if Distance == "Euclidean":
        print("\nHasil Euclidean Disini")
        distances = []
        length = len(testInstance)-1
        for x in range(len(trainingSet)):
            dist = euclideanDistance(testInstance, trainingSet[x], length)
            distances.append((trainingSet[x], dist))
        distancesNotSorted = np.array(distances, dtype=object)
        distances.sort(key=operator.itemgetter(1))
        neighbors = []
        for x in range(k):
            neighbors.append(distances[x][0])
        distances = np.array(distances, dtype=object) # Mengubah ke numpy array agar mudah diakses
        print("Ukuran Array Jarak Euclidean : ", distances.shape)
        print("Hasil Perhitungan Jarak Euclidean : ")
        for data in distances:
            print(data[1]) # Hasil Euclidean Distance [1]
        return distancesNotSorted, distances, neighbors

    # Manhattan v2
    else:
        print("\nHasil Manhattan Disini")
        distances = []
        length = len(testInstance) - 1
        for x in range(len(trainingSet)):
            dist = manhattanDistance(testInstance, trainingSet[x], length)
            distances.append((trainingSet[x], dist))
        distancesNotSorted = np.array(distances, dtype=object)
        distances.sort(key=operator.itemgetter(1))
        neighbors = []
        for x in range(k):
            neighbors.append(distances[x][0])
        distances = np.array(distances, dtype=object)  # Mengubah ke numpy array agar mudah diakses
        print("Ukuran Array Jarak Manhattan : ", distances.shape)
        print("Hasil Perhitungan Jarak Manhattan : ")
        for data in distances:
            print(data[1])  # Hasil Euclidean Distance [1]
        return distancesNotSorted, distances, neighbors

# Classification Result
def getResponse(neighbors):
    classVotes = {}
    for x in range(len(neighbors)):
        response = neighbors[x][-1]
        if response in classVotes:
            classVotes[response] += 1
        else:
            classVotes[response] = 1
    sortedVotes = sorted(classVotes.items(), key=operator.itemgetter(1), reverse=True)
    return sortedVotes[0][0]
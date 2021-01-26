import numpy as np
import random
from scipy.spatial import distance as dis

"""SMOTE Class"""


class Smote:
    def __init__(self, k, distance_calculator, new_instances):
        self.k = k
        self.distance_calculator = distance_calculator
        self.new_instances = new_instances

    def smote(self, sample_class):
        # for tracking number of generated synthetic samples
        newindex = 0
        synthetic = []
        n = self.new_instances
        while n != 0:
            i = random.randint(1, len(sample_class)) - 1
            self.populate(i, self.get_nearest_neighbor(sample_class[i], self.k, sample_class), self.k, synthetic,
                          newindex,
                          sample_class)
            n = n - 1
        synthetic = np.asarray(synthetic)
        synthetic = synthetic.astype('float64')
        return synthetic

    def populate(self, i, nn_array, k, synthetic, new_index, sample_class):
        nn = random.randint(1, k) - 1
        temp = []
        for feature_position in range(0, len(sample_class[0])):
            dif = sample_class[nn_array[nn]][feature_position] - \
                sample_class[i][feature_position]
            gap = random.random()
            temp.insert(feature_position,
                        sample_class[i][feature_position] + gap * dif)

        synthetic.insert(new_index, temp)
        new_index += 1
        return

    def get_nearest_neighbor(self, x_test, k, sample_class):
        distances = []
        targets_index = []
        for i in range(len(sample_class)):
            if (sample_class[i][:] != x_test).any():
                if self.distance_calculator == 'jaccard':
                    distance = dis.jaccard(x_test, sample_class[i][:])
                elif self.distance_calculator == 'dice':
                    distance = dis.dice(x_test, sample_class[i][:])
                elif self.distance_calculator == 'correlation':
                    distance = dis.correlation(x_test, sample_class[i][:])
                elif self.distance_calculator == 'yule':
                    distance = dis.yule(x_test, sample_class[i][:])
                elif self.distance_calculator == 'russelo-rao':
                    distance = dis.russellrao(x_test, sample_class[i][:])
                elif self.distance_calculator == 'sokal-michener':
                    distance = dis.sokalmichener(x_test, sample_class[i][:])
                elif self.distance_calculator == 'rogers-tanimoto':
                    distance = dis.rogerstanimoto(x_test, sample_class[i][:])
                elif self.distance_calculator == 'kulzinsky':
                    distance = dis.kulsinski(x_test, sample_class[i][:])
                distances.append([distance, i])

        # make a list of the k neighbors' targets
        distances.sort()
        for i in range(k):
            targets_index.append(distances[i][1])
        return targets_index



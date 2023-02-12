import numpy as np

def Freq1(b):
    d = {}
    for x in b:
        if x in d:
            d[x] += 1
        else:
            d[x] = 1 # Если ключ уже есть, прибавляем 1, если нет, записываем 1
    v = max(d, key=d.get)
    return v 


class KNNClassifier:
    """
    K-neariest-neighbor classifier using L1 loss
    """
    
    def __init__(self, k=1):
        self.k = k
    

    def fit(self, X, y):
        self.train_X = X
        self.train_y = y


    def predict(self, X, n_loops=0):
        """
        Uses the KNN model to predict clases for the data samples provided
        
        Arguments:
        X, np array (num_samples, num_features) - samples to run
           through the model
        num_loops, int - which implementation to use

        Returns:
        predictions, np array of ints (num_samples) - predicted class
           for each sample
        """
        
        if n_loops == 0:
            distances = self.compute_distances_no_loops(X)
        elif n_loops == 1:
            distances = self.compute_distances_one_loop(X)
        else:
            distances = self.compute_distances_two_loops(X)
        
        if len(np.unique(self.train_y)) == 2:
            return self.predict_labels_binary(distances)
        else:
            return self.predict_labels_multiclass(distances)


    def compute_distances_two_loops(self, X):
        """
        Computes L1 distance from every sample of X to every training sample
        Uses simplest implementation with 2 Python loops

        Arguments:
        X, np array (num_test_samples, num_features) - samples to run
        
        Returns:
        distances, np array (num_test_samples, num_train_samples) - array
           with distances between each test and each train sample
        """
        D = np.empty((len(X), len(self.train_X)))
        for j in range(len(self.train_X)):
            for i in range(len(X)):
                D[i, j] = np.sum(np.abs(X[i] - self.train_X[j]))
        
        return D


    def compute_distances_one_loop(self, X):
        """
        Computes L1 distance from every sample of X to every training sample
        Vectorizes some of the calculations, so only 1 loop is used

        Arguments:
        X, np array (num_test_samples, num_features) - samples to run
        
        Returns:
        distances, np array (num_test_samples, num_train_samples) - array
           with distances between each test and each train sample
        """
        
        D = np.sum(np.abs(X - self.train_X[0]), axis = 1)
        for i in range(1, len(self.train_X)):
            D = np.vstack((D, np.sum(np.abs(X - self.train_X[i]), axis = 1)))
        
        return D.T


    def compute_distances_no_loops(self, X):
        """
        Computes L1 distance from every sample of X to every training sample
        Fully vectorizes the calculations using numpy

        Arguments:
        X, np array (num_test_samples, num_features) - samples to run
        
        Returns:
        distances, np array (num_test_samples, num_train_samples) - array
           with distances between each test and each train sample
        """

        A1 = np.repeat(X, len(self.train_X), axis = 0)
        B1 = np.repeat(self.train_X.T, len(X), axis = 0).reshape((len(X[0]), len(X)*len(self.train_X))).T
        D1 = np.sum(np.abs(A1 - B1), axis = 1).reshape((len(X), len(self.train_X)))
        
        return D1


    def predict_labels_binary(self, distances):
        """
        Returns model predictions for binary classification case
        
        Arguments:
        distances, np array (num_test_samples, num_train_samples) - array
           with distances between each test and each train sample
        Returns:
        pred, np array of bool (num_test_samples) - binary predictions 
           for every test sample
        """

        n_train = distances.shape[1]
        n_test = distances.shape[0]
        prediction = np.zeros(n_test)

        for i in range(n_test):
            l = np.empty(self.k)
            d = distances[i].copy()
            #print(d)
            for j in range(self.k):
                #print(d)
                m = np.argmin(d)
                #print(m)
                l[j] = self.train_y[m]
                #print(l)
                d = np.delete(d, m)
            if self.k > 1:
                #print(l)
                prediction[i] = float(Freq1(l))
            else:
                prediction[i] = l[0]
        return prediction


    def predict_labels_multiclass(self, distances):
        """
        Returns model predictions for multi-class classification case
        
        Arguments:
        distances, np array (num_test_samples, num_train_samples) - array
           with distances between each test and each train sample
        Returns:
        pred, np array of int (num_test_samples) - predicted class index 
           for every test sample
        """

        n_train = distances.shape[0]
        n_test = distances.shape[0]
        prediction = np.zeros(n_test, np.int)

        for i in range(n_test):
            l = np.empty(self.k)
            d = distances[i].copy()
            #print(d)
            for j in range(self.k):
                #print(d)
                m = np.argmin(d)
                #print(m)
                l[j] = self.train_y[m]
                #print(l)
                d = np.delete(d, m)
            if self.k > 1:
                #print(l)
                prediction[i] = float(Freq1(l))
            else:
                prediction[i] = l[0]
        return prediction
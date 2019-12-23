import numpy as np
import os
from scipy.cluster.vq import vq, kmeans, whiten
import matplotlib.pyplot as plt

def knearestneighbor(test_data, train_data, train_label, k):
    
    length, num_features = np.shape(train_data)
    est_class = np.zeros((len(test_data),1))
    for i in range(len(test_data)):
        test_now = test_data[i]
        length = len(train_data)
        distance = np.zeros((length,1))
        for j in range(length):
            train = train_data[j]
            if (num_features == 1):
                distance[j] = np.sqrt(abs(train-test_now))
            else:
                distance[j] = np.sqrt(sum((train[k]-test_now[k])**2 for k in range(num_features)))
        sorted_index = np.argsort(distance,axis=0) 
        index = sorted_index[:k]
        add=1
        while distance[sorted_index[k]] == distance[sorted_index[k+add]]:
            add+=1
        if add==1:
           classcount = {}
           for a in range(k):
               label = train_label[index[a][0]]
               num_class = int(label)
               if num_class in classcount.keys():
                  classcount[num_class] +=1 
               else:
                   classcount[num_class] =1
        else:
            classcount = {}
            k1 = np.random.choice(np.arange(add-1))
            for a in range(k-1):
               label = train_label[index[a][0]]
               num_class = int(label)
               if num_class in classcount.keys():
                   classcount[num_class]+=1
               else:
                   classcount[num_class] =1
            label_choose = int(train_label[sorted_index[k+k1][0]])
            if label_choose in classcount.keys():
                classcount[label_choose]+=1
            else:
                classcount[label_choose] =1
        est_class[i] = max(zip(classcount.values(), classcount.keys()))[1]
        
    return est_class

def cross_validate(data, gt_labels, k, num_folds):
    
    length, num_features = np.shape(data)
    fold_accuracies = np.zeros((num_folds,1))
    avg_accuracy = 0
    index = np.arange(length)
    np.random.shuffle(index)
    conf_matrix = np.zeros((5,5))
    length_test = length//num_folds
    for k in range(num_folds):
        if k !=num_folds-1:
            test_index = index[k*length_test:(k+1)*length_test]
            train_index = np.hstack((index[0:k*length_test],index[(k+1)*length_test:]))
        else:
            test_index = index[k*length_test:]
            train_index = index[0:k*length_test]
        train_data = np.array([data[index] for index in train_index])
        test_data = np.array([data[index] for index in test_index])
        train_label = np.array([gt_labels[i] for i in train_index])
        est_class = knearestneighbor(test_data, train_data, train_label, 3)
        test_label = np.array([gt_labels[i] for i in test_index])
        num = 0
        for i in range(len(est_class)):
            if int(est_class[i]) == int(test_label[i]):
                num+=1
        fold_accuracies[k] = num/len(est_class)
        avg_accuracy +=fold_accuracies[k]
    avg_accuracy = avg_accuracy/num_folds

    for j in range(len(est_class)):
        conf_matrix[int(est_class[j])-1,int(test_label[j])-1] +=1

    return avg_accuracy, fold_accuracies, conf_matrix


def find_best_features(data, labels, k, num_folds):
    
    length, num_features = np.shape(data)
    avg_accuracy = np.array([])

    
    for i in range (num_features):
        
        accuracy, fold_accuracies, config_matrix =  cross_validate(np.reshape(data[:,i], (length, 1)), labels, k, num_folds)
        avg_accuracy = np.append(avg_accuracy, accuracy)

    sel_features = np.argsort(-avg_accuracy)
    sel_features = sel_features.astype(int)
    
    feature_index = np.amax(sel_features)
    
    return feature_index, sel_features   #We are returning sel_features here because call by object didn't work!


def select_features(data, labels, k, num_folds): 

    length, num_features = np.shape(data)
    data1 = np.array([])
    sel_features = np.array([])
    sel_feature_ind = np.array([])
    accuracy = np.array([])
    max_acc_so_far = 0.0
    feature_index, sel_features = find_best_features(data, labels, k, num_folds)
    
    data1 = np.reshape(data[:,feature_index], (length, 1))

    for i in sel_features:
    
        avg_accuracy, fold_accuracies, config_matrix = cross_validate(data1, labels, k, num_folds)
        data1 = np.hstack((data1, np.reshape(data[:,i], (length, 1))))

        if (avg_accuracy > max_acc_so_far):
            max_acc_so_far = avg_accuracy
            sel_feature_ind = np.append(sel_feature_ind, i)
            accuracy = np.append(accuracy, avg_accuracy)
    
    sel_feature_ind = sel_feature_ind.astype(int)


    plt.plot(accuracy)
    plt.xlabel('Number of Features')
    plt.ylabel('Accuracy')
    plt.show()

    return sel_feature_ind, accuracy


def evaluate(data,labels):
    features = np.loadtxt(os.path.join(os.path.dirname(__file__), 'data/feature_dict.txt'), delimiter=',', dtype=str)
    sel_feature_ind, accuracy = select_features(data, gt_labels, 3, 3)
    accuracies = np.array([])
    conf_matrices = np.zeros((5,5))
    
    print('Selected features for k = 3, numFolds = 3: ',features[sel_feature_ind])
    selected_data = np.take(data, sel_feature_ind, 1)
    
    avg_accuracy, fold_accuracy, conf_matrix1  = cross_validate(selected_data, labels, 1, 10)
    accuracies = np.append(accuracies, avg_accuracy)
    
    avg_accuracy, fold_accuracy, conf_matrix2  = cross_validate(selected_data, labels, 3, 10)
    accuracies = np.append(accuracies, avg_accuracy)

    avg_accuracy, fold_accuracy, conf_matrix3  = cross_validate(selected_data, labels, 7, 10)
    accuracies = np.append(accuracies, avg_accuracy)
    
    conf_matrices = np.array([conf_matrix1, conf_matrix2, conf_matrix3])
    
    return accuracies, conf_matrices




def kmeans_clustering(data, k):
    label_names = np.loadtxt(os.path.join(os.path.dirname(__file__), 'data/label_dict.txt'), delimiter=',', dtype=str)
    labels = np.loadtxt(os.path.join(os.path.dirname(__file__), 'data/labels.txt'), delimiter=' ', dtype=str)
    # running select features with k = 5 and folds = 10
    [sel_feature_ind, accuracy] = select_features(data, gt_labels, 5, 10)
    # sel_feature_ind = [0, 1,2, 4,5,6,8]
    selected_data = np.take(data, sel_feature_ind, 1)

    # normalization with unit variance for kmeans clustering
    whitened_data = whiten(selected_data)
    
    # initialize 5 random centroids
    np.random.seed(4)
    centroids = whitened_data[np.random.randint(whitened_data.shape[0], size=5), :]
    total_distance = 0
    c_labels = np.zeros(whitened_data.shape[0])
    iterations = 0
   
    while (True):
        iterations += 1 
        total_distance = 0
        dc_ccentroid = c_labels.copy()
        # calculate distances and closest centroid from each point in the whitened_data
        distances = np.sqrt(((whitened_data - centroids[:,np.newaxis,:])**2).sum(axis=2))
        c_labels = np.argmin(distances, axis=0)
        # adjust centroids 
        centroids = np.array([whitened_data[c_labels==k].mean(axis=0) for k in range(centroids.shape[0])])
        if np.array_equal(c_labels, dc_ccentroid):
            
            print('converged after iterations :',iterations)
            #for i in range(0,len(c_labels)):
            #    print(c_labels[i], labels[i])
    
            print(np.unique(c_labels, return_counts=True))
            print(np.unique(labels, return_counts=True))
            break
            
    return c_labels, centroids


if __name__ == "__main__": 

    pathdata = os.path.join(os.path.dirname(__file__), 'data/data.txt')
    pathlabel = os.path.join(os.path.dirname(__file__), 'data/labels.txt')
    gt_labels = np.loadtxt(pathlabel,dtype=str)
    data = np.loadtxt(pathdata, dtype=float)
    data = np.transpose(data)
    c_labels, centroids = kmeans_clustering(data, gt_labels)
    print('Cluster Labels : ', c_labels)
    print('Centroids : ')
    print(centroids)
    
    accuracies, conf_matrices = evaluate(data, gt_labels)
    
    print('Average accuracy & confusion matrix for k = 1, folds = 10 after feature selection :')
    print('Avg accuracy : ', accuracies[0])
    print('Confusion matrix : ')
    print(conf_matrices[0])


    print('Average accuracy & confusion matrix for k = 3, folds = 10 after feature selection :')
    print('Avg accuracy : ', accuracies[1])
    print('Confusion matrix : ')
    print(conf_matrices[1])

    print('Average accuracy & confusion matrix for k = 7, folds = 10 after feature selection :')
    print('Avg accuracy : ', accuracies[2])
    print('Confusion matrix : ')
    print(conf_matrices[2])



import numpy as np
import os
from scipy.cluster.vq import vq, kmeans, whiten

def knearestneighbor(test_data, train_data, train_label, k):
    est_class = np.zeros((len(test_data),))
    for i in range(len(test_data)):
        test_now = test_data[i]
        length = len(train_data)
        distance = np.zeros((length,1))
        for j in range(length):
            train = train_data[j]
            distance[j] = np.sqrt(sum((train[k]-test_now[k])**2 for k in range(10)))
        sorted_index = np.argsort(distance,axis=0)        
        index = sorted_index[:k]
        classcount = {}
        for a in range(k):
            label = train_label[index[a][0]]
            num_class= int(label)
            if num_class in classcount.keys():
                classcount[num_class] +=1
            else:
                classcount[num_class] =1
        est_class[i] = max(zip(classcount.values(), classcount.keys()))[1]
    
    return est_class

def cross_validate(data, gt_labels, k, num_folds):
    length,num_feature = np.shape(data)
    fold_accuracies = np.zeros((num_folds,1))
    avg_accuracy = 0
    conf_matrix = np.zeros((5,5))
    index = np.arange(length)
    np.random.shuffle(index)
    length_test = length//num_folds
    for k in range(num_folds):
        if k !=num_folds-1:
            test_index = index[k*length_test:(k+1)*length_test]
            train_index = np.hstack((index[0:k*length_test],index[(k+1)*length_test:]))
        else:
            test_index = index[k*length_test:]
            train_index = index[0:k*length_test]
        print(type(train_index))
        train_data = [data[index] for index in train_index]
        print(type(train_data))
        test_data = [data[index] for index in test_index]
        train_label = [gt_labels[i] for i in train_index]
        est_class = knearestneighbor(test_data, train_data, train_label, 3)
        print(est_class.shape)
        test_label = [gt_labels[i] for i in test_index]
        tp = 0
        fold_matrix = np.zeros((5,5))
        for i in range(len(est_class)):
            fold_matrix[int(test_label[i])-1,int(est_class[i])-1] += 1
            if int(est_class[i]) == int(test_label[i]):
                tp += 1

        conf_matrix = conf_matrix + fold_matrix
        fold_accuracies[k] = tp/len(est_class)
        avg_accuracy +=fold_accuracies[k]

    avg_accuracy = avg_accuracy/num_folds
    return avg_accuracy,fold_accuracies, conf_matrix

def select_features(data, labels, k, num_folds):

    return feature_index

def evaluate(data,labels):
    features = np.loadtxt(os.path.join(os.path.dirname(__file__), 'data/feature_dict.txt'), delimiter=',', dtype=str)
    
    # k1_features = select_features(data,labels,1,10)
    k1_features = [0,1,2]
    print('Selected features for k = 1: ',features[k1_features])
    selected_data = np.take(data, k1_features,0)
    print('Average accuracy & confusion matrix for k = 1, folds = 10 after feature selection :')
    print( cross_validate(selected_data, labels, 1, 10))

    # k3_features = select_features(data,labels,3,10)
    k3_features = [0,1,2]
    print('Selected features for k = 3: ',features[k3_features])
    selected_data = np.take(data, k3_features,0)
    print('Average accuracy & confusion matrix for k = 3, folds = 10 after feature selection :')
    print( cross_validate(selected_data, labels, 3, 10))

    # k7_features = select_features(data,labels,7,10)
    k7_features = [0,1,2]
    print('Selected features for k = 7: ',features[k7_features])
    selected_data = np.take(data, k7_features,0)
    print('Average accuracy & confusion matrix for k = 7, folds = 10 after feature selection :')
    print( cross_validate(selected_data, labels, 7, 10))

def kmeans_clustering(data, k):
    # normalization with unit variance for kmeans clustering
    print(data.shape)
    whitened_data = whiten(data)
    centroids, distortion = kmeans(whitened_data, k)
    clusters = vq(whitened_data, centroids)
    # clusters = clusters + 1
    print(clusters[0] + 1)


if __name__ == "__main__": 
    pathdata = os.path.join(os.path.dirname(__file__), 'data/data.txt')
    pathlabel = os.path.join(os.path.dirname(__file__), 'data/labels.txt')
    gt_labels = np.loadtxt(pathlabel,dtype=str)
    data = np.loadtxt(pathdata, dtype=float)
    data = np.transpose(data)
    kmeans_clustering(data, 5)
    # avg_accuracy,fold_accuracies = cross_validate(data, gt_labels, 3, 3)
    # print(avg_accuracy)
    # print(fold_accuracies)

    # evaluate(data, gt_labels)
    '''length,num_features= np.shape(data)
    index = np.arange(len(data))
    np.random.shuffle(index)    
    data1= cross_validate(data,gt_labels,3,3)
    
    train_index = index[0:400]
    test_index = index[400:]
    train_data = np.zeros((400,10))
    test_data = np.zeros((100,10))
    train_label = np.zeros((400,1))
    for i in range(len(train_index)):
        train_data[i,:] = data[train_index[i],:]
        index_label = train_index[i]
        train_label[i] =label[index_label]
    for j in range(len(test_index)):
        test_data[j,:] = data[test_index[j],:]
    est_class = knearestneighbor(test_data, train_data, train_label, 7)
    test_label = np.zeros((100,1))
    num=0
    for i in range(len(test_label)):
        test_label[i] =label[test_index[i]] 
        if test_label[i][0] == est_class[i]:
            num=num+1 ''' 
        
        

     
     
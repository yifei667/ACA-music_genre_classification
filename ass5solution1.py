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
    length,num_feature = np.shape(data)
    fold_accuracies = np.zeros((num_folds,num_feature))
    avg_accuracy = 0
    index = np.arange(length)
    np.random.shuffle(index)
    num_neighbour = k
    length_test = length//num_folds
    config_matrix = np.zeros((num_feature,5,5))
    for i in range(num_feature):
        feature = data[:,i]
        for k in range(num_folds):
            if k!=num_folds-1:
                test_index = index[k*length_test:(k+1)*length_test]
                train_index = np.hstack((index[0:k*length_test],index[(k+1)*length_test:]))
            else:
                test_index = index[k*length_test:]
                train_index = index[0:k*length_test]

            feature_train = [feature[index] for index in train_index]
            feature_train = np.reshape(feature_train, (len(train_index), 1))
            feature_test = [feature[index] for index in test_index]
            feature_test = np.reshape(feature_test, (len(test_index), 1))
            train_label = [gt_labels[index] for index in train_index]
            train_label = np.reshape(train_label, (len(train_index), 1))
            test_label = [gt_labels[index] for index in test_index]
            test_label = np.reshape(test_label, (len(test_index), 1))

            est_class = knearestneighbor(feature_test, feature_train, train_label,num_neighbour)
            num = sum(int(est_class[a])==int(test_label[a]) for a in range(len(est_class)))
            for j in range(len(est_class)):
                config_matrix[i,int(est_class[j])-1,int(test_label[j])-1] +=1
            fold_accuracies[k,i]= num/len(est_class)
            
    avg_accuracy = np.mean(fold_accuracies,axis=0)
    return avg_accuracy,fold_accuracies,config_matrix


def find_best_features(data, labels, k, num_folds):
    
    length, num_features = np.shape(data)
    [avg_accuracy, fold_accuracies, conf_matrix] = cross_validate(data, labels, k, num_folds)
    
    feature_index = np.argmax(avg_accuracy)
    print(feature_index)

    return feature_index



def cross_validate_overall(data, gt_labels, k, num_folds):

    length,num_feature = np.shape(data)
    fold_accuracies = np.zeros((num_folds,1))
    avg_accuracy = 0
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
        train_data = [data[index] for index in train_index]
        test_data = [data[index] for index in test_index]
        train_label = [gt_labels[i] for i in train_index]
        est_class = knearestneighbor(test_data, train_data, train_label, 3)
        test_label = [gt_labels[i] for i in test_index]
        num = 0
        for i in range(len(est_class)):
            #print(type(int(est_class[i])))
            #print(type(test_label[i]))
            if int(est_class[i]) == int(test_label[i]):
                num+=1
        #print(num)
        fold_accuracies[k] = num/len(est_class)
        avg_accuracy +=fold_accuracies[k]
    avg_accuracy = avg_accuracy/num_folds
    return avg_accuracy


def select_features(data, labels, k, num_folds): 

    length, num_features = np.shape(data)
    [avg_accuracy, fold_accuracies, conf_matrix] = cross_validate(data, labels, k, num_folds)

    data_ff = None
    acc = [0] * num_features
    sel_feature_indices = [0] * num_features
    
    
    for i in range (num_features):
        
        best_features = np.amax(avg_accuracy)
        feature_index = np.argmax(avg_accuracy)
        if(np.all(data_ff) == None):
            data_ff = data[:,feature_index]
            data_ff = np.reshape(data_ff, (length, 1))
        else:
            data_ff = np.hstack((data_ff, np.reshape(data[:,feature_index], (length, 1))))

        avg_accuracy1 = cross_validate_overall(data_ff, labels, k, num_folds)

        acc[i] = avg_accuracy1
        sel_feature_indices[i] = feature_index

        avg_accuracy[feature_index] = 0;
    
    accuracy = acc[0]
    sel_feature_ind = sel_feature_indices[0]
    acc_optimal = accuracy
    for i in range(0,num_features-1):
        if (acc[i+1] > acc_optimal):
            acc_optimal = acc[i+1]
            accuracy = np.append(accuracy, acc[i+1])
            sel_feature_ind = np.append(sel_feature_ind, sel_feature_indices[i+1])
    print(accuracy)

   
    return sel_feature_ind, accuracy


def evaluate(data,labels):
    features = np.loadtxt(os.path.join(os.path.dirname(__file__), 'data/feature_dict.txt'), delimiter=',', dtype=str)
    [sel_feature_ind, accuracy] = select_features(data, gt_labels, 3, 3)

    print('Selected features for k = 3, numFolds = 3: ',features[sel_feature_ind])
    print(data.shape)
    selected_data = np.take(data, sel_feature_ind, 1)
    print('--------------')
    print(selected_data.shape)

    print('Average accuracy & confusion matrix for k = 1, folds = 10 after feature selection :')
    avg_accuracy, fold_accuracy, conf_matrix  = cross_validate(selected_data, labels, 1, 10)
    print('Avg accuracy : ', avg_accuracy)
    print('Fold accuracy : ',fold_accuracy)
    print('Confusion matrix : ',conf_matrix)

    print('Average accuracy & confusion matrix for k = 3, folds = 10 after feature selection :')
    avg_accuracy, fold_accuracy, conf_matrix  = cross_validate(selected_data, labels, 3, 10)
    print('Avg accuracy : ', avg_accuracy)
    print('Fold accuracy : ',fold_accuracy)
    print('Confusion matrix : ',conf_matrix)

    print('Average accuracy & confusion matrix for k = 7, folds = 10 after feature selection :')
    avg_accuracy, fold_accuracy, conf_matrix  = cross_validate(selected_data, labels, 7, 10)
    print('Avg accuracy : ', avg_accuracy)
    print('Fold accuracy : ',fold_accuracy)
    print('Confusion matrix : ',conf_matrix)

def kmeans_clustering(data, k):
    # normalization with unit variance for kmeans clustering
    print(data.shape)
    whitened_data = whiten(data)
    
    # initialize random centroids
    centroids = whitened_data[np.random.randint(whitened_data.shape[0], size=5), :]
    print(centroids.shape)

    total_distance = 0
    closest_centroid = np.zeros(whitened_data.shape[0])
    while (True):
        total_distance = 0
        dc_ccentroid = closest_centroid.copy()
        
        # calculate distances and closest  centroid from each point in the whitened_data
        distances = np.sqrt(((whitened_data - centroids[:, np.newaxis])**2).sum(axis=2))
        print(distances.shape)
        closest_centroid = np.argmin(distances, axis=0)
        print(closest_centroid.shape)
        

        # adjust centroids 
        centroids = np.array([data[closest_centroid==k].mean(axis=0) for k in range(centroids.shape[0])])
        if np.array_equal(closest_centroid, dc_ccentroid):
            print('converged')
            print('Cluster labels :', closest_centroid)
            print('Corresponding distances from each cluster centroid :', distances)
            break

if __name__ == "__main__": 

    pathdata = os.path.join(os.path.dirname(__file__), 'data/data.txt')
    pathlabel = os.path.join(os.path.dirname(__file__), 'data/labels.txt')
    gt_labels = np.loadtxt(pathlabel,dtype=str)
    data = np.loadtxt(pathdata, dtype=float)
    data = np.transpose(data)
    kmeans_clustering(data, gt_labels)
    avg_accuracy,fold_accuracies,config_matrix = cross_validate(data, gt_labels, 3, 3)
    feature_index = find_best_features(data, gt_labels, 3, 3)
    [sel_feature_ind, accuracy] = select_features(data, gt_labels, 3, 3)   
    print (avg_accuracy)
    print (sel_feature_ind)
    plt.plot(accuracy)
    plt.xlabel('Number of Features')
    plt.ylabel('Accuracy')
    plt.show()



    
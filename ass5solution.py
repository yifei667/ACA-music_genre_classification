import numpy as np

def knearestneighbor(test_data, train_data, train_label, k):
    est_class = np.zeros((len(test_data),1))
    for i in range(len(test_data)):
        test_now = test_data[i]
        length = len(train_data)
        distance = np.zeros((length,1))
        for j in range(length):
            train = train_data[j]
            distance[j] = np.sqrt(abs(train-test_now))
        sorted_index = np.argsort(distance,axis=0) 
        index = sorted_index[:3]
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
    fold_accuracies = np.zeros((num_folds,num_feature))
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
            feature_test = [feature[index] for index in test_index]
            train_label = [gt_labels[index] for index in train_index]
            test_label = [gt_labels[index] for index in test_index]
            est_class = knearestneighbor(feature_test, feature_train, train_label,num_neighbour)
            num = sum(int(est_class[a])==int(test_label[a]) for a in range(len(est_class)))
            for j in range(len(est_class)):
                config_matrix[i,int(est_class[j])-1,int(test_label[j])-1] +=1
            fold_accuracies[k,i]= num/len(est_class)
    avg_accuracy = np.mean(fold_accuracies,axis=0)
    

    return avg_accuracy,fold_accuracies,config_matrix
    







def select_features(data, labels, k, num_folds):

    return feature_index

if __name__ == "__main__": 
    pathdata = '/Users/yuyifei/Desktop/6201/data/data.txt'
    pathlabel = '/Users/yuyifei/Desktop/6201/data/labels.txt'
    gt_labels = np.loadtxt(pathlabel,dtype=str)
    data = np.loadtxt(pathdata, dtype=float)
    data = np.transpose(data)
    avg_accuracy,fold_accuracies,config_matrix = cross_validate(data, gt_labels, 3, 3)
    print(fold_accuracies)
    print(avg_accuracy)
    print(config_matrix)
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
            num=num+1 
        
        

        train_data_total = [data[index] for index in train_index]
        print(train_data_total)
        test_data_total = [data[index] for index in test_index]
        train_label = [gt_labels[index] for index in train_index]
        test_label = [gt_labels[index] for index in test_index]
        est_class = np.zeros((num_feature))
        for i in range(num_feature):
            train_data = train_data_total[:,i]
            test_data = test_data_total[:,i]
            est_class = knearestneighbor(test_data, train_data, train_label, k)
            for a in range(len(test_label)):
                num = 0
                if int(test_label[a]) == int(est_class[a]):
                    num = num+1
            fold_accuracies[j,i] = num/len(test_label)'''     
     
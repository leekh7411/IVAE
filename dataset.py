from keras.datasets import mnist
import numpy as np

def load_interaction_mnist():
    # MNIST dataset
    (x_train, y_train), (x_test, y_test) = mnist.load_data()
    image_size = x_train.shape[1]
    original_dim = image_size * image_size
    x_train = np.reshape(x_train, [-1, original_dim])
    x_test = np.reshape(x_test, [-1, original_dim])
    x_train = x_train.astype('float32') / 255
    x_test = x_test.astype('float32') / 255
    print("input dimension is {}".format(original_dim))
    
    
    # Preprocess pair of summation is odd(...or even?)
    # here we choose even-number summation pair

    def preprocess_interaction_dataset(x_train, y_train, n_sample, mod, residue):
        inter_x_train = []
        inter_x1_train = []
        inter_x2_train = []
        n_inter_x_train = n_sample
        while True:

            train_indices1 = np.arange(len(x_train))
            train_indices2 = np.arange(len(x_train))
            np.random.shuffle(train_indices1)
            np.random.shuffle(train_indices2)

            for i1, i2 in zip(train_indices1, train_indices2):
                if ((y_train[i1]) + (y_train[i2])) % mod == residue:
                    
                    inter_x = np.concatenate([x_train[i1], x_train[i2]], axis=0)
                    inter_x_train.append(inter_x)
                    inter_x1_train.append(x_train[i1])
                    inter_x2_train.append(x_train[i2])

                    if len(inter_x_train) == n_inter_x_train:
                        break

            if len(inter_x_train) == n_inter_x_train:
                break

        inter_x_train = np.array(inter_x_train)
        inter_x1_train = np.array(inter_x1_train)
        inter_x2_train = np.array(inter_x2_train)

        return inter_x_train, inter_x1_train, inter_x2_train
    
    inter_x_train, inter_x1_train, inter_x2_train = preprocess_interaction_dataset(x_train, y_train, 30000, 2, 0)
    inter_x_test, inter_x1_test, inter_x2_test  = preprocess_interaction_dataset(x_test, y_test, 10000, 2, 0)
    
    print("interaction train x : {} / x1 : {} / x2 : {}".format(inter_x_train.shape, 
                                                                inter_x1_train.shape,
                                                                inter_x2_train.shape))
    print("interaction test  x : {} / x1 : {} / x2 : {}".format(inter_x_test.shape, 
                                                                inter_x1_test.shape,
                                                                inter_x2_test.shape))
        
    return (inter_x_train, inter_x1_train, inter_x2_train),(inter_x_test, inter_x1_test, inter_x2_test) 




def load_one2one_interaction_mnist():
    # MNIST dataset
    (x_train, y_train), (x_test, y_test) = mnist.load_data()
    image_size = x_train.shape[1]
    original_dim = image_size * image_size
    x_train = np.reshape(x_train, [-1, original_dim])
    x_test = np.reshape(x_test, [-1, original_dim])
    x_train = x_train.astype('float32') / 255
    x_test = x_test.astype('float32') / 255
    print("input dimension is {}".format(original_dim))
    
    
    # Preprocess pair of summation is odd(...or even?)
    # here we choose even-number summation pair

    def preprocess_interaction_dataset(x_train, y_train, n_sample, residue):
        inter_x_train = []
        inter_x1_train = []
        inter_x2_train = []
        n_inter_x_train = n_sample
        while True:
            
            train_indices1 = np.arange(len(x_train))
            train_indices2 = np.arange(len(x_train))
            np.random.shuffle(train_indices1)
            np.random.shuffle(train_indices2)

            for i1, i2 in zip(train_indices1, train_indices2):
                if ((y_train[i1]) - (y_train[i2])) == residue:
                    #print("{} and {}".format((y_train[i1]), (y_train[i2])))
                    inter_x = np.concatenate([x_train[i1], x_train[i2]], axis=0)
                    inter_x_train.append(inter_x)
                    inter_x1_train.append(x_train[i1])
                    inter_x2_train.append(x_train[i2])

                    if len(inter_x_train) == n_inter_x_train:
                        break

            if len(inter_x_train) == n_inter_x_train:
                break

        inter_x_train = np.array(inter_x_train)
        inter_x1_train = np.array(inter_x1_train)
        inter_x2_train = np.array(inter_x2_train)

        return inter_x_train, inter_x1_train, inter_x2_train
    
    inter_x_train, inter_x1_train, inter_x2_train = preprocess_interaction_dataset(x_train, y_train, 30000, 1)
    inter_x_test, inter_x1_test, inter_x2_test  = preprocess_interaction_dataset(x_test, y_test, 10000, 1)
    
    print("interaction train x : {} / x1 : {} / x2 : {}".format(inter_x_train.shape, 
                                                                inter_x1_train.shape,
                                                                inter_x2_train.shape))
    print("interaction test  x : {} / x1 : {} / x2 : {}".format(inter_x_test.shape, 
                                                                inter_x1_test.shape,
                                                                inter_x2_test.shape))
        
    return (inter_x_train, inter_x1_train, inter_x2_train),(inter_x_test, inter_x1_test, inter_x2_test) 
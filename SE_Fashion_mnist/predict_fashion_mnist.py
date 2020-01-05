import json
import numpy as np
import argparse
import sklearn.metrics as metrics
from scipy.optimize import minimize
from sklearn.metrics import log_loss
from models import wide_residual_net as WRN, dense_net as DN

from keras.datasets import fashion_mnist
import tensorflow.keras.backend as K
import keras.utils.np_utils as kutils
from itertools import combinations
import pandas as pd
import scipy.spatial.distance as ssd
import scipy.cluster.hierarchy as hcluster
from matplotlib import pyplot as plt

parser = argparse.ArgumentParser(description='FASHION MNIST Ensemble Prediction')

parser.add_argument('--optimize', type=int, default=0, help='Optimization flag. Set to 1 to perform a randomized search to maximise classification accuracy (otpimized ensemble weights).\n'
                                                            'Set to -1 to get non weighted classification accuracy (as in the paper)\n'
                                                            'Set to 0, default, predict using best weigths')

parser.add_argument('--num_tests', type=int, default=20, help='Number of tests to perform when optimizing the '
                                                              'ensemble weights for maximizing classification accuracy')

parser.add_argument('--model', type=str, default='wrn', help='Type of model to train')

# Wide ResNet Parameters
parser.add_argument('--wrn_N', type=int, default=2, help='Number of WRN blocks. Computed as N = (n - 4) / 6.')
parser.add_argument('--wrn_k', type=int, default=4, help='Width factor of WRN')

# DenseNet Parameters
parser.add_argument('--dn_depth', type=int, default=40, help='Depth of DenseNet')
parser.add_argument('--dn_growth_rate', type=int, default=12, help='Growth rate of DenseNet')

# Calculate Classifier Output Difference dendrogram
parser.add_argument('--cod', type= int, default=0, help='Classifier Output Difference dendrogram. Set 1 to compute it. Default is 0')

args = parser.parse_args()

# Change NUM_TESTS to larger numbers to get possibly better results
NUM_TESTS = args.num_tests

# Change to False to only predict
OPTIMIZE = args.optimize

CLASS_OUT_DIFF = args.cod

model_type = str(args.model).lower()
assert model_type in ['wrn', 'dn'], 'Model type must be one of "wrn" for Wide ResNets or "dn" for DenseNets'

if model_type == "wrn":
    n = args.wrn_N * 6 + 4
    k = args.wrn_k

    models_filenames = [r"weights/WRN-FASHIONMNIST-%d-%d-Best.h5" % (n, k),
                        r"weights/WRN-FASHIONMNIST-%d-%d-1.h5" % (n, k),
                        r"weights/WRN-FASHIONMNIST-%d-%d-2.h5" % (n, k),
                        r"weights/WRN-FASHIONMNIST-%d-%d-3.h5" % (n, k),
                        r"weights/WRN-FASHIONMNIST-%d-%d-4.h5" % (n, k),
                        r"weights/WRN-FASHIONMNIST-%d-%d-5.h5" % (n, k)]
else:
    depth = args.dn_depth
    growth_rate = args.dn_growth_rate

    models_filenames = [r"weights/DenseNet-FASHIONMNIST-%d-%d-Best.h5" % (depth, growth_rate),
                        r"weights/DenseNet-FASHIONMNIST-%d-%d-1.h5" % (depth, growth_rate),
                        r"weights/DenseNet-FASHIONMNIST-%d-%d-2.h5" % (depth, growth_rate),
                        r"weights/DenseNet-FASHIONMNIST-%d-%d-3.h5" % (depth, growth_rate),
                        r"weights/DenseNet-FASHIONMNIST-%d-%d-4.h5" % (depth, growth_rate),
                        r"weights/DenseNet-FASHIONMNIST-%d-%d-5.h5" % (depth, growth_rate),
                        r"weights/DenseNet-FASHIONMNIST-%d-%d-6.h5" % (depth, growth_rate)]

(trainX, trainY), (testX, testY) = fashion_mnist.load_data()

(valX, valY) = (testX[:2000], testY[:2000]) ## split into validation and test set, use validation set to optimize weights
(testX, testY) = (testX[2000:], testY[2000:])
nb_classes = len(np.unique(testY))

valX = valX.reshape(valX.shape[0], 28, 28, 1)
testX = testX.reshape(testX.shape[0], 28, 28, 1)

valX = valX.astype('float32')
valX /= 255.0
testX = testX.astype('float32')
testX /= 255.0

valY_cat = kutils.to_categorical(valY)
testY_cat = kutils.to_categorical(testY)

if K.image_data_format() == "th":
    init = (1, 28, 28)
else:
    init = (28, 28, 1)

if model_type == "wrn":
    model = WRN.create_wide_residual_network(init, nb_classes=10, N=args.wrn_N, k=args.wrn_k, dropout=0.00)

    model_prefix = 'WRN-FASHIONMNIST-%d-%d' % (args.wrn_N * 6 + 4, args.wrn_k)
else:
    model = DN.create_dense_net(nb_classes=10, img_dim=init, depth=args.dn_depth, nb_dense_block=1,
                                growth_rate=args.dn_growth_rate, nb_filter=16, dropout_rate=0.2)

    model_prefix = 'DenseNet-FASHIONMNIST-%d-%d' % (args.dn_depth, args.dn_growth_rate)


best_acc = 0.0
best_weights = None

val_preds = []
for fn in models_filenames:
    model.load_weights(fn)
    print("Predicting validation set values on model %s" % (fn))
    yPreds = model.predict(valX, batch_size=112, verbose=2)
    val_preds.append(yPreds)

test_preds = []
for fn in models_filenames:
    model.load_weights(fn)
    print("Predicting test set values on model %s" % (fn))
    yPreds = model.predict(testX, batch_size=112, verbose=2)
    test_preds.append(yPreds)


def calculate_weighted_accuracy():
    global weighted_predictions, weight, prediction, yPred, yTrue, accuracy, error
    weighted_predictions = np.zeros((testX.shape[0], nb_classes), dtype='float32')
    for weight, prediction in zip(prediction_weights, test_preds):
        weighted_predictions += weight * prediction
    yPred = np.argmax(weighted_predictions, axis=1)
    yTrue = testY
    accuracy = metrics.accuracy_score(yTrue, yPred) * 100
    error = 100 - accuracy
    print("Accuracy : ", accuracy)
    print("Error : ", error)
    exit()

# Calculate Classifier Output Difference metric (COD)
if CLASS_OUT_DIFF == 1:
    # binary function for COD
    def binary_function(lab_1, lab_2):
        if lab_1 == lab_2:
            return int(1)
        else:
            return int(0)

    # Classifier Output Difference function
    def classifier_output_difference(m, model_1, model_2):
        m.load_weights(model_1)
        y_pred_1 = []
        y_pred_1 = m.predict(testX, batch_size=128, verbose=2)

        m.load_weights(model_2)
        y_pred_2 = []
        y_pred_2 = m.predict(testX, batch_size=128, verbose=2)

        out = []
        out = [value for value in y_pred_1 if value in y_pred_2]
        return np.sum(out)/len(testX)


    # initialise matrix to calculate COD
    n_cod = len(models_filenames)-1

    cod = np.zeros((n_cod, n_cod), dtype = float)

    def calculate_cod(mat, m, model_filename):
        df = pd.DataFrame(mat)
        cc = list(combinations(df.columns,2))

        values = []
        for val in range(0, len(cc)):
            values.append(classifier_output_difference(m, model_filename[cc[val][0] + 1], model_filename[cc[val][1] + 1]))

        mat[np.triu_indices(len(mat), 1)] = values
        return mat + mat.T - np.diag(mat.diagonal())


    Class_Output_Difference_metric = calculate_cod(cod, model, models_filenames)

    labs = []
    for i in range(0, n_cod):
        labs.append('Model ' + str(i+1))

    # create dendrogram using COD
    distVec = ssd.squareform(Class_Output_Difference_metric)
    linkage = hcluster.linkage(distVec)
    dendro  = hcluster.dendrogram(linkage, labels = labs)
    plt.gcf()
    plt.ylabel('Classifier Output Difference')
    plt.xlabel('Model')
    plt.title('Hierarchical clustering of classifiers using COD metric')
    plt.savefig('images/dendrogram_cod.png')


# default: predict using model with best weights
if OPTIMIZE == 0:
    with open('weights/Ensemble weights %s.json' % model_prefix, mode='r') as f:
        dictionary = json.load(f)

    prediction_weights = dictionary['best_weights']
    calculate_weighted_accuracy()

# using equal weights for all models (as given in the paper)
elif OPTIMIZE == -1:
    prediction_weights = [1. / len(models_filenames)] * len(models_filenames)
    calculate_weighted_accuracy()

''' OPTIMIZATION REGION '''

print()

def log_loss_func(weights):
    ''' scipy minimize will pass the weights as a numpy array '''
    final_prediction = np.zeros((valX.shape[0], nb_classes), dtype='float32')

    for weight, prediction in zip(weights, val_preds):
        final_prediction += weight * prediction

    return log_loss(valY_cat, final_prediction)

# optimization of ensemble weights found minimizing the log_loss scores
for iteration in range(NUM_TESTS):
    prediction_weights = np.random.random(len(models_filenames))

    constraints = ({'type': 'eq', 'fun':lambda w: 1 - sum(w)})
    bounds = [(0, 1)] * len(val_preds)

    result = minimize(log_loss_func, prediction_weights, method='SLSQP', bounds=bounds, constraints=constraints)

    print('Best Ensemble Weights: {weights}'.format(weights=result['x']))

    weights = result['x']
    weighted_predictions = np.zeros((testX.shape[0], nb_classes), dtype='float32')

    for weight, prediction in zip(weights, test_preds):
        weighted_predictions += weight * prediction

    yPred = np.argmax(weighted_predictions, axis=1)
    yTrue = testY

    accuracy = metrics.accuracy_score(yTrue, yPred) * 100
    error = 100 - accuracy
    print("Iteration %d: Accuracy : " % (iteration + 1), accuracy)
    print("Iteration %d: Error : " % (iteration + 1), error)

    if accuracy > best_acc:
        best_acc = accuracy
        best_weights = weights

    print()

print("Best Accuracy : ", best_acc)
print("Best Weights : ", best_weights)

with open('weights/Ensemble weights %s.json' % model_prefix, mode='w') as f:
    dictionary = {'best_weights' : best_weights.tolist()}
    json.dump(dictionary, f)

''' END OF OPTIMIZATION REGION '''




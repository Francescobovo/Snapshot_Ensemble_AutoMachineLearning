import json
import numpy as np
import argparse
import sklearn.metrics as metrics
from scipy.optimize import minimize
from sklearn.metrics import log_loss
from models import wide_residual_net as WRN, dense_net as DN

from keras.datasets import cifar10
import tensorflow.keras.backend as K
import keras.utils.np_utils as kutils

parser = argparse.ArgumentParser(description='CIFAR 10 Ensemble Prediction')

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

args = parser.parse_args()

# Change NUM_TESTS to larger numbers to get possibly better results
NUM_TESTS = args.num_tests

# Change to False to only predict
OPTIMIZE = args.optimize

model_type = str(args.model).lower()
assert model_type in ['wrn', 'dn'], 'Model type must be one of "wrn" for Wide ResNets or "dn" for DenseNets'

if model_type == "wrn":
    n = args.wrn_N * 6 + 4
    k = args.wrn_k

    models_filenames = [r"weights/WRN-CIFAR10-%d-%d-Best.h5" % (n, k),
                        r"weights/WRN-CIFAR10-%d-%d-1.h5" % (n, k),
                        r"weights/WRN-CIFAR10-%d-%d-2.h5" % (n, k),
                        r"weights/WRN-CIFAR10-%d-%d-3.h5" % (n, k),
                        r"weights/WRN-CIFAR10-%d-%d-4.h5" % (n, k),
                        r"weights/WRN-CIFAR10-%d-%d-5.h5" % (n, k)]
else:
    depth = args.dn_depth
    growth_rate = args.dn_growth_rate

    models_filenames = [r"weights/DenseNet-CIFAR10-%d-%d-Best.h5" % (depth, growth_rate),
                        r"weights/DenseNet-CIFAR10-%d-%d-1.h5" % (depth, growth_rate),
                        r"weights/DenseNet-CIFAR10-%d-%d-2.h5" % (depth, growth_rate),
                        r"weights/DenseNet-CIFAR10-%d-%d-3.h5" % (depth, growth_rate),
                        r"weights/DenseNet-CIFAR10-%d-%d-4.h5" % (depth, growth_rate),
                        r"weights/DenseNet-CIFAR10-%d-%d-5.h5" % (depth, growth_rate),
                        r"weights/DenseNet-CIFAR10-%d-%d-6.h5" % (depth, growth_rate)]

(trainX, trainY), (testX, testY) = cifar10.load_data()

(valX, valY) = (testX[:2000], testY[:2000]) ## split into validation and test set, use validation set to optimize weights
(testX, testY) = (testX[2000:], testY[2000:])
nb_classes = len(np.unique(testY))

valX = valX.astype('float32')
valX /= 255.0
testX = testX.astype('float32')
testX /= 255.0

valY_cat = kutils.to_categorical(valY)
testY_cat = kutils.to_categorical(testY)

if K.image_data_format() == "th":
    init = (3, 32, 32)
else:
    init = (32, 32, 3)

if model_type == "wrn":
    model = WRN.create_wide_residual_network(init, nb_classes=10, N=args.wrn_N, k=args.wrn_k, dropout=0.00)

    model_prefix = 'WRN-CIFAR10-%d-%d' % (args.wrn_N * 6 + 4, args.wrn_k)
else:
    model = DN.create_dense_net(nb_classes=10, img_dim=init, depth=args.dn_depth, nb_dense_block=1,
                                growth_rate=args.dn_growth_rate, nb_filter=16, dropout_rate=0.2)

    model_prefix = 'DenseNet-CIFAR10-%d-%d' % (args.dn_depth, args.dn_growth_rate)

best_acc = 0.0
best_weights = None

val_preds = []
for fn in models_filenames:
    model.load_weights(fn)
    print("Predicting validation set values on model %s" % (fn))
    yPreds = model.predict(valX, batch_size=128, verbose=2)
    val_preds.append(yPreds)

test_preds = []
for fn in models_filenames:
    model.load_weights(fn)
    print("Predicting test set values on model %s" % (fn))
    yPreds = model.predict(testX, batch_size=128, verbose=2)
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

""" ########## TESTING
# binary function for COD
def binary_function(lab_1, lab_2):
    if lab_1 == lab_2:
        return int(1)
    else:
        return int(0)

# Classifier Output Difference function
def classifier_output_difference(m, model_1, model_2):
    mod_1 = m.load_weights(model_1)
    mod_2 = m.load_weights(model_2)
    y_pred_1 = []
    y_pred_2 = []

    y_pred_1 = mod_1.predict(testX, batch_size=128, verbose=2)
    y_pred_2 = mod_2.predict(testX, batch_size=128, verbose=2)

    out = [binary_function(x1, x2) for x1, x2 in zip(y_pred_1, y_pred_2)]
    return np.mean(out)

fn1 = models_filenames[1]
fn2 = models_filenames[2]
fn3 = models_filenames[3]
fn4 = models_filenames[4]
fn5 = models_filenames[5]

cod_12 = classifier_output_difference(model, fn1, fn2)
cod_13 = classifier_output_difference(model, fn1, fn3)
cod_14 = classifier_output_difference(model, fn1, fn4)
cod_15 = classifier_output_difference(model, fn1, fn5)

cod_23 = classifier_output_difference(model, fn2, fn3)
cod_24 = classifier_output_difference(model, fn2, fn4)
cod_25 = classifier_output_difference(model, fn2, fn5)

cod_34 = classifier_output_difference(model, fn3, fn4)
cod_35 = classifier_output_difference(model, fn3, fn5)

cod_45 = classifier_output_difference(model, fn4, fn5)

print('cod_12: ', cod_12)
print('cod_13: ', cod_13)
print('cod_14: ', cod_14)
print('cod_15: ', cod_15)
print('cod_23: ', cod_23)
print('cod_24: ', cod_24)
print('cod_25: ', cod_25)
print('cod_34: ', cod_34)
print('cod_35: ', cod_35)
print('cod_45: ', cod_45)
 """
# Snapshot Ensembles using COD in Keras

Implementation of the original paper [Snapshot Ensembles: Train 1, Get M for Free](http://openreview.net/pdf?id=BJYwwY9ll) in Keras 1.1.1

# Explanation 

Snapshot Ensemble is a method to obtain multiple neural network which can be ensembled at no additional training cost. This is achieved by letting a single neural network converge into several local minima along its optimization path and save the model parameters at certain epochs, therefore the weights being "snapshots" of the model. 

# Usage

The original paper uses several models such as ResNet-101, Wide Residual Network and DenseNet-40 and DenseNet-100. While DenseNets are the highest performing models, they are too large and take extremely long to train. Therefore, the current trained model is the Wide Residual Net (16-4) setting. This model performs poorly compared to the 34-4 version but trains several times faster.

The technique is simple to implement in Keras, using a custom callback. These callbacks can be built using the `SnapshotCallbackBuilder` class in `snapshot.py`. Other models can simply use this callback builder to other models to train them in a similar manner.

To use snapshot ensemble in other models : 
```
from snapshot import SnapshotCallbackBuilder

M = 5 # number of snapshots
nb_epoch = T = 200 # number of epochs
alpha_zero = 0.1 # initial learning rate
model_prefix = 'Model_'

snapshot = SnapshotCallbackBuilder(T, M, alpha_zero) 
...
model = Sequential() OR model = Model(ip, output) # Some model that has been compiled

model.fit(trainX, trainY, callbacks=snapshot.get_callbacks(model_prefix=model_prefix))
```

To train WRN or DenseNet models on FASHION MNIST (or use pre trained models):

1. Run the `train_fashion_mnist.py` script to train the WRN-16-4 model on FASHION MNIST dataset (not required since weights are provided)
2. Run the `predict_fashion_mnist.py` script to make an ensemble prediction.

According to the original paper, models trained on more complex datasets such as CIFAR 100 and Tiny ImageNet obtaines a greater boost from the ensemble model.

In the `predict_fashion_mnist.py` script is possible to compute the Classifier Output Difference (COD) dendrogram. COD is a metric that measures the number of observations on which a pair of classifiers yields a different prediction. A high value of COD indicates that two classifiers yield different predictions, hence they would be well suited to combine in an ensemble. This method of ensemble selection is well explained on the project paper developed.

## Parameters
Some parameters for WRN models from the original paper:
- M = 5
- nb_epoch = 200
- alpha_zero = 0.1
- wrn_N = 2 (WRN-16-4) or 4 (WRN-28-8)
- wrn_k = 4 (WRN-16-4) or 8 (WRN-28-8)

Some parameters for DenseNet models from the original paper:
- M = 6
- nb_epoch = 300
- alpha_zero = 0.2
- dn_depth = 40 (DenseNet-40-12) or 100 (DenseNet-100-24)
- dn_growth_rate = 12 (DenseNet-40-12) or 24 (DenseNet-100-24)

### train_fashion_mnist.py
```
--M              : Number of snapshots that will be taken. Optimal range is in between 4 - 8. Default is 5
--nb_epoch       : Number of epochs to train the network. Default is 200
--alpha_zero     : Initial Learning Rate. Usually 0.1 or 0.2. Default is 0.1

--model          : Type of model to train. Can be "wrn" for Wide ResNets or "dn" for DenseNet

--wrn_N          : Number of WRN blocks. Computed as N = (n - 4) / 6. Default is 2.
--wrn_k          : Width factor of WRN. Default is 12.

--dn_depth       : Depth of DenseNet. Default is 40.
--dn_growth_rate : Growth rate of DenseNet. Default is 12.
```

### predict_fashion_mnist.py
```
--optimize       : Flag to optimize the ensemble weights. 
                   Default is 0 (Predict using optimized weights).
                   Set to 1 to optimize ensemble weights (test for num_tests times).
                   Set to -1 to predict using equal weights for all models (As given in the paper).
               
--num_tests      : Number of times the optimizations will be performed. Default is 20

--model          : Type of model to train. Can be "wrn" for Wide ResNets or "dn" for DenseNet

--wrn_N          : Number of WRN blocks. Computed as N = (n - 4) / 6. Default is 2.
--wrn_k          : Width factor of WRN. Default is 12.

--dn_depth       : Depth of DenseNet. Default is 40.
--dn_growth_rate : Growth rate of DenseNet. Default is 12.

--cod            : Set to 1 to compute Classifier Output Difference dendrogram. Default is 0.
```

# Requirements

- Keras
- Theano (tested) / Tensorflow (not tested, weights not available but can be converted)
- scipy
- h5py
- sklearn

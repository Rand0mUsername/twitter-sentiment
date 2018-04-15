import pickle
import numpy as np

from dataset import TwitterDataset
from model import MultinomialNaiveBayes
from utils import calc_acc

nb_classes = 2  # (Positive, Negative)
nb_words = 10000  # Size of our bag of words
load_cached = True  # If false, regenerates features from the original dataset 

# Load the dataset
if load_cached:
    with open('dataset.pkl', 'rb') as pkl:
        print('Loading a cached dataset')
        dataset = pickle.load(pkl)
        print('Done loading')
else:
    with open('dataset.pkl', 'wb') as pkl:
        print('Preparing a new dataset')
        dataset = TwitterDataset('data/train.csv', nb_words, 0.1, 0.1)
        print('Done preparing the dataset, serializing')
        pickle.dump(dataset, pkl, pickle.HIGHEST_PROTOCOL)
        print('Done serializing')

# Fit several models with varying pseudocount parameter
models = dict()
for pseudocount in range(1, 30):
    # Fit the model
    print('Fitting a model with pseudocount={}'.format(pseudocount))
    model = MultinomialNaiveBayes(nb_classes, nb_words, pseudocount)
    model.fit(dataset.train)

    # Evaluate on train set
    preds_train = model.predict(dataset.train['x'])
    acc_train = calc_acc(dataset.train['y'], preds_train)
    print('Train set accuracy: {0:.4f}'.format(acc_train))

    # Evaluate on validation set
    preds_val = model.predict(dataset.val['x'])
    acc_val = calc_acc(dataset.val['y'], preds_val)
    print('Validation set accuracy: {0:.4f}'.format(acc_val))

    # Save the model
    models[model] = acc_val

# Find the best model (best validation set accuracy)
best_model = max(models, key=models.get)
print('Best pseudocount is {}'.format(best_model.pseudocount))

# Evaluate on test set
predictions = best_model.predict(dataset.test['x'])
acc_test = calc_acc(dataset.test['y'], predictions)
print('Test set accuracy for the final model: {}%'.format(round(100*acc_test)))
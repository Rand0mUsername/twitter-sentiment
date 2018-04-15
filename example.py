from model import MultinomialNaiveBayes
from utils import calc_acc

"""
    A toy example from a lecture:
    https://web.stanford.edu/class/cs124/lec/naivebayes.pdf

    Used to verify the model logic (by inspecting intermediate values).
"""

model = MultinomialNaiveBayes(nb_classes=2, nb_events=6, pseudocount=1)

# Words: Chinese Beijing Shangai Macao Tokyo Japan
# Class 0: China, Class 1: Japan

x_train = [{0:2, 1:1}, {0:2, 2:1}, {0:1, 3:1}, {0:1, 4:1, 5:1}]
y_train = [0, 0, 0, 1]
x_test = [{0:3, 4:1, 5:1}]

model.fit({'x': x_train, 'y': y_train})
predictions = model.predict(x_test)
print(predictions)
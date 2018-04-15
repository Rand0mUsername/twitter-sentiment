# Calculate accuracy
def calc_acc(labels, predictions):
        nb_correct = 0
        nb_total = len(labels)
        for i in range(nb_total):
            if labels[i] == predictions[i]:
                nb_correct += 1
        return nb_correct / nb_total
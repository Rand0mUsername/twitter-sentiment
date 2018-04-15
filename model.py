import math

class MultinomialNaiveBayes:
    """
    A Naive Bayes classifier that uses a categorical distribution to 
    estimate P(xi|C) likelihoods. Feature vector (x) is treated as
    a histogram of event frequencies so we have: 
        P(x|C)=product(P(xi|C))=product(pi^xi), 
    where pi are event probabilities from our categorical distribution.
    
    Therefore: log(P(C|x)) ~ log(P(C)) + sum(xi*log(pi))
    """

    def __init__(self, nb_classes, nb_events, pseudocount):
        self.nb_classes = nb_classes  # 2 in our case
        self.nb_events = nb_events  # nb_words in our case
        self.pseudocount = pseudocount  # additive smoothing parameter
    
    def fit(self, data):
        x, labels = data['x'], data['y']
        nb_examples = len(labels)

        # Calculate class priors
        self.priors = []
        for c in range(self.nb_classes):
            self.priors.append(labels.count(c) / nb_examples)

        # Sum event occurences for each class
        occs = [[0] * self.nb_events for _ in range(self.nb_classes)]
        for i in range(nb_examples):
            c = labels[i]
            for w, cnt in x[i].items():
                occs[c][w] += cnt
        
        # Calculate event likelihoods for each class
        self.likelihoods = [[0] * self.nb_events for _ in range(self.nb_classes)]
        for c in range(self.nb_classes):
            for w in range(self.nb_events):
                num = occs[c][w] + self.pseudocount
                den = sum(occs[c]) + self.nb_events*self.pseudocount
                self.likelihoods[c][w] = num / den
            
    def predict(self, xs):
        nb_examples = len(xs)
        preds = []
        for i in range(nb_examples):
            # Calculate log probabilities for each class
            log_probs = []
            for c in range(self.nb_classes):
                log_prob = math.log(self.priors[c])
                for w, cnt in xs[i].items():
                    log_prob += cnt * math.log(self.likelihoods[c][w])
                log_probs.append(log_prob)

            # Max log probability gives the prediction
            pred = log_probs.index(max(log_probs))
            preds.append(pred)
        return preds
import csv
import html
import re
from nltk.tokenize import word_tokenize
from nltk.stem import PorterStemmer
from nltk.corpus import stopwords
from nltk import FreqDist
from autocorrect import spell

class TwitterDataset:
    """
    Twitter sentiment analysis dataset from Kaggle:
        https://www.kaggle.com/c/twitter-sentiment-analysis2
    """

    def __init__(self, path, nb_words, ratio_val, ratio_test):
        # Load and process the data to get bags of words
        x, y = self.load_csv(path)
        x = self.clean(x)
        x = self.get_words(x)
        x = self.compute_bow_histograms(x, nb_words)

        # Make the train-validate-test split
        nb_total = len(x)
        nb_test = int(ratio_test * nb_total)
        nb_val = int(ratio_val * nb_total)
        nb_train = nb_total - nb_test - nb_val
        self.train = {
            'x': x[:nb_train], 
            'y': y[:nb_train]
        }
        self.val = {
            'x': x[nb_train:nb_train+nb_val],
            'y': y[nb_train:nb_train+nb_val]
        }
        self.test = {
            'x': x[nb_train+nb_val:],
            'y': y[nb_train+nb_val:]
        }

    def load_csv(self, path):
        """
        Loads 11 rows less than intended since lines 4290-4291, 5182-5183 and 8837-8846 
        are merged (mismatched quotes, dataset error) 
        """
        x, y = [], []
        with open(path, 'r', encoding='latin1') as f:
            reader = csv.reader(f, delimiter=',', quotechar='"')
            next(reader, None)
            for row in reader:
                y.append(int(row[1]))
                x.append(row[2])
        return x, y
    
    def clean(self, x):
        # Unescape and remove urls
        x = [html.unescape(tw) for tw in x]
        re_url = r'https?://(?:[-\w.]|(?:%[\da-fA-F]{2}))+'
        x = [re.sub(re_url, '', tw) for tw in x]
        
        # Remove special chars and convert to lowercase
        x = [re.sub(r'[^a-zA-Z0-9\s]', '', tw) for tw in x]
        x = [tw.lower() for tw in x]
        return x
    
    def get_words(self, x):
        # Tokenize and remove stopwords 
        x = [word_tokenize(tw) for tw in x]
        stopword_list = set(stopwords.words('english'))
        x = [[w for w in tw if not w in stopword_list] for tw in x]
        
        # Stem
        porter = PorterStemmer()
        x = [[porter.stem(w) for w in tw] for tw in x]
        return x
    
    def compute_bow_histograms(self, x, nb_words):
        # Take most common nb_words words and represent each
        # tweet as a histogram of word occurences
        freq = FreqDist([w for tw in x for w in tw])
        best_words, _ = zip(*freq.most_common(nb_words))
        x_bows = []
        for tw in x:
            bow = dict()
            for i in range(nb_words):
                cnt = tw.count(best_words[i])
                if cnt > 0:
                    bow[i] = cnt
            x_bows.append(bow)
        return x_bows
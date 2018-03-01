import numpy as np

from gensim import matutils
from gensim.models.ldamodel import LdaModel
from sklearn import linear_model
from sklearn.datasets import fetch_20newsgroups
from sklearn.feature_extraction.text import CountVectorizer


def print_features(clf, vocab, n=10):
    """ Print sorted list of non-zero features/weights. """
    coef = clf.coef_[0]
    print('positive features: %s' % (' '.join(['%s/%.2f' % (vocab[j], coef[j]) for j in np.argsort(coef)[::-1][:n] if coef[j] > 0])))
    print('negative features: %s' % (' '.join(['%s/%.2f' % (vocab[j], coef[j]) for j in np.argsort(coef)[:n] if coef[j] < 0])))


def fit_classifier(X, y, C=0.1):
    """ Fit L1 Logistic Regression classifier. """
    # Smaller C means fewer features selected.
    clf = linear_model.LogisticRegression(penalty='l1', C=C)
    clf.fit(X, y)
    return clf


def fit_lda(X, vocab, num_topics=5, passes=20):
    """ Fit LDA from a scipy CSR matrix (X). """
    print('fitting lda...')
    return LdaModel(matutils.Sparse2Corpus(X.T), num_topics=num_topics,
                    passes=passes,
                    id2word=dict([(i, s) for i, s in enumerate(vocab)]))


def print_topics(lda, vocab, n=10):
    """ Print the top words for each topic. """
    topics = lda.print_topics(num_topics=-1, num_words=n)
    for topic in topics:
        print(topic)


if __name__ == '__main__':
    # Load data.
    rand = np.random.mtrand.RandomState(8675309)
    cats = ['rec.sport.baseball', 'sci.crypt']
    data = fetch_20newsgroups(subset='train',
                              categories=cats,
                              shuffle=True,
                              random_state=rand)
    vec = CountVectorizer(min_df=10, stop_words='english')
    X = vec.fit_transform(data.data)
    vocab = vec.get_feature_names()

    # Fit classifier.
    clf = fit_classifier(X, data.target)
    print_features(clf, vocab)

    # Fit LDA.
    lda = fit_lda(X, vocab)
    print_topics(lda, vocab)

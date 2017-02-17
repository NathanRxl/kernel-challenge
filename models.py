from sklearn.linear_model import LogisticRegression as sk_LogisticRegression
from sklearn.svm import SVC
from sklearn.ensemble import RandomForestClassifier
from sklearn.tree import DecisionTreeClassifier


class LogisticRegression(sk_LogisticRegression):

    def __init__(self, **kwargs):
        super().__init__(**kwargs)


class SVM(SVC):

    def __init__(self, **kwargs):
        super().__init__(**kwargs)


class DecisionTree(DecisionTreeClassifier):

    def __init__(self, **kwargs):
        super().__init__(**kwargs)


class RandomForest(RandomForestClassifier):

    def __init__(self, **kwargs):
        super().__init__(**kwargs)
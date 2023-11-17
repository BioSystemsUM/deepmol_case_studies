from scipy.stats import spearmanr


def spearman(x, y):
    return spearmanr(x, y)[0]
import numpy as np


def TP(y_hat, y):
    return np.sum((y_hat == 1) & (y == 1))


def FP(y_hat, y):
    return np.sum((y_hat == 1) & (y == 0))


def TN(y_hat, y):
    return np.sum((y_hat == 0) & (y == 0))


def FN(y_hat, y):
    return np.sum((y_hat == 0) & (y == 1))


def accuracy(y_hat, y):
    """
    Calculates the accuracy of a prediction.

    Args:
        y_hat: The predicted values.
        y: The true values.

    Returns:
        The accuracy as a float between 0 and 1.
    """

    # YOUR CODE HERE
    tp = TP(y_hat, y)
    tn = TN(y_hat, y)
    fp = FP(y_hat, y)
    fn = FN(y_hat, y)

    return (tp + tn) / (tp + tn + fp + fn)


def f1_score_metric(y_hat, y):
    """
    Calculates the F1-score of a prediction.

    Args:
        y_hat: The predicted values.
        y: The true values.

    Returns:
        The F1-score as a float between 0 and 1.
    """

    pre = precision(y_hat, y)
    rec = recall(y_hat, y)

    return (2 * pre * rec) / (pre + rec)


def precision(y_hat, y):
    """
    Calculates the precision of a prediction.

    Args:
        y_hat: The predicted values.
        y: The true values.

    Returns:
        The precision as a float between 0 and 1.
    """

    tp = TP(y_hat, y)
    fp = FP(y_hat, y)

    return tp / (tp + fp)


def recall(y_hat, y):
    """
    Calculates the recall of a prediction.

    Args:
        y_hat: The predicted values.
        y: The true values.

    Returns:
        The recall as a float between 0 and 1.
    """

    tp = TP(y_hat, y)
    fn = FN(y_hat, y)

    return tp / (tp + fn)


def mse(y_hat, y):
    """
    Calculates the Mean Squared Error (MSE).

    Args:
        y_pred: The predicted values.
        y: The true values.

    Returns:
        The MSE.
    """

    return np.mean((y - y_hat) ** 2)


def rmse(y_hat, y):
    """
      Calculates the Root Mean Squared Error (RMSE).

      Args:
        y_hat: The predicted values.
        y: The true values.

      Returns:
        The RMSE.
    """

    return np.sqrt(mse(y_hat, y))


def mae(y_hat, y):
    """
    Calculates the Mean Absolute Error (MAE).

    Args:
        y_hat: The predicted values.
        y: The true values.

    Returns:
        The MAE.
    """

    return np.mean(np.abs(y - y_hat))


def r2_score(y_hat, y):
    """
    Calculates the R-squared score.

    Args:
        y_hat: The predicted values.
        y: The true values.

    Returns:
        The R-squared score.
    """

    return 1 - np.sum((y - y_hat) ** 2) / np.sum((y - np.mean(y)) ** 2)

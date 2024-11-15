# Part 1: Metrics for Evaluating Machine Learning Models

In this part of the lab, you will get familiar with the implementation of different evaluation metrics.

## 1. Definitions:
Provide definitions to the following terms. Next to your answer, write in parenthesis the source you used to answer (e.g. Wikipedia, a website, the references provided or none).

* **True positives**: a true positive (TP) is a correctly predicted positive outcome. (Rainio, Oona, Jarmo Teuho, and Riku Klén. "Evaluation metrics and statistical tests for machine learning." Scientific Reports 14.1 (2024): 6086)
* **True negatives**: a true negative (TN) is a correctly predicted negative outcome. (Rainio, Oona, Jarmo Teuho, and Riku Klén. "Evaluation metrics and statistical tests for machine learning." Scientific Reports 14.1 (2024): 6086)
* **False positives**: a false positive (FP) is a negative instance predicted to be positive. (Rainio, Oona, Jarmo Teuho, and Riku Klén. "Evaluation metrics and statistical tests for machine learning." Scientific Reports 14.1 (2024): 6086)
* **False negatives**: a false negative (FN) is a positive instance predicted to be negative. (Rainio, Oona, Jarmo Teuho, and Riku Klén. "Evaluation metrics and statistical tests for machine learning." Scientific Reports 14.1 (2024): 6086)
* **Confusion matrix**: 2x2 matrix containing the counts of TP, TN, FP, and FN observations. (Rainio, Oona, Jarmo Teuho, and Riku Klén. "Evaluation metrics and statistical tests for machine learning." Scientific Reports 14.1 (2024): 6086)

For the following terms, explain what they aim to measure along with the formula to estimate them. Next to your answer, write in parenthesis the source you used to answer (e.g. Wikipedia, a website, the references provided or none, if nothing was used).

* **Accuracy**: It is how close a given set of measurements (observations or readings) are to their true value. The formula is
$$\frac{TP + TN}{TP + TN + FP + FN}$$
(Wikipedia).
* **Precision**:Precision for a label is defined as the number of true positives divided by the number of predicted positives. It explains how many of the correctly predicted cases actually turned out to be positive. The formula is
$$\frac{TP}{TP + FP}$$
(<https://www.analyticsvidhya.com/blog/2021/07/metrics-to-evaluate-your-classification-model-to-take-the-right-decisions/#h-accuracy>).
* **Recall**: Recall for a label is defined as the number of true positives divided by the total number of actual positives.  It explains how many of the actual positive cases we were able to predict correctly with our model. The formula is
$$ \frac{TP}{TP + FN} $$
(<https://www.analyticsvidhya.com/blog/2021/07/metrics-to-evaluate-your-classification-model-to-take-the-right-decisions/#h-accuracy>).
* **F1-score**: F1 Score is the harmonic mean of precision and recall. It gives a combined idea about Precision and Recall metrics. It is maximum when Precision is equal to Recall. The formula is
$$\frac{2 \cdot (\text{precision} \cdot \text{recall})}{\text{precision} + \text{recall}}$$
(<https://www.analyticsvidhya.com/blog/2021/07/metrics-to-evaluate-your-classification-model-to-take-the-right-decisions/#h-accuracy>).
* **Mean squared error (MSE)**: Mean squared error (MSE) measures the amount of error in statistical models. It assesses the average squared difference between the observed and predicted values. The formula is
$$\frac{1}{n} \sum (y_{\text{true}} - y_{\text{pred}})^2$$
(<https://statisticsbyjim.com/regression/mean-squared-error-mse/>).
* **Root mean squared error (RMSE)**: Root mean square error or root mean square deviation is one of the most commonly used measures for evaluating the quality of predictions. It shows how far predictions fall from measured true values using Euclidean distance. The formula is
$$\sqrt{\frac{1}{n} \sum (y_{\text{true}} - y_{\text{pred}})^2}$$
(<https://statisticsbyjim.com/regression/mean-squared-error-mse/>).
* **Mean absolute error (MAE)**: Mean absolute error (MAE) measures the average magnitude of the errors in a set of predictions, without considering their direction. The formula is
$$ \frac{1}{n} \sum |y_{\text{true}} - y_{\text{pred}}| $$
(<https://www.deepchecks.com/glossary/mean-absolute-error/>)
* **Coefficient of determination ($R^2$)**: R-squared is a statistical metric for how much of the variation in the dependent variable can be attributed to the model. It may take on values between 0 and 1, with a greater value suggesting a more precise match between the model and the data. The formula is
$$ 1 - \frac{\sum (y_{\text{true}} - y_{\text{pred}})^2}{\sum (y_{\text{true}} - \bar{y}_{\text{true}})^2} $$
(<https://www.deepchecks.com/question/what-are-the-best-metrics-for-the-regression-model/>)

## 2. Implementation:
Open the file [metrics.py](./metrics.py) and implement the previously defined formulas in the respective function. You can run [test_script.py](./test_script.py) from a console to verify that your implementation is correct.

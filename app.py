import streamlit as st
import matplotlib.pyplot as plt
import numpy as np
from sklearn.utils import check_array

def smape(A, F):
    return A.size * np.sum(2 * np.abs(F - A) / (np.abs(A) + np.abs(F)))

def mape(y_true, y_pred): 
    # y_true, y_pred = check_array(y_true, y_pred)
    try:
        return np.mean(np.abs((y_true - y_pred) / y_true)) 
    except:
        return 'NAN'

def mse(A, F):
    return np.sum(np.abs(F - A) ** 2)


def mae(A, F):
    return np.sum(np.abs(F - A))


def mase(training_series, testing_series, prediction_series):
    """
    Computes the MEAN-ABSOLUTE SCALED ERROR forcast error for univariate time series prediction.
    
    See "Another look at measures of forecast accuracy", Rob J Hyndman
    
    parameters:
        training_series: the series used to train the model, 1d numpy array
        testing_series: the test series to predict, 1d numpy array or float
        prediction_series: the prediction of testing_series, 1d numpy array (same size as testing_series) or float
        absolute: "squares" to use sum of squares and root the result, "absolute" to use absolute values.
    
    """

    n = training_series.shape[0]
    d = np.abs(np.diff(training_series)).sum() / (n - 1)

    errors = np.abs(testing_series - prediction_series)
    return errors.mean() / d


st.title("ML Loss Explorer")
st.write("Understand your loss function")

# Sliders
# slider_mae = st.sidebar.slider("mae", 0, 1)
# slider_mse = st.sidebar.slider("mse", 0, 1)
# slider_mase = st.sidebar.slider("mase", 0, 1)
# slider_smape = st.sidebar.slider("smape", 0, 1)


# Section 1
st.header("Loss Calculator")
actual_slider = st.slider("Actual", 0, 10, step=1)
pred_slider = st.slider("Predict", -10, 10, step=1)
actual = np.array(actual_slider)
pred = np.array(pred_slider)
var_mae = mae(actual, pred)
var_mse = mse(actual, pred)
var_mape = mape(actual, pred)
# var_mase = mase(actual, pred)
smape_val = smape(actual, pred)
st.write("MAPE:", var_mape)
# st.write('MSE:', mse)
st.write("SMAPE:", smape_val)
# st.write('MSE:', mse)
fig, ax = plt.subplots(1, 2)
xs = np.linspace(-50, 50, 100)
actuals = np.repeat(actual, len(xs))
ys = []
for i , xi in enumerate(xs):
    ys.append(smape(actual, xi))
ys = np.array(ys)


print('ys',ys)
print(xs)
ax[0].plot(xs, ys)
ax[0].set_title('SMAPE')
ax[0].set_xlabel('prediction')


ys=[]
for i , xi in enumerate(xs):
    ys.append(mape(actual, xi))
ys = np.array(ys)
ax[1].plot(xs, ys)
ax[1].set_title('MAPE')
ax[1].set_xlabel('prediction')
st.write(fig)


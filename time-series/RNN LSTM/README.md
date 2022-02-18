# Air-temperature model for indoor climate



### Overview
With the goal of demonstrating data science techniques, this project focuses on building a RNN-LSTM model to forecast hourly indoor air temperatures over a 24-hour period. By preprocessing the input data and training the model with adam optimization, an RMSE of 0.60 is achieved, improving on a benchmark of 1.20. Those results are visualized and discussed below.

![#f03c15](https://via.placeholder.com/15/f03c15/000000?text=+) **Note:** This mini-project is intended for hobby and educational purposes, so some formatting and design protocols are ignored in the interest of readability. For example, the functions are spread out sequentially through the script and the default arguments are included in the function calls.



### Context
- What this is about
  - able to control/automate indoor temperatures
- Why use RNN LSTM (demonstration)
- Why this is useful (forecasting in general)



### Datasets and Inputs
The data provided for this research consists of CSV log files:
- weather.csv - Outdoor weather measurements
  - *time*: Timestamps (Excel format)
  - *Tout*: Outside temperature (°C)
  - *Rhout*: Outside relative humidity (%)
  - *Windsp*: Wind speed (m/s)
- climate.csv - Indoor climate measurements and hardware setpoints
  - *time*: Timestamps (Excel format)
  - *Tair*: Indoor air temperature (°C)
  - *Rhair*: Indoor relative humidity (%)
  - *VentLee*: Leeward air vents opening (%)
  - *VentWind*: Windward air vents opening (%)



### Software Requirements

This project uses **Python 3.8** and the following libraries:
- matplotlib==3.3.2
- numpy==1.19.3
- pandas==1.1.4
- seaborn==0.11.0
- tensorflow==2.7.0



### Run
In a terminal or command window, navigate to the 'RNN LSTM' project directory and run the command: python model.py



### Preprocessing
With the aim of making the script easier to test and also flexible enough to run with new data, the preprocessing is largely contained in functions. Some of the transformations include:
- Loading the CSVs into pandas DataFrames and merging the _weather_ and _climate_ data
- Checking there are no missing timestamps (thus also ensuring the two datasets run over identical timespans)
- Reviewing general statistics to assess the distribution and identify outliers. For example:
  - All values for Rhair, Rhout, VentLee, Ventwind are between 0-100
  - The means are sensible and logical for the fields they represent
  - No extreme max/mins that lie beyond a reasonable range
- Linearly interpolating null values (across comparatively short timespans) to ensure continous data
- Converting the excel datetime format to YYYY-MM-DD hh:mm:ss, with a date offset and rounded to the hour.
- Reviewing the general behaviour of some fields over the first few days of recorded data:
![plot_features](https://github.com/callumc789/data-science/blob/main/time-series/RNN%20LSTM/graphs/01%20plot_features.png)
- Converting dates and times to sine and cosine, to create a cyclical nature for the representation of time:
![datetime_sin_cos](https://github.com/callumc789/data-science/blob/main/time-series/RNN%20LSTM/graphs/02%20datetime_sin_cos.png)
- Splitting data chronologically into train/validation/test sets
- Normalizing the values so that the varying scales do not bias results towards certain fields:
![violin_plot_normed](https://github.com/callumc789/data-science/blob/main/time-series/RNN%20LSTM/graphs/03%20violin_plot_normed.png)



### Modelling
As an overview, this model works by feeding windows of input and label data into a recurrent neural network (RNN) using Long Short-Term Memory (LSTM) to learn the order and sequence of the input fields. By keeping information from the past and weighting certain features through the network nodes, the model can correlate data across a range of time, therefore enabling us to use current data (t<=0) to predict future data (t>0).

To predict the indoor temperature, _Tair_, we can isolate the features that are useful indicators (including previous values of _Tair_) to use as the model inputs.

With this approach, the results are shown below over a 24-hour period of hourly predictions, with a confidence interval from P10 to P90. Visually, we can observe that the predicted values lie close to the labels: ![plot_denorm](https://github.com/callumc789/data-science/blob/main/time-series/RNN%20LSTM/graphs/04%20plot_denorm.png)

This gives an RMSE of 0.60, an improvement from the benchmark of 1.20 from a baseline model (not plotted), indicating that the RNN-LSTM model has improved on the predictive capability of forecasting _Tair_.
The benchmark was calculated by simply using the last 24 hours of input data as the label data, i.e. using the previous day as a direct prediction for the following day.

To test the model on new data, one can update the filepaths for the _load_csv()_ function and run the script.



### Model Choices

#### RNN LSTM
After investigating temperature model research and finding a [paper](https://orca.cardiff.ac.uk/123835/1/Weng-Mourshed_camera_ready.pdf) modelling indoor temperatures with RNN-based LSTM time series forecasting, this approach was chosen in order to capture the complexities and multiple influences on temperature variations.

#### Model Parameters
- The mean square error is used as the loss function, directly related to the RMSE described below.
- Adam is an often-used and effective optimizer that can avoid problems such as locating local minima instead of minimizing the loss to the global minimum.
- Through some simple re-running of the script, a learning rate of 0.01 proved to minimize the loss efficiently.
- Setting patience=2 had the effect of allowing the model to keep training if the loss did not immediately decrease, but prevents the model running for too long without improvement.

#### RMSE
The root mean square error (RMSE) measures the average magnitude of the error between the predicted and actual values. Since the errors are squared before they are averaged, the RMSE gives higher weights to larger errors and thus penalizes large errors more than, for example, the mean absolute error. This means the RMSE is useful when large errors are particularly undesirable, such as extreme changes in indoor temperatures.



### Remaining Work
Given the purpose of demonstrating data science principles, focus was pointed towards creating a working model with some simple visualizations and a clear narrative structure. This included:
- Researching temperature models and influences
- Planning the model structure
- EDA
- Creating a basic script
- Optimizing performance
- Modularization
- Documentation

With this, sufficient attention has not yet been given to other topics, such as:
- Feature importance (and choosing the optimal input fields)
- Hyperparameter tuning and cross-validation, e.g. with GridSearchCV
  - Window size (# days for training)
  - Patience, learning rate, # epochs
  - Model layers, # LSTM layer nodes
- Reverting the time values on the final plot to datetime



### Limitations
Some of the shortcomings of this model include:
- The confidence intervals are approximated from the RMSE. Ideally, we would run the model several times to calculate the sample mean and standard deviation of each data point, from which we could then get the confidence interval as: mean ± z(std/sqrt(n))
- The input data does not cover a whole year, so the model is currently unaware of year-round seasonal trends. However, it could feasibly identify this with the current script, since there are also sin/cos fields for the day-of-the-year number.
- Precipitation is not factored into the inputs, which could potentially affect wind, outdoor humidity and outdoor temperature.


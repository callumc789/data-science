'''
Using a RNN LSTM model to forecast climate variables over 24 hours.
'''


import datetime
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
import pandas as pd
import tensorflow as tf


''' LOAD DATA '''


def load_csv(path=''):
    ''' Loads CSV file into pandas dataframe '''
    df = pd.read_csv(path)
    return df


def merge_select(df1, df2, merge_col='', cols=[]):
    ''' Inner join two dataframes into one, and keep only the specified columns'''
    df_m = df1.merge(df2, on=merge_col)
    df = df_m[cols].copy()
    return df


weather_path = 'data/weather.csv'
climate_path = 'data/climate.csv'
df_w = load_csv(weather_path)
df_g = load_csv(climate_path)

# take columns we believe to affect temperature
cols = ['time', 'Windsp', 'VentLee', 'Ventwind', 'Rhout', 'Rhair', 'Tout', 'Tair']
df = merge_select(df_w, df_g, merge_col='time', cols=cols)


''' CLEAN & TRANSFORM DATA '''


def EDA(df=df):
    ''' Basic EDA
    Checking for missing timestamps, inspecting dataset info, reviewing general stats'''

    # check if any timestamps are missing
    # convert time format to show minutes
    df_eda = df.copy()
    df_eda['time_mins'] = pd.to_datetime(df_eda['time'], unit='D', origin='1899-12-30').round('min')
    df_eda['timediff'] = df_eda['time_mins'].diff()
    print('\n timediff:')
    print(df_eda[df_eda['timediff'] != '0 days 00:05:00'])  # this only returns 1st row, so all rows are 5 mins apart

    # inspect dataset
    print('\n info:')
    print(df.info())

    # check spread of data (basic outlier detection)
    print('\n Describe')
    with pd.option_context('display.max_columns', 10):  # display all fields
        print(df.describe())


def replace_nulls(df_nulls=df):
    ''' Replaces nulls with linearly interpolated values '''

    print('\n nulls:')
    print(df_nulls[df_nulls.isna().any(axis=1)])  # returns 71 rows with NaN values

    if len(df_nulls[df_nulls.isna().any(axis=1)]) > 0:
        df_nulls.interpolate(inplace=True)  # linear interpolation for null values
    return df_nulls


def excel_to_datetime(df_time=df):
    ''' Converts Excel time format to datetime '''

    # round to hours, so model can predict the desired 1 hour intervals
    df_time['time'] = pd.to_datetime(df_time['time'], unit='D', origin='1899-12-30').round('H')
    df_time = df_time.groupby(['time'])[cols[1:]].mean().reset_index()  # Group rows by hour

    return df_time


plot_cols = ['Windsp', 'Rhout', 'Rhair', 'Tout', 'Tair']


def plot_features(df_plot=df.copy(), plot_cols=plot_cols, days=4):
    ''' Plots features over the first n specified days of the given time series data.
    Args:
        df_plot: pandas dataframe with time series data
        plot_cols: list of fields to plot
        days: # of days to plot
    '''

    plot_features = df_plot[plot_cols][:day * days]
    plot_features.index = df_plot['time'][:day * days]
    _ = plot_features.plot(subplots=True)
    plt.xlabel('Date')
    plt.show()


def datetime_sin_cos(df=df):
    ''' Transform time field into sine and cosine values for the hours and the day numbers.
    - Want model to know that the time loops around, i.e. that 23:00 and 00:00 are consecutive
    - Can convert datetime to sine wave, to have periodicity
    - Sine wave repeats values (e.g. sin(0) = sin(180) = 0), so also use cosine wave
    - Each point can then be uniquely described by using both values
    '''
    hour = df['time'].dt.hour  # hour of day
    df['day_sin'] = np.sin((hour/24) * 2 * np.pi)
    df['day_cos'] = np.cos((hour/24) * 2 * np.pi)

    day_num = df['time'].dt.dayofyear
    df['year_sin'] = np.sin((day_num/365) * 2 * np.pi)
    df['year_cos'] = np.cos((day_num/365) * 2 * np.pi)

    # times = df['time'].copy()  # store datetimes to use later
    df.drop(columns='time', inplace=True)

    plt.plot(np.array(df['day_sin'])[:day])  # plot for one day
    plt.plot(np.array(df['day_cos'])[:day])
    plt.xlabel('Time / hour')
    plt.title('Time-of-day Cycle')
    plt.show()

    return df


def train_test_val(df=df):
    ''' Split dataset into training, validation and testing sets.
    Args:
        df: pandas dataframe
    Returns:
        train_df: data used for training model
        val_df: data used for validating model performance
        test_df: data used for testing model performance
    '''

    n = len(df)
    train_df = df[0:int(n * 0.7)].copy()  # train on first 70% of data
    val_df = df[int(n * 0.7):int(n * 0.9)].copy()  # validate on next 20%
    test_df = df[int(n * 0.9):].copy()  # test on last (unseen) 10

    return train_df, val_df, test_df


EDA()
df = replace_nulls(df)
df = excel_to_datetime(df)
day = 24  # 24 hourly data points per day
plot_features(df)
df = datetime_sin_cos(df)
train_df, val_df, test_df = train_test_val(df)



''' NORMALIZATION '''


def normalize(train_df=train_df, val_df=val_df, test_df=test_df):
    ''' Normalize model input values to ensure scale does not influence results.
    Args:
        train_df, val_df, test_df: outputs of train_test_val() function
    Returns:
        train_df_norm, val_df_norm, test_df_norm: normalized datasets (shape preserved)
        train_mean: mean value of each training data field
        train_std: standard deviation of each training data field

    '''

    # Subtract mean and divide by standard deviation
    # To prevent data leakage, this step comes after train/val/test split and only uses train stats
    train_mean = train_df.mean()
    train_std = train_df.std()

    train_df_norm = (train_df - train_mean) / train_std
    val_df_norm = (val_df - train_mean) / train_std
    test_df_norm = (test_df - train_mean) / train_std

    return train_df_norm, val_df_norm, test_df_norm, train_mean, train_std


train_df, val_df, test_df, train_mean, train_std = normalize(train_df, val_df, test_df)


def violin_plot_normed(df=df, train_mean=train_mean, train_std=train_std):
    ''' Violin plot of normalized data, to show distribution of model input fields. '''

    df_norm = (df - train_mean) / train_std
    df_norm = df_norm.melt(var_name='Field', value_name='Normalized')
    plt.figure(figsize=(12, 6))
    ax = sns.violinplot(x='Field', y='Normalized', data=df_norm)
    _ = ax.set_xticklabels(df.keys(), rotation=90)
    plt.show()


violin_plot_normed()


''' MODEL INPUT FUNCTIONS '''


def split_window(window):
    '''Splits a window of data into inputs (beginning section of window) and labels (end section of window).
    Args:
        window: a slice (or stack of slices) of time series data
    Returns:
        inputs: the inputs slice of the window
        labels: the labels slice of the window
    '''

    inputs = window[:, input_slice, :]
    labels = window[:, label_slice, :]
    labels = tf.stack([labels[:, :, train_col_index[col]] for col in label_cols], axis=-1)

    inputs.set_shape([None, input_width, None])  # set shape for easier inspection if needed
    labels.set_shape([None, label_width, None])

    return inputs, labels


def tf_dataset(df):
    ''' Converts time series data to tensorflow dataset of (input window, label window) pairs using split_window().
    Args:
        df: time series dataframe
    Returns:
        tf_ds: a tensorflow "tf.data.Dataset" dataset object
    '''

    df_arr = np.array(df, dtype=np.float32)  # convert df to array

    # create dataset for a window with 1-step period intervals (i.e. every hour)
    tf_ds = tf.keras.utils.timeseries_dataset_from_array(data=df_arr, targets=None, sequence_length=total_window_size,
                                                      sequence_stride=1, shuffle=True, batch_size=32,)
    tf_ds = tf_ds.map(split_window)
    return tf_ds


''' MODEL INPUTS '''

label_cols = ['Tair']
label_col_index = {name: i for i, name in enumerate(label_cols)}  # numbering all the label columns
train_col_index = {name: i for i, name in enumerate(train_df.columns)}  # column index in the training data

input_width = day*10  # hours of training data
label_width = day
shift = day  # how far ahead to forecast
total_window_size = input_width + shift

input_slice = slice(0, input_width)  # slice object for the window input
input_range = np.arange(input_width)

label_start = total_window_size - label_width   # starting index of labels
label_slice = slice(label_start, None)  # slice object for the window label (at the end of the window)
label_range = np.arange(total_window_size - label_width, total_window_size)

# tensorflow datasets:
train = tf_dataset(train_df)
val = tf_dataset(val_df)
test = tf_dataset(test_df)


''' INSTANTIATE MODEL '''

epochs = 20
num_features = train_df.shape[1]

# group a stack of layers onto keras model with Sequential()
# LSTM layer: RNN LSTM algorithm
# Dense layer: deeply-connected hidden layer, transforms the input into a suitable output
# Reshape layer: reshapes inputs to desired output shape
model_LSTM = tf.keras.Sequential([
    tf.keras.layers.LSTM(32),
    tf.keras.layers.Dense(label_width*num_features, kernel_initializer=tf.initializers.zeros()),
    tf.keras.layers.Reshape([label_width, num_features])
])


''' PREDICT '''


def fit_model(model, patience=2):
    ''' Fits model on training data with defined loss function, optimizer and metrics.
    Args:
        model: keras model with defined layers
        patience: # epochs without loss improvement, after which training is stopped
    Returns:
        history: model fitting history across epochs
    '''

    # stop model training at min loss value (for validation data)
    early_stopping = tf.keras.callbacks.EarlyStopping(monitor='val_loss', patience=patience, mode='min')

    # define loss function, optimizer (and learning rate), metrics
    model.compile(loss=tf.losses.MeanSquaredError(), optimizer=tf.optimizers.Adam(0.01),
                  metrics=[tf.metrics.RootMeanSquaredError()])

    history = model.fit(train, epochs=epochs, validation_data=val, callbacks=[early_stopping])

    print('\n Model evaluation (validation set):')
    print(model_LSTM.evaluate(val))

    print('\n Model evaluation (test set):')
    print(model_LSTM.evaluate(test))

    return history  # preserves model fitting history


history = fit_model(model_LSTM)


''' PLOT RESULTS '''

''' Create windows to plot '''

win_train = np.array(train_df[:total_window_size])
win_val = np.array(val_df[:total_window_size])
win_test = np.array(test_df[:total_window_size])

plot_window = tf.stack([win_train, win_val, win_test])
plot_inputs, plot_labels = split_window(plot_window)  # split windows into input and label fields


''' Denormalize y-values '''
n = 0


def denormalize(n=n, plot_col='Tair'):
    '''Denormalizes the inputs and outputs for the label field, for window with index n.
    Args:
        n: window index (from stacked list of windows)
        plot_col: which label field to plot
    Returns:
        inputs_denorm: denormalized inputs
        labels_denorm: denormalized labels
        preds_denorm: denormalized predictions
        rmse: the recalculated RMSE for the denormalized model predictions
        ci_low: P10 prediction interval
        ci_high: P90 prediction interval
    '''

    # denormalization: (value * STD) + mean

    # INPUT
    inputs = plot_inputs.numpy()[n, :, train_col_index[plot_col]]  # normalized plot_col inputs
    inputs_denorm = [(i*train_std[train_col_index[plot_col]]) + train_mean[train_col_index[plot_col]] for i in inputs]

    # LABELS
    labels = plot_labels.numpy()[n, :, label_col_index[plot_col]]  # normalized plot_col labels
    labels_denorm = [(i*train_std[train_col_index[plot_col]]) + train_mean[train_col_index[plot_col]] for i in labels]

    # PREDICTIONS
    preds = model_LSTM(plot_inputs).numpy()[n, :, label_col_index[plot_col]]  # normalized plot_col predictions
    preds_denorm = [(i * train_std[train_col_index[plot_col]]) + train_mean[train_col_index[plot_col]] for i in preds]


    # PREDICTION INTERVAL
    # 90% confidence -> z = 1.645
    # CI = z * RMSE
    # Need to recalculate RMSE, because model RMSE is from normalized data
    # RMSE = ( SUM( (y_pred_i - y_actual_i)**2) / n )**0.5
    rmse_list = []
    for i in range(len(labels)):
        rmse_list.append((preds_denorm[i] - labels_denorm[i])**2)
    rmse = (sum(rmse_list)/len(labels))**0.5

    ci_low  = [preds_denorm[i] - (1.645 * rmse) for i in range(len(preds_denorm))]  # P10 CI
    ci_high = [preds_denorm[i] + (1.645 * rmse) for i in range(len(preds_denorm))]  # P90 CI

    return inputs_denorm, labels_denorm, preds_denorm, rmse, ci_low, ci_high


def plot_denorm(plot_col='Tair', subplots=len(plot_inputs)):
    ''' Plots the de-normalized input, label and predicted values, with a subplot per window.
    Args:
        plot_col: which label field to plot
        subplots: # windows to plot, with 1 subplot per window
    '''

    plt.figure(figsize=(12, 8))
    for n in range(subplots):
        plt.subplot(subplots, 1, n + 1)
        plt.gca().set_title(f'Window {n+1}')

        if plot_col=='Tair':
            plt.ylabel(f'{plot_col} / $^\circ$C')  # degrees symbol
        elif plot_col=='Rhair':
            plt.ylabel(f'{plot_col} (%)')

        inputs_denorm, labels_denorm, preds_denorm, rmse, ci_low, ci_high = denormalize(n=n, plot_col=plot_col)

        plt.plot(input_range, inputs_denorm, label='Input')  # inputs
        plt.plot(label_range, labels_denorm, label='Label')  # labels
        plt.plot(label_range, preds_denorm, label='Prediction')  # predictions
        plt.fill_between(label_range, ci_low, ci_high, color='b', alpha=0.2)  # P10-P90

        if n == 0:
            plt.legend(loc='upper left')

    plt.xlabel('Time / hours')
    plt.show()


plot_denorm()


''' BASELINE MODEL COMPARISON '''
# Basic model, use the last 24 hours as the prediction (i.e. repeat the last day)

baseline_input, baseline_preds, model_preds, model_rmse = denormalize(n=0, plot_col='Tair')[0:4]  # using the first window
baseline_input = baseline_input[-len(baseline_preds):]  # take length of "predictions", i.e last day

baseline_rmse_list = []
for i in range(len(baseline_preds)):
    baseline_rmse_list.append((baseline_input[i] - baseline_preds[i]) ** 2)
baseline_rmse = (sum(baseline_rmse_list) / len(baseline_preds)) ** 0.5

print('\n Baseline RMSE: ', baseline_rmse)
print('\n Model RMSE: ', model_rmse)


''' OPTIMIZATION '''
# Hyperparameter tuning:
# window size (# days for training)
# epochs & patience
# LSTM layer nodes, model layers, learning rate
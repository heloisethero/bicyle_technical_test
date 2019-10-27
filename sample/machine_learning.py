"""This script generates the different tables and figures of
the machine learning part."""


from sklearn.linear_model import Ridge
from sklearn.model_selection import train_test_split
from sklearn.model_selection import cross_val_score

import numpy as np
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt


def get_shuffled_data(file_name, path):
    """From the dataframe with the predictors and their features, returns the
    dataframe with the predictors (df) and the variable to predict (y)"""
    df = pd.read_csv(path + file_name)
    df = df.sample(frac=1, random_state=0)
    y = df["count"].values
    return df, y


def get_one_hot_encoder(colname, df):
    class_dummies = pd.get_dummies(df[colname], prefix='split_' + colname)
    df = df.join(class_dummies)
    del df[colname]
    return df


def get_predictors(df, columns_predictors, columns_to_encode):
    """From the data, returns the matrix of predictors without the additional features,
    with the categorical ones transformed through the one-hot encoder."""
    X = data[columns_predictors]
    for column_index in columns_to_encode:
        X = get_one_hot_encoder(column_index, X)
    return X.values


def get_score(X, y, model):
    """Get the r2 score of a ridge model from the predictors and the variable to predict."""
    r2 = cross_val_score(model, X, y, scoring="r2", cv=5)
    return np.mean(r2)


def save_scores(scores, table_name, path):
    """Save the table with the scores of the two models on slide 15"""
    df = pd.DataFrame(data={'Model without features': [scores[0]],
                            'Model with features': [scores[1]]})
    df.to_csv(path + table_name)


def plot_real_predicted_y(y, y_hat):
    """Plot the real values the predicted value distributions"""
    bins = np.linspace(0, 1000, 100)
    plt.hist(y, bins, alpha=0.5, label='Real values', color="slateblue")
    plt.hist(y_hat, bins, alpha=0.5, label='Predicted values', color="orange")
    plt.legend(loc='upper right')
    plt.xlim(0, 1000)
    plt.yticks((0, 500, 1000))


def fit_the_model(X, y, model):
    """Train the model and make predictions"""
    fitted_model = model.fit(X, y)
    y_hat = fitted_model.predict(X)
    return fitted_model, y_hat


def plot_residuals(y, y_hat):
    """Plot the distribution of the real minus predicted values"""
    bins = np.linspace(-500, 500, 100)
    plt.hist(y - y_hat, bins, alpha=0.5, color="limegreen",
             label='Residuals \n(real values - \npredicted values)')
    plt.legend(loc='upper right')
    plt.xlim(-500, 500)
    plt.yticks((0, 200, 400, 600))


def save_prediction_histogram(y, y_hat, figure_name, path):
    """Save the prediction histogram figure on slide 15"""
    plt.figure(num=None, figsize=[12.8, 3], dpi=100, facecolor='w', edgecolor='k')
    plt.subplot(1,2,1)
    plot_real_predicted_y(y, y_hat)
    plt.subplot(1,2,2)
    plot_residuals(y, y_hat)
    plt.savefig(path + figure_name, transparent=True)


def save_parameter_hour(data, model, figure_name, path):
    """Save the parameter and hour effect figure on slide 16"""
    dataToPlot = []
    for value in np.sort(data["hour"].unique()):
        dataToPlot.append(data[data.hour == value]["count"].median())
    fig, ax1 = plt.subplots(figsize=(10, 4))
    ax2 = ax1.twinx()
    ax2.plot(dataToPlot, color='slateblue')
    ax1.plot(model.coef_[55:], color='orange')
    ax1.set_xlabel('Hour', size='large')
    ax2.set_ylabel('Count (median)', color='slateblue', size='large')
    ax1.set_ylabel('Parameters of the linear model', color='darkorange', size='large')
    plt.xticks(np.arange(0,24))
    plt.savefig(path + figure_name, transparent=True)


def save_predicted_interaction(data, y_hat, figure_name, path):
    """"""
    data["count_predicted"] = pd.Series(y_hat, index=data.index)

    plt.figure(figsize=[10, 4], dpi=100, facecolor='w', edgecolor='k')
    labels = ["workingday", "weekend", "holiday"]
    colors = ["slateblue", "limegreen", "orange"]

    for specialday_value in np.sort(data["specialday"].unique()):
        data_tmp = data[data.specialday == specialday_value]
        dataToPlot = []
        for value in np.sort(data["hour"].unique()):
            dataToPlot.append(data_tmp[data_tmp.hour == value]["count_predicted"].median())
        plt.plot(dataToPlot, label=labels[specialday_value], color=colors[specialday_value])

    plt.xticks(tuple(range(data["hour"].nunique())),
               tuple(data["hour"].unique()))
    plt.xlabel("Hour")
    plt.ylabel("Predicted count (median)")
    plt.ylim(-20, 500)
    plt.legend(loc='upper left')

    plt.savefig(path + figure_name, transparent=True)


if __name__ == "__main__":
    DATA_PATH = "./data/"
    OUTPUT_PATH = "./graphs/"

    data, y = get_shuffled_data(file_name="bike_data_with_features.csv", path=DATA_PATH)
    model = Ridge(normalize=True, alpha=0.01)

    X_1 = get_predictors(df=data,
                         columns_predictors=["season", "holiday", "workingday",
                                             "weather", "temp", "atemp", "humidity", "windspeed"],
                         columns_to_encode=['season', 'weather'])
    score_1 = get_score(X_1, y, model)
    X_2 = get_predictors(df=data,
                         columns_predictors=["season", "weather", "temp",
                                             "atemp", "humidity", "windspeed",
                                             "day", "month", "year", "hour",
                                             "weekday", "specialday"],
                         columns_to_encode=['season', 'specialday', 'weather',
                                            'day', 'month', 'year', 'weekday', 'hour'])
    score_2 = get_score(X_2, y, model)
    save_scores(scores=[score_1, score_2], path=OUTPUT_PATH,
                table_name="slide15_performance-table.csv")

    fitted_model, y_hat = fit_the_model(X=X_2, y=y, model=model)
    save_prediction_histogram(y=y, y_hat=y_hat,
                              figure_name="slide15_prediction-histogram.png", path=OUTPUT_PATH)
    save_parameter_hour(data=data, model=fitted_model,
                        figure_name="slide16_prediction-histogram.png", path=OUTPUT_PATH)
    save_predicted_interaction(data=data, y_hat=y_hat,
                               figure_name="slide17_predicted-interaction.png", path=OUTPUT_PATH)

    print("The analysis is done :)")
    print("You can go see the graphics in the", OUTPUT_PATH, "folder.")

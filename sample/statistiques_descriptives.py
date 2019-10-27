"""This script generates the different tables and figures of
the descriptive statistics part."""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt


def save_column_description(df, table_name, path):
    """Create the colmun summary table on slide 3"""
    column_description = pd.concat([bike_df.min(axis=0), bike_df.max(axis=0), bike_df.median(axis=0)],
                                   axis=1, sort=False)
    column_description = column_description.rename(columns={0: "minimum", 1: "maximum", 2: "median"}).T
    column_description.to_csv(path + table_name, header=True)


def save_count_histogram(df, figure_name, path):
    """Create the count histogram on slide 4"""
    plt.figure(figsize=[12.8, 4.5], dpi=100, facecolor='w', edgecolor='k')
    plt.hist(df["count"].values, 100, color="slateblue")
    plt.plot([145, 145], [0, 1300], color="orange", linewidth=2, linestyle='dashed')
    plt.xlim(0, 1000)
    plt.xlabel("Values")
    plt.ylabel("Frequency")
    plt.title("Histogram of the count variable")
    plt.savefig(path + figure_name, transparent=True)


def plot_with_boxplot(df, columnName, xTickNames, nb=0, sigma=0.01, x_label=""):
    """Create the template for the boxplot subfigures"""
    # This is to plot the data points cloud around the box plot
    plt.plot(df[columnName].values.astype(int) - nb
             + np.random.normal(0, sigma, len(bike_df)),
             df["count"].values, color='slateblue',
             marker='.', markersize=1, linestyle='', alpha=0.1)

    # This is to plot the box plot
    dataToPlot = []
    for value in np.sort(df[columnName].unique()):
        dataToPlot.append(df[bike_df[columnName] == value]["count"].values)
    plt.boxplot(dataToPlot, showfliers=False)

    plt.xticks(tuple(range(1, df[columnName].nunique()+1)), xTickNames)
    plt.xlabel(x_label)
    plt.ylim(-50, 1000)
    plt.yticks((0, 250, 500, 750, 1000))


def save_categorical_boxplot(df, figure_name, path):
    """Create the boxplots showing the effects of categorical predictors on slide 5"""
    plt.figure(figsize=[12.8, 5.2], dpi=100, facecolor='w', edgecolor='k')

    plt.subplot(1,4,1)
    plot_with_boxplot(df=df, columnName="season",
                    xTickNames=('winter', 'spring', 'summer', 'fall'),
                    sigma = 0.03, x_label = "Season")
    plt.ylabel('Count')

    plt.subplot(1,4,2)
    plot_with_boxplot(df=df, columnName="holiday",
                    xTickNames=('not a\nholiday', 'holiday'),
                    nb = -1, sigma = 0.01)
    plt.yticks((0, 250, 500, 750, 1000), (''))

    plt.subplot(1,4,3)
    plot_with_boxplot(df=df, columnName="workingday",
                    xTickNames=('not a\nworkingday', 'workingday'),
                    nb = -1, sigma = 0.01)
    plt.yticks((0, 250, 500, 750, 1000), (''))

    plt.subplot(1,4,4)
    plot_with_boxplot(df=df, columnName="weather",
                    xTickNames=('clear', 'mist', 'light\nrain', 'heavy\nrain'),
                    sigma = 0.03, x_label = "Weather")
    plt.yticks((0, 250, 500, 750, 1000), (''))

    plt.savefig(path + figure_name, transparent=True)


def plot_with_glidding_average(df, columnName, x_label=""):
    """Create the scatterplot with glidding average template for the subfigures"""
    df_tmp = df.sort_values(by=[columnName])
    plt.plot(df_tmp[columnName].values,
             df_tmp["count"].rolling(500, center=True).median().values,
             color="orange", linewidth=2)
    plt.plot(df[columnName].values, df["count"].values,
             color='slateblue', marker='.', markersize=1, linestyle='', alpha=0.2)
    plt.xlabel(x_label)
    plt.ylim(-50, 1000)
    plt.yticks((0, 250, 500, 750, 1000))


def save_continuous_scatterplot(df, figure_name, path):
    """Create the scatterplots with glidding average on slide 6"""
    plt.figure(figsize=[12.8, 4.5], dpi=100, facecolor='w', edgecolor='k')

    plt.subplot(1,4,1)
    plot_with_glidding_average(df, 'temp', x_label="Temperature (°C)")
    plt.ylabel('Count')

    plt.subplot(1,4,2)
    plot_with_glidding_average(df, 'atemp', x_label="'Feels like' temperature (°C)")
    plt.yticks((0, 250, 500, 750, 1000), (''))

    plt.subplot(1,4,3)
    plot_with_glidding_average(df, 'humidity', x_label="Relative humidity")
    plt.yticks((0, 250, 500, 750, 1000), (''))

    plt.subplot(1,4,4)
    plot_with_glidding_average(df, 'windspeed', x_label="Wind speed")
    plt.yticks((0, 250, 500, 750, 1000), (''))

    plt.savefig(path + figure_name, transparent=True)


def save_correlation_matrix(df, figure_name, path):
    """Create the correlation matrix between the continuous variables on slide 7"""
    plt.figure(figsize=[7, 6], dpi=100, facecolor='w', edgecolor='k')

    labels = ['count', 'temp', 'atemp', 'humidity', 'windspeed']
    corrMatrix = np.corrcoef(df[labels].values.T)
    for i in range(len(labels)):
        for j in range(len(labels)):
            if i < j:
                corrMatrix[i, j] = 0

    ax = plt.gca()
    im = ax.imshow(corrMatrix, cmap='PuOr', vmin=-1, vmax=1)
    cbar = ax.figure.colorbar(im, ax=ax, shrink=0.94)
    cbar.ax.set_ylabel(r'$\rho$', rotation=0)
    for i in range(len(labels)):
        for j in range(len(labels)):
            if (i, j) == (4, 0) or (i, j) == (3, 1) or (i, j) == (4, 1) or (i, j) == (3, 2) or (i, j) == (4, 2):
                colorText = "k"
            else:
                colorText = "w"
            plt.text(j, i, np.round_(corrMatrix[i, j], decimals=2),
                     ha="center", va="center", color=colorText)
    plt.yticks(range(len(labels)), tuple(labels))
    plt.xticks(range(len(labels)), tuple(labels), rotation=45, ha="right")

    plt.savefig(path + figure_name, transparent=True)


def extract_from_datetime(df):
    """Extract different features from the datetime column such as the year, the month,
    the day of the month, the hour, the time number and the day of the week"""

    df["year"] = df["datetime"].apply(lambda x: x.split(" ")[0].split("-")[0])
    df["month"] = df["datetime"].apply(lambda x: x.split(" ")[0].split("-")[1])
    df["day"] = df["datetime"].apply(lambda x: x.split(" ")[0].split("-")[2])
    df["hour"] = df["datetime"].apply(lambda x: x.split(" ")[1].split(":")[0])

    # Now we extract the time number, i.e., the number of hours that have passed since the 2011-01-01 00:00:00
    day_per_month = [31, 28, 31, 30, 31, 30, 31, 31, 30, 31, 30, 31]
    df["timenumber"] = ((df["year"].astype(int)-2011) * 365 * 24 +
                         df["month"].astype(int).
                         apply(lambda x:sum(day_per_month[0:x-1])) * 24 +
                         (df["day"].astype(int) - 1) * 24 +
                         df["hour"].astype(int))
    # Here we compensate for the fact that 2012 is a leap year
    df.loc[df["datetime"] >= "2012-02-29 00:00:00", 'timenumber'] = df["timenumber"] + 24

    # From the time number, we can compute the day of the week for each datetime
    df["weekday"] = ((np.floor(df["timenumber"]/24) + 5) % 7).astype(int)

    return df


def save_count_across_time(df, figure_name, path):
    """Create the count across time figure on slide 8"""
    plt.figure(figsize=[12.8, 4.5], dpi=100, facecolor='w', edgecolor='k')
    plt.plot(df["timenumber"].values, df["count"].values,
             color='slateblue', marker='.', markersize=1, linestyle='', alpha=0.5)
    plt.plot(df["timenumber"].values, df["count"].rolling(500, center=True).median().values,
             color="orange", linewidth=2)
    plt.xticks((0, 2880, 5832, 8760, 11664, 14616, 17232),
               ("2011-Jan", "2011-May", "2011-Sep", "2012-Jan",
                "2012-May", "2012-Sep", "2012-Dec"))
    plt.ylabel('Count')
    plt.xlabel('Datetime')
    plt.yticks((0, 250, 500, 750, 1000))
    plt.savefig(path + figure_name, transparent=True)


def save_datetime_boxplot(df, figure_name, path):
    """Create the datetime features boxplots on slide 9"""
    plt.figure(figsize=[12.8, 10], dpi=100, facecolor='w', edgecolor='k')

    plt.subplot(3, 1, 1)
    plot_with_boxplot(df, columnName="hour",
                      xTickNames=tuple(bike_df["hour"].unique()),
                      sigma=0.08, nb=-1, x_label="Hour")
    plt.ylabel('Count')

    plt.subplot(3, 1, 2)
    plot_with_boxplot(df, columnName="day",
                      xTickNames=tuple(bike_df["day"].unique()),
                      sigma=0.08, x_label="Day of the month")
    plt.ylabel('Count')

    plt.subplot(3, 2, 5)
    plot_with_boxplot(df, columnName="month",
                      xTickNames=('Jan', 'Feb', 'Mar', 'Apr', 'May', 'Jun',
                                  'Jul', 'Aug', 'Sep', 'Oct', 'Nov', 'Dec'),
                      sigma =0.05, x_label="Month")
    plt.ylabel('Count')

    plt.subplot(3, 6, 16)
    plot_with_boxplot(df, columnName="year", xTickNames=('2011', '2012'), nb=2010, x_label="Year")
    plt.yticks((0, 250, 500, 750, 1000), (''))

    plt.subplot(3, 3, 9)
    plot_with_boxplot(df, columnName="weekday", xTickNames=("Mon", "Tue", "Wed", "Thu", "Fri", "Sat", "Sun"),
                      nb=-1, sigma=0.06, x_label="Day of the week")
    plt.yticks((0, 250, 500, 750, 1000), (''))

    plt.savefig(path + figure_name, transparent=True)


def extract_specialday(df):
    """Extract the specialday feature from the workingday and holiday columns"""
    df["specialday"] = df["holiday"] - df["workingday"] + 1
    return df


def save_specialday_table(df, table_name, path):
    """Create the table showing the relationship between
    the variables workingday, holiday and specialday"""
    df = df[["workingday", "holiday", "specialday"]].drop_duplicates().sort_values("specialday")
    df.to_csv(path + table_name, header=True, index=False)


def save_hour_day_interaction(df, figure_name, path):
    """Create the hour x specialday interaction figure on slide 10"""
    plt.figure(figsize=[12, 3.8], dpi=100, facecolor='w', edgecolor='k')

    labels = ["workingday", "weekend", "holiday"]
    colors = ["slateblue", "limegreen", "orange"]
    for specialday_value in np.sort(df["specialday"].unique()):
        df_tmp = df[df.specialday == specialday_value]
        dataToPlot = []
        for value in np.sort(df["hour"].unique()):
            dataToPlot.append(df_tmp[df_tmp.hour == value]["count"].median())
        plt.plot(dataToPlot,
                 label=labels[specialday_value],
                 color=colors[specialday_value])

    plt.xticks(tuple(range(bike_df["hour"].nunique())),
               tuple(bike_df["hour"].unique()))
    plt.xlabel("Hour")
    plt.ylabel("Count (median)")
    plt.ylim(-20, 600)
    plt.yticks((0, 200, 400, 600))
    plt.legend(loc='upper left')

    plt.savefig(path + figure_name, transparent=True)


def save_df_with_features(df, table_name, path):
    """Save the dataframe with the extracted features to be used by the linear model"""
    list_predictors_to_keep = ['season', 'holiday', 'workingday', 'weather', 'temp',
                               'atemp', 'humidity', 'windspeed', 'day', 'month',
                               'year', 'hour', 'weekday', 'specialday', 'count']
    bike_df[list_predictors_to_keep].to_csv(path + table_name, index=False)


if __name__ == "__main__":
    DATA_PATH = "./data/"
    OUTPUT_PATH = "./output/"

    bike_df = pd.read_csv(DATA_PATH + "bike_data.csv")
    save_column_description(df=bike_df, table_name="slide03_column-description.csv", path=OUTPUT_PATH)
    save_count_histogram(df=bike_df, figure_name="slide04_count-histogram.png", path=OUTPUT_PATH)
    save_categorical_boxplot(df=bike_df, figure_name="slide05_categorical-boxplot.png", path=OUTPUT_PATH)
    save_continuous_scatterplot(df=bike_df, figure_name="slide06_continuous-scatterplot.png", path=OUTPUT_PATH)
    save_correlation_matrix(df=bike_df, figure_name="slide07_correlation-matrix.png", path=OUTPUT_PATH)

    bike_df_with_datetime_features = extract_from_datetime(df=bike_df)
    save_count_across_time(df=bike_df_with_datetime_features,
                           figure_name="slide08_count-across-time.png", path=OUTPUT_PATH)

    save_datetime_boxplot(df=bike_df_with_datetime_features,
                          figure_name="slide09_datetime-boxplots.png", path=OUTPUT_PATH)

    bike_df_with_datetime_specialday_features = extract_specialday(df=bike_df_with_datetime_features)
    save_specialday_table(df=bike_df_with_datetime_specialday_features,
                          table_name="slide10_specialday-table.csv", path=OUTPUT_PATH)
    save_hour_day_interaction(df=bike_df_with_datetime_specialday_features,
                              figure_name="slide10_hour-day-interaction.png", path=OUTPUT_PATH)

    save_df_with_features(df=bike_df_with_datetime_specialday_features,
                          table_name="bike_data_with_features.csv", path=DATA_PATH)

    print("The statistical analysis is done :)")
    print("You can go see the graphics in the", OUTPUT_PATH, "folder.")

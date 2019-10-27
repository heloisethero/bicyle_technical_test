# The bicycle technical test

## Context

This repository contains the technical test I did to apply for a data science consulting company. The goal of the test was to evaluate competences in development and data science, and more precisely the ability to do descriptive statistics followed by a predictive model.

The dataset is based on the [_Bike Sharing Demand_ Kaggle competition](https://www.kaggle.com/c/bike-sharing-demand). The data is about a shared bike system (as the Velib system in Paris). The goal of the exercise was two-fold:
* First describe the dataset and which factors seem to influence the bike demand with a few graphics
* Then train a model to predict the `count` variable (the number of bikes rented per hour), and describe its performance

At the end of the technical test, I was supposed to send a folder with my codes and a presentation of about 10 slides. You can see my presentation in the `docs` folder (it is named _bicycle_exercice_results.pdf_).

## My approach

I wanted to spend not that much time on the statistical description but when I started digging around, I found many interesting things and I actually ended up arranging a lot the graphics for the presentation.

Because I wanted time to clean the code at the end, I was left with only a few hours to train the model. I decided to do a one-hot-encoder on many variables and try a linear model because I have read [here](https://www.eyrolles.com/Informatique/Livre/data-science-fondamentaux-et-etudes-de-cas-9782212142433/) that such model could perform quite well to predict a time series variable. I was happy to obtain a R2 of 0.7 and decided to keep this model. I regretted a bit not having time to run a random forest or an XGBoost.

I wanted to show my ability to deliver clean code so I spent a lot of time in the end creating functions with docstrings. I also created a docker service so my code could be run on different OS and also because I wanted to try Docker on a small project like this.

## How to run the code

There are two ways to run the code: either directly on python, or within a docker.

### with Python

This code was developed and tested with Python 3.6.5. If necessary you need to install Python.

First you need to install the libraries described in the requirements.txt, for example through pip with this command:
```
pip install -r requirements.txt
```

To have the figures and the tables corresponding to the first part of the presentation, you need to launch:
```
python ./sample/statistiques_descriptives.py
```
You will see the output files (the graphs and tables from slides 3 to 10) appear in the `graphs` folder.

For the figures and the tables corresponding to the second part, this is the command to launch:
```
python ./sample/machine_learning.py
```
You will see the output files (the graphs and tables from slides 15 to 17) appear in the `graphs` folder.

To run both scripts at once, you can use this command:
```
./main.sh
```

### with Docker

Docker allows to the code to run without needing to install Python locally.

First you need to install Docker if necessary.

To run the code, you need to first build the docker:
```
docker-compose -f docker-compose.yml build python-test-bicycle
```

Then the code will be run with this command:
```
docker-compose -f docker-compose.yml up -d
```
You will then see all the output files in the `graphs` folder.

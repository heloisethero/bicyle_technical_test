# A technical challenge I did for a Data Scientist application, in Python (pandas, matplotlib, scikit-learn) and Docker

### Context

This repository contains the technical challenge I did to apply as a data scientist for a consulting company. I had two weeks to do it, and I could only work on it at nights and on week-ends because I was working at another job back then. I also had no internet at the time, so I had to crash at my friends' house to work on it (that part was actually quite fun).

The dataset is based on the [_Bike Sharing Demand_ Kaggle competition](https://www.kaggle.com/c/bike-sharing-demand). The data is about a shared bike system (as the Velib system in Paris). The goal of the test was to evaluate competences in development and data science. More precisely it was two-fold:
* First describe the dataset and which factors seem to influence the bike demand with a few graphics,
* Then train a model to predict the `count` variable (the number of bikes rented per hour), and describe its performance.

At the end of the technical test, I was supposed to send a folder with my code and a presentation of about 10 slides. You can see my presentation [here](https://github.com/heloisethero/bicyle_technical_test/blob/master/docs/bicycle_exercise_results.pdf) (Yes I know it is more than 10 slides, I tend to talk too much...).

### My approach

I had planned to spend not too much time on the statistical description part. But when I started digging around, I actually found many interesting things, and ended up working a lot on the graphics. Because the job was for a consulting compagny, I really wanted my presentation to be clear, interesting and nice to see.

I was finally left with only a few hours to train the model. I decided to try a linear model while doing a one-hot-encoder on many variables because I have read [in this book](https://www.eyrolles.com/Informatique/Livre/data-science-fondamentaux-et-etudes-de-cas-9782212142433/) that such model could perform quite well to predict a time series variable. I was happy to obtain a R2 of 0.7 and decided to keep this model. I regretted a bit not having time to run a random forest or an XGBoost.

I wanted to also show my ability to deliver clean code so I spent a lot of time in the end creating small functions with docstrings. I also wanted to try Docker on this project, because I had never used it on my own before.

In the end I did not get this job, but I am still proud of the code.

### How to run the code

There are two ways to run the code: either directly on python, or within a docker. But first you should get the repo:

```
git clone https://github.com/heloisethero/bicyle_technical_test
cd bicyle_technical_test
```

#### with Python

This code was developed and tested with Python 3.6.5. If necessary you need to install Python.

You should install the libraries described in the requirements.txt, for example through pip:
```
pip install -r requirements.txt
```

To have the figures and the tables corresponding to the first part of the presentation, you need to run:
```
python ./sample/statistiques_descriptives.py
```
You will see the output files (the graphs and tables from slides 3 to 10) appear in the `output` folder.

For the figures and the tables corresponding to the second part, this is the command:
```
python ./sample/machine_learning.py
```
You will see the output files (the graphs and tables from slides 15 to 17) appear in the `output` folder.

To run both scripts at once, you can use this:
```
./main.sh
```

#### with Docker

Docker allows you to run the code without needing to install Python locally.

You need to first install Docker if necessary, then build the docker image:
```
docker build -t python-test-bicycle .
```

Finally running it will create the figures and tables in the `output` folder:
```
docker run -v "$(pwd)"/output:/app/output -d python-test-bicycle
```

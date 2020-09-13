# lstm-time-series-forecasting
Description:
These are two LSTM neural networks that perform time series forecasting for a household's energy consumption
The first performs prediction of a variable in the future given as input one variable (univariate).
The second performs prediction of a variable in the future given as input three variables (multivariate). 

Dependencies:

    Python (3.6)
    Tensorflow (2.1.0)
    Keras (2.3.1)
    Pillow (7.1.1)
    matplotlib (3.2.1)
    np (1.0.2)
    scikit-learn (0.22.2.post)
    numpy (1.18.2)
    os

Instructions to run the program:

    1. Clone the project from: https://github.com/gdimitriou/lstm-time-series-forecasting.git
    2. Import it to your favorite IDE
    3. Download the dependencies
    4. Download the dataset from: https://www.kaggle.com/uciml/electric-power-consumption-data-set
    5. Rename the dataset as household_power_consumption.csv
    6. Import it under project's parent directory
    7. Run uni-variate.py or multy-variate.py

Expecting output:
    
    1. For the univariate model, you may see a graph of the predicted variable (Global_reactive_power):
    
        1. the immediate next time.
        2. a timeseries of it, 12 hr in the future
    
    2. For the multivariate model, you may see a graph of the predicted variable (Global_reactive_power) 12 hr in the future


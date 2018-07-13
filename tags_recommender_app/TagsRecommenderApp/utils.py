import pandas as pd
import logging as lg
import numpy as np
import pdb

from .models import load_joblib, load_data, Airports, Origins, Dests

def airport_list():
    origin_list = []
    dest_list = []

    for airport in Origins.query.all():
        origin_list.append([airport.iata, airport.city, airport.state])

    for airport in Dests.query.all():
        dest_list.append([airport.iata, airport.city, airport.state])

    return origin_list, dest_list

def load_models(departed):
    if departed=="True":
        model = load_joblib('lr_past_departed.sav')
        meta  = load_joblib('meta_past_departed.pkl')
        score = load_joblib('score_lr_past_departed.pkl')
        df = load_data('past_train_df_departed.csv')
    else:
        model = load_joblib('lr_past_.sav')
        meta  = load_joblib('meta_past_.pkl')
        score = load_joblib('score_lr_past_.pkl')
        df = load_data('past_train_df_.csv')

    return model, meta, score, df

def predict(origin_iata, dest_iata, date, time, departed, carrier):

    model, meta, score, df = load_models(departed)

    input_columns = meta
    rmse_score_test = score['rmse_score_test']

    # prepare a new input vector
    input_vector = np.zeros(len(input_columns))
    time_period = 400
    date_pd = pd.to_datetime(date, format="%Y-%m-%d")
    time_split = time.split(":")
    time_int = int("".join(time_split))
    time_index = int(time_int/time_period)

    # Monday is 1 but pandas delivers Monday = 0
    dayofweek = date_pd.dayofweek + 1
    dayofmonth = date_pd.day

    time_features = [ "DEP_TIME_NIGHT",
                      "DEP_TIME_TWILIGHT",
                      "DEP_TIME_MORNING",
                      "DEP_TIME_NOON",
                      "DEP_TIME_AFTERNOON",
                      "DEP_TIME_EVENING"]
    delay_features = ['CARRIER_DELAY',
                      'WEATHER_DELAY',
                      'NAS_DELAY',
                      'SECURITY_DELAY',
                      'LATE_AIRCRAFT_DELAY',
                      'DEP_DELAY',
                      'ARR_DELAY']

    # core core_features
    input_vector[input_columns["DAY_OF_MONTH"]] = int(dayofmonth)
    input_vector[input_columns["DAY_OF_WEEK"]] = int(dayofweek)

    try:
        input_vector[input_columns[str(time_features[time_index])]] = 1
    except:
        pass

    try:
        input_vector[input_columns['CARRIER_'+str(carrier)]] = 1
    except:
        pass
    # check for existence of the flight
    flight_exist = (Airports.query.filter(Airports.origin==str(origin_iata),
                                      Airports.dest==str(dest_iata),
                                      Airports.carrier==str(carrier)).
                                      scalar() is not None)

    # Prepare input vector
    df_delay = df[(df.ORIGIN==str(origin_iata)) & (df.DEST==str(dest_iata))]

    origin_degree = (Origins.query.filter(Origins.iata==str(origin_iata)).
                    first().degree)
    dest_degree = (Dests.query.filter(Dests.iata==str(dest_iata)).
                  first().degree)
    
    input_vector[input_columns['ORIGIN_DEGREE']] = int.from_bytes(origin_degree,
                                                   byteorder='little')
    input_vector[input_columns['DEST_DEGREE']] = int.from_bytes(dest_degree,
                                                   byteorder='little')

    for feature in delay_features:
        #if df_delay is empty, we will imputate values with the median of
        # df[feature]
        if df_delay.empty:
            input_vector[input_columns['MEDIAN_'+ feature]] = df['MEDIAN_'+ feature].median()
            input_vector[input_columns['MEAN_'+ feature]] = df['MEAN_'+ feature].median()
            input_vector[input_columns['Q0_'+ feature]] = df['Q0_'+ feature].median()
            input_vector[input_columns['Q1_'+ feature]] = df['Q1_'+ feature].median()
            input_vector[input_columns['Q3_'+ feature]] = df['Q3_'+ feature].median()
            input_vector[input_columns['Q95_'+ feature]] = df['Q95_'+ feature].median()

            if departed=="True" and feature != "ARR_DELAY":
                input_vector[input_columns[feature]] = df[feature].median()

        else:
            input_vector[input_columns['MEDIAN_'+ feature]] = df_delay['MEDIAN_'+ feature]
            input_vector[input_columns['MEAN_'+ feature]] = df_delay['MEAN_'+ feature]
            input_vector[input_columns['Q0_'+ feature]] = df_delay['Q0_'+ feature]
            input_vector[input_columns['Q1_'+ feature]] = df_delay['Q1_'+ feature]
            input_vector[input_columns['Q3_'+ feature]] = df_delay['Q3_'+ feature]
            input_vector[input_columns['Q95_'+ feature]] = df_delay['Q95_'+ feature]

            if departed=="True" and feature != "ARR_DELAY":
                input_vector[input_columns[feature]] = df_delay[feature]

    # prediction
    y_pred = model.predict(input_vector.reshape(1, -1))

    # delay threshold
    y_delay = y_pred + rmse_score_test

    if departed=="True":
        #we consider the threshold to be 15 in the case of departed flight
        y_delay_thr = 15.
    else:
        y_delay_thr = 30.

    y_delayed = y_delay > y_delay_thr

    return y_pred, rmse_score_test, flight_exist, y_delayed
    #prediction = model.predict()

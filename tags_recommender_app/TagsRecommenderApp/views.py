#! /usr/bin/python
# -*- coding:utf-8 -*-

from flask import Flask, request, redirect, render_template
from flask import url_for, flash, session
import logging as lg
import pdb

app = Flask(__name__)

app.config.from_object('config')
from .utils import airport_list, predict, load_data
from .forms import AirplaneForm

@app.route('/', methods= ['GET','POST'])
@app.route('/index', methods= ['GET','POST'])
def index():
    form = AirplaneForm()
    origin_airports, dest_airports = airport_list()

    form.origin.choices = ([(airport[0]+","+airport[1]+","+airport[2],
                           airport[0]+ " ("+airport[1]+", "+airport[2]+")")
                           for airport in origin_airports])

    form.dest.choices = ([(airport[0]+","+airport[1]+","+airport[2],
                        airport[0]+ " ("+airport[1]+", "+airport[2]+")")
                        for airport in dest_airports])

    if request.method == 'POST':
        if form.validate() == False:
            flash('All fields are required.')
            return render_template('index.html',
                                    title = "Airline Delay Predictor",
                                    form = form)
        else:
            return redirect('result.html')

    return render_template('index.html',
                            title = "Airline Delay Predictor",
                            form = form)

@app.route('/result', methods= ['GET','POST'])
def result():
    form = AirplaneForm()

    origin_iata = request.form["origin"]
    dest_iata = request.form["dest"]
    time = request.form["dep_time"]
    date = request.form["dep_day"]
    departed = request.form["departed"]
    carrier = request.form["carrier"]

    origin = origin_iata.split(",")
    dest = dest_iata.split(",")
    carrier = carrier.split(",")
    # predict
    y_pred, rmse_score_test, flight_exist, y_delayed = predict(origin[0],
                                                               dest[0],
                                                               date, time,
                                                               departed,
                                                               carrier[0])
    y_pred = int(y_pred[0])
    rmse_score_test = int(rmse_score_test)

    # written forms
    w_origin = origin[0]+" ("+ origin[1]+", "+origin[2] +")"
    w_dest = dest[0]+" ("+ dest[1]+", "+dest[2] +")"
    w_carrier = carrier[1]+" ("+ carrier[0]+")"

    return render_template('result.html', title = "Airline Delay Predictor",
                            form=form, origin_iata=w_origin, dest_iata=w_dest,
                            time=time, date=date, departed=departed,
                            carrier=w_carrier, y_pred=y_pred,
                            rmse_score_test=rmse_score_test,
                            flight_exist=flight_exist, y_delayed=y_delayed)

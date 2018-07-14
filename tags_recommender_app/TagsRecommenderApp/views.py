#! /usr/bin/python
# -*- coding:utf-8 -*-

from flask import Flask, request, redirect, render_template
from flask import url_for, flash, session
import logging as lg
import pdb

app = Flask(__name__)

app.config.from_object('config')
#from .utils import airport_list, predict, load_data
from .forms import RecommenderForm

@app.route('/', methods= ['GET','POST'])
@app.route('/index', methods= ['GET','POST'])
def index():
    form = RecommenderForm()
    return render_template('index.html',
                            form = form)
def index2():
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
                            title = "Tag Recommender",
                            form = form)


@app.route('/result', methods= ['GET','POST'])
def result():
    form = RecommenderForm()

    title= request.form["title"]
    lg.warning(title)
    pdb.set_trace()
    body = request.form["wmd-input"]
    wmd_preview = request.form["wmd-preview"]
    lg.warning(body)
    # predict

    #y_pred = int(y_pred[0])
    #rmse_score_test = int(rmse_score_test)

    # written forms
    #w_origin = origin[0]+" ("+ origin[1]+", "+origin[2] +")"
    #w_dest = dest[0]+" ("+ dest[1]+", "+dest[2] +")"
    #w_carrier = carrier[1]+" ("+ carrier[0]+")"

    return render_template('result.html', title = title,
                            form=form, body=body)

#! /usr/bin/python
# -*- coding:utf-8 -*-

from flask import Flask, request, redirect, render_template
from flask import url_for, flash, session
import logging as lg
from flask_pagedown import PageDown
from flask_misaka import markdown

app = Flask(__name__)
pagedown = PageDown(app)
app.config.from_object('config')
from .utils import run_predict
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
    pagedown_text = request.form["pagedown"]
    markdown_content = markdown(pagedown_text)
    # run predict
    y_pred = run_predict(title, markdown_content)
    lg.warning(y_pred)

    return render_template('result.html', title = title,
                            form=form, pagedown_text=pagedown_text)

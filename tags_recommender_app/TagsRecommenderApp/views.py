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
    website_title = "StackOverflow"

    if request.method == 'POST':
        if form.validate_on_submit()==False:
            flash('All fields are required.')
            return render_template('index.html',form = form,
                    website_title=website_title)
        else:
            return redirect('result.html')

    return render_template('index.html',
                            form = form, website_title=website_title)

@app.route('/result', methods= ['GET','POST'])
def result():
    form = RecommenderForm()
    website_title = "StackOverflow"
    title= request.form["title"]
    pagedown_text = request.form["pagedown"]
    markdown_content = markdown(pagedown_text)
    # run predict
    rec_tags = run_predict(title, markdown_content)[0]

    return render_template('result.html', website_title=website_title,
                            title = title, form=form,
                            pagedown_text=pagedown_text,
                            rec_tags=rec_tags)

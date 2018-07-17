from flask_wtf import FlaskForm
from wtforms import SubmitField, TextField
from flask_pagedown.fields import PageDownField
from wtforms import validators
#from .utils import load_data
import datetime

class RecommenderForm(FlaskForm):
    #pagedown = PageDownField('Ask your question...')
    title = TextField('Title', [validators.DataRequired()])
    pagedown = PageDownField('Body', [validators.DataRequired()])
    submit = SubmitField("Recommend tags")

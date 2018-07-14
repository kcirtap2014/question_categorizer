from flask_wtf import FlaskForm
from wtforms import SelectField, SubmitField, BooleanField, TextField
from wtforms.fields.html5 import DateField, TimeField, SearchField
from flask_pagedown.fields import PageDownField
from wtforms import validators
#from .utils import load_data
import datetime

class RecommenderForm(FlaskForm):
    #pagedown = PageDownField('Ask your question...')
    title = TextField('Title', [validators.Length(min=4)])
    submit = SubmitField("Recommend tags")

    def check_validity(self, df, origin_iata, dest_iata):
        new_df = df[(df.ORIGIN_IATA==origin_iata) & (df.DEST_IATA == dest_iata)]
        if new_df:
            raise validators.ValidationError("Check your flight origin and destination")

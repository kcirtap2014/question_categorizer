from flask_wtf import FlaskForm
from wtforms import SelectField, SubmitField, BooleanField
from wtforms.fields.html5 import DateField, TimeField, SearchField
from wtforms import validators
from .utils import load_data
import datetime

class AirplaneForm(FlaskForm):
    origin = SelectField("Origin",
                         validators= [validators.DataRequired()],
                         id="origin",
                         render_kw={"placeholder": "Enter name or code"})
    dest = SelectField("Destination",
                        validators= [validators.DataRequired()],
                        id="dest",
                        render_kw={"placeholder": "Enter name or code"})
    carrier = SelectField("Carrier",
                          validators= [validators.DataRequired()],
                          id="carrier",
    choices= [("AA,American Airlines ","American Airlines (AA)"),
                            ("AS,Alaska Airlines","Alaska Airlines (AS)"),
                            ("B6,JetBlue Airways","JetBlue Airways (B6)"),
                            ("DL,Delta Airlines","Delta Airlines (DL)"),
                            ("EV,Atlantic Southeast Airlines","Atlantic Southeast Airlines (EV)"),
                            ("F9,Frontier Airlines","Frontier Airlines (F9)"),
                            ("HA,Hawaiian Airlines","Hawaiian Airlines (HA)"),
                            ("NK,Spirit Airlines"," Spirit Airlines (NK)"),
                            ("OO,SkyWest Airlines","SkyWest Airlines (OO)"),
                            ("UA,United Airlines","United Airlines (UA)"),
                            ("VX,Virgin America","Virgin America (VX)"),
                            ("WN,Southwest Airlines","Southwest Airlines (WN)")])
    dep_day = DateField("Departure Date",
                        validators= [validators.DataRequired()],
                        id="dep_date",
                        default=datetime.datetime.today,
                        format = '%d-%m-%Y')
    dep_time = TimeField("Departure Time",
                         validators= [validators.DataRequired()],
                         id="dep_time",
                         format = '%H:%M')
    departed = BooleanField("Departed", id="departed")
    submit = SubmitField("Submit")

    def check_validity(self, df, origin_iata, dest_iata):
        new_df = df[(df.ORIGIN_IATA==origin_iata) & (df.DEST_IATA == dest_iata)]
        if new_df:
            raise validators.ValidationError("Check your flight origin and destination")

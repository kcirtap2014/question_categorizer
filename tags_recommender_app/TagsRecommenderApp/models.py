from flask_sqlalchemy import SQLAlchemy
import config as CONFIG
import pandas as pd
import logging as lg
from sklearn.externals import joblib
import pdb

from .views import app
# Create database connection object

db = SQLAlchemy(app)

class Airports(db.Model):
    id = db.Column(db.Integer, primary_key=True)
    origin = db.Column(db.String(3), nullable=False)
    dest = db.Column(db.String(3), nullable=False)
    carrier = db.Column(db.String(2), nullable=False)

    def __init__(self, origin, dest, carrier):
        self.origin = origin
        self.dest = dest
        self.carrier = carrier

class Origins(db.Model):
    __bind_key__ = 'origin'
    iata = db.Column(db.String(3), primary_key=True)
    city = db.Column(db.String(30), nullable=False)
    state = db.Column(db.String(5), nullable=False)
    degree = db.Column(db.LargeBinary, nullable=False)

    def __init__(self, iata, city, state, degree):
        self.iata = iata
        self.city = city
        self.state = state
        self.degree = degree

class Dests(db.Model):
    __bind_key__ = 'dest'
    iata = db.Column(db.String(3), primary_key=True)
    city = db.Column(db.String(30), nullable=False)
    state = db.Column(db.String(5), nullable=False)
    degree = db.Column(db.LargeBinary, nullable=False)

    def __init__(self, iata, city, state, degree):
        self.iata = iata
        self.city = city
        self.state = state
        self.degree = degree

def load_data(filename):
    data = pd.read_csv(CONFIG.DATABASE_URI+filename)
    return data

def load_joblib(filename):
    data = joblib.load(CONFIG.DATABASE_URI+filename)
    return data

def init_db():
    db.drop_all()
    db.create_all()
    data = load_data("apsearch_US.csv")
    data_origin = (data.drop_duplicates(subset=["ORIGIN_IATA"]).
                    sort_values(by="ORIGIN_IATA"))
    data_dest = (data.drop_duplicates(subset=["DEST_IATA"]).
                    sort_values(by="DEST_IATA"))
    for index in range(len(data)):
        db.session.add(Airports(data.loc[index,"ORIGIN_IATA"],
            data.loc[index,"DEST_IATA"], data.loc[index,"UNIQUE_CARRIER"]))

    for index in range(len(data_origin)):
        db.session.add(Origins(data_origin.iloc[index]["ORIGIN_IATA"],
            data_origin.iloc[index]["ORIGIN_CITY"],
            data_origin.iloc[index]["ORIGIN_STATE"],
            data_origin.iloc[index]["ORIGIN_DEGREE"]))

    for index in range(len(data_dest)):
        db.session.add(Dests(data_dest.iloc[index]["DEST_IATA"],
            data_dest.iloc[index]["DEST_CITY"],
            data_dest.iloc[index]["DEST_STATE"],
            data_dest.iloc[index]["DEST_DEGREE"]))
    db.session.commit()
    lg.warning('Database initialized!')

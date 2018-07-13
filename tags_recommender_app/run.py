#! /usr/bin/env python
from TagsRecommenderApp import app
import config as CONFIG

if __name__ == "__main__":
    app.secret_key = CONFIG.SECRET_KEY
    app.run(debug=True, host='localhost', port=5000)

from flask import Flask, redirect, url_for, request, render_template
from flask import jsonify
from flask_cors import CORS
import search_algorithms as search_algorithms


import json

app = Flask(__name__)
CORS(app)

@app.route('/search', methods=['POST'])
def add_obstacle():
    data = request.json
    alg = data["algorithm"]
    start = data["start"]
    dest = data["dest"]
    return search_algorithms.search(alg, start, dest)

@app.route('/')
def index():
    return render_template("index.html")

if __name__ == '__main__':
    app.run(debug=True)
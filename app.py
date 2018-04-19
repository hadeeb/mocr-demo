import numpy as np
from flask import Flask, render_template, request, jsonify

from mocr.predict import predict as pdt

app = Flask(__name__)
app.config['SECRET_KEY'] = '22334455'


@app.route('/predict', methods=['POST'])
def predict():
    img = request.get_json(force=True)['img']
    data = pdt(img)
    data = data[np.argsort(data[:, 1])][::-1, ...]
    # print(data)
    return jsonify(data.tolist())


@app.route('/')
def index():
    return render_template('index.html')


if __name__ == '__main__':
    app.run()

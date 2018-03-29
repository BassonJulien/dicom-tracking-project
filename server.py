from flask import Flask, request, render_template
from flask_cors import CORS
from catheter_predictor import CatheterPredictor

app = Flask(__name__)
CORS(app)


@app.route("/")
def index():
    # predictor = CatheterPredictor()
    return render_template('input-path.html')

@app.route('/', methods=['POST'])
def index_post():
    text = request.form['text']
    processed_text = text.upper()
    return processed_text

if __name__ == '__main__':
    app.run(port=4000)

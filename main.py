from flask import Flask, request, flash, redirect
from flask import jsonify, render_template
from werkzeug.utils import secure_filename
import os
import os.path
from Unsupervised_Learning.Cluster_MP_SKLearn import execute_all
from Supervised_Learning.Supervised_Learning import execute

UPLOAD_FOLDER = 'Static/uploads/'

app = Flask(__name__, template_folder='Static/Template', static_folder='Static')
app.secret_key = "secret key"
app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER


@app.route('/', methods=['GET', 'POST'])
def function1():
    return render_template("index.html")


@app.route('/upload', methods=['GET', 'POST'])
def upload():
    video1 = request.files.get('video1')
    video2 = request.files.get('video2')
    video1.save(os.path.join(app.config['UPLOAD_FOLDER'], "video1.mp4"))
    video2.save(os.path.join(app.config['UPLOAD_FOLDER'], "video2.mp4"))
    print(video1, video2)
    return {"video1": "video1.mp4", "video2": "video2.mp4"}


@app.route('/process')
def process():
    var1, var2 = execute_all("video1.mp4", "video2.mp4")
    # execute()
    return {'vid1': var2[0], 'vid2': var2[1], 'images': var1}


if __name__ == '__main__':
    app.run()

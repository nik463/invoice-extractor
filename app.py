from crypt import methods
from flask import Flask, render_template,jsonify
from flask_wtf import FlaskForm
from wtforms import FileField, SubmitField
from werkzeug.utils import secure_filename
import os 
#from wtform.validators import InputRequired

from main import json_op
import numpy as np

import pandas as pd

##

from preprocess import split_txt,stopword,extract_form,address,preprocess_img,pre_image





app = Flask(__name__)
app.config['SECRET_KEY'] = 'supersecretkey'
app.config['UPLOAD_FOLDER'] = 'static/files'

class UploadFileForm(FlaskForm):
    file = FileField("File")
    submit = SubmitField("Upload File")


@app.route('/', methods=['GET','POST'])
@app.route('/home', methods=['GET','POST'])
def home():
    form = UploadFileForm()
    if form.validate_on_submit():
        file = form.file.data
        file.save(os.path.join(os.path.abspath(os.path.dirname(__file__)),app.config['UPLOAD_FOLDER'],secure_filename(file.filename)))
        path  = os.path.join(os.path.abspath(os.path.dirname(__file__)),app.config['UPLOAD_FOLDER'],secure_filename(file.filename))
        #table,cons_details = json_op(path)
        #image = preprocess_img(path)
        #ext_text,table = detection_block(image,0.5)
        #cons_details = address(ext_text[3])
        #rece_details = address(ext_text[2])
        #invoice_details = address(ext_text[0])
        #df = table_extraction(table)
        #json_table = df.to_json()
        #total = { 'consignee': {},'receiver': {},'inv': {}, 'table': {}}
        #total['consignee'] = cons_details
        #total['receiver'] = rece_details
        #total['inv'] = invoice_details
        #total['table'] = table
        total = json_op(path)
        return total
    return render_template("index.html",form=form)

if __name__ == "__main__":
    app.run(host="127.0.0.1" ,port=8080 , debug=True)

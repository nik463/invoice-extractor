import re
import time

from flask import Flask, render_template,jsonify
from flask_wtf import FlaskForm
from nltk import word_tokenize
from wtforms import FileField, SubmitField
from werkzeug.utils import secure_filename
import os 
#from wtform.validators import InputRequired
from detect import detection_block, detection_column
from main import json_op
import numpy as np

import pandas as pd

##

from preprocess import split_txt, stopword, extract_form, address, preprocess_img, pre_image, convert_pdf_to_image, \
    extract_todict, clean, slice_text, extract_as_dict

app = Flask(__name__)
app.config['SECRET_KEY'] = 'supersecretkey'
app.config['UPLOAD_FOLDER'] = 'static'

class UploadFileForm(FlaskForm):
    file = FileField("File")
    submit = SubmitField("Upload File")


@app.route('/', methods=['GET','POST'])
@app.route('/home', methods=['GET','POST'])
def home():
    global consignee_details_dict, receiver_details, consignee_details, receiver_details_dict, invoice_details_dict, table_details_dict, inv_total_amount, enitity_details, ext_text, table_img
    form = UploadFileForm()
    if form.validate_on_submit():
        file = form.file.data
        file.save(os.path.join(os.path.abspath(os.path.dirname(__file__)),app.config['UPLOAD_FOLDER'],secure_filename(file.filename)))
        path  = os.path.join(os.path.abspath(os.path.dirname(__file__)),app.config['UPLOAD_FOLDER'],secure_filename(file.filename))
        image = path
        image = convert_pdf_to_image(image)
        image = np.array(image[0])
            # image = cv2.cvtColor(image, 1, cv2.COLOR_BAYER_BG2GRAY)
            # ret, image = cv2.threshold(image, 100, 255, cv2.THRESH_TOZERO)

        print("Image is detected")
        try:
            print("Invoice details extracting...")
            ext_text, table_img = detection_block(image, 0.4)
            print("Extracted", "len(ext_text)")

        except:
            print("Some error occurred in detecting")

        try:
            print("Invoice details extracting...")
            invoice_details = [(str(i).strip()) for i in ext_text[0]]
            invoice_details = [i.split(",") for i in invoice_details if len(i) >= 2]
            invoice_details_dict = extract_as_dict(invoice_details)
            print("Extracted")
        except:
            print("Invoice details not detected")

        try:
            print("Reciever details extracting...")
            receiver_details = [(str(i).strip()) for i in ext_text[1]]
            receiver_details = [i.split(",") for i in receiver_details if len(i) >= 2]
            receiver_details_dict = extract_todict(receiver_details)
            print("Extracted!")
        except:
            print("Can't Extracted")

        try:
            print("Consignee details extracting...")
            consignee_details = [(str(i).strip()) for i in ext_text[2]]
            consignee_details = [i.split(",") for i in consignee_details if len(i) >= 2]
            consignee_details_dict = extract_todict(consignee_details)
            print("Extracted!")
        except:
            pass
        try:
            print("Entity details extracting...")
            enitity_details = " ".join(ext_text[4][1:])
            print("Extracted!")
        except:
            print("entity details not detected")
        try:
            inv_total_amount = re.sub("[^a-zA-Z0-9:/\s,.]", "", str(ext_text[5]).split(" ")[-1])
        except:
            pass
        try:
            print("Table details extracting...")
            table_details_dict = detection_column(table_img, 0.4)
            print("Extracted!")
        except:
            print("Error:Table not detected!")
        try:
            print("Adress formatting...")
            tokens = word_tokenize(str(consignee_details))

            consignee_details_dict["address"] = re.sub("[^a-zA-Z0-9:/\s,.]", "",
                                                       clean(slice_text(["address"], ["contact"], tokens)))
            consignee_details_dict["address"] = [i.strip() for i in consignee_details_dict["address"].split(",") if
                                                 len(i) >= 1]
            consignee_details_dict["address"] = ", ".join(consignee_details_dict["address"])
            print("address is formatted")

            tokens1 = word_tokenize(str(receiver_details))
            print("tokenized")
            receiver_details_dict["address"] = re.sub("[^a-zA-Z0-9:/\s,.]", "",
                                                      clean(slice_text(["address"], ["state"], tokens1)))
            receiver_details_dict["address"] = [i.strip() for i in receiver_details_dict["address"].split(",") if
                                                len(i) >= 1]
            receiver_details_dict["address"] = ", ".join(receiver_details_dict["address"])
            print("address is formatted")
        except:
            print("Address is not recognizing")

        consignee_df = pd.DataFrame.from_dict(consignee_details_dict, orient='index').T
        receiver_df = pd.DataFrame.from_dict(receiver_details_dict, orient='index').T
        invoice_df = pd.DataFrame.from_dict(invoice_details_dict, orient='index').T
        table_df = pd.DataFrame.from_dict(table_details_dict, orient='index').T

        receiver_df['contact_number'] = None
        print("json converted to dataframe")

        # print(invoice_df.po_number)
        try:
            invoice_df["so_number"] = invoice_df.apply(
                lambda x: x["po_number"] if len(x["so_number"]) < 5 else x["so_number"], axis=1)
            invoice_df["po_date"] = invoice_df.apply(
                lambda x: x["po_date"] if len(x["po_date"]) > 8 else x["invoice_dt"], axis=1)
            print("if so number is not presented then po copy")
        except:
            print("there is no change in so number")

        try:
            print("Adress is splitting...")
            consignee_df["location"] = [i.split(",")[-5].strip() for i in consignee_df["address"]]
            consignee_df["city"] = [i.split(",")[-4].strip() for i in consignee_df["address"]]
            consignee_df["state"] = [i.split(",")[-3].strip() for i in consignee_df["address"]]
            consignee_df["pincode"] = [i.split(",")[-1].strip() for i in consignee_df["address"]]

            receiver_df["location"] = [i.split(",")[-5].strip() for i in receiver_df["address"]]
            receiver_df["city"] = [i.split(",")[-4].strip() for i in receiver_df["address"]]
            receiver_df["state"] = [i.split(",")[-3].strip() for i in receiver_df["address"]]
            receiver_df["pincode"] = [i.split(",")[-1].strip() for i in receiver_df["address"]]
            print("address is split into city,state,pincode")
        except:
            print("Error: Not splitted")

        temp_df = pd.DataFrame()
        try:
            temp_df["SKU Name"] = table_df["description"].dropna()
        except:
            print("Not detected table properly")
        try:
            temp_df["Package Level"] = table_df["rate_uom"].dropna()
        except:
            print("Not detected table properly")
        try:
            temp_df["Qty/Boxes"] = table_df["qty"].dropna()

            print("new dataframe created: temp_df")
        except:
            print("Not detected table properly")

        final_csv = pd.DataFrame(
            columns=["SO Number", "SO Date", "SO Value", "Billed From - Entity Code", "Billed From - Entity Name",
                     "Consignee Code - Entity Code", "Consignee Name", "Consignee Address", "Consignee Email",
                     "Consignee Phone", "Consg Location", "Consg City", "Consg State", "Consg Pincode",
                     "SKU Number/Code", "SKU Name", "Package Level", "Qty/Boxes", "Actual Weight", "Units",
                     "Ship To/Destination - Entity Name", "Dest Address", "Dest Email", "Dest Phone", "Dest Location",
                     "Dest City", "Dest State", "Dest Pincode"])
        print("final csv df created")
        try:
            final_csv["SO Number"] = invoice_df['so_number']
        except:
            final_csv["SO Number"] = None

        try:
            final_csv["SO Date"] = invoice_df['po_date']
        except:
            final_csv["SO Date"] = None
        try:
            final_csv["SO Value"] = inv_total_amount
        except:
            print("So value not detected")
        try:
            final_csv["Billed From - Entity Code"] = None
        except:
            print("some error occure")

        try:
            final_csv["Billed From - Entity Name"] = enitity_details
        except:
            print("some error occure")

        try:
            final_csv["Consignee Code - Entity Code"] = receiver_df['customer_no']
        except:
            print("some error occure")
        try:
            final_csv["Consignee Address"] = receiver_df["address"]

        except:
            print("some error occure")

        try:
            final_csv["Consignee Email"] = None
        except:
            print("some error occure")
        try:
            final_csv["Consignee Phone"] = receiver_df['contact_number']
        except:
            print("some error occure")
        try:
            final_csv["Consignee Name"] = receiver_df['name']
        except:
            print("some error occure")
        try:
            final_csv["Consg Location"] = receiver_df["location"]
        except:
            print("some error occure")
        try:
            final_csv["Consg City"] = receiver_df["city"]
        except:
            print("some error occure")
        try:
            final_csv["Consg State"] = receiver_df["state"]
        except:
            print("some error occure")
        try:
            final_csv["Consg Pincode"] = receiver_df["pincode"]
        except:
            print("some error occure")
        try:
            final_csv["SKU Number/Code"] = None
        except:
            print("some error occure")
        try:
            final_csv["SKU Name"] = None
        except:
            print("some error occure")
        try:
            final_csv["Actual Weight"] = None
        except:
            print("some error occure")
        try:
            final_csv["Units"] = "KG"
        except:
            print("some error occure")
        try:
            final_csv["Ship To/Destination - Entity Name"] = consignee_df["name"]
        except:
            print("some error occure")
        try:
            final_csv["Dest Address"] = consignee_df["address"]
        except:
            print("some error occure")
        try:
            final_csv["Dest Email"] = None
        except:
            print("some error occure")
        try:
            final_csv["Dest Phone"] = consignee_df['contact_number']
        except:
            print("some error occure")
        try:
            final_csv["Dest Location"] = consignee_df["location"]
        except:
            print("some error occure")
        try:
            final_csv["Dest City"] = consignee_df["city"]
        except:
            print("some error occure")
        try:
            final_csv["Dest State"] = consignee_df["state"]
        except:
            print("some error occure")
        try:
            final_csv["Dest Pincode"] = consignee_df["pincode"]
            print("final csv generated")
        except:
            print("some error occure")
        new = pd.concat([final_csv, temp_df], axis=0)
        new = new.ffill()
        new = new.iloc[1:]
        date_string = time.strftime("%Y-%m-%d-%H:%M")
        output = 'output'
        new.to_csv(os.path.join(output, secure_filename(file.filename)+ ".csv"), index=False)
        return "PDF is successfully converted to CSV Format"
    return render_template("index.html",form=form)

if __name__ == "__main__":
    app.run(host="127.0.0.1" ,port=8080 , debug=True)

from concurrent.futures import process
import pandas as pd
import numpy as np
import re
from thefuzz import fuzz, process

########### PDF TO IMAGES###################
import pdf2image
import cv2


def convert_pdf_to_image(document):
    images = []
    images.extend(list(map(lambda image: cv2.cvtColor(np.asarray(image), code=cv2.COLOR_RGB2BGR),
                           pdf2image.convert_from_path(document), )))
    return images


### IMAGE PRE-PROCESS  ###

def set_image_dpi(image):
    image = cv2.imread(image)
    length_x, width_y, dim = image.shape
    factor = min(1, float(1024.0 / length_x))
    size = int(factor * length_x), int(factor * width_y)
    image = cv2.resize(image, size)
    return image


def pre_image(image):
    ret, image = cv2.threshold(image, 120, 255, cv2.THRESH_TRUNC)
    return image


### TABLE PRE-PROCESS ####

def pre_table(arr):
    # df = pd.DataFrame(arr) df.drop(df.index[:2], inplace=True) df = df.iloc[:-11] df1 = pd.concat([df[0].str.split(
    # '\n', expand=True), df[1].str.split('\n', expand=True)], axis=1) dummy_columns = [str(i) for i in range(len(
    # df1.columns)-11)] df1.columns = ["Discount","Total","Rate","UOM","QTY","HSNS AC CODE","Description","SL.No",
    # "Amount","IGST Amount","IGST Rate"]+ dummy_columns df1 = df.drop(labels=dummy_columns, axis=1) df = df[2:-11]
    arr = arr[2:-11]

    def dfe(arr):
        df_ls = []
        for i in arr:
            df_ls.append(i[0].split() + i[1].split())
        return df_ls

    table = dfe(arr)
    finl = pd.DataFrame(df_ls)
    finl.columns = ["Total", "RATE/UOM", "QTY", "HSNS AC CODE", "Description", "Amount", "IGST Rate", "Discount",
                    "Rate", "Taxable/Value", ]

    return finl


#############################################

# pre process the image
def preprocess_img(image):
    image = cv2.imread(image)
    image = cv2.cvtColor(image, 1, cv2.COLOR_BAYER_BG2GRAY)
    ret, image = cv2.threshold(image, 100, 255, cv2.THRESH_TOZERO)
    return image


# preprocess image - resize
def pre_img_size(image):
    # gray = cv2.cvtColor(image,0, cv2.COLOR_BGR2GRAY)
    gray = cv2.resize(image, None, fx=0.5, fy=0.5, interpolation=cv2.INTER_AREA)
    thresh = cv2.threshold(gray, 1, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)[1]
    thresh = 255 - thresh
    return thresh


##############################################

# preprocess text

stopwords = ["", "\x0c", "[0, '"]


def split_txt(text):
    a = str(text).split("\n\n")
    a = str(text).split("\\n")
    return a


def stopword(text):
    x = []
    for i in text:
        if i not in stopwords:
            x.append([i])
    return x


def address(adr):
    adr = split_txt(adr)
    adr = stopword(adr)
    adr = extract_form(adr)
    return adr


json_tag_text = ["address", "contact_number", "customer_no", "gstin", "state", "state_code", "cin", "charge",
                 "ewaybil_no", "invoice_dt", "invoice_serial_no", "l.r._no", "no.of_pkgs", "pick_up_address",
                 "place_of_supply", "po_date", "po_number",
                 "so_number", "tax_is_payable_on_reverse", "name"
                                                           "transporter_name", "vehicle no", ]


def extract_form(text):
    form = {}
    for i in text:
        a = i[0].split(":")[0]

        try:
            form[a.lower()] = i[0].split(":")[1]
        except:
            form[a.lower()] = ""
    return form


#################################################


json_tag_text = ["address", "contact_number", "customer_no", "gstin", "state", "state_code", "cin", "charge",
                 "ewaybil_no", "invoice_dt", "invoice_serial_no", "l.r._no", "no.of_pkgs", "pick_up_address",
                 "place_of_supply", "po_date", "po_number",
                 "so_number", "tax_is_payable_on_reverse", "name",
                 "transporter_name", "vehicle no", "po_date"]


def extract_as_dict(text):
    form = {}
    for i in text:
        if len(i) != 0:
            a = i[0].split(":")[0]
            try:
                form[process.extract(a.lower(), json_tag_text, limit=1)[0][0]] = i[0].split(":")[1].strip()
            except:
                form[process.extract(a.lower(), json_tag_text, limit=1)[0][0]] = ""
    return form

def extract_todict(text):
    form = {}
    for i in text:
        a = i[0].split(":")
        key = a[0]
        if len(a)>=2:
            try:
                form[process.extract(key.lower(),json_tag_text,limit=1)[0][0]]=a[1:][0].strip()
            except:
                form[process.extract(key.lower(),json_tag_text,limit=1)[0][0]]=None
    return form

def slice_text(current_tag, next_tag, tokens):
    name_list = []
    end = []
    for idx, token in enumerate(tokens):
        if token.lower().startswith(tuple(next_tag)) or token.lower().endswith(tuple(next_tag)) and idx < (
                len(tokens) - 1):
            end.append(idx)
    for idx, token in enumerate(tokens):
        if token.lower().startswith(tuple(current_tag)) or token.lower().endswith(tuple(current_tag)) and idx < (
                len(tokens) - 1):
            name = tokens[idx:end[0]]
            name_list.extend(name)
    return name_list


def clean(name_list):
    if name_list[1] == ":":
        name_list = name_list[2:]
    else:
        name_list = name_list[1:]
    f = " ".join(name_list)
    g = f.split(",")
    fnl = []
    for i in g:
        text = (re.sub("'", "", i))
        text = (re.sub("  ", "", text))
        fnl.append(re.sub("`", "", text).strip())
    fn = []
    for i in fnl:
        if len(i) != 0:
            fn.append(i)
    final = ",".join(fn)
    return final

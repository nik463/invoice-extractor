import os
import numpy as np
import tensorflow as tf
import cv2
import time
import re
import pandas as pd

os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'  # Suppress TensorFlow logging (1)
os.environ['CUDA_VISIBLE_DEVICES'] = "0"
# import tesserocr
from PIL import Image
import pytesseract
from object_detection.utils import label_map_util
from object_detection.utils import config_util
from object_detection.builders import model_builder
from thefuzz import process

# from object_detection.utils import visualization_utils as viz_utils
if os.name == 'nt':
    pytesseract.pytesseract.tesseract_cmd = r'C:\Program Files\Tesseract-OCR\tesseract.exe'
CUSTOM_MODEL_NAME = 'my_ssd_mobnet'
PRETRAINED_MODEL_NAME = 'ssd_mobilenet_v2_fpnlite_320x320_coco17_tpu-8'
PRETRAINED_MODEL_URL = 'http://download.tensorflow.org/models/object_detection/tf2/20200711/ssd_mobilenet_v2_fpnlite_320x320_coco17_tpu-8.tar.gz'
TF_RECORD_SCRIPT_NAME = 'generate_tfrecord.py'
LABEL_MAP_NAME = 'label_map.pbtxt'
paths = {
    'WORKSPACE_PATH': os.path.join('Tensorflow', 'workspace'),
    'SCRIPTS_PATH': os.path.join('Tensorflow', 'scripts'),
    'APIMODEL_PATH': os.path.join('Tensorflow', 'models'),
    'ANNOTATION_PATH': os.path.join('Tensorflow', 'workspace', 'annotations'),
    'IMAGE_PATH': os.path.join('Tensorflow', 'workspace', 'images'),
    'MODEL_PATH': os.path.join('Tensorflow', 'workspace', 'models'),
    'PRETRAINED_MODEL_PATH': os.path.join('Tensorflow', 'workspace', 'pre-trained-models'),
    'CHECKPOINT_PATH': os.path.join('Tensorflow', 'workspace', 'models', CUSTOM_MODEL_NAME),
    'OUTPUT_PATH': os.path.join('Tensorflow', 'workspace', 'models', CUSTOM_MODEL_NAME, 'export'),
    'TFJS_PATH': os.path.join('Tensorflow', 'workspace', 'models', CUSTOM_MODEL_NAME, 'tfjsexport'),
    'TFLITE_PATH': os.path.join('Tensorflow', 'workspace', 'models', CUSTOM_MODEL_NAME, 'tfliteexport'),
    'PROTOC_PATH': os.path.join('Tensorflow', 'protoc')
}

files = {
    'PIPELINE_CONFIG': os.path.join('Tensorflow', 'workspace', 'models', CUSTOM_MODEL_NAME, 'pipeline.config'),
    'PIPELINE_CONFIG_1': os.path.join('Tensorflow', 'workspace', 'models', CUSTOM_MODEL_NAME, 'pipeline1.config'),
    'TF_RECORD_SCRIPT': os.path.join(paths['SCRIPTS_PATH'], TF_RECORD_SCRIPT_NAME),
    'LABELMAP': os.path.join(paths['ANNOTATION_PATH'], LABEL_MAP_NAME)
}

# Load pipeline config and build a detection model
configs = config_util.get_configs_from_pipeline_file(files['PIPELINE_CONFIG'])
detection_model = model_builder.build(model_config=configs['model'], is_training=False)

# Restore checkpoint
ckpt = tf.compat.v2.train.Checkpoint(model=detection_model)
ckpt.restore(os.path.join(paths['CHECKPOINT_PATH'], 'ckpt-51')).expect_partial()


@tf.function
def detect_fn(image):
    image, shapes = detection_model.preprocess(image)
    prediction_dict = detection_model.predict(image, shapes)
    detections = detection_model.postprocess(prediction_dict, shapes)
    return detections


# detection block

def detection_block(image, detection_threshold):
    image_np = np.array(image)
    input_tensor = tf.convert_to_tensor(np.expand_dims(image_np, 0), dtype=tf.float32)
    detections = detect_fn(input_tensor)
    num_detections = int(detections.pop("num_detections"))
    detections = {key: value[0, :num_detections].numpy() for key, value in detections.items()}
    detections['num_detections'] = num_detections
    detections['detection_classes'] = detections['detection_classes'].astype(np.int64)
    image_np_with_detections = image_np.copy()
    width = image.shape[1]
    height = image.shape[0]
    image = image_np_with_detections
    scores = list(filter(lambda x: x > detection_threshold, detections['detection_scores']))
    boxes = detections['detection_boxes'][:len(scores)]
    classes = detections['detection_classes'][:len(scores)]
    crop = []
    for idx, box in enumerate(boxes):
        roi = box * [height, width, height, width]
        region = image[int(roi[0]):int(roi[2]), int(roi[1]):int(roi[3])]
        crop.append(region)
    ext_text = [[i] for i in range(len(classes))]
    for i in range(len(classes)):
        if classes[i] != 3:
            # gray = cv2.cvtColor(crop[i], cv2.COLOR_BGR2GRAY)
            # gray = cv2.resize(gray, None, fx=1.8, fy=1.8, interpolation=cv2.INTER_AREA)
            # thresh = cv2.threshold(gray, 1, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)[1]
            # image = 255 - thresh
            # text = Image.fromarray(thresh)
            # text = tesserocr.image_to_text(text)

            image = cv2.cvtColor(crop[i], cv2.COLOR_BGR2GRAY)
            image = cv2.adaptiveThreshold(image, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY, 11, 15)
            image = cv2.resize(image, None, fx=1.8, fy=1.8, interpolation=cv2.INTER_AREA)
            text = pytesseract.image_to_string(image, config='--psm 4').strip()
            text = text.split("\n")
            # text = re.sub("[^a-zA-Z0-9:,.]\s", "", text)
            # text=pytesseract.image_to_string(crop[i],lang='eng',config='--psm 6')
            ext_text[classes[i]].extend(text)
        else:
            table_img = crop[i]
    return ext_text, table_img


#################################################################################################################


# Load pipeline config and build a detection model
configs1 = config_util.get_configs_from_pipeline_file(files['PIPELINE_CONFIG_1'])
detection_model_column = model_builder.build(model_config=configs1['model'], is_training=False)

# Restore checkpoint
ckpt1 = tf.compat.v2.train.Checkpoint(model=detection_model_column)
ckpt1.restore(os.path.join(paths['CHECKPOINT_PATH'], 'ckpt-21')).expect_partial()


@tf.function
def detect_fn_table(image):
    image, shapes = detection_model_column.preprocess(image)
    prediction_dict = detection_model_column.predict(image, shapes)
    detections = detection_model_column.postprocess(prediction_dict, shapes)
    return detections


def detection_column(image, detection_threshold):
    image_np = np.array(image)
    input_tensor = tf.convert_to_tensor(np.expand_dims(image_np, 0), dtype=tf.float32)
    detections = detect_fn_table(input_tensor)

    num_detections = int(detections.pop("num_detections"))
    detections = {key: value[0, :num_detections].numpy() for key, value in detections.items()}
    detections['num_detections'] = num_detections
    detections['detection_classes'] = detections['detection_classes'].astype(np.int64)
    image_np_with_detections = image_np.copy()
    width = image.shape[1]
    height = image.shape[0]
    image = image_np_with_detections
    scores = list(filter(lambda x: x > detection_threshold, detections['detection_scores']))
    boxes = detections['detection_boxes'][:len(scores)]
    classes = detections['detection_classes'][:len(scores)]
    json_tag_table = ["sl_no", "description", "qty", "hsn_sac_code","uom","rate", "rate/uom", "total", "discount", "taxable_value",
                      "amount","istin"]
    dict = {}
    table_details = []
    for idx, box in enumerate(boxes):
        roi = box * [height, width, height, width]
        region = image[int(roi[0]):int(roi[2]), int(roi[1]):int(roi[3])]
        #region = cv2.cvtColor(region, cv2.COLOR_BGR2GRAY)
        #image = cv2.adaptiveThreshold(region, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY, 11, 15)
        #region = cv2.resize(region, None, fx=1.8, fy=1.8, interpolation=cv2.INTER_AREA)
        out = pytesseract.image_to_string(region, config='--psm 4 oem').strip()
        #out = tostring(region)
        # ee= Image.fromarray(region)
        # out = tesserocr.image_to_text(ee).strip()
        # out = re.sub("[\\n]", " ",out)
        #
        out = re.sub("[|]", "", out)
        out = out.split("\n")
        table_details.append(out)
        for i in range(len(out)):
            #dict.update({process.extract(re.sub("[^a-zA-Z0-9]:", "",out[0]), json_tag_table, limit=1)[0][0]: out[1:]})
            dict.update({process.extract(re.sub("[^a-zA-Z0-9]:", "",out[0]), json_tag_table, limit=1)[0][0]:[i.strip() for i in out[1:] if len(i.strip())!=0]})
    return dict,table_details

def tostring(img):
    #img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    ret, thresh1 = cv2.threshold(img, 0, 255, cv2.THRESH_OTSU | cv2.THRESH_BINARY_INV)
    rect_kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (18, 18))
    dilation = cv2.dilate(thresh1, rect_kernel, iterations = 1)
    contours, hierarchy = cv2.findContours(dilation, cv2.RETR_EXTERNAL,
                                                 cv2.CHAIN_APPROX_NONE)
    im2 = img.copy()
    #image = cv2.cvtColor(im2, cv2.COLOR_BGR2GRAY)
    image = cv2.adaptiveThreshold(image, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY, 11, 15)
    image = cv2.resize(image, None, fx=1.8, fy=1.8, interpolation=cv2.INTER_AREA) 
    for cnt in contours:
        x, y, w, h = cv2.boundingRect(cnt)
        rect = cv2.rectangle(im2, (x, y), (x + w, y + h), (0, 255, 0), 2)
        cropped = image[y:y + h, x:x + w]
        text = pytesseract.image_to_string(cropped, config = '--psm 4')
    return text
'''
def table_extraction(img):
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    thresh, img_bin = cv2.threshold(img, 128, 255, cv2.THRESH_BINARY | cv2.THRESH_OTSU)
    img_bin = 255 - img_bin
    kernel_len = np.array(img).shape[1] // 100
    ver_kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (1, kernel_len))
    hor_kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (kernel_len, 1))
    kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (2, 2))
    image_1 = cv2.erode(img_bin, ver_kernel, iterations=3)
    vertical_lines = cv2.dilate(image_1, ver_kernel, iterations=3)
    image_2 = cv2.erode(img_bin, hor_kernel, iterations=3)
    horizontal_lines = cv2.dilate(image_2, hor_kernel, iterations=3)
    img_vh = cv2.addWeighted(vertical_lines, 0.5, horizontal_lines, 0.5, 0.0)
    img_vh = cv2.erode(~img_vh, kernel, iterations=2)
    thresh, img_vh = cv2.threshold(img_vh, 128, 255, cv2.THRESH_BINARY | cv2.THRESH_OTSU)
    bitxor = cv2.bitwise_xor(img, img_vh)
    bitnot = cv2.bitwise_not(bitxor)
    contours, hierarchy = cv2.findContours(img_vh, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)

    def sort_contours(cnts, method="left-to-right"):
        reverse = False
        i = 0
        if method == "right-to-left" or method == "bottom-to-top":
            reverse = True
        if method == "top-to-bottom" or method == "bottom-to-top":
            i = 1
        boundingBoxes = [cv2.boundingRect(c) for c in cnts]
        (cnts, boundingBoxes) = zip(*sorted(zip(cnts, boundingBoxes),
                                            key=lambda b: b[1][i], reverse=reverse))
        return (cnts, boundingBoxes)

    contours, boundingBoxes = sort_contours(contours, method="top-to-bottom")
    heights = [boundingBoxes[i][3] for i in range(len(boundingBoxes))]
    mean = np.mean(heights)
    box = []
    for c in contours:
        x, y, w, h = cv2.boundingRect(c)
        if (w < 10000 and h < 500):
            image = cv2.rectangle(img, (x, y), (x + w, y + h), (0, 255, 0), 2)
            box.append([x, y, w, h])
    row = []
    column = []
    j = 0

    for i in range(len(box)):
        if (i == 0):
            column.append(box[i])
            previous = box[i]
        else:
            if (box[i][1] <= previous[1] + mean / 2):
                column.append(box[i])
                previous = box[i]
                if (i == len(box) - 1):
                    row.append(column)
            else:
                row.append(column)
                # column=[]
                previous = box[i]
                column.append(box[i])
    countcol = 0
    for i in range(len(row)):
        countcol = len(row[i])
        if countcol > countcol:
            countcol = countcol
    center = [int(row[i][j][0] + row[i][j][2] / 2) for j in range(len(row[i])) if row[0]]
    center = np.array(center)
    center.sort()
    finalboxes = []
    for i in range(len(row)):
        lis = []
        for k in range(countcol):
            lis.append([])
        for j in range(len(row[i])):
            diff = abs(center - (row[i][j][0] + row[i][j][2] / 4))
            minimum = min(diff)
            indexing = list(diff).index(minimum)
            lis[indexing].append(row[i][j])
        finalboxes.append(lis)
    outer = []
    for i in range(len(finalboxes)):
        for j in range(len(finalboxes[i])):
            inner = ''
            if (len(finalboxes[i][j]) == 0):
                outer.append(' ')
            else:
                for k in range(len(finalboxes[i][j])):
                    y, x, w, h = finalboxes[i][j][k][0], finalboxes[i][j][k][1], finalboxes[i][j][k][2], \
                                 finalboxes[i][j][k][3]
                    finalimg = bitnot[x:x + h, y:y + w]
                    kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (2, 1))
                    border = cv2.copyMakeBorder(finalimg, 2, 2, 2, 2, cv2.BORDER_CONSTANT, value=[255, 255])
                    resizing = cv2.resize(border, None, fx=2, fy=2, interpolation=cv2.INTER_CUBIC)
                    dilation = cv2.dilate(resizing, kernel, iterations=1)
                    erosion = cv2.erode(dilation, kernel, iterations=2)
                    erosion = Image.fromarray(erosion)
                    out = tesserocr.image_to_text(erosion)
                    # out = pytesseract.image_to_string(erosion,config='--psm 6')
                    # out = re.sub("[^a-zA-Z0-9:]", "", out)
                    if (len(out) == 0):
                        out = tesserocr.image_to_text(erosion)
                        # out = pytesseract.image_to_string(erosion, config='--psm 6')
                        # out = re.sub("[^a-zA-Z0-9:]", "", out)
                    inner = inner + " " + out
                outer.append(inner)
    arr = np.array(outer)
    arr = arr.reshape(len(row), countcol)
    arr = pd.DataFrame(arr)
    # df_ls=[] for i in outer[2:-12]: df_ls.append(i[0].split()+i[1].split()) finl = pd.DataFrame(df_ls) finl.columns
    # = ["Total","RATE/UOM","QTY","HSNS AC CODE","Description","Amount","IGST Rate","Discount","Rate","Taxable/Value",]
    return arr

'''
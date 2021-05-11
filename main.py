import os.path
from flask import Flask, flash, request, redirect, url_for, render_template
from flask import Flask, session
from werkzeug.utils import secure_filename
from flask import Flask, session
import sqlite3
import os

# comment out below line to enable tensorflow logging outputs
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'
import time
import math
import tensorflow as tf
import core.utils as utils
from core.yolov4 import filter_boxes
from tensorflow.python.saved_model import tag_constants
from core.config import cfg
from PIL import Image
import cv2
import numpy as np
import matplotlib.pyplot as plt
from tensorflow.compat.v1 import ConfigProto
from tensorflow.compat.v1 import InteractiveSession
# deep sort imports
from deep_sort import preprocessing, nn_matching
from deep_sort.detection import Detection
from deep_sort.tracker import Tracker
from tools import generate_detections as gdet

UPLOAD_FOLDER = './uploads'
app = Flask(__name__, static_folder='static', template_folder='templates')
app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER


@app.route('/', methods=['GET', 'POST'])
def home_page():
    session['filename'] = ''
    upload = False
    if request.method == 'POST':
        # check if the post request has the file part
        if 'file' not in request.files:
            flash('No file part')
            return redirect(request.url)
        file = request.files['file']
        # if user does not select file, browser also
        # submit an empty part without filename
        if file.filename == '':
            flash('No selected file')
            return redirect(request.url)
        if file:
            filename = secure_filename(file.filename)
            file.save(os.path.join(app.config['UPLOAD_FOLDER'], filename))
            session['filename'] = filename

        if os.path.isfile('uploads/' + session['filename']):
            upload = True
        else:
            upload = False
    return render_template('home_page.html', uploaded=upload)


@app.route('/demo')
def demo():
    return render_template('demo.html')


@app.route('/info')
def info():
    return render_template('info.html')


@app.route('/processing')
def processing():
    return render_template('info.html')


@app.route('/result')
def detector():
    # Definition of the parameters
    max_cosine_distance = 0.4
    nn_budget = None
    nms_max_overlap = 1.0
    # initialize deep sort
    model_filename = 'model_data/mars-small128.pb'
    encoder = gdet.create_box_encoder(model_filename, batch_size=1)
    # calculate cosine distance metric
    metric = nn_matching.NearestNeighborDistanceMetric("cosine", max_cosine_distance, nn_budget)
    # initialize tracker
    tracker = Tracker(metric)

    # load configuration for object detector
    config = ConfigProto()
    config.gpu_options.allow_growth = True
    input_size = 416

    # video_path = 'data/video/vertical.mp4'
    video_path = 'uploads/' + session['filename']

    # print(video_path)

    # load tflite model if flag is set

    # TINY
    saved_model_loaded = tf.saved_model.load('./checkpoints/yolov4-tiny-416', tags=[tag_constants.SERVING])
    # NORMAL
    # saved_model_loaded = tf.saved_model.load('./checkpoints/yolov4-416', tags=[tag_constants.SERVING])

    infer = saved_model_loaded.signatures['serving_default']

    # begin video capture
    try:
        vid = cv2.VideoCapture(int(video_path))
    except:
        vid = cv2.VideoCapture(video_path)

    out = None

    # get video ready to save locally if flag is set
    if './outputs/output.avi':
        # by default VideoCapture returns float instead of int
        width = int(vid.get(cv2.CAP_PROP_FRAME_WIDTH))
        height = int(vid.get(cv2.CAP_PROP_FRAME_HEIGHT))
        fps = int(vid.get(cv2.CAP_PROP_FPS))
        codec = cv2.VideoWriter_fourcc(*'XVID')
        out = cv2.VideoWriter('./outputs/output.avi', codec, fps, (width, height))

    frame_num = 0
    # while video is running

    data = cv2.VideoCapture(video_path)

    fps = int(data.get(cv2.CAP_PROP_FPS))

    print("fps: " + str(fps))

    bus_counter = []
    car_counter = []
    truck_counter = []
    sum_counter = []

    myDict = {}
    while True:
        cur_minute = math.floor(frame_num / (60 * fps)) + 1

        return_value, frame = vid.read()
        if return_value:
            frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            image = Image.fromarray(frame)
        else:
            print('Video has ended or failed, try a different video format!')
            break

        if fps >= 25:
            if frame_num % 3 != 0:
                frame_num += 1
                continue

        frame_num += 1

        print('Frame #: ', frame_num)
        print('Minute : ', cur_minute)
        frame_size = frame.shape[:2]
        image_data = cv2.resize(frame, (input_size, input_size))
        image_data = image_data / 255.
        image_data = image_data[np.newaxis, ...].astype(np.float32)
        start_time = time.time()

        # run detections
        batch_data = tf.constant(image_data)
        pred_bbox = infer(batch_data)
        for key, value in pred_bbox.items():
            boxes = value[:, :, 0:4]
            pred_conf = value[:, :, 4:]

        boxes, scores, classes, valid_detections = tf.image.combined_non_max_suppression(
            boxes=tf.reshape(boxes, (tf.shape(boxes)[0], -1, 1, 4)),
            scores=tf.reshape(
                pred_conf, (tf.shape(pred_conf)[0], -1, tf.shape(pred_conf)[-1])),
            max_output_size_per_class=50,
            max_total_size=50,
            iou_threshold=0.45,
            score_threshold=0.50
        )

        # convert data to numpy arrays and slice out unused elements
        num_objects = valid_detections.numpy()[0]
        bboxes = boxes.numpy()[0]
        bboxes = bboxes[0:int(num_objects)]
        scores = scores.numpy()[0]
        scores = scores[0:int(num_objects)]
        classes = classes.numpy()[0]
        classes = classes[0:int(num_objects)]

        # format bounding boxes from normalized ymin, xmin, ymax, xmax ---> xmin, ymin, width, height
        original_h, original_w, _ = frame.shape
        bboxes = utils.format_boxes(bboxes, original_h, original_w)

        # store all predictions in one parameter for simplicity when calling functions
        pred_bbox = [bboxes, scores, classes, num_objects]

        # read in all class names from config
        class_names = utils.read_class_names(cfg.YOLO.CLASSES)

        # by default allow all classes in .names file
        allowed_classes = list(class_names.values())

        # loop through objects and use class index to get class name, allow only classes in allowed_classes list
        names = []
        deleted_indx = []
        for i in range(num_objects):
            class_indx = int(classes[i])
            class_name = class_names[class_indx]
            if class_name not in allowed_classes:
                deleted_indx.append(i)
            else:
                names.append(class_name)
        names = np.array(names)
        count = len(names)

        # encode yolo detections and feed to tracker
        features = encoder(frame, bboxes)
        detections = [Detection(bbox, score, class_name, feature) for bbox, score, class_name, feature in
                      zip(bboxes, scores, names, features)]

        # initialize color map
        cmap = plt.get_cmap('tab20b')
        colors = [cmap(i)[:3] for i in np.linspace(0, 1, 20)]

        # run non-maxima supression
        boxs = np.array([d.tlwh for d in detections])
        scores = np.array([d.confidence for d in detections])
        classes = np.array([d.class_name for d in detections])
        indices = preprocessing.non_max_suppression(boxs, classes, nms_max_overlap, scores)
        detections = [detections[i] for i in indices]

        # Call the tracker
        tracker.predict()
        tracker.update(detections)

        # update tracks
        for track in tracker.tracks:
            if not track.is_confirmed() or track.time_since_update > 1:
                continue
            bbox = track.to_tlbr()
            class_name = track.get_class()

            # draw bbox on screen
            color = colors[int(track.track_id) % len(colors)]
            color = [i * 255 for i in color]

            cv2.rectangle(frame, (int(bbox[0]), int(bbox[1])), (int(bbox[2]), int(bbox[3])), color, 2)
            cv2.rectangle(frame, (int(bbox[0]), int(bbox[1] - 30)),
                          (int(bbox[0]) + (len(class_name) + len(str(track.track_id))) * 17, int(bbox[1])), color,
                          -1)
            cv2.putText(frame, class_name + "-" + str(track.track_id), (int(bbox[0]), int(bbox[1] - 10)), 0, 0.75,
                        (255, 255, 255), 2)

            # if enable info flag then print details about each track

            height, width, _ = frame.shape

            # Veritical

            cv2.line(frame, (0, int(3 * height / 6)), (width, int(3 * height / 6)), (0, 255, 0), thickness=2)
            center_y = int(((bbox[1]) + (bbox[3])) / 2)
            if center_y <= int(3 * height / 6 + height / 30) and center_y >= int(3 * height / 6 - height / 30):
                if class_name == 'car':
                    car_counter.append(int(track.track_id))
                    sum_counter.append(int(track.track_id))
                    if cur_minute not in myDict:
                        myDict[cur_minute] = []
                    myDict[cur_minute].append(track.track_id)

                elif class_name == 'truck':
                    truck_counter.append(int(track.track_id))
                    sum_counter.append(int(track.track_id))
                    if cur_minute not in myDict:
                        myDict[cur_minute] = []
                    myDict[cur_minute].append(track.track_id)
                elif class_name == 'bus':
                    bus_counter.append(int(track.track_id))
                    sum_counter.append(int(track.track_id))
                    if cur_minute not in myDict:
                        myDict[cur_minute] = []
                    myDict[cur_minute].append(track.track_id)

            # Horizontal

            cv2.line(frame, (int(3 * width / 6), 0), (int(3 * width / 6), height), (0, 255, 0), thickness=2)
            center_x = int(((bbox[0]) + (bbox[2])) / 2)
            if center_x <= int(3 * width / 6 + width / 30) and center_x >= int(3 * width / 6 - width / 30):
                if class_name == 'car':
                    car_counter.append(int(track.track_id))
                    sum_counter.append(int(track.track_id))
                    if cur_minute not in myDict:
                        myDict[cur_minute] = []
                    myDict[cur_minute].append(track.track_id)
                elif class_name == 'truck':
                    truck_counter.append(int(track.track_id))
                    sum_counter.append(int(track.track_id))
                    if cur_minute not in myDict:
                        myDict[cur_minute] = []
                    myDict[cur_minute].append(track.track_id)
                elif class_name == 'bus':
                    bus_counter.append(int(track.track_id))
                    sum_counter.append(int(track.track_id))
                    if cur_minute not in myDict:
                        myDict[cur_minute] = []
                    myDict[cur_minute].append(track.track_id)

        car_count = len(set(car_counter))
        truck_count = len(set(truck_counter))
        bus_count = len(set(bus_counter))

        total_count = len(set(sum_counter))
        print(total_count)

        cv2.putText(frame, "Total Vehicle Count: " + str(total_count), (0, 130), 0, 1, (0, 0, 255), 2)

        # calculate frames per second of running detections
        fps_running = 1.0 / (time.time() - start_time)
        # print("FPS: %.2f" % fps_running)
        result = np.asarray(frame)
        result = cv2.cvtColor(frame, cv2.COLOR_RGB2BGR)

        if './outputs/tiny.avi':
            out.write(result)
        if cv2.waitKey(1) & 0xFF == ord('q'): break
    cv2.destroyAllWindows()

    # ========================================================================================

    name_db = session['filename']

    # return render_template('result.html', result=number_of_vehicles)

    first_sum = len(set(myDict[1]))
    second_sum = 0
    third_sum = 0
    fourth_sum = 0
    fifth_sum = 0

    if 2 in myDict:
        second_sum = len(set(myDict[2]))
    if 3 in myDict:
        third_sum = len(set(myDict[3]))
    if 4 in myDict:
        fourth_sum = len(set(myDict[4]))
    if 5 in myDict:
        fifth_sum = len(set(myDict[4]))

    conn = sqlite3.connect('database.db')
    cursor = conn.cursor()
    cursor.execute("INSERT INTO vehicles VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?)",
                   (name_db, car_count, bus_count, truck_count, first_sum, second_sum, third_sum, fourth_sum, fifth_sum,
                    total_count))
    conn.commit()
    conn.close()

    return render_template('result.html', car=car_count, bus=bus_count, truck=truck_count, total=total_count,
                           first=first_sum, second=second_sum, third=third_sum, fourth=fourth_sum, fifth=fifth_sum,
                           length=cur_minute)


if __name__ == "__main__":
    app.secret_key = 'super secret key'
    app.config['SESSION_TYPE'] = 'filesystem'

    app.run(debug=True)

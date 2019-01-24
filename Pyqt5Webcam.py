import cv2
import sys
import numpy as np
import os
import six.moves.urllib as urllib
import tarfile
import tensorflow as tf
import zipfile

from io import StringIO
from matplotlib import pyplot as plt
from PIL import Image
from collections import defaultdict

from collections import Counter
from PyQt5.QtCore import *
from PyQt5.QtGui import *
from PyQt5.QtWidgets import *

#object detection imports
from object_detection.utils import label_map_util
from object_detection.utils import visualization_utils as vis_util

sys.path.append(r"C:\Users\yan_h\Desktop\object_detection")
threshold = 0.7
# What model to download.
MODEL_NAME = 'ssd_mobilenet_v1_coco_2018_01_28'

# Path to frozen detection graph. This is the actual model that is used for the object detection.
PATH_TO_CKPT = os.path.join(r'C:\Users\yan_h\Desktop\object_detection',MODEL_NAME,'frozen_inference_graph.pb')

# List of the strings that is used to add correct label for each box.
PATH_TO_LABELS = os.path.join('data', 'mscoco_label_map.pbtxt')

NUM_CLASSES = 90

# Load a (frozen) Tensorflow model into memory.
detection_graph = tf.Graph()
with detection_graph.as_default():
  od_graph_def = tf.GraphDef()
  with tf.gfile.GFile(PATH_TO_CKPT, 'rb') as fid:
    serialized_graph = fid.read()
    od_graph_def.ParseFromString(serialized_graph)
    tf.import_graph_def(od_graph_def, name='')

label_map = label_map_util.load_labelmap(os.path.join(r'C:\Users\yan_h\Desktop\object_detection\data', 'mscoco_label_map.pbtxt'))
categories = label_map_util.convert_label_map_to_categories(label_map, max_num_classes=NUM_CLASSES, use_display_name=True)
category_index = label_map_util.create_category_index(categories)

# matrix = ['person','person','apple','pen','pen','person','horse']

class MainWindow(QWidget):
    def __init__(self, fps=60):
        super().__init__()

        # self.capture = cv2.VideoCapture(camera_index)
        self.table = QTableWidget(self)
        #set fixed size of table widget
        self.table.setFixedSize(500,500)
        self.image = QLabel(self)
        self.image.setFixedSize(500,500)
        self.control = QPushButton('Start')

        layout = QHBoxLayout(self)
        layout.addWidget(self.image)
        layout.addWidget(self.table)
        layout.addWidget(self.control)
        # layout.addWidget(self.table)

        self.timer = QTimer(self)
        self.timer.setInterval(int(1000/fps))
        self.timer.timeout.connect(self.get_frame)
        self.control.clicked.connect(self.controlTimer)
        # timer.timeout.connect(self.createTable)
        # self.timer.start()

    def get_frame(self):
        _, frame = self.capture.read()
        #image in cv2 format
        image = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        height, width, channel = image.shape
        bytesPerLine = 3 * width
        # image = QImage(frame, *frame.shape[1::-1], QImage.Format_RGB888).rgbSwapped()

        with detection_graph.as_default():
            with tf.Session(graph=detection_graph) as sess:

                    #inserted code from capture.py
                    image_np_expanded = np.expand_dims(image, axis=0)
                    image_tensor = detection_graph.get_tensor_by_name('image_tensor:0')
                    # Each box represents a part of the image where a particular object was detected.
                    boxes = detection_graph.get_tensor_by_name('detection_boxes:0')
                    # Each score represent how level of confidence for each of the objects.
                    # Score is shown on the result image, together with the class label.
                    scores = detection_graph.get_tensor_by_name('detection_scores:0')
                    classes = detection_graph.get_tensor_by_name('detection_classes:0')
                    num_detections = detection_graph.get_tensor_by_name('num_detections:0')
                    # Actual detection.
                    (boxes, scores, classes, num_detections) = sess.run(
                        [boxes, scores, classes, num_detections],
                        feed_dict={image_tensor: image_np_expanded})

                    self.objects = []
                    for index, value in enumerate(classes[0]):
                        object_dict = {}
                        if scores[0,index] > threshold:
                            object_dict = category_index.get(value).get('name')
                            self.objects.append(object_dict)
                            print(self.objects)
                    # # Visualization of the results of a detection.
                    vis_util.visualize_boxes_and_labels_on_image_array(
                        image,
                        np.squeeze(boxes),
                        np.squeeze(classes).astype(np.int32),
                        np.squeeze(scores),
                        category_index,
                        use_normalized_coordinates=True,
                        line_thickness=8)

        #convert from cv2 image to qimage

        qImg = QImage(image.data, width, height, bytesPerLine, QImage.Format_RGB888)
        pixmap = QPixmap.fromImage(qImg)
        self.image.setPixmap(pixmap)

    def createTable(self):
       # Create table
        cnt = Counter()
        for item in self.objects:
            cnt[item] += 1
        newMatrix = Counter(cnt)

        # Create table
        appleCost = 0.6
        personCost = 100
        penCost = 1.5
        horseCost = 6
        appleTotal = 0
        personTotal = 0
        penTotal = 0
        horseTotal = 0
        length = len(newMatrix.keys()) + 1
        self.table.setRowCount(length)
        self.table.setColumnCount(4)
        self.table.setHorizontalHeaderLabels(["Item","Quantity", "Cost (EA)","Cost"])
        self.table.horizontalHeaderItem(0).setTextAlignment(Qt.AlignHCenter)
        self.table.horizontalHeaderItem(1).setTextAlignment(Qt.AlignHCenter)
        self.table.horizontalHeaderItem(2).setTextAlignment(Qt.AlignHCenter)

        index = 0
        for i in newMatrix.keys():
            self.table.setItem(index, 0,QTableWidgetItem(str(i)))
            self.table.setItem(index, 1,QTableWidgetItem(str(newMatrix[i])))

            if i == 'apple':
                self.table.setItem(index, 2,QTableWidgetItem(str(appleCost)))
                appleTotal = newMatrix[i] * appleCost
                self.table.setItem(index, 3,QTableWidgetItem(str(appleTotal)))
            if i == 'person':
                self.table.setItem(index, 2,QTableWidgetItem(str(personCost)))
                personTotal = newMatrix[i] * personCost
                self.table.setItem(index, 3,QTableWidgetItem(str(personTotal)))
            if i == 'pen':
                self.table.setItem(index, 2,QTableWidgetItem(str(penCost)))
                penTotal = newMatrix[i] * penCost
                self.table.setItem(index, 3,QTableWidgetItem(str(penTotal)))
            if i == 'horse':
                self.table.setItem(index, 2,QTableWidgetItem(str(horseCost)))
                horseTotal = newMatrix[i] * horseCost
                self.table.setItem(index, 3,QTableWidgetItem(str(horseTotal)))

            index += 1

        total = personTotal + appleTotal + penTotal + horseTotal
        self.table.setItem(index, 2,QTableWidgetItem(str("Total Bill : ")))
        self.table.setItem(index, 3,QTableWidgetItem(str(total)))

    # start/stop timer
    def controlTimer(self):
        # if timer is stopped
        if not self.timer.isActive():
            # create video capture
            self.capture = cv2.VideoCapture(0)
            # start timer
            self.timer.start(5)
            # update control_bt text
            self.control.setText("Stop")
        # if timer is started
        else:
            # stop timer
            self.timer.stop()
            # release video capture
            self.capture.release()
            self.createTable()
            # update control_bt text
            self.control.setText("Start")

app = QApplication([])
win = MainWindow()
win.show()
app.exec()

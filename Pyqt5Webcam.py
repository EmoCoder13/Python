import cv2
import sys
from collections import Counter
from PyQt5.QtCore import *
from PyQt5.QtGui import *
from PyQt5.QtWidgets import *

matrix = ['people','people','apple','pen','pen','people','horse']

class MainWindow(QWidget):
    def __init__(self, camera_index=0, fps=30):
        super().__init__()

        self.capture = cv2.VideoCapture(camera_index)
        self.table = QTableWidget(self)
        self.image = QLabel(self)

        layout = QHBoxLayout(self)
        layout.addWidget(self.image)
        layout.addWidget(self.table)

        timer = QTimer(self)
        timer.setInterval(int(1000/fps))
        timer.timeout.connect(self.get_frame)
        timer.timeout.connect(self.createTable)
        timer.start()

    def get_frame(self):
        _, frame = self.capture.read()
        image = QImage(frame, *frame.shape[1::-1], QImage.Format_RGB888).rgbSwapped()
        pixmap = QPixmap.fromImage(image)
        self.image.setPixmap(pixmap)

    def createTable(self):
       # Create table
        cnt = Counter()
        for item in matrix:
            cnt[item] += 1
        newMatrix = Counter(cnt)

        length = len(newMatrix.keys())
        self.table.setRowCount(length)
        self.table.setColumnCount(3)
        index = 0
        for i in newMatrix.keys():
            self.table.setItem(index, 0,QTableWidgetItem(str(i)))
            self.table.setItem(index, 1,QTableWidgetItem(str(newMatrix[i])))

            if i == 'apple':
                self.table.setItem(index, 2,QTableWidgetItem('0.95'))
            if i == 'people':
                self.table.setItem(index, 2,QTableWidgetItem('2.00'))
            if i == 'pen':
                self.table.setItem(index, 2,QTableWidgetItem('1.5'))
            if i == 'horse':
                self.table.setItem(index, 2,QTableWidgetItem('6'))

            index += 1

app = QApplication([])
win = MainWindow()
win.show()
app.exec()

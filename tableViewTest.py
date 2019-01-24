import sys
from collections import Counter
from collections import Counter
from PyQt5.QtCore import *
from PyQt5.QtGui import *
from PyQt5.QtWidgets import *

import cv2

matrix = ['person','person','apple','apple','pen','pen','person','horse','horse']
from PyQt5.QtWidgets import QMainWindow, QApplication, QWidget, QAction, QTableWidget,QTableWidgetItem,QVBoxLayout
from PyQt5.QtGui import QIcon
from PyQt5.QtCore import pyqtSlot

matrix = ['people','people','apple','pen','pen','people','horse']

class App(QWidget):

    def __init__(self):
        super().__init__()
        self.title = 'Checkout'
        self.left = 100
        self.top = 100
        self.width = 500
        self.height = 500

        self.initUI()

    def initUI(self):
        #set a horizontal Layout
        self.setLayout(QVBoxLayout())
        h_layout = QHBoxLayout()
        #add horizontal layout to main window
        self.layout().addLayout(h_layout)
        #Define Label
        camera = QLabel('Camera View')
        #define tableWidget
        self.table = QTableWidget()
        self.timer = QTimer()

        #add wigets to horizontal layout
        h_layout.addWidget(camera)
        h_layout.addWidget(self.table)
        self.createTable(self.sorter())
        #adding button to main window
        start = QPushButton('Start')
        self.layout().addWidget(start)

        self.setWindowTitle(self.title)
        self.setGeometry(self.left, self.top, self.width, self.height)
        self.createTable(self.sorter())
        # Add box layout, add table to box layout and add box layout to widget
        self.layout = QVBoxLayout()
        self.layout.addWidget(self.tableWidget)
        self.setLayout(self.layout)

        # Show widget
        self.show()

    def createTable(self,newMatrix):
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

        self.tableWidget = QTableWidget()
        length = len(newMatrix.keys())
        self.tableWidget.setRowCount(length)
        self.tableWidget.setColumnCount(3)
        index = 0
        for i in newMatrix.keys():
            self.table.setItem(index, 0,QTableWidgetItem(str(i)))
            self.table.setItem(index, 1,QTableWidgetItem(str(newMatrix[i])))
            print(i +" "+ str(newMatrix[i]))
            self.tableWidget.setItem(index, 0,QTableWidgetItem(str(i)))
            self.tableWidget.setItem(index, 1,QTableWidgetItem(str(newMatrix[i])))

            if i == 'apple':
                self.table.setItem(index, 2,QTableWidgetItem(str(appleCost)))
                appleTotal = newMatrix[i] * appleCost
                self.table.setItem(index, 3,QTableWidgetItem(str(appleTotal)))
            if i == 'person':
                self.table.setItem(index, 2,QTableWidgetItem(str(personCost)))
                personTotal = newMatrix[i] * personCost
                self.table.setItem(index, 3,QTableWidgetItem(str(personTotal)))
                self.tableWidget.setItem(index, 2,QTableWidgetItem('0.95'))
            if i == 'people':
                self.tableWidget.setItem(index, 2,QTableWidgetItem('2.00'))
            if i == 'pen':
                self.table.setItem(index, 2,QTableWidgetItem(str(penCost)))
                penTotal = newMatrix[i] * penCost
                self.table.setItem(index, 3,QTableWidgetItem(str(penTotal)))
                self.tableWidget.setItem(index, 2,QTableWidgetItem('1.5'))
            if i == 'horse':
                self.table.setItem(index, 2,QTableWidgetItem(str(horseCost)))
                horseTotal = newMatrix[i] * horseCost
                self.table.setItem(index, 3,QTableWidgetItem(str(horseTotal)))
                self.tableWidget.setItem(index, 2,QTableWidgetItem('6'))

            index += 1

        total = personTotal + appleTotal + penTotal + horseTotal
        self.table.setItem(index, 2,QTableWidgetItem(str("Total Bill : ")))
        self.table.setItem(index, 3,QTableWidgetItem(str(total)))

    def sorter(self):
        cnt = Counter()
        for item in matrix:
            cnt[item] += 1
        newMatrix = Counter(cnt)
        print(newMatrix)
        return newMatrix


if __name__ == '__main__':
    app = QApplication(sys.argv)
    ex = App()
    sys.exit(app.exec_())

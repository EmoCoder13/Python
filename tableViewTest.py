import sys
from collections import Counter
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
        self.tableWidget = QTableWidget()
        length = len(newMatrix.keys())
        self.tableWidget.setRowCount(length)
        self.tableWidget.setColumnCount(3)
        index = 0
        for i in newMatrix.keys():
            print(i +" "+ str(newMatrix[i]))
            self.tableWidget.setItem(index, 0,QTableWidgetItem(str(i)))
            self.tableWidget.setItem(index, 1,QTableWidgetItem(str(newMatrix[i])))

            if i == 'apple':
                self.tableWidget.setItem(index, 2,QTableWidgetItem('0.95'))
            if i == 'people':
                self.tableWidget.setItem(index, 2,QTableWidgetItem('2.00'))
            if i == 'pen':
                self.tableWidget.setItem(index, 2,QTableWidgetItem('1.5'))
            if i == 'horse':
                self.tableWidget.setItem(index, 2,QTableWidgetItem('6'))

            index += 1

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

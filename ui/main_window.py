from PyQt5 import QtGui, QtWidgets
from PyQt5.QtCore import QRect
from PyQt5.QtWidgets import QMainWindow

from constants import MAIN_WINDOW_TITLE
from ui.widgets.mask_menu import MaskMenuWidget


class MainWindow(QMainWindow):
    def resizeEvent(self, a0: QtGui.QResizeEvent) -> None:
        rect = QRect(0, 0, a0.size().width(), a0.size().height())
        if self.widget_mask_menu.isVisible() and self.widget_mask_menu.geometry():
            self.widget_mask_menu.setGeometry(rect)

    def mousePressEvent(self, a0: QtGui.QMouseEvent) -> None:
        self.widget_mask_menu.mousePressEvent(a0)

    def mouseMoveEvent(self, a0: QtGui.QMouseEvent) -> None:
        self.widget_mask_menu.mouseMoveEvent(a0)

    def mouseReleaseEvent(self, a0: QtGui.QMouseEvent) -> None:
        self.widget_mask_menu.mouseReleaseEvent(a0)

    def __init__(self, app):
        super(MainWindow, self).__init__()
        self.setWindowTitle(MAIN_WINDOW_TITLE)
        self.widget_mask_menu = MaskMenuWidget(self)
        self.widget_mask_menu.show()
        self.resize(self.widget_mask_menu.size())

        self.progress_bar = QtWidgets.QProgressBar()
        self.statusBar().addPermanentWidget(self.progress_bar)
        self.progress_bar.hide()

        self.progress_bar.setFixedSize(self.geometry().width() - 250, 16)

        self.app = app

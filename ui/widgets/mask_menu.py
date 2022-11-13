import math
from typing import List, Union, Dict
import cv2
import numpy as np
from PyQt5 import QtGui
from PyQt5.QtCore import Qt, QRect, QPoint, QPointF
from PyQt5.QtGui import QPixmap, QImage, QCursor
from PyQt5.QtWidgets import QFileDialog, QMessageBox, QWidget
from catboost import CatBoostClassifier
import pickle
from constants import MAX_PIXMAP_SIDE
from ui.qtdesigner import Ui_MaskMenu
from enum import Enum
import pandas as pd
import tensorflow as tf
import os
import json
from tensorflow import keras
from tensorflow.keras import datasets, layers, models
from tensorflow.keras.preprocessing.image import ImageDataGenerator


js = json.load(open('settings.json'))
proba_thresh = js['proba_thresh']
step = js['step']


size = 576
model = tf.keras.applications.resnet50.ResNet50(include_top=True,
                                              input_shape=(size, size, 3),
                                              pooling=None,
                                              weights=None,
                                              classes=102)
model = tf.keras.models.load_model('cnn93.hdf5')
model_ex = keras.Model(inputs=model.inputs, outputs=model.get_layer(name="avg_pool").output)

#cb_model = CatBoostClassifier()
#cb_model.load_model('cb.pkl')
cb_model = pickle.load(open('cb.pkl', 'rb'))

class ImageView(Enum):
    ORIGINAL_IMAGE = 0
    INVERTED_MASKED_IMAGE = 1
    MASKED_IMAGE = 2
    ONLY_MASK = 3


def dist(p1, p2):
    return ((p1[0] - p2[0]) ** 2 + (p1[1] - p2[1]) ** 2) ** 0.5


def get_angle(p1, p2, p3):
    """p2 = center"""
    d1 = dist(p1, p2)
    d2 = dist(p2, p3)
    angle = math.acos(min(d1, d2) / max(d1, d2))
    return math.degrees(angle)


class MaskMenuWidget(QWidget, Ui_MaskMenu):
    def get_warp_data(self, group, subgroup):
        img_data = self.data[group]['img_data'][subgroup]
        warp_data = {}
        is_180_rot_global = False
        all_180 = []
        for img_name, mask in img_data.items():
            image_path = os.path.join(group, subgroup, img_name)
            img = cv2.imread(image_path)
            mask = mask.copy()

            h, w = mask.shape
            if h > w:
                img = np.rot90(img)
                mask = np.rot90(mask)

            ret, thresh = cv2.threshold(mask, 127, 255, 0)
            contours, hierarchy = cv2.findContours(thresh, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)

            cnt = contours[0]
            epsilon = 0.001 * cv2.arcLength(cnt, True)
            while True:
                epsilon += epsilon
                approx = cv2.approxPolyDP(cnt, epsilon, True)
                if len(approx) <= 4:
                    break

            left = list(approx[approx[:, :, 0].argmin()][0])
            right = list(approx[approx[:, :, 0].argmax()][0])
            other = np.array([elem[0] for elem in approx.tolist() if elem[0] not in [left, right]]).astype(int)
            if len(approx) == 3:
                if dist(left, other[0]) >= dist(right, other[0]):
                    start = left
                    end = right
                else:
                    start = right
                    end = left
            elif len(approx) == 4:
                subsequence = [(tuple(elem[0], )) for elem in approx.tolist()]
                if abs(subsequence.index(tuple(other[0])) - subsequence.index(tuple(other[1]))) != 1:
                    # если две точки не соединены между собой, то там четырехугольник, то есть только тело
                    if get_angle(other[0], left, other[1]) >= get_angle(other[0], right, other[1]):
                        start = left
                        end = right
                    else:
                        start = right
                        end = left
                else:
                    # если две точки соединены между собой, то там треугольник, то есть хвост
                    if dist(left, other[0]) + dist(left, other[1]) >= dist(right, other[0]) + dist(right, other[1]):
                        start = left
                        end = right
                    else:
                        start = right
                        end = left

            is_180_rot = start < end

            rect = cv2.minAreaRect(contours[0])
            box = cv2.boxPoints(rect)
            box = np.int0(box)
            box = np.where(box > 0, box, 0)

            dtl = math.inf
            for point in box:
                distance = dist([0, 0], point)
                if distance < dtl:
                    tl = point
                    dtl = distance
            elems = [tuple(elem) for elem in box.tolist()]
            ind = elems.index(tuple(tl))
            tl = elems[ind]
            tr = elems[(ind + 1) % len(elems)]
            br = elems[(ind + 2) % len(elems)]
            bl = elems[(ind + 3) % len(elems)]

            box = np.array([tr, tl, bl, br]).astype(np.float32)

            widthA = np.sqrt(((br[0] - bl[0]) ** 2) + ((br[1] - bl[1]) ** 2))
            widthB = np.sqrt(((tr[0] - tl[0]) ** 2) + ((tr[1] - tl[1]) ** 2))
            heightA = np.sqrt(((tr[0] - br[0]) ** 2) + ((tr[1] - br[1]) ** 2))
            heightB = np.sqrt(((tl[0] - bl[0]) ** 2) + ((tl[1] - bl[1]) ** 2))
            maxWidth = max(int(widthA), int(widthB))
            maxHeight = max(int(heightA), int(heightB))
            dst = np.array([
                [maxWidth - 1, 0],
                [0, 0],
                [0, maxHeight - 1],
                [maxWidth - 1, maxHeight - 1]], dtype="float32")
            M = cv2.getPerspectiveTransform(box, dst)
            warp_data[img_name] = [M, (maxWidth, maxHeight)]
            all_180.append(is_180_rot)
        s = [int(elem) for elem in all_180]
        s = sum(s) / len(s)
        is_180_rot_global = s > 0.5
        return is_180_rot_global, warp_data

    def save_prediction(self):
        """Сохранение в файл со всеми известными данными"""
        out_path = QFileDialog.getSaveFileName(filter=f'CSV file (*.csv)')[0]
        if not any(out_path):
            return
        data = []
        for group_path, group_data in self.data.items():
            preds = group_data['pred']
            group = os.path.split(group_path)[-1]
            data.append([group] + preds)
        df = pd.DataFrame(data, columns=['name', 'top1', 'top2', 'top3', 'top4', 'top5'])
        df.to_csv(out_path, index=False)

    def predict_all(self):
        if len(self.image_paths) == 0:
            return
        #self.parent.statusBar().showMessage("processing...", 0)
        self.parent.progress_bar.show()
        self.parent.statusBar().show()
        #self.parent.progress_bar.setValue(count)
        for group, group_data in self.data.items():
            img_data = group_data['img_data']

            preds = []
            for subgroup, subgroup_data in img_data.items():
                #self.parent.statusBar().showMessage('warping data...', 0)
                #self.parent.app.processEvents()
                #is_180_rot, warp_data = self.get_warp_data(group, subgroup)
                self.parent.progress_bar.setValue(0)
                self.parent.statusBar().showMessage(f'processing "{os.path.split(group)[-1]}/{subgroup}"...', 0)
                self.parent.app.processEvents()
                l = list(subgroup_data.items())[::step]
                for i, (img_name, mask) in enumerate(l):
                    image_path = os.path.join(group, subgroup, img_name)
                    #img = cv2.imread(image_path)

                    warp_mask = mask.copy()
                    warp_img = cv2.imread(image_path)
                    h, w, *_ = warp_mask.shape
                    #if h > w:
                    #    warp_img = np.rot90(warp_img)
                    #    warp_mask = np.rot90(warp_mask)
                    #warp_mask = cv2.warpPerspective(warp_mask, warp_data[img_name][0], warp_data[img_name][1])
                    #warp_img = cv2.warpPerspective(warp_img, warp_data[img_name][0], warp_data[img_name][1])

                    #h, w, *_ = warp_mask.shape
                    #if h > w:
                    #    continue
                    #if is_180_rot >= 0.5:
                    #    warp_img = warp_img[:, ::-1]
                    #    warp_mask = warp_mask[:, ::-1]

                    img = warp_img
                    img = img / 255
                    img = cv2.resize(img, (576, 576))
                    img = img.reshape(1, 576, 576, 3)
                    x = tf.expand_dims(img, axis=0)
                    y = model_ex(x[0])
                    y = y[0].numpy()
                    proba = max(cb_model.predict_proba(y))
                    #print(cb_model.predict(y))
                    if proba <= proba_thresh:
                        pred = 0
                    else:
                        pred = int(cb_model.predict(y)[0])

                    preds.append(pred)

                    self.parent.progress_bar.setValue(int(i / len(l) * 100))
                    self.parent.app.processEvents()
            print(os.path.split(group)[-1], preds)
            unique, counts = np.unique(preds, return_counts=True)
            x = list(sorted([tuple(elem) for elem in np.asarray((counts, unique)).T.tolist()], reverse=True))
            x = [elem[1] for elem in x]
            if len(x) < 5:
                x = x + [0] * (5 - len(x))
            group_data['pred'] = [x[0], 0, 0, 0, 0]#x[:5]
        self.parent.progress_bar.hide()
        self.parent.statusBar().hide()
        self.classSpinBox.setValue(self.data[self.cur_group]['pred'][0])

    def go_main_menu(self):
        self.parent.go_main_menu()

    def prev_img_id(self):
        self.cur_img_id -= 1

    def next_img_id(self):
        self.cur_img_id += 1

    def notify_none_cur_img(self):
        QMessageBox().warning(self, 'Warning', 'Cannot read image or path')

    def load_data(self):
        folder = QFileDialog.getExistingDirectory()
        if not any(folder):
            return
        self.parent.progress_bar.show()
        self.parent.statusBar().show()
        data = {}
        for group in os.listdir(folder):
            group_path = os.path.join(folder, group)
            if not os.path.isdir(group_path):
                continue
            data[group_path] = {'pred': [0, 0, 0, 0, 0], 'img_data': {}}
            for subgroup in os.listdir(group_path):
                self.parent.statusBar().showMessage(f'loading "{os.path.split(group)[-1]}/{subgroup}"...', 0)
                self.parent.progress_bar.setValue(0)
                self.parent.app.processEvents()
                subgroup_path = os.path.join(group_path, subgroup)
                if not os.path.isdir(subgroup_path):
                    continue
                data[group_path]['img_data'][subgroup] = {}
                filepaths = os.listdir(subgroup_path)
                for i, filepath in enumerate(filepaths):
                    full_filepath = os.path.join(subgroup_path, filepath)
                    if '.jpg' not in filepath or filepath.replace('.jpg', '.png') not in filepaths:
                        continue
                    data[group_path]['img_data'][subgroup][filepath] = cv2.imread(full_filepath.replace('.jpg', '.png'), cv2.IMREAD_GRAYSCALE)
                    self.parent.progress_bar.setValue(int(i / len(filepaths) * 100))
                    self.parent.app.processEvents()
            self.parent.app.processEvents()
        self.data = data
        self.group_id = 0
        #self.groupsList.setCurrentRow(0)
        self.parent.progress_bar.hide()
        self.parent.statusBar().hide()

    def filename_row_changed(self, row):
        self.cur_img_id = row

    def image_view_changed(self, index):
        self.image_view = ImageView(index)

    def group_row_changed(self, row):
        if row == -1:
            return
        self.group_id = row

    def subgroup_row_changed(self, row):
        if row == -1:
            return
        self.subgroup_id = row

    def paintEvent(self, event):
        if (self.cur_img is None and self.image_view != ImageView.ONLY_MASK) or \
                (self.cur_mask is None and self.image_view != ImageView.ORIGINAL_IMAGE):
            pixmap = QPixmap()
        else:
            if self.image_view == ImageView.ORIGINAL_IMAGE:
                img = self.cur_img
            else:
                # отрисовка изображения, связанного с маской
                mask = self.cur_mask
                # преобразование маски в изображение
                if self.image_view == ImageView.ONLY_MASK:
                    img = cv2.cvtColor(mask, cv2.COLOR_GRAY2RGB)
                else:
                    if self.image_view == ImageView.INVERTED_MASKED_IMAGE:
                        mask = 255 - mask
                    img = cv2.bitwise_and(self.cur_img, self.cur_img, mask=mask)
            img = cv2.resize(img, self.pixmap_size)
            h, w, ch = img.shape
            qt_format = QImage(img.data, w, h, ch * w, QImage.Format_RGB888)
            pixmap = QPixmap.fromImage(qt_format)
        self.imageLabel.setPixmap(pixmap)

    def draw_mask_line(self, from_point: Union[QPoint, QPointF], to_point: Union[QPoint, QPointF],
                       color: int, radius: int):
        if len(self.image_paths) == 0:
            return
        if isinstance(from_point, QPoint):
            from_point = self.get_percentage_point(from_point)
        if isinstance(to_point, QPoint):
            to_point = self.get_percentage_point(to_point)
        mask = self.data[self.cur_group]['img_data'][self.cur_subgroup][self.image_paths[self.cur_img_id]]
        if mask is None:
            return
        h, w = mask.shape
        start_point = (int(from_point.x() * w), int(from_point.y() * h))
        end_point = (int(to_point.x() * w), int(to_point.y() * h))
        cv2.line(mask, start_point, end_point, color, radius)

    def mousePressEvent(self, a0: QtGui.QMouseEvent) -> None:
        image_point = self.get_image_point()
        self.last_image_point = image_point
        if a0.button() == Qt.LeftButton and 0 <= image_point.x() <= self.pixmap_size[0]:
            self._drawing_mask_color = 255
            self._is_drawing_mask = True
        elif a0.button() == Qt.RightButton and 0 <= image_point.y() <= self.pixmap_size[1]:
            self._drawing_mask_color = 0
            self._is_drawing_mask = True
        else:
            self._is_drawing_mask = False
        if self._is_drawing_mask:
            self.draw_mask_line(image_point, self.last_image_point, self._drawing_mask_color, self.drawingRadiusSpinBox.value())

    def mouseMoveEvent(self, a0: QtGui.QMouseEvent) -> None:
        image_point = self.get_image_point()
        if self._is_drawing_mask:
            self.draw_mask_line(image_point, self.last_image_point, self._drawing_mask_color, self.drawingRadiusSpinBox.value())
        self.last_image_point = image_point

    def mouseReleaseEvent(self, a0: QtGui.QMouseEvent) -> None:
        image_point = self.get_image_point()
        self._is_drawing_mask = False
        self._drawing_mask_color = 0
        self.last_image_point = image_point

    def get_image_point(self) -> QPoint:
        if self.cur_img is None:
            return QPoint(0, 0)
        image_point = self.imageLabel.mapFromGlobal(QCursor.pos())
        return image_point

    def get_actual_point(self, point: QPointF) -> QPoint:
        if self.cur_img is None:
            return QRect()
        return QPoint(int(point.x() * self.pixmap_size[0]), int(point.y() * self.pixmap_size[1]))

    def get_percentage_point(self, point: QPoint) -> QPointF:
        if self.cur_img is None:
            return QRect()
        return QPointF(point.x() / self.pixmap_size[0], point.y() / self.pixmap_size[1])

    def delete_cur_img(self):
        if len(self.image_paths) == 0:
            return
        del self.data[self.cur_group]['img_data'][self.cur_subgroup][self.image_paths[self.cur_img_id]]
        self.image_paths = self.image_paths[:self.cur_img_id] + self.image_paths[self.cur_img_id + 1:]

    def class_spin_box_changed(self, a0):
        if len(self.image_paths) == 0:
            return
        self.data[self.cur_group]['pred'][0] = a0
        self.update_top_predictions()

    def update_top_predictions(self):
        self.topPredictionsLabel.setText(f'Топ 5 предсказаний: {",".join([str(pred) for pred in self.data[self.cur_group]["pred"]])}')
        if self.data[self.cur_group]['pred'][0] == 0:
            self.notRecognizedWhaleLabel.show()
        else:
            self.notRecognizedWhaleLabel.hide()

    @property
    def image_paths(self):
        return self._image_paths

    @image_paths.setter
    def image_paths(self, value: List[str]):
        self._image_paths = value

        self.imageFilesList.clear()
        self.imageFilesList.addItems(value)

        # cur_img_id устанавливается после imageFilesList, потому что использует изменение индекса в imageFilesList
        self.cur_img_id = 0

    @property
    def cur_img_id(self):
        return self._cur_img_id

    @cur_img_id.setter
    def cur_img_id(self, value: int):
        if len(self._image_paths) == 0:
            self._cur_img_id = 0
            self._cur_img = None
            self.classSpinBox.setValue(1)
        else:
            self._cur_img_id = value % len(self._image_paths)

            group = self.cur_group
            subgroup = self.cur_subgroup
            img_name = self._image_paths[self._cur_img_id]
            #img_data = self.data[group]['img_data'][img_name]
            self._cur_img = cv2.imread(os.path.join(group, subgroup, img_name))
            if self._cur_img is not None:
                self._cur_img = cv2.cvtColor(self._cur_img, cv2.COLOR_BGR2RGB)
            else:
                self.notify_none_cur_img()
            self.imageFilesList.setCurrentRow(self._cur_img_id)
            self.classSpinBox.setValue(self.data[group]['pred'][0])

    @property
    def cur_img(self):
        return self._cur_img

    @property
    def cur_mask(self):
        if len(self.image_paths) == 0:
            return MAX_PIXMAP_SIDE, MAX_PIXMAP_SIDE
        filename = os.path.split(self.image_paths[self.cur_img_id])[-1]
        mask = self.data[self.cur_group]['img_data'][self.cur_subgroup][self.image_paths[self.cur_img_id]]
        if mask is None and self.cur_img is not None:
            mask = np.zeros(self.cur_img.shape[:2], dtype=np.uint8)
            self.masks[filename] = mask
        if self.cur_img is not None:
            h, w, _ = self.cur_img.shape
            mask = cv2.resize(mask, (w, h))
        return mask

    @property
    def pixmap_size(self):
        """w, h"""
        if self.cur_img is None:
            return MAX_PIXMAP_SIDE, MAX_PIXMAP_SIDE
        h_cur, w_cur, _ = self.cur_img.shape
        if h_cur > w_cur:
            k = MAX_PIXMAP_SIDE / h_cur
        else:
            k = MAX_PIXMAP_SIDE / w_cur
        return int(k * w_cur), int(k * h_cur)

    @property
    def group_id(self) -> int:
        return self._group_id

    @group_id.setter
    def group_id(self, val):
        self.groupsList.clear()

        if len(self.data.keys()) != 0:
            self._group_id = val % len(self.data.keys())
            self.groupsList.addItems(self.data.keys())
            # self.groupsList.setCurrentRow(self._group_id)
        else:
            self._group_id = 0

        self.subgroup_id = 0

    @property
    def cur_group(self):
        return list(self.data.keys())[self._group_id]

    @property
    def subgroup_id(self) -> int:
        return self.subgroup_id

    @subgroup_id.setter
    def subgroup_id(self, val):
        self.subgroupsList.clear()
        if len(self.data[self.cur_group].keys()) != 0:
            self._subgroup_id = val % len(self.data[self.cur_group]['img_data'].keys())
            self.subgroupsList.addItems(self.data[self.cur_group]['img_data'].keys())
            #self.subgroupsList.setCurrentRow(self._subgroup_id)
        else:
            self._subgroup_id = 0

        self.image_paths = list(self.data[self.cur_group]['img_data'][self.cur_subgroup].keys())

    @property
    def cur_subgroup(self):
        return list(self.data[self.cur_group]['img_data'].keys())[self._subgroup_id]

    def __init__(self, parent):
        super().__init__(parent)
        self.setupUi(self)
        self.parent = parent

        self.data = {}

        self.image_view = ImageView(self.imageViewComboBox.currentIndex())
        self.last_image_point = self.imageLabel.mapFromGlobal(QCursor.pos())
        self.masks: Dict[str, np.ndarray] = {}

        self._image_paths = []
        self._group_id = 0
        self._cur_img_id = 0
        self._cur_img = None

        self._is_drawing_mask = False
        self._drawing_mask_color = 0

        # ui
        self.prevButton.clicked.connect(self.prev_img_id)
        self.deleteButton.clicked.connect(self.delete_cur_img)
        self.nextButton.clicked.connect(self.next_img_id)

        self.imageViewComboBox.currentIndexChanged.connect(self.image_view_changed)

        self.imageFilesList.currentRowChanged.connect(self.filename_row_changed)
        self.groupsList.currentRowChanged.connect(self.group_row_changed)
        self.subgroupsList.currentRowChanged.connect(self.subgroup_row_changed)

        self.loadDataButton.clicked.connect(self.load_data)
        self.predictAllButton.clicked.connect(self.predict_all)
        self.savePredictionButton.clicked.connect(self.save_prediction)

        self.classSpinBox.valueChanged.connect(self.class_spin_box_changed)

        self.notRecognizedWhaleLabel.hide()

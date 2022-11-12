import pandas as pd
import cv2
import numpy as np
from matplotlib import pyplot as plt
import math
import os
import time
from skimage import exposure
from tqdm import tqdm

def dist(p1, p2):
    return ((p1[0] - p2[0]) ** 2 + (p1[1] - p2[1]) ** 2) ** 0.5


def get_angle(p1, p2, p3):
    """p2 = center"""
    d1 = dist(p1, p2)
    d2 = dist(p2, p3)
    angle = math.acos(min(d1, d2) / max(d1, d2))
    return math.degrees(angle)


for class_path in tqdm(os.listdir('WhaleReId')):
    folder1 = f'WhaleReId/{class_path}'
    for crop_path in os.listdir(folder1):
        folder2 = f'{folder1}/{crop_path}'
        data = {}
        for img_path in filter(lambda x: '.png' in x, os.listdir(folder2)):
            path = f'{folder2}/{img_path}'
            img = cv2.imread(path.replace('.png', '.jpg'))
            mask = cv2.imread(path)
            mask = cv2.cvtColor(mask, cv2.COLOR_BGR2GRAY)
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
               approx = cv2.approxPolyDP(cnt, epsilon,True)
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
                subsequence = [(tuple(elem[0],))for elem in approx.tolist()]
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
            #print(is_180_rot, start, end, path)
            #if is_180_rot:
            #    img = img[:, ::-1]
            #    mask = mask[:, ::-1]
            #    thresh = thresh[:, ::-1]
            #    contours, hierarchy = cv2.findContours(mask, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
            #    h, w = mask.shape
            #    start[0] = w - start[0]
            #    end[0] = w - end[0]

            rect = cv2.minAreaRect(contours[0])
            (x, y), (w, h), a = rect
            box = cv2.boxPoints(rect)
            box = np.int0(box)
            box = np.where(box > 0, box, 0)

            dtl = math.inf
            h, w = mask.shape
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
	            [maxWidth - 1, maxHeight - 1]], dtype = "float32")
            M = cv2.getPerspectiveTransform(box, dst)

            warp_mask = cv2.imread(path)
            warp_mask = cv2.cvtColor(warp_mask, cv2.COLOR_BGR2GRAY)
            warp_img = cv2.imread(path.replace('.png', '.jpg'))
            h, w, *_ = warp_mask.shape
            if h > w:
                warp_img = np.rot90(warp_img)
                warp_mask = np.rot90(warp_mask)
            warp_mask = cv2.warpPerspective(warp_mask, M, (maxWidth, maxHeight))
            warp_img = cv2.warpPerspective(warp_img, M, (maxWidth, maxHeight))
            h, w, *_ = warp_mask.shape
            if h > w:
                warp_img = np.rot90(warp_img, 3)
                warp_mask = np.rot90(warp_mask, 3)

            out_path = 'converted_' + path
            os.makedirs(os.path.join(*os.path.split(out_path)[:-1]), exist_ok=True)
            data[out_path] = [is_180_rot, warp_img, warp_mask]
        s = [int(is_180_rot) for (is_180_rot, _, _) in data.values()]
        s = sum(s) / len(s)
        for out_path, (is_180_rot, warp_img, warp_mask) in data.items():
            if s >= 0.5:
                warp_img = warp_img[:, ::-1]
                warp_mask = warp_mask[:, ::-1]
            cv2.imwrite(out_path, warp_mask)
            cv2.imwrite(out_path.replace('.png', '.jpg'), warp_img)
            #warp = cv2.cvtColor(warp, cv2.COLOR_BGR2RGB)

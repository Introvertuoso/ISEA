import cv2
import numpy as np

kernel = np.ones((2, 2), np.uint8)
rect_kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (2, 2))
rect_kernel_3_3 = cv2.getStructuringElement(cv2.MORPH_RECT, (3, 3))


def check_illumination_changes(img):
    if img is None:
        return False
    contours, hierarchy = cv2.findContours(img, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
    if len(contours) == 0:
        return False
    largest_areas = sorted(contours, key=cv2.contourArea)
    if cv2.contourArea(largest_areas[-1]) >= (img.shape[0] * img.shape[1]) / 4:
        return True


def cso_img_process(img):
    img = cv2.cvtColor(img, cv2.COLOR_RGB2GRAY)
    bilateral_filter = cv2.bilateralFilter(img, 10, 50, 50)
    thresh = cv2.adaptiveThreshold(bilateral_filter, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY, 57, 5)
    invert = cv2.bitwise_not(thresh, img.shape)
    return invert


def draw_largest_contour(img):
    mask = np.zeros(img.shape, np.uint8)
    contours, hierarchy = cv2.findContours(img, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
    if len(contours) == 0:
        return mask
    largest_areas = sorted(contours, key=cv2.contourArea)
    cv2.drawContours(mask, [largest_areas[-1]], 0, 255, -1)
    return mask


def pre_process(img):
    lab = cv2.cvtColor(img, cv2.COLOR_RGB2LAB)
    Lum, A, B = cv2.split(lab)
    eq = cv2.equalizeHist(Lum)
    gaussian = cv2.GaussianBlur(eq, (3, 3), 0)
    lab = cv2.merge([gaussian, A, B])
    img = cv2.cvtColor(lab, cv2.COLOR_LAB2RGB)
    gray = cv2.cvtColor(img, cv2.COLOR_RGB2GRAY)
    open = cv2.morphologyEx(gray, cv2.MORPH_OPEN, kernel, iterations=1)
    # cv2.imshow('open', open)
    img_gradient = sobel(open)
    cv2.imshow('img_gradient', img_gradient)
    return img_gradient


def sobel(img):
    scale = 1
    delta = 0
    ddepth = cv2.CV_16S
    grad_x = cv2.Sobel(img, ddepth, 1, 0, ksize=3, scale=scale, delta=delta, borderType=cv2.BORDER_DEFAULT)
    grad_y = cv2.Sobel(img, ddepth, 0, 1, ksize=3, scale=scale, delta=delta, borderType=cv2.BORDER_DEFAULT)
    mag = np.hypot(grad_x, grad_y)
    mag = mag / mag.max() * 255
    mag = np.uint8(mag)
    return mag


def diff_process(diff):
    diff = cv2.morphologyEx(diff, cv2.MORPH_OPEN, rect_kernel)
    diff = cv2.morphologyEx(diff, cv2.MORPH_CLOSE, rect_kernel, iterations=2)
    diff = draw_contours(diff)
    return diff


def mog_mask_processing(mask, kernel):
    mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, kernel)
    mask = cv2.threshold(mask, 0, 255, cv2.THRESH_BINARY)[1]
    return mask


def draw_contours(imgray, minSize=300):
    mask = np.zeros(imgray.shape, np.uint8)
    contours, hierarchy = cv2.findContours(imgray, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
    if len(contours) != 0 and len(hierarchy) != 0:
        for cnt in contours:
            if cv2.contourArea(cnt) > minSize:
                cv2.drawContours(mask, [cnt], 0, 255, -1)
    lines = lines_segments(mask, minSize)
    return cv2.absdiff(lines, mask)


def lines_segments(imgray, minSize):
    mask = np.zeros(imgray.shape, np.uint8)
    thresh = 0.3
    contours = cv2.findContours(imgray, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)[0]
    if len(contours) != 0:
        for cnt in contours:
            if cv2.contourArea(cnt) > minSize:
                rr = cv2.minAreaRect(cnt)
                bp = cv2.boxPoints(rr)
                l1 = np.sqrt(np.power((bp[0][0] - bp[1][0]), 2) + np.power((bp[0][1] - bp[1][1]), 2))
                l2 = np.sqrt(np.power((bp[0][0] - bp[2][0]), 2) + np.power((bp[0][1] - bp[2][1]), 2))
                l3 = np.sqrt(np.power((bp[0][0] - bp[3][0]), 2) + np.power((bp[0][1] - bp[3][1]), 2))
                length_list = [l1, l2, l3]
                length_list.sort()
                width = length_list[0]
                height = length_list[1]
                aspect_ratio = width / height
                if aspect_ratio < thresh:
                    cv2.drawContours(mask, [cnt], 0, 255, -1)
    return mask


def enlargeBB(rows, cols, x, y, w, h, padding):
    newx = x - padding
    newy = y - padding
    neww = w + (padding * 2)
    newh = h + (padding * 2)
    if newx < 0:
        newx = 0
    if newy < 0:
        newy = 0
    if newx + neww >= cols:
        neww = cols - newx
    if newy + newh >= rows:
        newh = rows - newy
    return newx, newy, neww, newh

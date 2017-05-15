import cv2
import numpy as np

THRESHOLD = 120
OPENING_SIZE = 3
BLUR_SIZE = 5
MIN_CONTOUR_AREA = 80
APPROX_PERI_RATIO = 0.045


class Contour:
    def __init__(self, contour):
        self.cv_contour = contour
        self.area = cv2.contourArea(contour)
        self.approx = Contour.approximate(contour).reshape(-1, 2)
        self.centroid = np.mean(self.approx, axis=0)

    def draw(self, image, color=(0, 255, 255), line_width=2):
        cv2.drawContours(image, self.cv_contour, -1, color, line_width)

    @staticmethod
    def draw_all(contours, image, color=(0, 255, 255), line_width=2):
        for contour in contours:
            contour.draw(image, color, line_width)

    @property
    def vertex_count(self):
        return len(self.approx)

    @staticmethod
    def approximate(cv_contour):
        peri = cv2.arcLength(cv_contour, True)
        return cv2.approxPolyDP(cv_contour, APPROX_PERI_RATIO * peri, True)

    @staticmethod
    def find_triangles_and_squares(img):
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        _, thresh = cv2.threshold(gray, THRESHOLD, 255, cv2.THRESH_BINARY_INV)
        opening_kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (OPENING_SIZE, OPENING_SIZE))
        opening = cv2.morphologyEx(thresh, cv2.MORPH_OPEN, opening_kernel)
        blurred = cv2.GaussianBlur(opening, (BLUR_SIZE, BLUR_SIZE), 0)
        edges = cv2.Canny(blurred, 100, 150)
        contours = cv2.findContours(edges.copy(), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)[1]
        objects = map(Contour, contours)
        triangles = filter(lambda c: c.area > MIN_CONTOUR_AREA and c.vertex_count == 3, objects)
        squares = filter(lambda c: c.area > MIN_CONTOUR_AREA and c.vertex_count == 4, objects)
        return triangles, squares

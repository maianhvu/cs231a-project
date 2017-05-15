import cv2
import numpy as np


class ReferenceLine:
    def __init__(self, triangles):
        points = np.reshape(map(lambda t: t.centroid, triangles), (-1, 2))
        points = np.hstack([points, np.ones((points.shape[0], 1))])
        _, _, vt = np.linalg.svd(points)
        self.vector_form = vt[-1:, :].reshape(3, -1)

    def draw(self, image, color=(0, 255, 0), thickness=1):
        vector = self.vector_form.ravel()
        left_y = int(-vector[2] / vector[1])
        right_y = int((-vector[2] - (image.shape[1] - 1) * vector[0]) / vector[1])
        cv2.line(image, (0, left_y), (image.shape[1] - 1, right_y), color, thickness, cv2.LINE_AA)

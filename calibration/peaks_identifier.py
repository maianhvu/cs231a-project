import numpy as np
import cv2
from reference_line import ReferenceLine


class PeaksIdentifier:
    def __init__(self, triangles, ref_line=None):
        self.triangles = triangles
        if ref_line is None:
            self.ref_line = ReferenceLine(triangles)
        else:
            self.ref_line = ref_line

    def identify(self):
        approx = map(lambda t: t.approx, self.triangles)
        vertices = reduce(lambda a, b: np.vstack([a, b]), approx)
        vertices = np.hstack([
            vertices,
            np.ones((vertices.shape[0], 1))
        ])
        relative_positions = vertices.dot(self.ref_line.vector_form)
        peaks = vertices[np.where(relative_positions < 0)[0]][:, :2]

        if len(peaks) > len(vertices) - len(peaks):
            peaks = vertices[np.where(relative_positions > 0)[0]][:, :2]

        peaks = sorted(peaks, key=lambda pk: pk[0])
        result = {}
        for (i, p) in enumerate(peaks):
            result[i] = p

        return result

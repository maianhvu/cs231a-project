import numpy as np
import cv2

PEAKS = {
    0: [100, -60, 100],
    1: [360, -60, 100],
    2: [620, -60, 100]
}
SQUARES = {
    0: [[-160, 0, 0],   [-40, 0, 0],   [-160, 120, 0], [-40, 120, 0]],
    1: [[-160, 160, 0], [-40, 160, 0], [-160, 280, 0], [-40, 280, 0]],
    2: [[-160, 320, 0], [-40, 320, 0], [-160, 440, 0], [-40, 440, 0]],
    3: [[760, 0, 0],   [880, 0, 0],   [760, 120, 0], [880, 120, 0]],
    4: [[760, 160, 0], [880, 160, 0], [760, 280, 0], [880, 280, 0]],
    5: [[760, 320, 0], [880, 320, 0], [760, 440, 0], [880, 440, 0]]
}
AXIS_COLORS = [(0, 0, 255), (0, 255, 0), (255, 0, 0)]


class CalibrationRig:
    def __init__(self, labeled_peaks, labeled_squares):
        self.peaks = labeled_peaks
        self.squares = labeled_squares
        self.__M = None

    @property
    def camera_matrix(self):
        if self.__M is None:
            P = np.empty((0, 12))
            for i, peak in self.peaks.iteritems():
                point3d_h = np.hstack([PEAKS[i], [1]]).reshape(1, 4)
                matrix = np.vstack([
                    np.hstack([point3d_h, np.zeros((1, 4)), -peak[0] * point3d_h]),
                    np.hstack([np.zeros((1, 4)), point3d_h, -peak[1] * point3d_h])
                ])
                P = np.vstack([P, matrix])

            for i, sqr in self.squares.iteritems():
                point3d_h = np.hstack([np.mean(SQUARES[i], axis=0), [1]]).reshape(1, 4)
                sqr_centroid = sqr.centroid.ravel()
                matrix = np.vstack([
                    np.hstack([point3d_h, np.zeros((1, 4)), -sqr_centroid[0] * point3d_h]),
                    np.hstack([np.zeros((1, 4)), point3d_h, -sqr_centroid[1] * point3d_h])
                ])
                P = np.vstack([P, matrix])

            _, _, VT = np.linalg.svd(P)
            self.__M = VT[-1:, :].reshape(3, 4)

        return self.__M

    def draw_axes(self, image, axis_length=100):
        M = self.camera_matrix
        point3ds = np.vstack([
            [0, 0, 0, 1],
            np.hstack([
                np.eye(3) * axis_length,
                np.ones((3, 1))
            ])
        ])
        points = point3ds.dot(M.T)
        points = (points / points[:, 2:])[:, :2]
        for i in xrange(1, 4):
            cv2.line(image, tuple(np.int0(points[0])), tuple(np.int0(points[i])),
                     AXIS_COLORS[i-1], 2, cv2.LINE_AA)

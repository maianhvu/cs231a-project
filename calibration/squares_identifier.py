import numpy as np
import cv2


class SquaresIdentifier:
    def __init__(self, squares, ref_line):
        self.squares = squares
        self.ref_line = ref_line

    def identify(self, image, triangles):
        image_center = np.flip(image.shape[:2], axis=0) / 2
        triangle_centroid_dists = np.sum(np.square(
            np.array(map(lambda t: t.centroid, triangles)) - image_center
        ), axis=1)
        min_triangle = triangles[np.argmin(triangle_centroid_dists)]
        center_x = min_triangle.centroid[0]
        ref_line_vector = self.ref_line.vector_form.ravel()
        center_y = (ref_line_vector[0] * center_x + ref_line_vector[2]) / -ref_line_vector[1]
        center = np.array([center_x, center_y])
        square_centroids = np.array(map(lambda s: s.centroid, self.squares))

        def normalize(vs): return vs / np.linalg.norm(vs, axis=0)

        vectors = normalize(center - square_centroids)
        line_vector = center - [0, -ref_line_vector[2] / ref_line_vector[1]]
        line_vector /= np.linalg.norm(line_vector)
        angles = np.arccos(np.sum(vectors * line_vector, axis=1))
        square_centroids_homo = np.hstack([
            square_centroids,
            np.ones((square_centroids.shape[0], 1))
        ])
        distances = np.linalg.norm(square_centroids_homo * self.ref_line.vector_form.T, axis=1)

        left_indexes = np.where(angles < 1.5)[0]
        right_indexes = np.where(angles >= 1.5)[0]
        squares = np.array(self.squares)
        left_squares = squares[left_indexes[np.argsort(distances[left_indexes])]]
        right_squares = squares[right_indexes[np.argsort(distances[right_indexes])]]
        result = {}
        for (i, s) in enumerate(left_squares):
            result[i] = s

        for (i, s) in enumerate(right_squares):
            result[i+3] = s

        return result

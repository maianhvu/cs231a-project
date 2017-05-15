import cv2
import numpy as np
import imutils
from calibration import Contour, ReferenceLine, SquaresIdentifier, PeaksIdentifier, CalibrationRig

USE_VIDEO = True

def process_image(image):
    triangles, squares = Contour.find_triangles_and_squares(image)
    ref_line = ReferenceLine(triangles)
    # ref_line.draw(image)

    peaks_identifier = PeaksIdentifier(triangles, ref_line)
    peaks = peaks_identifier.identify()
    # for label, peak in peaks.iteritems():
    #     peak = np.int0(peak)
    #     cv2.circle(image, tuple(peak), 5, (0, 0, 255), 2, lineType=cv2.LINE_AA)
    #     cv2.putText(image, str(label), (peak[0], peak[1] - 10), cv2.FONT_HERSHEY_SIMPLEX,
    #                 0.4, (0, 0, 255), 2)

    sqr_identifier = SquaresIdentifier(squares, ref_line)
    identified_squares = sqr_identifier.identify(image, triangles)
    # for label, s in identified_squares.iteritems():
    #     cv2.putText(image, str(label), tuple(np.int0(s.centroid)), cv2.FONT_HERSHEY_SIMPLEX,
    #                 0.5, (0, 255, 255), 2)

    calib = CalibrationRig(peaks, identified_squares)
    calib.draw_axes(image)

    return image


if USE_VIDEO:
    cap = cv2.VideoCapture("assets/video.mp4")
    wait_dur = 0

    while True:
        ret, frame = cap.read()
        if not ret:
            break

        image = process_image(frame)
        cv2.imshow("Frame", image)
        if cv2.waitKey(wait_dur) & 0xFF == ord('q'):
            break

        wait_dur = 1

    cap.release()
    cv2.destroyAllWindows()

else:
    image = cv2.imread("assets/IMG_4205.JPG")
    image = imutils.resize(image, width=720)
    image = process_image(image)
    cv2.imshow("Image", image)
    cv2.waitKey(0)

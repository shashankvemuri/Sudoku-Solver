import cv2
import numpy as np


def show(*args):
    for i, j in enumerate(args):
        cv2.imshow(str(i), j)
    cv2.waitKey(0)
    cv2.destroyAllWindows()


def process(img):
    kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (2, 2))
    greyscale = img if len(img.shape) == 2 else cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    denoise = cv2.GaussianBlur(greyscale, (9, 9), 0)
    thresh = cv2.adaptiveThreshold(
        denoise, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY, 11, 2
    )
    inverted = cv2.bitwise_not(thresh, 0)
    morph = cv2.morphologyEx(inverted, cv2.MORPH_OPEN, kernel)
    dilated = cv2.dilate(morph, kernel, iterations=1)
    return dilated


def get_corners(img):
    contours, hire = cv2.findContours(img, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    contours = sorted(contours, key=lambda x: cv2.contourArea(x), reverse=True)
    largest_contour = np.squeeze(contours[0])

    sums = [sum(i) for i in largest_contour]
    differences = [i[0] - i[1] for i in largest_contour]

    top_left = np.argmin(sums)
    top_right = np.argmax(differences)
    bottom_left = np.argmax(sums)
    bottom_right = np.argmin(differences)

    corners = [
        largest_contour[top_left],
        largest_contour[top_right],
        largest_contour[bottom_left],
        largest_contour[bottom_right],
    ]
    return corners


def display_points(in_img, points, radius=20, color=(0, 0, 255)):
    # https://gist.github.com/mineshpatel1/22e86200eee86ebe3e221343b26fc3f3#file-display_points-py
    img = in_img.copy()

    # Dynamically change to a color image if necessary
    if len(color) == 3:
        if len(img.shape) == 2:
            img = cv2.cvtColor(img, cv2.COLOR_GRAY2BGR)
        elif img.shape[2] == 1:
            img = cv2.cvtColor(img, cv2.COLOR_GRAY2BGR)

    for point in points:
        img = cv2.circle(img, tuple(int(x) for x in point), radius, color, -1)
    return img


def main(path, out_path, framerate, dims):
    cap = cv2.VideoCapture(path)

    fourcc = cv2.VideoWriter_fourcc(*"XVID")
    out = cv2.VideoWriter(out_path, fourcc, framerate, dims)

    while True:
        # Capture frame-by-frame
        ret, frame = cap.read()

        if frame is None:
            break

        processed = process(frame)
        corners = get_corners(processed)
        points = display_points(frame, corners)

        out.write(points)
        if cv2.waitKey(1) & 0xFF == ord("q"):
            break

    # When everything done, release the capture
    cap.release()
    cv2.destroyAllWindows()


if __name__ == "__main__":
    main("sudoku.mp4", "corners.avi", 30.0, (1920, 1080))


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


def get_edges(img):
    contours, hire = cv2.findContours(img, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    contours = sorted(contours, key=lambda x: cv2.contourArea(x), reverse=True)
    largest_contour = np.squeeze(contours[0])
    return largest_contour


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
        contour = get_edges(processed)
        cv2.drawContours(frame, [contour], 0, (0, 0, 255), 3)

        out.write(frame)
        if cv2.waitKey(1) & 0xFF == ord("q"):
            break

    # When everything done, release the capture
    cap.release()
    cv2.destroyAllWindows()


if __name__ == "__main__":
    main("sudoku.mp4", "edges.avi", 30.0, (1920, 1080))


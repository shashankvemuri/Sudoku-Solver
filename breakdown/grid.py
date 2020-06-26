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


def transform(pts, img):  # TODO: Spline transform, remove this
    pts = np.float32(pts)
    top_l, top_r, bot_l, bot_r = pts[0], pts[1], pts[2], pts[3]

    def pythagoras(pt1, pt2):
        return np.sqrt((pt2[0] - pt1[0]) ** 2 + (pt2[1] - pt1[1]) ** 2)

    width = int(max(pythagoras(bot_r, bot_l), pythagoras(top_r, top_l)))
    height = int(max(pythagoras(top_r, bot_r), pythagoras(top_l, bot_l)))
    square = max(width, height) // 9 * 9  # Making the image dimensions divisible by 9

    dim = np.array(
        ([0, 0], [square - 1, 0], [square - 1, square - 1], [0, square - 1]), dtype="float32"
    )
    matrix = cv2.getPerspectiveTransform(pts, dim)
    warped = cv2.warpPerspective(img, matrix, (square, square))
    return warped


def get_grid_lines(img, length=12):
    horizontal = np.copy(img)
    cols = horizontal.shape[1]
    horizontal_size = cols // length
    horizontal_structure = cv2.getStructuringElement(cv2.MORPH_RECT, (horizontal_size, 1))
    horizontal = cv2.erode(horizontal, horizontal_structure)
    horizontal = cv2.dilate(horizontal, horizontal_structure)

    vertical = np.copy(img)
    rows = vertical.shape[0]
    vertical_size = rows // length
    vertical_structure = cv2.getStructuringElement(cv2.MORPH_RECT, (1, vertical_size))
    vertical = cv2.erode(vertical, vertical_structure)
    vertical = cv2.dilate(vertical, vertical_structure)

    return vertical, horizontal


def create_grid_mask(img, vertical, horizontal):
    grid = cv2.add(horizontal, vertical)
    grid = cv2.adaptiveThreshold(
        grid, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY, 235, 2
    )
    grid = cv2.dilate(grid, cv2.getStructuringElement(cv2.MORPH_RECT, (3, 3)), iterations=2)
    # pts = cv2.HoughLines(grid, 0.1, np.pi / 90, 200)

    # try:
    #     img = cv2.cvtColor(img, cv2.COLOR_GRAY2BGR)
    # except:
    #     pass

    # def draw_lines(im, pts):
    #     im = np.copy(im)
    #     pts = np.squeeze(pts)
    #     for r, theta in pts:
    #         a = np.cos(theta)
    #         b = np.sin(theta)
    #         x0 = a * r
    #         y0 = b * r
    #         x1 = int(x0 + 1000 * (-b))
    #         y1 = int(y0 + 1000 * a)
    #         x2 = int(x0 - 1000 * (-b))
    #         y2 = int(y0 - 1000 * a)
    #         cv2.line(im, (x1, y1), (x2, y2), (0, 0, 255), 2)
    #     return im

    x = np.zeros((grid.shape[0], grid.shape[0]), dtype=np.uint8)
    for i, a in enumerate(x):
        for j, _ in enumerate(a):
            x[i][j] = 150

    # show(cv2.bitwise_and(x, grid))

    grid = cv2.cvtColor(grid, cv2.COLOR_GRAY2RGB)
    grid_red = grid.copy()

    for i, a in enumerate(grid_red):
        for j, v in enumerate(a):
            grid_red[i][j][0] = 0
            grid_red[i][j][1] = 0

    # lines = draw_lines(img, pts)

    return grid, grid_red


def inverse_perspective(img, dst_img, pts):
    pts_source = np.array(
        [
            [0, 0],
            [img.shape[1] - 1, 0],
            [img.shape[1] - 1, img.shape[0] - 1],
            [0, img.shape[0] - 1],
        ],
        dtype="float32",
    )
    h, status = cv2.findHomography(pts_source, pts)
    warped = cv2.warpPerspective(img, h, (dst_img.shape[1], dst_img.shape[0]))
    cv2.fillConvexPoly(dst_img, np.ceil(pts).astype(int), 0, 16)
    dst_img = dst_img + warped
    return dst_img


def extract_digits(img, img_rgb):
    contours, _ = cv2.findContours(img, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    img_area = img.shape[0] * img.shape[1]
    # Reversing contours list to loop with y coord ascending, and removing small bits of noise
    contours_denoise = [i for i in contours[::-1] if cv2.contourArea(i) > img_area * 0.0005]
    _, y_compare, _, _ = cv2.boundingRect(contours_denoise[0])
    digits = []
    row = []

    for i in contours_denoise:
        x, y, w, h = cv2.boundingRect(i)
        cropped = img[y : y + h, x : x + w]

        cv2.rectangle(img_rgb, (x, y), (x + w, y + h), (0, 0, 255), 2)

        if y - y_compare > img.shape[1] // 40:
            row = [i[0] for i in sorted(row, key=lambda x: x[1])]

            for j in row:
                digits.append(j)
            row = []

        row.append((cropped, x))

        y_compare = y
    # Last loop doesn't add row
    row = [i[0] for i in sorted(row, key=lambda x: x[1])]
    for i in row:
        digits.append(i)

    return img_rgb


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
        warped = transform(corners, processed)
        warped_rgb = transform(corners, frame)
        vertical_lines, horizontal_lines = get_grid_lines(warped)
        grid, grid_red = create_grid_mask(warped_rgb, vertical_lines, horizontal_lines)

        warped_rgb[np.where(grid > 200)] = grid_red[np.where(grid > 200)]
        warped_inverse = inverse_perspective(warped_rgb, frame.copy(), np.array(corners))

        out.write(warped_inverse)
        if cv2.waitKey(1) & 0xFF == ord("q"):
            break

    # When everything done, release the capture
    cap.release()
    cv2.destroyAllWindows()


if __name__ == "__main__":
    main("sudoku.mp4", "grid.avi", 30.0, (1920, 1080))


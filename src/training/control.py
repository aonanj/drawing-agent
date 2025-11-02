import cv2
def canny_map(png_path, out_path):
    img = cv2.imread(png_path, cv2.IMREAD_GRAYSCALE)
    edges = cv2.Canny(img, 60, 180)  # type: ignore
    cv2.imwrite(out_path, edges)

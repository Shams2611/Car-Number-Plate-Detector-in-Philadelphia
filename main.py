import cv2
import imutils
import numpy as np
import easyocr
import matplotlib.pyplot as plt


def load_and_preprocess_image(image_path):
    """
    Load and preprocess the input image.

    :param image_path: Path to the input image.
    :return: Original image, thresholded image, and grayscale image.
    """

    # Load and preprocess the input image
    img = cv2.imread(image_path)
    gray_scale = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    bifilter = cv2.bilateralFilter(gray_scale, 11, 17, 17)
    adaptive_thresh = cv2.adaptiveThreshold(bifilter, 255, cv2.ADAPTIVE_THRESH_MEAN_C, cv2.THRESH_BINARY, 11, 2)
    return img, adaptive_thresh, gray_scale


def find_license_plate_contour(image):
    """
    Find the license plate contour in the image.

    :parameter image: Grayscale image.
    :return: Contour coordinates of the license plate.
    """
    keypoints = cv2.findContours(image.copy(), cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
    contours = imutils.grab_contours(keypoints)
    contours = sorted(contours, key=cv2.contourArea, reverse=True)

    location = None
    for contour in contours:
        approx = cv2.approxPolyDP(contour, 10, True)
        if len(approx) == 4:
            x, y, w, h = cv2.boundingRect(approx)
            aspect_ratio = w / h
            if aspect_ratio > 1.5:
                location = approx
                break

    return location


''
def draw_contours_and_crop_roi(img, adaptive_thresh, gray_scale):
    """
    Draw contours on the original image and crop the region of interest.

    :param img: Original image.
    :param adaptive_thresh: Thresholded image.
    :param gray_scale: Grayscale image.
    :return: Original image with contours, cropped license plate image, and license plate contour coordinates.
    """

    # Find and filter contours based on aspect ratio
    location = find_license_plate_contour(adaptive_thresh)

    # Draw contours on the image
    cv2.drawContours(img, [location], 0, (0, 255, 0), 2)
    cropped_image = crop_license_plate_region(gray_scale, location)
    return img, cropped_image, location





def create_contour_mask(image, contour):
    """
    Create a binary mask based on the contour.

    :param image: Input image.
    :param contour: Contour coordinates.
    :return: Binary mask.
    """
    masking = np.zeros(image.shape, np.uint8)
    cv2.drawContours(masking, [contour], 0, [255], -1)
    return masking


def crop_license_plate_region(image, contour):
    """
    Crop the license plate region from the image based on the contour.

    :param image: Input image.
    :param contour: Contour coordinates.
    :return: Cropped license plate region.
    """
    (x, y, w, h) = cv2.boundingRect(contour)
    cropped_image = image[y:y + h, x:x + w]
    return cropped_image







def apply_threshold_and_read_text(cropped_image):
    """
    Apply adaptive thresholding on the cropped image and use EasyOCR to read text.

    :param cropped_image: Cropped license plate image.
    :return: Recognized text.
    """
    cropped_adaptive_thresh = cv2.adaptiveThreshold(cropped_image, 255, cv2.ADAPTIVE_THRESH_MEAN_C, cv2.THRESH_BINARY,
                                                    11, 2)
    reader = easyocr.Reader(['en'])
    result = reader.readtext(cropped_adaptive_thresh)
    text = result[0][-2]
    return text


def display_result(img, text, location):
    """
    Display the recognized text on the original image.

    :param img: Original image.
    :param text: Recognized text.
    :param location: Contour coordinates.
    """
    font = cv2.FONT_HERSHEY_COMPLEX
    res = cv2.putText(img, text=text, org=(location[3][0][0], location[1][0][1] + 60), fontFace=font, fontScale=1,
                      color=(0, 255, 0), thickness=2, lineType=cv2.LINE_AA)
    plt.imshow(cv2.cvtColor(res, cv2.COLOR_BGR2RGB))
    plt.show()


def read_license_plate(image_path):
    """
    Read the license plate from an image and display the result.

    :param image_path: Path to the input image file.
    """
    img, adaptive_thresh, gray_scale = load_and_preprocess_image(image_path)
    new_image, cropped_image, location = draw_contours_and_crop_roi(img, adaptive_thresh, gray_scale)
    text = apply_threshold_and_read_text(cropped_image)
    display_result(new_image, text, location)


# Example usage
read_license_plate('image1.jpg')
read_license_plate('image2.jpg')
read_license_plate('image3.jpg')
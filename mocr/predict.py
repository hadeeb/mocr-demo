import base64
import cv2
import numpy as np
from keras.models import load_model

IMAGE_SIZE = 32


def predict(img):
    img = process_image(img)
    data = apply_nn(img)
    return data


def process_image(img):
    image = data_uri_to_cv2_img(img)

    # greyscale
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

    # binary
    (__, img_bw) = cv2.threshold(gray, 128, 255, cv2.THRESH_BINARY_INV)

    # dilation

    # find contours
    __, ctrs, __ = cv2.findContours(img_bw.copy(), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    # sort contours
    char_ctr = sorted(ctrs, key=lambda ctr: (cv2.boundingRect(ctr)[2] * cv2.boundingRect(ctr)[3]),
                      reverse=True)[0]
    # Get bounding box
    x, y, w, h = cv2.boundingRect(char_ctr)

    # Getting ROI
    roi = img_bw[y:y + h, x:x + w]
    return skeletize(crop(roi, IMAGE_SIZE))


def crop(image, desired_size):
    """Crop and pad to req size"""
    old_size = image.shape[:2]  # old_size is in (height, width) format
    ratio = float(desired_size) / max(old_size)
    new_size = tuple([int(x * ratio) for x in old_size])

    # new_size should be in (width, height) format
    im = cv2.resize(image, (new_size[1], new_size[0]))

    delta_w = desired_size - new_size[1]
    delta_h = desired_size - new_size[0]
    top, bottom = delta_h // 2, delta_h - (delta_h // 2)
    left, right = delta_w // 2, delta_w - (delta_w // 2)

    color = [0, 0, 0]
    new_im = cv2.copyMakeBorder(im, top, bottom, left, right, cv2.BORDER_CONSTANT,
                                value=color)

    return new_im


def skeletize(img):
    size = np.size(img)
    skel = np.zeros(img.shape, np.uint8)
    element = cv2.getStructuringElement(cv2.MORPH_CROSS, (3, 3))
    done = False

    while not done:
        eroded = cv2.erode(img, element)
        temp = cv2.dilate(eroded, element)
        temp = cv2.subtract(img, temp)
        skel = cv2.bitwise_or(skel, temp)
        img = eroded.copy()

        zeroes = size - cv2.countNonZero(img)
        if zeroes == size:
            done = True

    return skel


def apply_nn(data):
    image_data = data
    dataset = np.asarray(image_data)
    dataset = dataset.reshape((-1, 32, 32, 1)).astype(np.float32)
    model = load_model('mocr/model.h5')
    a = model.predict(dataset)[0]

    classes = np.genfromtxt('mocr/classes.csv', delimiter=',')[:, 1].astype(int)

    res = np.column_stack((classes, a)).astype(int)
    return res


def data_uri_to_cv2_img(uri):
    encoded_data = uri.split(',')[1]
    nparr = np.fromstring(base64.b64decode(encoded_data), np.uint8)
    img = cv2.imdecode(nparr, cv2.IMREAD_UNCHANGED)
    alpha_channel = img[:, :, 3]
    rgb_channels = img[:, :, :3]

    # White Background Image
    white_background_image = np.ones_like(rgb_channels, dtype=np.uint8) * 255

    # Alpha factor
    alpha_factor = alpha_channel[:, :, np.newaxis].astype(np.float32) / 255.0
    alpha_factor = np.concatenate((alpha_factor, alpha_factor, alpha_factor), axis=2)

    # Transparent Image Rendered on White Background
    base = rgb_channels.astype(np.float32) * alpha_factor
    white = white_background_image.astype(np.float32) * (1 - alpha_factor)
    final_image = base + white
    return final_image.astype(np.uint8)

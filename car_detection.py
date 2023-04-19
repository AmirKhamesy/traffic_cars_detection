import time
import urllib.request

import cv2
import numpy as np
import requests

gaglardi_url = 'https://ns-webcams.its.sfu.ca/public/images/gaglardi-current.jpg'
clark_url = 'https://trafficcams.vancouver.ca/cameraimages/clark1east.jpg'

car_cascade = cv2.CascadeClassifier(
    'car.xml')


def get_image(url):
    img_resp = requests.get(url)
    img_arr = np.array(bytearray(img_resp.content), dtype=np.uint8)
    img = cv2.imdecode(img_arr, cv2.IMREAD_COLOR)
    # Resize the image to the same dimensions
    return cv2.resize(img, (600, 450))


def detect_cars():
    while True:
        gaglardi_img = get_image(gaglardi_url)
        clark_img = get_image(clark_url)

        # Detect cars in Gaglardi image
        gaglardi_gray = cv2.cvtColor(gaglardi_img, cv2.COLOR_BGR2GRAY)
        gaglardi_cars = car_cascade.detectMultiScale(
            gaglardi_gray, scaleFactor=1.1, minNeighbors=4)  # Lower minNeighbors to help detect more cars

        # Draw boxes around the detected cars in Gaglardi image
        for (x, y, w, h) in gaglardi_cars:
            cv2.rectangle(gaglardi_img, (x, y), (x + w, y + h), (0, 0, 255), 2)

        # Detect cars in Clark image
        clark_gray = cv2.cvtColor(clark_img, cv2.COLOR_BGR2GRAY)
        clark_cars = car_cascade.detectMultiScale(
            clark_gray, scaleFactor=1.1, minNeighbors=2)  # Lower minNeighbors to help detect more cars

        # Draw boxes around the detected cars in Clark image
        for (x, y, w, h) in clark_cars:
            cv2.rectangle(clark_img, (x, y), (x + w, y + h), (0, 0, 255), 2)

        # Concatenate the two images side by side
        both_img = np.concatenate((gaglardi_img, clark_img), axis=1)

        # Show the combined image with detected cars
        cv2.imshow('Gaglardi and Clark Traffic Cameras', both_img)

        # Exit if 'q' is pressed
        if cv2.waitKey(1) == ord('q'):
            break

        # Wait for 5 seconds before retrieving and processing the next set of images
        time.sleep(5)

    cv2.destroyAllWindows()


detect_cars()

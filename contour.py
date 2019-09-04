import cv2
import imutils
import four_point_transform as fp
def contour_detect(edged, image):
    orig = image.copy()
    ratio = image.shape[0] / 500.0
    # show the original image and the edge detected image
    print("STEP 1: Edge Detection")
    # find the contours in the edged image, keeping only the
    # largest ones, and initialize the screen contour
    cnts = cv2.findContours(edged.copy(), cv2.RETR_LIST, cv2.CHAIN_APPROX_SIMPLE)
    print(cnts[0][0])
    print("-------------------------------------")
    cnts = imutils.grab_contours(cnts)
    print(cnts[0])
    exit()
    cnts = sorted(cnts, key = cv2.contourArea, reverse = True)[:5]
    screenCnt = None
    # loop over the contours
    for c in cnts:
        # approximate the contour
        peri = cv2.arcLength(c, True)
        approx = cv2.approxPolyDP(c, 0.02 * peri, True)
    
        # if our approximated contour has four points, then we
        # can assume that we have found our screen
        if len(approx) == 4:
            screenCnt = approx
            break
    if screenCnt is not None:
        # show the contour (outline) of the piece of paper
        print("STEP 2: Find contours of paper")
        cv2.drawContours(image, [screenCnt], -1, (0, 255, 0), 2)
        # apply the four point transform to obtain a top-down
        # view of the original image
        orig = fp.four_point_transform(orig, screenCnt.reshape(4, 2) * ratio)
    return orig
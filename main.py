import cv2 #openCV
import numpy as np #numpy library

#temp variable
alert = 0
count_frame = 0

#input
mode = "night" #night / day
#koordinat ROI (Range of Interest)
x1 = 0
x2 = 1280
y1 = 0
y2 = 720
frame_skip = 5
detection_size = 600
#haarcascade
scale_factor = 2 #atur sensitivity haarcascade
#alert
frame_countdown = 10

# cap = cv2.VideoCapture(0)
# cap = cv2.VideoCapture("in.avi")
cap = cv2.VideoCapture("test5a.mp4")
# cap = cv2.VideoCapture("test7.avi")

#HOG Descriptor
hog = cv2.HOGDescriptor()
hog.setSVMDetector(cv2.HOGDescriptor_getDefaultPeopleDetector())

#Haarcascade
fullbody_cascade = cv2.CascadeClassifier("haarcascade_fullbody.xml")
upperbody_cascade = cv2.CascadeClassifier("haarcascade_upperbody.xml")
lowerbody_cascade = cv2.CascadeClassifier("haarcascade_lowerbody.xml")

#output
out = cv2.VideoWriter(
    "output.avi", cv2.VideoWriter_fourcc(*"MJPG"), 15, (1080, 720))

#read video feed
ret, frame1 = cap.read()
ret, frame2 = cap.read()

#Resize to ROI
frame1 = frame1[y1: y2, x1: x2]
frame2 = frame2[y1: y2, x1: x2]

while cap.isOpened():
    try:
        frame1 = cv2.resize(frame1, (1080, 720))
        frame2 = cv2.resize(frame2, (1080, 720))
    except Exception:
        break

    #Find contour
    diff = cv2.absdiff(frame1, frame2)
    diff_gray = cv2.cvtColor(diff, cv2.COLOR_BGR2GRAY)

    blur = cv2.GaussianBlur(diff_gray, (5, 5), 0) #Reduce noise

    dilated = cv2.dilate(blur, None, iterations=3) #Reduce noise
    _, thresh = cv2.threshold(dilated, 20, 255, cv2.THRESH_BINARY) #Black / White only

    cv2.imshow("Masking", thresh)

    contours, _ = cv2.findContours(
        thresh, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)

    motion = 0
    for contour in contours:
        (x, y, w, h) = cv2.boundingRect(contour)
        if cv2.contourArea(contour) < detection_size:
            continue
        cv2.rectangle(frame1, (x, y), (x+w, y+h), (0, 0, 255), 2)
        motion += 1

    #Haarcascade and HOG Descriptor
    gray = cv2.cvtColor(frame1, cv2.COLOR_BGR2GRAY)
    gray = cv2.equalizeHist(gray) #Increase Contrast

    if mode == "day":
        boxes, weights = hog.detectMultiScale(gray, winStride=(8, 8))

        boxes = np.array([[x, y, x + w, y + h] for (x, y, w, h) in boxes])

        for (xA, yA, xB, yB) in boxes:
            cv2.rectangle(frame1, (xA, yA), (xB, yB), (255, 0, 0), 2)

        hog_detect = len(boxes)

        if motion + hog_detect > 0:
            cv2.putText(frame1, f"{motion} Movement Dectected!", (10, 20),
                        cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2)
            cv2.putText(frame1, f"{hog_detect} Humans Dectected! (HOG)", (10, 80),
                        cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 0, 0), 2)

    elif mode == "night":
        full_bodies = fullbody_cascade.detectMultiScale(
            gray, scale_factor, 2, minSize=(40, 40))
        for (x, y, w, h) in full_bodies:
            cv2.rectangle(frame1, (x, y), (x+w, y+h), (0, 255, 0), 2)

        upper_bodies = upperbody_cascade.detectMultiScale(
            gray, scale_factor, 2, minSize=(15, 15))
        for (x, y, w, h) in upper_bodies:
            cv2.rectangle(frame1, (x, y), (x+w, y+h), (0, 255, 0), 2)

        lower_bodies = lowerbody_cascade.detectMultiScale(
            gray, scale_factor, 2, minSize=(15, 15))
        for (x, y, w, h) in lower_bodies:
            cv2.rectangle(frame1, (x, y), (x+w, y+h), (0, 255, 0), 2)

        haar_detect = len(full_bodies) + len(upper_bodies) + len(lower_bodies)

        if motion + haar_detect > 0 or alert == 1:
            count_frame += 1
            if (count_frame > frame_countdown or alert == 1):
                alert = 1
                cv2.putText(frame1, "Movement Dectected!", (10, 20),
                            cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2)
        else:
            count_frame = 0
    else:
        raise ValueError("Input mode is invalid")

    out.write(frame1.astype("uint8"))
    cv2.imshow("frame", frame1)
    frame1 = frame2
    for i in range(frame_skip):
        ret, frame2 = cap.read()
        try:
            frame2 = frame2[y1: y2, x1: x2]
        except:
            break
    if cv2.waitKey(1) & 0xFF == ord("q"):
        break

# When everything done, release the capture
out.release()
cap.release()
cv2.destroyAllWindows()
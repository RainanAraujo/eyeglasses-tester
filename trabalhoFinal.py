import cv2
import math
import json

from numpy import around

face_cascade = cv2.CascadeClassifier('./assets/haarcascades/haarcascade_frontalface_alt.xml')
eye_cascade = cv2.CascadeClassifier('./assets/haarcascades/haarcascade_eye.xml')
capture = cv2.VideoCapture(0)

fileJson = open('glasses.json')
data = json.load(fileJson)

# Change index of data for change the glass (0 or 1 or 2)
glasses = data["glasses"][2]
glass = cv2.imread(glasses["path"], cv2.IMREAD_UNCHANGED)

# Change value of your head width in mm
your_face_width_mm = 150

last_width_glass = 0
last_pivot_glass = (0, 0)
buffer_to_move_glass = 0


def divSize(value):
    tmp_value = value / 2
    if value % 2 == 0:
        return round(tmp_value)
    return math.ceil(tmp_value)


def getWidthGlassInPx(face_width_mm, glass_width_mm, face_width_px):
    return int((glass_width_mm * face_width_px) / face_width_mm)


def insertGlass(frame, face_ret, glass_pivot, glass, your_face_mm):

    glass_width = smoothSizeWidthGlass(getWidthGlassInPx(your_face_mm, glasses["width"], face_ret[2]))
    glass_height = int(glasses["height"] * (glass_width / float(glasses["width"])))

    pivot_glass_in_frame = (face_ret[0] + glass_pivot[0], face_ret[1] + glass_pivot[1])
    glass_resized = cv2.resize(glass, (glass_width, glass_height), interpolation=cv2.INTER_AREA)

    start_set_x = pivot_glass_in_frame[0] - int(glass_width/2)
    start_set_y = pivot_glass_in_frame[1] - int(glass_height/2)
    end_set_x = pivot_glass_in_frame[0] + divSize(glass_width)
    end_set_y = pivot_glass_in_frame[1] + divSize(glass_height)

    frame[start_set_y:end_set_y, start_set_x:end_set_x] = frame[start_set_y:end_set_y, start_set_x:end_set_x] * \
        (1 - glass_resized[:, :, 3:] / 255) + glass_resized[:, :, :3] * (glass_resized[:, :, 3:] / 255)


def getEyes(face_gray, limit_y_for_detect):
    eyes_pre_detected = eye_cascade.detectMultiScale(face_gray)
    eyes = []
    for (ex, ey, ew, eh) in eyes_pre_detected:
        eye_pivot_y = int(ey+(eh/2))
        if eye_pivot_y < limit_y_for_detect:
            eyes.append((ex, ey, ew, eh))
    return eyes[0:2]


def smoothPivotGlass(pivot_glass, smooth_factor=0.1):
    pivot_smooth = (int(last_pivot_glass[0]*(1-smooth_factor) + pivot_glass[0]*smooth_factor),
                    int(last_pivot_glass[1]*(1-smooth_factor) + pivot_glass[1]*smooth_factor))
    return pivot_smooth


def smoothSizeWidthGlass(width_glass, smooth_factor=0.1):
    return int(last_width_glass*(1-smooth_factor) + width_glass * smooth_factor)


def getPivotGlasses(eyes):
    global last_pivot_glass
    if len(eyes) == 2:
        eye_one_pivot = (int(eyes[0][0]+(eyes[0][1]/2)), int(eyes[0][2]+(eyes[0][3]/2)))
        eye_two_pivot = (int(eyes[1][0]+(eyes[1][1]/2)), int(eyes[1][2]+(eyes[1][3]/2)))
        pivot = (int((eye_one_pivot[0]+eye_two_pivot[0])/2), int((eye_one_pivot[1]+eye_two_pivot[1])/2) + 2)
        if last_pivot_glass == (0, 0):
            last_pivot_glass = pivot
        else:
            last_pivot_glass = smoothPivotGlass(pivot)


while capture.isOpened():
    ret, frame = capture.read()
    if ret is True:

        frameGray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        faces = face_cascade.detectMultiScale(frameGray, 1.3, 3, minSize=(100, 100))

        for (x, y, w, h) in faces:
            # cv2.rectangle(frame, (x, y), (x + w, y + h), (0, 255, 0), 2)
            face_gray = frameGray[y:y+h, x:x+w]
            face_color = frame[y:y+h, x:x+w]

            if buffer_to_move_glass == 10:
                eyes = getEyes(face_gray, (h / 2))
                getPivotGlasses(eyes)
                buffer_to_move_glass = 0
            else:
                buffer_to_move_glass += 1

            if last_width_glass == 0:
                last_width_glass = getWidthGlassInPx(your_face_width_mm, glasses["width"], w)

            # for (ex, ey, ew, eh) in eyes:
            #cv2.rectangle(face_color, (ex, ey), (ex+ew, ey+eh), (0, 0, 255), 2)

            insertGlass(frame, (x, y, w, h), last_pivot_glass, glass, your_face_width_mm)

            last_width_glass = getWidthGlassInPx(your_face_width_mm, glasses["width"], w)
            # cv2.circle(frame, (x + last_pivot_glasses[0], y + last_pivot_glasses[1]), 5, (255, 0, 0), -1)

        cv2.imshow("Video", frame)
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break
    else:
        break

capture.release()
cv2.destroyAllWindows()

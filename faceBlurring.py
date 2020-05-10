import numpy as np
import argparse
import cv2

# construct the argument parse and parse the arguments
ap = argparse.ArgumentParser()
# ap.add_argument("-l", "--image", required=True,
#                 help="path to input image")
ap.add_argument("-p", "--prototxt", required=True,
                help="path to Caffe 'deploy' prototxt file")
ap.add_argument("-m", "--model", required=True,
                help="path to Caffe pre-trained model")
ap.add_argument("-c", "--confidence", type=float, default=0.4,
                help="minimum probability to filter weak detections")
args = vars(ap.parse_args())

net = cv2.dnn.readNetFromCaffe(args['prototxt'], args['model'])

# image = cv2.imread(args['image'])

cap = cv2.VideoCapture(0)

while True:
    ret, frame = cap.read()
    if not ret:
        break

    (h, w) = frame.shape[:2]
    blob = cv2.dnn.blobFromImage(cv2.resize(
        frame, (300, 300)), 1.0, (300, 300), (104.0, 177.0, 123.0))

    net.setInput(blob)

    detections = net.forward()
    faces = []
    for i in range(detections.shape[2]):
        confidence = detections[0, 0, i, 2]
        arr = np.array([w, h, w, h])
        detection = detections[0, 0, i, 3:7]

        if confidence > args['confidence']:
            try:

                box = detection * arr
                (startX, startY, endX, endY) = box.astype('int')
                text = "{:.2f}%".format(confidence * 100)
                y = startY - 10 if startY - 10 > 10 else startY + 10
                print("BOUNDS: ", startX, endX, startY, endY)
                faceROI = frame[startY:endY, startX:endX]
                width, height = endX-startX, endY-startY
                print(faceROI.shape, height, width)
                if(width > 0 and height > 0):
                    faceROI = cv2.GaussianBlur(faceROI, (11, 11), 5)
                    faceROI = cv2.resize(
                        faceROI, (6, 6), interpolation=cv2.INTER_LINEAR)
                    faceROI = cv2.resize(
                        faceROI, (width, height), interpolation=cv2.INTER_NEAREST)
                    print("FACE ROI", faceROI.shape)
                    frame[startY:endY, startX:endX] = faceROI
                cv2.rectangle(frame, (startX, startY), (endX, endY),
                              (0, 0, 255), 2)

                cv2.putText(frame, text, (startX, y),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.45, (0, 0, 255), 2)
            except Exception:
                pass
    # show the output frame
    cv2.imshow("Output", frame)

    key = cv2.waitKey(1) & 0xFF

    # if the `q` key was pressed, break from the loop
    if key == ord("q"):
        break
cv2.destroyAllWindows()
cap.release()

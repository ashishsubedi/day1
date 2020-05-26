import numpy as np
import argparse
import cv2

# construct the argument parse and parse the arguments
ap = argparse.ArgumentParser()
# ap.add_argument("-l", "--image", required=True,
#                 help="path to input image")
ap.add_argument("-p", "--prototxt",
                help="path to Caffe 'deploy' prototxt file", default='deploy.prototxt.txt')
ap.add_argument("-m", "--model",
                help="path to Caffe pre-trained model", default='res10_300x300_ssd_iter_140000.caffemodel')
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
    frame = cv2.flip(frame, 1)
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

                faceROI = frame[startY:endY, startX:endX]
                width, height = endX-startX, endY-startY
                newImg = np.random.randint(
                    0, 255, (faceROI.shape[0], faceROI.shape[1], 1), dtype=np.uint8)
                if(width > 0 and height > 0):
                    faceROI = newImg

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
    if key == 27:
        break
cv2.destroyAllWindows()
cap.release()

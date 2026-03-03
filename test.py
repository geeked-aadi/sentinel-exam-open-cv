import cv2
import numpy as np
import time

violation_start = None

net = cv2.dnn.readNetFromCaffe(
    "models/deploy.prototxt",
    "models/res10_300x300_ssd_iter_140000.caffemodel"
)

cap = cv2.VideoCapture(0)

while True:
    ret, frame = cap.read()
    frame = cv2.flip(frame, 1)
    h, w = frame.shape[:2]

    blob = cv2.dnn.blobFromImage(
        cv2.resize(frame, (300, 300)),
        1.0,
        (300, 300),
        (104.0, 177.0, 123.0)
    )

    net.setInput(blob)
    detections = net.forward()

    count = 0

    for i in range(detections.shape[2]):
        confidence = detections[0, 0, i, 2]

        if confidence > 0.6:
            count += 1

            box = detections[0, 0, i, 3:7] * np.array([w, h, w, h])
            (x1, y1, x2, y2) = box.astype("int")

            cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 2)

    # update status once per frame (not inside the loop)
    status = "Normal"
    if count == 0:
        status = "No Faces"
        color = (0, 0, 255)   # Red
    elif count == 1:
        status = "Single Face"
        color = (0, 255, 0)   # Green
    else:
        status = "Multiple Faces"
        color = (0, 165, 255) # Orange

    cv2.putText(frame, f"Status: {status}", (20, 80),
                cv2.FONT_HERSHEY_SIMPLEX, 0.8, color, 2)
    cv2.putText(
        frame,
        f"Faces detected: {count}",
        (20, 40),
        cv2.FONT_HERSHEY_SIMPLEX,
        1,
        color,
        2
    )

    current_time = time.time()
    if count == 0 or count > 1:
        # start or continue violation timer
        if violation_start is None:
            violation_start = current_time
        elapsed = current_time - violation_start
        remaining = max(0, 5 - int(elapsed))

        cv2.putText(
            frame,
            f"Violation! Exiting in {remaining}s",
            (20, 100),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.8,
            (0, 0, 255),
            2
        )

        if current_time - violation_start >= 5:
            print("Violation sustained for 5 seconds. Exiting...")
            break
    else:
        # reset timer when condition returns to normal
        violation_start = None  # reset if normal

# Condition: no face OR more than one face
    
    print("Faces detected:", count)

    cv2.imshow("DNN Face Detection", frame)

    if cv2.waitKey(1) == 27:
        break

cap.release()
cv2.destroyAllWindows()
# Pasito a pasito, suave suavecito
# Nos vamo' pegando, poquito a poquito
# Cuando tú me besas con esa destreza
# Veo que eres malicia con delicadez
import cv2
import os
import uuid

os.chdir("D:/Facial Recognition/")

POSITIVES = os.path.join("data", "positive")
NEGATIVES = os.path.join("data", "negative")
ANCHORS = os.path.join("data", "anchors")
dirs = [POSITIVES, NEGATIVES, ANCHORS]
for dir in dirs :
    os.makedirs(dir, exist_ok = True)

cap = cv2.VideoCapture(0)
while True:
    ret, frame = cap.read()
    cv2.imshow("Frame", frame)
    if cv2.waitKey(1) & 0xFF == ord('a'):
        imgname = os.path.join(ANCHORS, "{}.jpg".format(uuid.uuid1()))
        cv2.imwrite(imgname, frame)
        # collecting the positives
    if (cv2.waitKey(1) & 0xFF == ord('p')):
        imgname = os.path.join(POSITIVES, "{}.jpg".format(uuid.uuid1()))
        cv2.imwrite(imgname, frame)
        # collecting the negatives
    if (cv2.waitKey(1) & 0xFF == ord('n')):
        imgname = os.path.join(NEGATIVES, "{}.jpg".format(uuid.uuid1()))
        cv2.imwrite(imgname, frame)
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break
cap.release()
cv2.destroyAllWindows()
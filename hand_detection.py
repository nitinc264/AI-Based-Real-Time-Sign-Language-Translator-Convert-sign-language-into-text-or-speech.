import cv2
import csv                           # ← add this
from cvzone.HandTrackingModule import HandDetector    

finger = {0:"thumb",1:"index",2:"middle",3:"four_finger",4:"fifth_finger"}

detector = HandDetector(detectionCon=0.8 , maxHands=1)
video = cv2.VideoCapture(0)

# Open dataset.csv in append mode
# Make sure dataset.csv already has this header row:
# Frame,thumb,index,middle,four_finger,fifth_finger,total_up
csv_file = open('dataset.csv', 'a', newline='')
writer = csv.writer(csv_file)
frame_idx = 0

def define(i, lmlist):
    if lmlist[(i+1)*4 - 2][1] < lmlist[(i+1)*4][1]:
        return "down"
    return "up"

while True:
    ret, frame = video.read()
    if not ret:
        break

    frame = cv2.flip(frame, 1)
    frame = cv2.resize(frame, (800, 500))
    hands, img = detector.findHands(frame)

    if hands:
        hand = hands[0]
        lmlist = hand['lmList']
        fingerup = detector.fingersUp(hand)

        # compute and save states
        states = []
        for i in range(len(fingerup)):
            st = define(i, lmlist)
            states.append(1 if st == 'up' else 0)

        finger_count = sum(states)
        # write row: [frame_idx, thumb, index, middle, four_finger, fifth_finger, total_up]
        writer.writerow([frame_idx] + states + [finger_count])

        # on‑screen annotations
        for i, st_flag in enumerate(states):
            st = "up" if st_flag else "down"
            color = (0,255,0) if st_flag else (0,0,255)
            cv2.putText(img, f"{finger[i]} {st}", (20,(i+1)*80),
                        cv2.FONT_HERSHEY_COMPLEX, 1, color, 1, cv2.LINE_AA)

        cv2.putText(img, f'Finger Count: {finger_count}', (20,40),
                    cv2.FONT_HERSHEY_COMPLEX, 1, (255,255,255), 1, cv2.LINE_AA)

    cv2.imshow("Frame", img)
    if cv2.waitKey(1) == ord('q'):
        break

# Cleanup
csv_file.close()
video.release()
cv2.destroyAllWindows()

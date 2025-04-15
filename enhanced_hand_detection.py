import cv2
import csv
from cvzone.HandTrackingModule import HandDetector

# Initialize detector and video
detector = HandDetector(detectionCon=0.8, maxHands=1)
cap = cv2.VideoCapture(0)

# Label mapping (key: label)
LABELS = {
    ord('a'): 'A',
    ord('b'): 'B',
    ord('c'): 'C',
    ord('d'): 'D',
    ord('e'): 'E',
    ord('f'): 'F',
    ord('g'): 'G',
    ord('h'): 'H',
    ord('i'): 'I',
    ord('j'): 'J',
    ord('k'): 'K',
    ord('l'): 'L',
    ord('m'): 'M',
    ord('n'): 'N',
    ord('o'): 'O',
    ord('p'): 'P',
    ord('r'): 'R',
    ord('s'): 'S',
    ord('t'): 'T',
    ord('u'): 'U',
    ord('v'): 'V',
    ord('w'): 'W',
    ord('x'): 'X',
    ord('y'): 'Y',
    ord('z'): 'Z',
}

# CSV setup
CSV_FILE = 'labeled_data.csv'
with open(CSV_FILE, 'a', newline='') as f:
    writer = csv.writer(f)
    if f.tell() == 0:  # Write header only once
        writer.writerow(['thumb', 'index', 'middle', 'ring', 'pinky', 'total_up', 'label'])

def get_finger_states(hand):
    lmlist = hand['lmList']
    states = []
    for i in range(5):  # For each finger (thumb to pinky)
        if lmlist[(i+1)*4 - 2][1] < lmlist[(i+1)*4][1]:
            states.append(1)  # Up
        else:
            states.append(0)  # Down
    return states

while True:
    success, img = cap.read()
    if not success:
        break
    
    img = cv2.flip(img, 1)
    hands, _ = detector.findHands(img)
    
    if hands:
        hand = hands[0]
        states = get_finger_states(hand)
        total_up = sum(states)
        
        # Display finger states
        y_start = 40
        for i, state in enumerate(states):
            text = f"{['Thumb','Index','Middle','Ring','Pinky'][i]}: {'Up' if state else 'Down'}"
            cv2.putText(img, text, (10, y_start), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0,255,0), 2)
            y_start += 30
        cv2.putText(img, f'Total Up: {total_up}', (10, y_start+20), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255,255,255), 2)
    
    cv2.imshow("Data Collection", img)
    
    key = cv2.waitKey(1)
    if key == ord('q'):
        break
    elif key in LABELS:
        label = LABELS[key]
        if hands:
            with open(CSV_FILE, 'a', newline='') as f:
                writer = csv.writer(f)
                writer.writerow(states + [total_up, label])
            print(f"Saved sample: {label}")

cap.release()
cv2.destroyAllWindows()
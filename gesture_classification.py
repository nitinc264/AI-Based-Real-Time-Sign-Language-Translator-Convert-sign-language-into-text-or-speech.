import cv2
import joblib
from cvzone.HandTrackingModule import HandDetector

# Load trained model
model = joblib.load('gesture_classifier.pkl')

# Initialize detector
detector = HandDetector(detectionCon=0.8, maxHands=1)
cap = cv2.VideoCapture(0)

def get_finger_states(hand):
    lmlist = hand['lmList']
    states = []
    for i in range(5):
        if lmlist[(i+1)*4 - 2][1] < lmlist[(i+1)*4][1]:
            states.append(1)
        else:
            states.append(0)
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
        
        # Predict
        prediction = model.predict([states + [total_up]])[0]
        cv2.putText(img, f'Sign: {prediction}', (10, 50), 
                   cv2.FONT_HERSHEY_SIMPLEX, 1.5, (0,255,0), 3)
    
    cv2.imshow("Sign Language Translator", img)
    if cv2.waitKey(1) == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()
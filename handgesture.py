import cv2
import mediapipe as mp
import numpy as np
from ctypes import cast, POINTER
from comtypes import CLSCTX_ALL
from pycaw.pycaw import AudioUtilities, IAudioEndpointVolume



mp_hands = mp.solutions.hands
hands = mp_hands.Hands(max_num_hands=1)
mp_draw = mp.solutions.drawing_utils



devices = AudioUtilities.GetSpeakers()
interface = devices.Activate(IAudioEndpointVolume._iid_, CLSCTX_ALL, None)
volume = cast(interface, POINTER(IAudioEndpointVolume))
volume_range = volume.GetVolumeRange()
min_vol = volume_range[0]  
max_vol = volume_range[1]  


cap = cv2.VideoCapture(0)

while True:
    success, img = cap.read()
    img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    results = hands.process(img_rgb)


    if results.multi_hand_landmarks:
        for hand_landmarks in results.multi_hand_landmarks:
            
            mp_draw.draw_landmarks(img, hand_landmarks, mp_hands.HAND_CONNECTIONS)

           

            landmarks = hand_landmarks.landmark
            thumb_tip = landmarks[4]
            index_tip = landmarks[8]

            

            h, w, _ = img.shape
            thumb_pos = (int(thumb_tip.x * w), int(thumb_tip.y * h))
            index_pos = (int(index_tip.x * w), int(index_tip.y * h))

           

            cv2.circle(img, thumb_pos, 15, (255, 0, 0), cv2.FILLED)
            cv2.circle(img, index_pos, 15, (255, 0, 0), cv2.FILLED)

           

            cv2.line(img, thumb_pos, index_pos, (0, 255, 0), 3)

           
            distance = np.linalg.norm(np.array(thumb_pos) - np.array(index_pos))

            

            vol = np.interp(distance, [30, 300], [min_vol, max_vol])
            volume.SetMasterVolumeLevel(vol, None)

            
            vol_percent = np.interp(distance, [30, 300], [0, 100])
            cv2.putText(img, f'Volume: {int(vol_percent)}%', (50, 50), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 0, 0), 3)

  

    cv2.imshow('Hand Volume Control', img)

    
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break


cap.release()
cv2.destroyAllWindows()

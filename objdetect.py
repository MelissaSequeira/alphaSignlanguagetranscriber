import cv2
import mediapipe as mp
import time
import warnings
import uuid
# Suppress specific warning from protobuf
warnings.filterwarnings("ignore", category=UserWarning, module='google.protobuf.symbol_database')

# Video capture object
cap = cv2.VideoCapture(0)

# MediaPipe Hands object
mpHands = mp.solutions.hands
hands = mpHands.Hands()
mpDraw = mp.solutions.drawing_utils
IMAGE_PATH='CollectedImages'

labels=['Hello','Yes','No','Thanks','IloveYou','Please']

number_of_images=20
while True:
    success, img = cap.read()

    # Check if the frame was captured successfully
    if not success:
        print("Error: Could not read frame.")
        break

    # Convert the image color to RGB
    imgRGB = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

    # Process the RGB image to find hands
    results = hands.process(imgRGB)

    # If hands are found
    if results.multi_hand_landmarks:
        for handLms in results.multi_hand_landmarks:
            # Draw hand landmarks and connections on the original image
            mpDraw.draw_landmarks(img, handLms, mpHands.HAND_CONNECTIONS)
            
        
    # Display the image
    cv2.imshow("Image", img)

    # Break the loop if 'q' is pressed
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# Release the capture and close windows
cap.release()
cv2.destroyAllWindows()

import cv2
from cvzone.HandTrackingModule import HandDetector
from cvzone.ClassificationModule import Classifier
import numpy as np
import math
import pyttsx3
from flask import Flask, render_template, Response, request

# Initialize Flask app
app = Flask(__name__)

# Initialize OpenCV and HandDetector
cap = cv2.VideoCapture(0)
detector = HandDetector(maxHands=1)
classifier = Classifier(r"keras_model.h5", r"labels.txt")
offset = 20
imgSize = 300
text_speech = pyttsx3.init()

labels = ['A', 'B', 'C', 'D', 'E', 'F', 'G', 'H', 'I', 'J', 'K', 'L', 'M', 'N', 'O', 'P', 'Q', 'R', 'S', 'space', 'T', 'U', 'V', 'W', 'X', 'Y', 'Z']

answer = ""  # Initialize answer variable

# Route to render the home page (HTML)
@app.route('/')
def index():
    return render_template('index.html')

# Function to stream camera feed to the browser
def gen_frames():
    global answer
    while True:
        success, img = cap.read()
        if not success:
            break

        imgOutput = img.copy()
        hands, img = detector.findHands(img)
        if hands:
            hand = hands[0]
            x, y, w, h = hand['bbox']

            # Ensure the cropping coordinates are within the image bounds
            y_start = max(0, y - offset)
            y_end = min(img.shape[0], y + h + offset)
            x_start = max(0, x - offset)
            x_end = min(img.shape[1], x + w + offset)

            imgCrop = img[y_start:y_end, x_start:x_end]

            if imgCrop.size > 0:  # Proceed only if imgCrop is valid
                imgWhite = np.ones((imgSize, imgSize, 3), np.uint8) * 255

                aspectRatio = h / w

                if aspectRatio > 1:
                    k = imgSize / h
                    wCal = math.ceil(k * w)
                    imgResize = cv2.resize(imgCrop, (wCal, imgSize))
                    wGap = math.ceil((imgSize - wCal) / 2)
                    imgWhite[:, wGap:wCal + wGap] = imgResize
                    prediction, index = classifier.getPrediction(imgWhite, draw=False)
                else:
                    k = imgSize / w
                    hCal = math.ceil(k * h)
                    imgResize = cv2.resize(imgCrop, (imgSize, hCal))
                    hGap = math.ceil((imgSize - hCal) / 2)
                    imgWhite[hGap:hCal + hGap, :] = imgResize
                    prediction, index = classifier.getPrediction(imgWhite, draw=False)

                if 0 <= index < len(labels):  # Ensure the index is within the valid range
                    cv2.rectangle(imgOutput, (x - offset, y - offset - 70), (x - offset + 400, y - offset + 60 - 50), (0, 255, 0), cv2.FILLED)
                    cv2.putText(imgOutput, labels[index], (x, y - 30), cv2.FONT_HERSHEY_COMPLEX, 2, (0, 0, 0), 2)
                    cv2.rectangle(imgOutput, (x - offset, y - offset), (x + w + offset, y + h + offset), (0, 255, 0), 4)
                    answer = labels[index]  # Update the answer with the detected label
                else:
                    answer = ""  # Reset the answer if index is invalid

        # Convert image to JPEG and stream it
        ret, buffer = cv2.imencode('.jpg', imgOutput)
        if not ret:
            continue
        frame = buffer.tobytes()
        yield (b'--frame\r\n'
               b'Content-Type: image/jpeg\r\n\r\n' + frame + b'\r\n\r\n')

# Route to stream the camera feed
@app.route('/video_feed')
def video_feed():
    return Response(gen_frames(), mimetype='multipart/x-mixed-replace; boundary=frame')

# Route to handle gesture recognition requests
@app.route('/recognize', methods=['POST'])
def recognize():
    global answer
    if request.form.get('action') == 'recognize':
        return answer
    return '', 204

# Route to handle audio output requests
@app.route('/play_audio', methods=['POST'])
def play_audio():
    full_text = request.form.get('text')
    text_speech.say(full_text)
    text_speech.runAndWait()
    return '', 204

if __name__ == '__main__':
    app.run(debug=True, host='0.0.0.0', port=5000)




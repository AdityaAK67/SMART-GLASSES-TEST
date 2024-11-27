import pytesseract
import cv2
import pyttsx3
import os

# Initialize the text-to-speech engine
engine = pyttsx3.init()
engine.setProperty('rate', 125)  # Adjust speech speed for clarity

# Initialize webcam
webcam = cv2.VideoCapture(0)
webcam.set(3, 640)  # Set width
webcam.set(4, 480)  # Set height

print("Press 'z' to capture an image and process text.")
print("Press 'q' to exit the program.")

def preprocess_image(image):
    """
    Preprocess the image to improve OCR accuracy.
    - Convert to grayscale
    - Apply thresholding
    - Remove noise using morphological operations
    """
    # Convert to grayscale
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

    # Apply thresholding (binarization)
    _, binary = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)

    # Optionally remove noise with morphological operations
    kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (3, 3))
    cleaned = cv2.morphologyEx(binary, cv2.MORPH_CLOSE, kernel)

    return cleaned

while True:
    # Capture frame-by-frame from the webcam
    ret, frame = webcam.read()

    if not ret:
        print("Error: Unable to access the camera.")
        break

    # Display the video feed
    cv2.imshow("Live Video Feed", frame)

    # Check for key press
    key = cv2.waitKey(1) & 0xFF  # Use bitwise AND for cross-platform compatibility

    if key == ord('z'):  # Press 'z' to capture an image
        print("Image captured! Processing for text...")

        # Save the captured image temporarily
        image_path = "captured_image.jpg"
        cv2.imwrite(image_path, frame)

        # Perform OCR on the captured image
        if os.path.exists(image_path):
            try:
                # Read the captured image
                image = cv2.imread(image_path)

                # Preprocess the image for OCR
                preprocessed_image = preprocess_image(image)

                # Use pytesseract to extract text
                detected_text = pytesseract.image_to_string(preprocessed_image, lang='eng')

                if detected_text.strip():  # Check if text was detected
                    print("Text Detected:")
                    print(detected_text)

                    # Output detected text in audio form
                    engine.say("Here is the text detected in the image:")
                    engine.say(detected_text)
                    engine.runAndWait()
                else:
                    print("No text detected in the captured image.")
            except Exception as e:
                print(f"An error occurred during OCR processing: {e}")
        else:
            print(f"Error: Could not save the captured image.")

    elif key == ord('q'):  # Press 'q' to quit the program
        print("Exiting the program.")
        break

# Release resources
webcam.release()
cv2.destroyAllWindows()




# SMART-GLASSES-TEST


from PIL import Image
import pytesseract
import os

# Path to the image
image = '/home/pi/Smart_Glasses/image_scan.jpg'
file = '/home/pi/Smart_Glasses/content.txt'

# Check if the file exists
if not os.path.isfile(image):
    print(f"Error: File {image} not found!")
else:
    # Process the image
    pic = Image.open(image)
    text = pytesseract.image_to_string(pic)

    # Print the extracted text
    print(text)

    # Save the extracted text to a file
    with open(file, 'w') as f:
        f.write(text)
import cv2
import pytesseract
import pyttsx3
import time
from gpiozero import AngularServo

# Servo Motor Initialization
servo = AngularServo(18, initial_angle=0, min_pulse_width=0.0006, max_pulse_width=0.0023)

# Text-to-Speech Engine Setup
engine = pyttsx3.init()
engine.setProperty('rate', 125)  # Speech speed

# COCO Class Names for Object Detection
classNames = []
classFile = "/home/pi/Desktop/Object_Detection_Files/coco.names"
with open(classFile, "rt") as f:
    classNames = f.read().rstrip("\n").split("\n")

# Object Detection Model Paths
configPath = "/home/pi/Desktop/Object_Detection_Files/ssd_mobilenet_v3_large_coco_2020_01_14.pbtxt"
weightsPath = "/home/pi/Desktop/Object_Detection_Files/frozen_inference_graph.pb"

# Load Object Detection Model
net = cv2.dnn_DetectionModel(weightsPath, configPath)
net.setInputSize(320, 320)
net.setInputScale(1.0 / 127.5)
net.setInputMean((127.5, 127.5, 127.5))
net.setInputSwapRB(True)

# Set up Tesseract path (adjust if necessary)
pytesseract.pytesseract.tesseract_cmd = r'C:\Program Files\Tesseract-OCR\tesseract.exe'

def preprocess_image(image):
    """
    Preprocess image to improve OCR accuracy.
    Converts image to grayscale and applies thresholding.
    """
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    _, thresh = cv2.threshold(gray, 150, 255, cv2.THRESH_BINARY)
    return thresh

def getObjects(img, thres=0.5, nms=0.3, draw=True, objects=[]):
    """
    Detect objects in the given frame using the loaded model.
    """
    classIds, confs, bbox = net.detect(img, confThreshold=thres, nmsThreshold=nms)
    objectInfo = []
    if len(objects) == 0:
        objects = classNames
    if len(classIds) != 0:
        for classId, confidence, box in zip(classIds.flatten(), confs.flatten(), bbox):
            className = classNames[classId - 1]
            if className in objects:
                objectInfo.append([box, className])
                if draw:
                    cv2.rectangle(img, box, color=(0, 255, 0), thickness=2)
                    cv2.putText(img, className.upper(), (box[0] + 10, box[1] + 30),
                                cv2.FONT_HERSHEY_COMPLEX, 1, (0, 255, 0), 2)
                    cv2.putText(img, str(round(confidence * 100, 2)), (box[0] + 200, box[1] + 30),
                                cv2.FONT_HERSHEY_COMPLEX, 1, (0, 255, 0), 2)
                    # Servo Motor Action
                    servo.angle = -90
                    time.sleep(0.5)
                    servo.angle = 90
                # Speak detected object
                engine.say(f"Detected {className}")
                engine.runAndWait()
    return img, objectInfo

def processText(image):
    """
    Perform OCR on the captured image and read the detected text aloud.
    """
    try:
        processed_img = preprocess_image(image)
        text = pytesseract.image_to_string(processed_img, lang='eng')
        if text.strip():
            print("Detected Text:")
            print(text)
            engine.say("Here is the text detected in the image.")
            engine.say(text)
            engine.runAndWait()
        else:
            print("No text detected in the captured image.")
            engine.say("No text detected in the captured image.")
            engine.runAndWait()
    except Exception as e:
        print(f"Error during OCR: {e}")

if __name__ == "__main__":
    # Initialize Webcam
    cap = cv2.VideoCapture(0)
    cap.set(3, 640)  # Width
    cap.set(4, 480)  

    print("Press 'z' to capture an image for OCR (text detection).")
    print("Press 'q' to quit the program.")

    while True:
        success, img = cap.read()

        if not success:
            print("Error: Unable to access the camera.")
            break

        # Perform Live Object Detection
        result, objectInfo = getObjects(img, thres=0.5, nms=0.3, objects=['person', 'cup', 'book'])

        # Display the Live Video Feed
        cv2.imshow("Live Object Detection", img)

        # Handle Key Events
        key = cv2.waitKey(1) & 0xFF
        if key == ord('z'):  # Press 'z' to capture image for OCR
            print("Capturing image for text detection...")
            processText(img)
        elif key == ord('q'):  # Press 'q' to quit the program
            print("Exiting the program.")
            break

    # Release Resources
    cap.release()
    cv2.destroyAllWindows()

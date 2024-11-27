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
    Preprocess the image to improve OCR accuracy:
    - Convert to grayscale
    - Apply adaptive thresholding
    - Remove noise
    - Optionally resize the image for better OCR
    """
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    binary = cv2.adaptiveThreshold(
        gray, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY, 11, 2
    )
    cleaned = cv2.medianBlur(binary, 3)  # Reduce noise
    resized = cv2.resize(cleaned, None, fx=1.5, fy=1.5, interpolation=cv2.INTER_LINEAR)
    return resized

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

                # Debug: Display the preprocessed image
                cv2.imshow("Preprocessed Image", preprocessed_image)
                cv2.waitKey(0)

                # Use pytesseract to extract text
                custom_config = r'--oem 3 --psm 6'
                detected_text = pytesseract.image_to_string(preprocessed_image, config=custom_config)

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

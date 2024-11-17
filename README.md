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

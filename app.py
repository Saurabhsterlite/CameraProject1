import cv2
import numpy as np
from flask import Flask, render_template, request

app = Flask(__name__)

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/detect', methods=['POST'])
def detect():
    if request.method == 'POST':
        # Get the uploaded image file from the request
        file = request.files['image']
        # Read the uploaded file into the 'image' variable
        image = cv2.imdecode(np.fromstring(file.read(), np.uint8), cv2.IMREAD_UNCHANGED)
        image = ~image

        # Get the dimensions of the input image
        height, width, channels = image.shape
        print("Input image dimensions (height x width x channels):", height, "x", width, "x", channels)

        # Convert the image to grayscale
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

        # Apply a threshold to the image to separate the object from the background
        thresh = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY | cv2.THRESH_OTSU)[1]

        # Find the contours in the image
        contours, _ = cv2.findContours(thresh, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

        # Find the largest contour (i.e., the object)
        largest_contour = max(contours, key=cv2.contourArea)

        # Get the dimensions of the bounding box around the object
        x, y, w, h = cv2.boundingRect(largest_contour)

        # Display the dimensions of the input image and the detected object
        cv2.rectangle(image, (x, y), (x+w, y+h), (0, 255, 0), 2)
        cv2.putText(image, f"Image dimensions: {width}x{height}", (10, 20), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 255), 2)
        cv2.putText(image, f"Object dimensions: {w}x{h}", (10, 40), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 255), 2)

        # Save the resulting image to a file
        filename = 'result.jpg'
        cv2.imwrite(filename, image)

        # Calculate the object height and pass it to the template
        object_height = 121 * float(h / height)
        return render_template('result.html', object_height=object_height, filename=filename)

if __name__ =="__main__":
    app.run(host='0.0.0.0', port=8080, debug=True)

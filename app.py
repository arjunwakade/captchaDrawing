import cv2
import base64
from PIL import Image
import numpy as np
import io
from flask import Flask, render_template, request
from flask import send_file

app = Flask(__name__)

def compare_with_circle(contour):
    # Create a reference perfect circle contour
    circle_img = np.zeros((100, 100), dtype=np.uint8)
    cv2.circle(circle_img, (50, 50), 40, 255, thickness=-1)
    circle_contour, _ = cv2.findContours(circle_img, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    
    # Calculate Hu moments for both contours
    shape_moments = cv2.HuMoments(cv2.moments(contour)).flatten()
    circle_moments = cv2.HuMoments(cv2.moments(circle_contour[0])).flatten()
    
    # Compare using some distance metric (e.g., Euclidean distance)
    distance = np.linalg.norm(shape_moments - circle_moments)
    return distance

def is_captcha_valid(captcha_data):
    # Decode the base64 image data
    captcha_data = captcha_data.split(',')[1]
    image_data = base64.b64decode(captcha_data)

    with open("output_image.png", "wb") as f:
        f.write(image_data)

    image = Image.open("output_image.png")
    white_bg = Image.new("RGBA", image.size, (255, 255, 255, 255))
    combined_image = Image.alpha_composite(white_bg, image)
    image_np = np.array(combined_image)
    image_np = cv2.cvtColor(image_np, cv2.COLOR_RGB2GRAY)

    # Preprocessing
    blurred = cv2.GaussianBlur(image_np, (5, 5), 2)
    edges = cv2.Canny(blurred, 50, 150)

    # Detect circles using Hough Transform
    circles = cv2.HoughCircles(edges, cv2.HOUGH_GRADIENT, dp=1.2, minDist=50, param1=100, param2=50, minRadius=30, maxRadius=80)
    if circles is not None:
        circles = np.round(circles[0, :]).astype("int")
        for (x, y, r) in circles:
            # Create a mask to extract the detected circle's region
            mask = np.zeros_like(image_np)
            cv2.circle(mask, (x, y), r, 255, thickness=-1)
            circle_region = cv2.bitwise_and(image_np, mask)

            # Find contours within the detected circle region
            contours, _ = cv2.findContours(circle_region, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
            if contours:
                contour = contours[0]
                # Use Hu Moments to compare the shape to a perfect circle
                distance = compare_with_circle(contour)
                print(distance)

                # Set a threshold for accepting an imperfect circle
                if distance < 0.0001:  # Adjust the threshold as necessary
                    return True  # Imperfect circle detected

    return False  # No valid circle detected

#uncomment code below and comment out line 83 if you want to try using MS Paint
# def load_image_and_encode_to_base64(image_path):
#     with Image.open(image_path) as img:
#         buffered = io.BytesIO()
#         img.save(buffered, format="PNG")
#         img_str = base64.b64encode(buffered.getvalue()).decode()
#         return f"data:image/png;base64,{img_str}"

# # Load and encode the perfect circle image
# image_path = 'perfect_circle.png'  # Ensure the .png image is in the same directory
# captcha_data = load_image_and_encode_to_base64(image_path)

@app.route('/', methods=['GET', 'POST'])
def index():
    if request.method == 'POST':
        name = request.form['name']
        captcha_data = request.form['captcha']
        if not captcha_data:
            return "Captcha data missing. Please try again"
        if is_captcha_valid(captcha_data):
            print("success!")
            return render_template('welcome.html', name=name)
        else:
            print("failure")
            return "CAPTCHA failed. Please try again."
    return render_template('form.html')

if __name__ == '__main__':
    app.run(debug=True)
from flask import Flask, request, render_template, jsonify
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing import image
import numpy as np
import io

# Initialize the Flask app
app = Flask(__name__)

# Load the pre-trained model
model = load_model('image_recognition_model.h5')

# Class names for CIFAR-10 dataset
class_names = ['airplane', 'automobile', 'bird', 'cat', 'deer', 
               'dog', 'frog', 'horse', 'ship', 'truck']

# Define the route for the homepage
@app.route('/')
def index():
    return render_template('index.html')

# Define a route for processing image input
@app.route('/chatbot', methods=['POST'])
def chatbot():
    try:
        # Check if an image was uploaded
        if 'image' not in request.files:
            return jsonify({'error': 'No image provided'}), 400

        # Retrieve the image file from the request
        file = request.files['image']
        
        # Load the image using PIL directly from the file-like object
        img = image.load_img(io.BytesIO(file.read()), target_size=(32, 32))
        img_array = image.img_to_array(img)
        img_array = np.expand_dims(img_array, axis=0)  # Add batch dimension
        img_array = img_array / 255.0  # Normalize

        # Predict the class
        predictions = model.predict(img_array)
        predicted_class = class_names[np.argmax(predictions[0])]

        response = f'The image is classified as: {predicted_class}'
        return jsonify({'response': response})
    except Exception as e:
        print(f"Error: {e}")
        return jsonify({"error": "Internal Server Error"}), 500

# Run the app
if __name__ == '__main__':
    app.run(debug=True)

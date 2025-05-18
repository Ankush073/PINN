from flask import Flask, request, jsonify, render_template
import numpy as np
import tensorflow as tf
from keras.models import load_model
from Pinn_Model import PINN  # Import the PINN class from a separate file

app = Flask(__name__)

# Load the trained model, ensuring that the custom class is available
try:
    model = load_model("./models/pinn_model_recreated.keras", custom_objects={'PINN': PINN})
except Exception as e:
    print(f"Error loading model: {e}")

@app.route('/')
def home():
    return render_template("index.html")  # Ensure index.html is in the templates folder

@app.route("/predict", methods=["POST"])
def predict():
    try:
        # Get input data
        data = request.get_json()

        if not data:
            return jsonify({"error": "No input data provided"}), 400

        x_values = np.array(data.get("x", []), dtype=np.float32).reshape(-1, 1)
        t_values = np.array(data.get("t", []), dtype=np.float32).reshape(-1, 1)

        # Validate input
        if x_values.size == 0 or t_values.size == 0:
            return jsonify({"error": "Invalid input. Provide 'x' and 't' as non-empty lists."}), 400

        # Convert to TensorFlow tensors
        input_tensor = tf.convert_to_tensor(np.hstack((x_values, t_values)), dtype=tf.float32)

        # Predict using the model
        predictions = model.predict(input_tensor).tolist()

        return jsonify({"predictions": predictions})

    except Exception as e:
        print(f"Error in prediction: {e}")
        return jsonify({"error": str(e)}), 500

if __name__ == "__main__":
    app.run(host="0.0.0.0", port=5000, debug=True)

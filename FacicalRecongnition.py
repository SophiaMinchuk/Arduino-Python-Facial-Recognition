import cv2               # For capturing video and processing images
import serial            # For communication with the Arduino over serial
import time              # For adding delays in the code
import atexit            # For cleanup actions when the program exits
from keras.models import load_model   # To load the pre-trained facial recognition model
from PIL import Image, ImageOps      # For image processing and preparation for the model
import numpy as np       # For numerical operations required for the model
from tensorflow.keras.layers import DepthwiseConv2D  # Optional, if you are using a specific type of neural network layer

# Setup Arduino communication
arduino = serial.Serial(port='COM3', baudrate=9600, timeout=1)
time.sleep(2)  # Allow Arduino to initialize

def send_command(command):
    """Send a command to the Arduino."""
    arduino.write(f"{command}\n".encode())  # Send the command with a newline
    time.sleep(0.05)  # Allow Arduino to process
    
    response = arduino.readline().decode().strip()  # Read the response
    return response

# Define cleanup function to run on program exit
def cleanup():
    """Cleanup function to turn off LEDs and close Arduino connection."""
    print("Sending EXIT command to Arduino...")
    send_command("EXIT")  # Send EXIT to turn off all LEDs
    arduino.close()
    print("Arduino connection closed.")

# Register cleanup function to execute when program exits
atexit.register(cleanup)

# Load the model
class CustomDepthwiseConv2D(DepthwiseConv2D):
    def __init__(self, *args, **kwargs):
        kwargs.pop("groups", None)  # Ignore 'groups' argument
        super().__init__(*args, **kwargs)

model = load_model(
    "C:/Users/sophi/OneDrive/Desktop/MyCode/keras_model.h5",
    compile=False,
    custom_objects={"DepthwiseConv2D": CustomDepthwiseConv2D}
)

# Load the labels
class_names = open("C:/Users/sophi/OneDrive/Desktop/MyCode/labels.txt", "r").readlines()

def process_frame(frame):
    """Prepare and predict class for a video frame."""
    # Resize the frame
    size = (224, 224)
    image = Image.fromarray(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))
    image = ImageOps.fit(image, size, Image.Resampling.LANCZOS)

    # Normalize the image
    image_array = np.asarray(image)
    normalized_image_array = (image_array.astype(np.float32) / 127.5) - 1

    # Load the image into the array
    data = np.ndarray(shape=(1, 224, 224, 3), dtype=np.float32)
    data[0] = normalized_image_array

    # Predict with the model
    prediction = model.predict(data)
    index = np.argmax(prediction)
    class_name = class_names[index].strip()
    confidence_score = prediction[0][index]
    return class_name, confidence_score

try:
    # Open a connection to the camera
    camera = cv2.VideoCapture(0)  # Use the first connected camera

    while True:
        ret, frame = camera.read()  # Capture a frame
        if not ret:
            print("Failed to capture frame. Exiting...")
            break

        # Display the frame
        cv2.imshow("Camera Feed", frame)

        # Process the frame for prediction
        class_name, confidence = process_frame(frame)

        # Debugging: Print the predicted class and confidence
        print(f"Class: {class_name}, Confidence: {confidence}")

        # Check prediction and decide Arduino command
        if class_name == "0 1":  # If it's your face (change this based on your label)
            print("Your face detected, turning on green light.")
            send_command("GREEN_ON")  # Turn on the green light
            send_command("RED_OFF")   # Ensure red light is off
        else:
            print("Not your face detected, turning on red light.")
            send_command("RED_ON")    # Turn on the red light
            send_command("GREEN_OFF") # Ensure green light is off

        # Exit if 'q' is pressed
        if cv2.waitKey(1) & 0xFF == ord('q'):
            print("Exiting program...")
            break

finally:
    # Release resources
    camera.release()
    cv2.destroyAllWindows()

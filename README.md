# Arduino-Python-Facial-Recognition-Control

## Dataset
This project uses a facial recognition model trained using Teachable Machine to identify whether a face is present. The model is used to control an LED light through Arduino.

## References
- [Teachable Machine](https://teachablemachine.withgoogle.com/)
- [PySerial Documentation](https://pyserial.readthedocs.io/)
- [TensorFlow Documentation](https://www.tensorflow.org/)
- [Arduino Documentation](https://docs.arduino.cc/)

## Project Steps

1. **Train the Facial Recognition Model:**
   - Go to [Teachable Machine](https://teachablemachine.withgoogle.com/).
   - Create a new image model with two classes:
     - Class 1: Your face (upload multiple pictures).
     - Class 2: Other faces/no face (or background images).
   - Export the model in TensorFlow/Keras format.

2. **Upload Arduino Code:**
   - Open the Arduino IDE and upload the basic code that listens for commands from Python to control the LED light.

3. **Set Up Python-Arduino Communication:**
   - Install the required Python libraries using `pip install pyserial tensorflow opencv-python`.
   - Use Python to send serial signals to the Arduino. When the model detects your face, it sends a signal to turn on the green LED. When no face is detected, it turns on the red LED.

4. **Test the System:**
   - Train and test your facial recognition model.
   - Test the communication between Python and Arduino to ensure the LED control works based on face detection.

5. **Refine the Model (if needed):**
   - Adjust the facial recognition model’s sensitivity if required.
   - Ensure that the system behaves as expected during runtime.

## Versions Used:
- **Python:** 3
- **TensorFlow:** 2.18
- **Keras:** Latest
- **OpenCV:** 4.10
- **PySerial:** 3.5
- **Arduino IDE:** Latest
- **Hardware:** Arduino Mega 2560

# Arduino-Python-Facial-Recognition-Control

## Dataset
This project uses two models trained using Teachable Machine:
1. Facial Recognition Model – Identifies whether the user's face is present.
2. Stop Hand Model – Detects a "stop hand" gesture to control the behavior of the system.
These models work in tandem to control LED light's through Arduino.

## References
- [Teachable Machine](https://teachablemachine.withgoogle.com/)
- [PySerial Documentation](https://pyserial.readthedocs.io/)
- [TensorFlow Documentation](https://www.tensorflow.org/)
- [Arduino Documentation](https://docs.arduino.cc/)

## Project Steps

1. **Train the Models:**
- Facial Recognition Model
   - Go to [Teachable Machine](https://teachablemachine.withgoogle.com/).
   - Create a new image model with two classes:
   - Class 1: Your face (upload multiple pictures).
   - Class 2: Other faces/no face (or background images).
   - Export the model in TensorFlow/Keras format.
  
- Stop Hand Model
   - Create another model in Teachable Machine with two classes:
   - Class 1: Stop hand (capture images of a stop gesture).
   - Class 2: Neutral (background images and your face in a neutral postion).
   - Export this model in TensorFlow/Keras format.
  
2. **Upload Arduino Code:**
- Open the Arduino IDE and upload the basic code that listens for commands from Python to control the LED light.
- The code should:
  -Turn on the green LED when your face is detected.
  -Turn on the red LED when no face is detected.
  -Respond to a stop hand gesture by exiting the program and turning off both LEDs.
     
3. **Set Up Python-Arduino Communication:**
- Install the required Python libraries using: pip install pyserial tensorflow opencv-python
   - Use Python to send serial signals to the Arduino:
   - When the model detects your face, Python sends a signal to turn on the green LED.
   - When no face is detected, Python sends a signal to turn on the red LED.
   - When a stop hand gesture is detected with high confidence, Python sends a signal to exit the program and stop 
       the LED control.

4. **Test the System:**
- Train and test both models (facial recognition and stop hand detection).
- Test the communication between Python and Arduino to ensure the LED control works based on:
  - Facial recognition (green for your face, red for no face).
  - Stop hand detection to stop the program.

5. **Refine the Model (if needed):**
- Adjust the facial recognition model’s sensitivity if required.
- Adjust the stop hand model's sensitivity to ensure it correctly detects the stop gesture.
- Ensure that the system behaves as expected during runtime.

## Versions Used:
- **Python:** 3
- **TensorFlow:** 2.18
- **Keras:** Latest
- **OpenCV:** 4.10
- **PySerial:** 3.5
- **Arduino IDE:** Latest
- **Hardware:** Arduino Mega 2560

import cv2
import streamlit as st


# Create a function to capture video from the webcam and display it in a window within the Streamlit app
def show_webcam():
    # Open a video capture object
    cap = cv2.VideoCapture(0)

    # Check if the camera is opened successfully
    if not cap.isOpened():
        st.error("Failed to open camera.")
        return

    # Create a Streamlit window to display the video
    st.write("Live webcam stream:")
    my_slot = st.empty()

    # Loop over frames while the camera is open
    while cap.isOpened():
        # Read a frame from the camera
        ret, frame = cap.read()

        if ret:
            # Display the frame in the Streamlit window
            my_slot.image(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB), channels="RGB")
        else:
            # End the loop if there are no more frames to read
            break

    # Release the camera and close the Streamlit app
    cap.release()
    st.write("Camera stream ended.")


# Call the function to display the live webcam window in the Streamlit app
show_webcam()

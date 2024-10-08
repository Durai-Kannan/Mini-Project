1. Imports:

cv2: OpenCV library for computer vision tasks like frame capture and drawing.
numpy as np: NumPy for numerical computations and array manipulation.
mediapipe as mp: MediaPipe library for hand landmark detection and processing.
screen_brightness_control as sbc: Library to control screen brightness (Windows/Linux specific).
from math import hypot: Imports the hypotenuse function for calculating distances between points.
from pycaw.pycaw import AudioUtilities, IAudioEndpointVolume: Pycaw library to access and control system volume.
from ctypes import cast, POINTER: ctypes library for system-level interactions.
from comtypes import CLSCTX_ALL: comtypes library for interacting with Windows components.
import pyautogui: Library to control mouse movement and clicks.
import time: Library for handling time-related functions.


2. main Function:

Sets up volume control using Pycaw:
Gets the default audio device.
Retrieves the volume interface and its volume range (min, max, muted).
Sets up hand detection using MediaPipe Hands:
Creates a hand detection model with specific parameters (static image mode=False, model complexity=1, etc.).
Initializes drawing utilities and video capture from webcam (index 0).
Gets the screen width and height using pyautogui.
Tracks the last click time for double-click detection.
Enters a loop that continues until the 'q' key is pressed:
Captures a frame from the webcam.
Flips the frame horizontally (mirrors the image).
Converts the frame to RGB format (required by MediaPipe).
Processes the frame to detect hands using MediaPipe.
Calls get_left_right_landmarks to extract landmarks for left and right hands (if detected).
Processes left hand gestures:
If left hand is detected and index and middle fingers are close:
Calculates the scrolling position based on index finger Y coordinate.
Scrolls the screen using pyautogui.scroll.
Displays scrolling position text on the frame.
Otherwise (fingers not close):
Calculates distance between specific landmarks for brightness control.
Maps the distance to a brightness level (0-100%).
Sets the screen brightness using sbc.set_brightness.
Displays brightness level text on the frame.
Processes right hand gestures:
If right hand is detected and index and middle fingers are close:
Calculates cursor position based on index finger coordinates.
Moves the mouse cursor using pyautogui.moveTo.
Displays "Cursor Mode" text on the frame.
Checks for left click:
Calculates distance between index finger and thumb.
Checks for double-click based on time since last click.
Performs left click or double-click using pyautogui.click or pyautogui.doubleClick.
Displays "Left Click" or "Double Click" text on the frame (depending on action).
Checks for right click:
Calculates distance between middle finger and thumb.
Performs right click using pyautogui.click(button='right').
Displays "Right Click" text on the frame.
Otherwise (fingers far apart):
Calculates distance between specific landmarks for volume control.
Maps the distance to a volume level (min-max range).
Sets the system volume using volume.SetMasterVolumeLevel.
Displays volume level percentage text on the frame.
Displays the processed frame with text overlays using cv2.imshow.
Exits the loop when 'q' key is pressed.
Releases the video capture and closes all OpenCV windows.


3. Helper Functions:

get_left_right_landmarks:
Extracts and stores landmark data for the left and right hands (if detected).
get_distance_for_control:
Calculates the distance between two landmarks used for brightness or volume control.
get_distance_between_fingers:
Calculates the distance between two specified finger tips.
4. if __name__ == '__main__':

To run the entire program.

Flow of work:
Frame Capture:
Webcam captures a frame.
Frame is converted to RGB format.
Hand Detection:
MediaPipe's Hand Detection model processes the frame.
Detects hands and their landmarks.
Landmark Extraction:
Extracts specific landmarks (e.g., index finger tip, middle finger tip, thumb tip).
Gesture Recognition:
Calculates distances between landmarks to determine gestures.
Examples:
Distance between index and middle fingers: Scrolling or cursor control.
Distance between index finger and thumb: Left click.
Distance between middle finger and thumb: Right click.
Action Execution:
Based on the detected gesture, performs actions:
Scrolling: Uses pyautogui to scroll the screen.
Volume control: Sets the system volume using Pycaw.
Mouse movements: Moves the mouse cursor using pyautogui.
Clicks: Performs left or right clicks using pyautogui.
Display:
Draws the detected hands and landmarks on the frame.
Displays relevant information (e.g., brightness level, volume percentage, gesture status) on the frame.
Shows the processed frame in a window.
import tkinter as tk
from tkinter import filedialog, messagebox
from tkinter import *
from PIL import Image, ImageTk
import numpy as np
import cv2
from keras.models import load_model
import time

# Load the pre-trained model and hand detection cascade
model = load_model("SignLanguageDetection.keras")

# Use a standard cascade file for testing
cascade_path = 'haarcascade_hand.xml'  # Assuming the file is in the same directory
hand_cascade = cv2.CascadeClassifier(cascade_path)

# Check if the cascade file is loaded correctly
if hand_cascade.empty():
    raise IOError(f"Failed to load cascade file from {cascade_path}")

# Initialize Tkinter window
screen = tk.Tk()
screen.geometry('800x600')
screen.title("Sign Detection GUI")
screen.configure(background="#2C3E50")

# Set window icon
screen.iconphoto(False, tk.PhotoImage(file='contract.png'))

# Create labels for displaying detected sign and images
label1 = Label(screen, background="#2C3E50", foreground="white", font=("Helvetica", 15, "bold"))
label2 = Label(screen, background="#2C3E50", foreground="white", font=("Helvetica", 15, "bold"))
sign_image = Label(screen)

# Mapping function for detected sign
def map_sign(sign):
    if 0 <= sign <= 9:
        return str(sign)
    elif 10 <= sign <= 35:
        return chr(sign + 55)  # Convert to 'A'-'Z'
    elif sign == 36:
        return '_'
    else:
        return '?'

def process_image(image):
    # Convert image to RGB if it's not already
    if image.mode != 'RGB':
        image = image.convert('RGB')
    
    # Convert the image to a numpy array
    image_np = np.array(image)
    
    # Detect hands in the image
    gray = cv2.cvtColor(image_np, cv2.COLOR_RGB2GRAY)
    hands = hand_cascade.detectMultiScale(gray, 1.1, 4)
    
    if len(hands) > 0:
        # Assume the first detected hand is the one we want
        x, y, w, h = hands[0]
        hand = image_np[y:y+h, x:x+w]
    else:
        # If no hand is detected, assume the image is just a hand
        hand = image_np
    
    # Resize the hand image to (50, 50) for the model
    hand = cv2.resize(hand, (50, 50))
    
    # Normalize the image
    hand = hand / 255.0
    
    # Add batch dimension
    hand = np.expand_dims(hand, axis=0)
    
    return hand

# Function to detect hand and predict sign from image
def Detect(file_path):
    global label_packed
    try:
        # Open the image file
        image = Image.open(file_path)
        
        # Process the image
        processed_image = process_image(image)
        
        # Debugging: Print the shape of the processed image
        print(f"Processed image shape: {processed_image.shape}")
        
        # Predict the sign
        pred = model.predict(processed_image)
        sign = np.argmax(pred, axis=1)[0]
        mapped_sign = map_sign(sign)
        
        # Debugging: Print the prediction result
        print(f"Prediction result: {pred}, Mapped sign: {mapped_sign}")
        
        # Update the label with the detected sign
        label1.configure(foreground="#011638", text=f"Detected Sign: {mapped_sign}")
        label1.pack(side="bottom", expand=True)
    except IOError:
        messagebox.showerror("Error", "Unable to open image file.")
    except PermissionError:
        messagebox.showerror("Error", "Permission denied. Please check your file permissions.")
    except ValueError as e:
        messagebox.showerror("Error", str(e))
    except Exception as e:
        messagebox.showerror("Error", f"An error occurred: {e}")

# Function to show image
def show_image():
    global file_path
    file_path = filedialog.askopenfilename()
    uploaded = Image.open(file_path)
    uploaded.thumbnail(((screen.winfo_width()/2.25), (screen.winfo_height()/2.25)))
    im = ImageTk.PhotoImage(uploaded)
    
    sign_image.configure(image=im)
    sign_image.image = im
    label1.pack(side="bottom", expand=True)
    sign_image.pack(side="bottom", expand=True)


def process_frame(frame):
    # Convert the frame to RGB
    frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    
    # Detect hands in the frame
    gray = cv2.cvtColor(frame_rgb, cv2.COLOR_RGB2GRAY)
    hands = hand_cascade.detectMultiScale(gray, 1.1, 4)
    
    if len(hands) > 0:
        # Assume the first detected hand is the one we want
        x, y, w, h = hands[0]
        hand = frame_rgb[y:y+h, x:x+w]
    else:
        # If no hand is detected, assume the frame is just a hand
        hand = frame_rgb
    
    # Resize the hand image to (50, 50) for the model
    hand = cv2.resize(hand, (50, 50))
    
    # Normalize the image
    hand = hand / 255.0
    
    # Add batch dimension
    hand = np.expand_dims(hand, axis=0)
    
    return hand

def capture_video():
    cap = cv2.VideoCapture(0)
    detected_signs = []
    last_detected_sign = None
    ret, prev_frame = cap.read()
    prev_gray = cv2.cvtColor(prev_frame, cv2.COLOR_BGR2GRAY)
    
    while True:
        ret, frame = cap.read()
        if not ret:
            break
        
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        diff = cv2.absdiff(prev_gray, gray)
        _, thresh = cv2.threshold(diff, 30, 255, cv2.THRESH_BINARY)
        non_zero_count = cv2.countNonZero(thresh)
        
        if non_zero_count > 5000:  # Threshold to detect significant changes
            try:
                # Process the frame
                processed_frame = process_frame(frame)
                
                # Debugging: Print the shape of the processed frame
                print(f"Processed frame shape: {processed_frame.shape}")
                
                # Predict the sign
                result = model.predict(processed_frame)
                sign = np.argmax(result, axis=1)[0]
                confidence = np.max(result)
                
                # Only consider predictions with high confidence
                if confidence > 0.8:
                    mapped_sign = map_sign(sign)
                    
                    # Debugging: Print the prediction result
                    print(f"Prediction result: {result}, Mapped sign: {mapped_sign}")
                    
                    if mapped_sign != last_detected_sign:
                        detected_signs.append(mapped_sign)
                        last_detected_sign = mapped_sign
                    
                    # Display the detected sign on the frame
                    cv2.putText(frame, f"Detected Sign: {mapped_sign}", (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2, cv2.LINE_AA)
                    label1.configure(foreground="white", text=f"Detected Sign: {mapped_sign}")
            except Exception as e:
                print(f"An error occurred: {e}")
        
        cv2.imshow('Video', frame)
        prev_gray = gray
        
        # Press 'q' to quit the video capture
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break
    
    cap.release()
    cv2.destroyAllWindows()
    
    # Show the combined output in a message box
    combined_output = ''.join(detected_signs)
    messagebox.showinfo("Combined Output", f"Detected Signs: {combined_output}")

# Add buttons to upload image, capture video, and detect sign
upload_button = Button(screen, text="Upload an Image", command=show_image, padx=10, pady=5)
upload_button.configure(background="#1ABC9C", foreground="white", font=("Helvetica", 10, "bold"))
upload_button.pack(side="bottom", pady=20)

video_button = Button(screen, text="Capture Video", command=capture_video, padx=10, pady=5)
video_button.configure(background="#1ABC9C", foreground="white", font=("Helvetica", 10, "bold"))
video_button.pack(side="bottom", pady=20)

# Use a global variable to store the file path for the uploaded image
detect_button = Button(screen, text="Detect Sign", command=lambda: Detect(file_path), padx=10, pady=5)
detect_button.configure(background="#1ABC9C", foreground="white", font=("Helvetica", 10, "bold"))
detect_button.pack(side="bottom", pady=20)

# Run the Tkinter main loop
screen.mainloop()
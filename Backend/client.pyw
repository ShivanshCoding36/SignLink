# Import required modules
import socket
import threading
import customtkinter as ctk
import tkinter as tk
from tkinter import messagebox, filedialog
import cv2
import mediapipe as mp
import numpy as np
import pickle
from NEW_normal_text_to_ASL import generate_video, extract_text_from_pdf

print('LIBRARY DONE')

# Load necessary models and data
with open('transformer2.pickle', 'rb') as f:
    transformer = pickle.load(f)

mp_hands = mp.solutions.hands
mp_drawing = mp.solutions.drawing_utils
mp_drawing_styles = mp.solutions.drawing_styles

hands = mp_hands.Hands(static_image_mode=True, min_detection_confidence=0.3)
Reading = True
Text_ = ''

selected_option = 'some value'
toggle = 0

HOST = '127.0.0.1'
PORT = 1234

DARK_GREY = '#121212'
MEDIUM_GREY = '#1F1B24'
OCEAN_BLUE = '#464EB8'
WHITE = "white"
FONT = ("Helvetica", 17)
BUTTON_FONT = ("Helvetica", 15)
SMALL_FONT = ("Helvetica", 13)

# Load models
with open(r'models/random_forest_model.pickle', 'rb') as f:
    model = pickle.load(f)

with open(r'models/RFC_ASIJE.pickle', 'rb') as f:
    model_ASIJE = pickle.load(f)

with open(r'models/RFC_MN.pickle', 'rb') as f:
    model_MN = pickle.load(f)

with open(r'models/RFC_PQ.pickle', 'rb') as f:
    model_PQ = pickle.load(f)

with open(r'models/RFC_URVEK.pickle', 'rb') as f:
    model_URVEK = pickle.load(f)

with open(r'models/RFC_XTLS.pickle', 'rb') as f:
    model_XTLS = pickle.load(f)
print('MODELS LOADED')
# Creating a socket object
client = socket.socket(socket.AF_INET, socket.SOCK_STREAM)

# Helper functions
def add_message(message):
    message_box.configure(state=ctk.NORMAL)
    message_box.insert(ctk.END, message + '\n')
    message_box.configure(state=ctk.DISABLED)

def connect():
    try:
        client.connect((HOST, PORT))
        print("Successfully connected to server")
        add_message("[SERVER] Successfully connected to the server")
    except:
        messagebox.showerror("Connection Error", f"Unable to connect to server {HOST} {PORT}")

    username = username_textbox.get()
    if username != '':
        client.sendall(username.encode())
    else:
        messagebox.showerror("Invalid Username", "Username cannot be empty")

    threading.Thread(target=listen_for_messages_from_server, args=(client,)).start()
    username_textbox.configure(state=ctk.DISABLED)
    username_button.configure(state=ctk.DISABLED)


def send_message():
    message = message_textbox.get()
    if message != '':
        client.sendall(message.encode())
        message_textbox.delete(0, len(message))
    else:
        messagebox.showerror("Empty message", "Message cannot be empty")

def open_popup(callback):
    popup = ctk.CTkToplevel(root)
    popup.geometry('300x200')
    popup.title("Choose an Option")

    Fr1 = ctk.CTkFrame(popup)
    Fr1.pack(fill=tk.BOTH, expand=True)

    label = ctk.CTkLabel(Fr1, text="Select an Option", font=FONT)
    label.pack(pady=10)

    def button1_pressed():
        popup.destroy()
        callback('camera')

    def button2_pressed():
        popup.destroy()
        callback('video')

    button1 = ctk.CTkButton(Fr1, text="Open Camera", command=button1_pressed, width=220, height=40)
    button1.pack(pady=(10, 10))

    button2 = ctk.CTkButton(Fr1, text="Upload Video", command=button2_pressed, width=220, height=40)
    button2.pack(pady=(10, 20))

def predict(frame):
    global Reading, Text_  # Declare Reading as global

    data_aux = [0.0] * 42
    x_ = []
    y_ = []
    frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    H, W, _ = frame.shape
    results = hands.process(frame_rgb)
    
    # Reset Reading to True if no hand gestures are detected
    if not results.multi_hand_landmarks:
        Reading = True
        return None, frame

    for hand_landmarks in results.multi_hand_landmarks:
        mp_drawing.draw_landmarks(
            frame,  # image to draw
            hand_landmarks,  # model output
            mp_hands.HAND_CONNECTIONS,  # hand connections
            mp_drawing_styles.get_default_hand_landmarks_style(),
            mp_drawing_styles.get_default_hand_connections_style())

        for i in range(len(hand_landmarks.landmark)):
            x = hand_landmarks.landmark[i].x
            y = hand_landmarks.landmark[i].y
            x_.append(x)
            y_.append(y)
            data_aux[i * 2] = x
            data_aux[i * 2 + 1] = y

    x1 = int(min(x_) * W) - 10
    y1 = int(min(y_) * H) - 10

    x2 = int(max(x_) * W) - 10
    y2 = int(max(y_) * H) - 10
    prediction = model.predict(np.expand_dims(data_aux, axis=0))[0]
    print(f'Earlier prediction is {prediction}')
    if prediction in ['A','I','J','S','E']:
        prediction = model_ASIJE.predict(np.expand_dims(data_aux, axis=0))[0]
    elif prediction in ['M','N']:
        prediction = model_MN.predict(np.expand_dims(data_aux, axis=0))[0]
    elif prediction in ['P','Q']:
        prediction = model_PQ.predict(np.expand_dims(data_aux, axis=0))[0]
    elif prediction in ['X','T','L','S']:
        prediction = model_XTLS.predict(np.expand_dims(data_aux, axis=0))[0]
    elif prediction in ['U','R','V','E','K']:
        prediction = model_URVEK.predict(np.expand_dims(data_aux, axis=0))[0]
    print("Before manipulation:", Text_)  # Add this line for debugging purposes
    print(prediction, Reading)
    if Reading:
        if prediction == 'space':
            prediction = ' '
            Text_ += prediction
        elif prediction == 'del':
            Text_ = Text_[:-1]
        else:
            Text_ += prediction
        Reading = False
    cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 0, 0), 4)
    print("After manipulation:", Text_)  # Add this line for debugging purposes
    return [Text_, frame]

def videoWork(callback):
    popup = ctk.CTkToplevel(root)
    popup.geometry('300x200')
    popup.title("Choose an Option")

    Fr1 = ctk.CTkFrame(popup)
    Fr1.pack(fill=tk.BOTH, expand=True)

    label = ctk.CTkLabel(Fr1, text="Select an Option", font=FONT)
    label.pack(pady=10)

    def op_video():
        video_path = filedialog.askopenfilename(filetypes=[("Video Files", "*.mp4;*.avi;*.mkv")])
        Send = False
        global Text_
        if video_path:
            video = cv2.VideoCapture(video_path)
            while video.isOpened():
                _, frame = video.read()
                prediction_result = predict(frame)
                if prediction_result:
                    Text_ = prediction_result[0]
                    frame = prediction_result[1]
                if cv2.waitKey(1) & 0xFF == ord('q'):
                    callback(Text_)
                    Send = True
                    break
        if not Send:
            callback(Text_)

    button1 = ctk.CTkButton(Fr1, text="Upload Video", command=op_video, width=220, height=40)
    button1.pack(pady=(10, 20))


def CameraWork(callback):
    popup = ctk.CTkToplevel(root)
    popup.geometry('300x200')
    popup.title("Camera")

    Fr1 = ctk.CTkFrame(popup)
    Fr1.pack(fill=tk.BOTH, expand=True)

    label = ctk.CTkLabel(Fr1, text="Starting Camera", font=FONT)
    label.pack(pady=10)

    def op_cam():
        global Text_
        video = cv2.VideoCapture(0)
        while video.isOpened():
            _, frame = video.read()
            prediction_result = predict(frame)
            if prediction_result:
                frame = prediction_result[1]
            cv2.imshow('Frame', frame)
            if cv2.waitKey(1) & 0xFF == ord('q'):
                try:
                    callback(Text_)
                except Exception as e:
                    pass
                break
        Text_ = ''

    op_cam()


def UseAI():
    def Get_Text(Text):
        print('Output is :', Text)
        message_textbox.delete(0, tk.END)
        message_textbox.insert(tk.END, Text)

    def print_option(option):
        print(option)
        if option == 'camera':
            CameraWork(Get_Text)
        elif option == 'video':
            videoWork(Get_Text)

    open_popup(print_option)

def convert():
    texts = message_box.get('1.0', tk.END)
    texts_rg = []
    for tet in texts.split('\n'):
        parts = tet.split(' ')
        if len(parts) == 2:
            sender, val = parts
            if sender == '[SERVER]':
                continue
            else:
                path = generate_video(val, r'asl_alphabet_test\asl_alphabet_test')
                texts_rg.append((sender, path))
        else:
            print(f"Ignoring malformed message: {tet}")
    message_box.configure(state=tk.NORMAL)
    message_box.delete('1.0', tk.END)
    for x, y in texts_rg:
        tex = f'{x} {y}'
        message_box.insert(tk.END, tex + '\n')
    message_box.configure(state=tk.DISABLED)

def listen_for_messages_from_server(client):
    while True:
        try:
            message = client.recv(2048).decode('utf-8')
            if message != '':
                username, content = message.split("~")
                add_message(f"[{username}] {content}")
            else:
                messagebox.showerror("Error", "Received empty message from server.")
        except Exception as e:
            print(f"Error: {e}")
            break

import customtkinter as ctk
from tkinter import filedialog
from PIL import Image, ImageDraw

# Function to handle document upload
def accept_document():
    file_path = filedialog.askopenfilename(
        title="Select a PDF File",
        filetypes=[("PDF Files", "*.pdf")]
    )
    if file_path:
        print(f"Selected file: {file_path}")
        extract_text_from_pdf(file_path)

# Placeholder function for home button
def home_action():
    print("Home button clicked")

def extraAI():
    def op_cam():
        global Text_
        video = cv2.VideoCapture(0)
        while video.isOpened():
            _, frame = video.read()
            prediction_result = predict(frame)
            if prediction_result:
                frame = prediction_result[1]
            cv2.imshow('Frame', frame)
            if cv2.waitKey(1) & 0xFF == ord('q'):
                break

    op_cam()
    tk.messagebox.showinfo(title='Translated Message',message=prediction_result[0])

# Constants
ctk.set_appearance_mode("system")  # Set mode: system, light, dark
ctk.set_default_color_theme("blue")  # Set theme
FONT = ("Helvetica", 14)
BUTTON_FONT = ("Helvetica", 12, "bold")

# Initialize root window
root = ctk.CTk()
root.geometry("900x700")  # Adjust size to fit the navigation bar
root.title("Sign Link")
root.resizable(False, False)

# Load button images
def load_circular_image(image_path, size):
    img = Image.open(image_path).resize(size, Image.Resampling.LANCZOS)
    mask = Image.new("L", img.size, 0)
    mask_draw = ImageDraw.Draw(mask)
    mask_draw.ellipse((0, 0, size[0], size[1]), fill=255)
    img.putalpha(mask)
    return ctk.CTkImage(img)


home_icon = load_circular_image("imgz\home_icon.png", (50, 50))
doc_icon = load_circular_image("imgz\document_icon.png", (50, 50))

# Left-side navigation bar
nav_frame = ctk.CTkFrame(root, width=200, corner_radius=10, fg_color="transparent")
nav_frame.pack(side="left", fill="y", padx=10, pady=10)

nav_label = ctk.CTkLabel(nav_frame, text="Navigation", font=BUTTON_FONT)
nav_label.pack(pady=10)

home_button = ctk.CTkButton(
    nav_frame,
    image=home_icon,
    text="",
    # fg_color="transparent",
    # hover_color="transparent",
    command=home_action
)
home_button.pack(pady=5, fill="x", padx=10)

upload_button = ctk.CTkButton(
    nav_frame,
    image=doc_icon,
    text="",
    # fg_color="transparent",
    # hover_color="transparent",
    command=accept_document
)
upload_button.pack(pady=5, fill="x", padx=10)

ExtraAI = ctk.CTkButton(
    nav_frame,
    text="AI",
    # fg_color="transparent",
    # hover_color="transparent",
    command=extraAI
)
ExtraAI.pack(pady=5, fill="x", padx=10)

# Main UI layout (updated with side placement)
main_frame = ctk.CTkFrame(root, corner_radius=10)
main_frame.pack(side="right", fill="both", expand=True, padx=10, pady=10)

# Top frame
top_frame = ctk.CTkFrame(main_frame, corner_radius=10)
top_frame.pack(pady=10, padx=10, fill="x")

username_label = ctk.CTkLabel(top_frame, text="Enter Username:", font=FONT)
username_label.pack(side="left", padx=10)

username_textbox = ctk.CTkEntry(top_frame, font=FONT, width=200)
username_textbox.pack(side="left", padx=10)

username_button = ctk.CTkButton(top_frame, text="Join", font=BUTTON_FONT, command=connect)
username_button.pack(side="left", padx=10)

convert_button = ctk.CTkButton(top_frame, text="Convert", font=BUTTON_FONT, command=convert)
convert_button.pack(side="left", padx=10)

# Middle frame
middle_frame = ctk.CTkFrame(main_frame, corner_radius=10)
middle_frame.pack(pady=10, padx=10, fill="both", expand=True)

message_box = ctk.CTkTextbox(middle_frame, font=("Helvetica", 12), state="disabled", height=400)
message_box.pack(pady=10, padx=10, fill="both", expand=True)

# Bottom frame
bottom_frame = ctk.CTkFrame(main_frame, corner_radius=10)
bottom_frame.pack(pady=10, padx=10, fill="x")

message_textbox = ctk.CTkEntry(bottom_frame, font=("Helvetica", 12), width=400)
message_textbox.pack(side="left", padx=10)

AI_button = ctk.CTkButton(bottom_frame, text="Record", font=BUTTON_FONT, command=UseAI)
AI_button.pack(side="left", padx=10)

message_button = ctk.CTkButton(bottom_frame, text="Send", font=BUTTON_FONT, command=send_message)
message_button.pack(side="left", padx=10)

# Main function
def main():
    root.mainloop()

if __name__ == "__main__":
    main()

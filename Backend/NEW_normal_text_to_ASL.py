import cv2
import os
import pyttsx3
import speech_recognition as sr
import fitz  # PyMuPDF

def generate_video(message, folder_path):
    def load_images(folder_path):
        images = {}
        for filename in sorted(os.listdir(folder_path)):
            if filename.endswith(".jpg") or filename.endswith(".png"):
                letter = os.path.splitext(filename)[0].split("_")[0].lower()
                images[letter] = cv2.imread(os.path.join(folder_path, filename))
        return images

    def create_video(frames, filename):
        if len(filename)>35:
            filename = 'TRANSLATED_FILE'
        video_frames = []
        for char in message:
            char_lower = char.lower()
            if char_lower in frames:
                video_frames.append(frames[char_lower])
            else:
                print(f"No ASL image found for character: {char}")

        if video_frames:
            video_height, video_width, _ = video_frames[0].shape
            video_writer = cv2.VideoWriter(f"{filename}.mp4", cv2.VideoWriter_fourcc(*'mp4v'), 2, (video_width, video_height))  # Decreased frame rate to 2 frames per second
            for frame in video_frames:
                video_writer.write(frame)
            video_writer.release()
            print(f"ASL video generated successfully as '{filename}.mp4'.")
            os.startfile(f'{filename}.mp4')
        else:
            print("No valid ASL images found for the input message.")

    # Load images from folder
    images = load_images(folder_path)

    # Generate video filename
    filename = message.replace(" ", "_").lower()  # Replace spaces with underscores and convert to lowercase

    # Generate video
    create_video(images, filename)
    
    return filename + '.mp4'

def take_voice_input():
    engine = pyttsx3.init()
    engine.say("Please speak your message.")
    engine.runAndWait()

    recognizer = sr.Recognizer()
    with sr.Microphone() as source:
        audio_data = recognizer.listen(source)
        print("Recognizing...")

        try:
            message = recognizer.recognize_google(audio_data)
            print(f"Message recognized: {message}")
            return message
        except sr.UnknownValueError:
            print("Sorry, could not understand the audio.")
            return None
        except sr.RequestError as e:
            print(f"Could not request results; {e}")
            return None

def take_input():
    while True:
        choice = input("Do you want to enter the message using voice or typing? (voice/typing): ").lower()
        if choice == "voice":
            message = take_voice_input()
            if message:
                return message
        elif choice == "typing":
            return input("Enter your message: ")
        else:
            print("Invalid choice. Please enter 'voice' or 'typing'.")

def extract_text_from_pdf(pdf_path):
    doc = fitz.open(pdf_path)
    text = ""
    for page_num in range(doc.page_count):
        page = doc.load_page(page_num)
        text += page.get_text()
    folder_path = r"asl_alphabet_test\asl_alphabet_test"
    generate_video(text, folder_path)



# Main function
if __name__ == "__main__":
    print(extract_text_from_pdf('Sign-Link_synopsis.pdf'))
    choice = input("Do you want to enter a message or provide a PDF? (message/pdf): ").lower()
    
    if choice == "pdf":
        pdf_path = input("Enter the path to the PDF: ")
        message = extract_text_from_pdf(pdf_path)
        print(f"Extracted text: {message}")
    else:
        message = take_input()
    
    folder_path = r"asl_alphabet_test\asl_alphabet_test"  # Update folder path with correct location
    generate_video(message, folder_path)


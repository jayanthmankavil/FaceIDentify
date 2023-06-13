import cv2
import face_recognition

# Load reference photos and their corresponding names
reference_images = [
    ("person1.jpg", "Jayanth")
    # Add more reference photos here
]

known_face_encodings = []
known_face_names = []

# Extract face encodings from reference photos
for image_path, name in reference_images:
    image = face_recognition.load_image_file(image_path)
    face_encoding = face_recognition.face_encodings(image)[0]
    known_face_encodings.append(face_encoding)
    known_face_names.append(name)

# Initialize the webcam
video_capture = cv2.VideoCapture(0)

# Initialize face detector from OpenCV
face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + "haarcascade_frontalface_default.xml")

while True:
    # Capture frame from the webcam
    ret, frame = video_capture.read()

    # Convert the frame to grayscale
    gray_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

    # Detect faces in the frame using OpenCV's Haar cascade classifier
    face_locations = face_cascade.detectMultiScale(gray_frame, scaleFactor=1.1, minNeighbors=5, minSize=(30, 30))

    # Iterate over detected faces
    for (x, y, w, h) in face_locations:
        # Extract face encodings from the detected face
        face_encoding = face_recognition.face_encodings(frame, [(y, x + w, y + h, x)])[0]

        # Compare face encodings with the reference encodings
        matches = face_recognition.compare_faces(known_face_encodings, face_encoding)
        name = "Unknown"

        # Check for a match
        if True in matches:
            # Find the index of the matched face
            match_index = matches.index(True)
            name = known_face_names[match_index]

        # Draw a bounding box and name label around the face
        cv2.rectangle(frame, (x, y), (x + w, y + h), (0, 0, 255), 2)
        cv2.putText(frame, name, (x, y - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0, 0, 255), 2)

    # Display the resulting frame
    cv2.imshow('Face Recognition', frame)

    # Exit loop on 'q' key press
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# Release the webcam and close windows
video_capture.release()
cv2.destroyAllWindows()

import argparse
import cv2 as cv
import torch
import os
from network import Net
from transforms import ValidationTransform
from PIL import Image

# NOTE: This will be the live execution of your pipeline


def live(args):
    # Load the model checkpoint
    checkpoint = torch.load("model.pt")
    net = Net(len(checkpoint["classes"]))
    net.load_state_dict(checkpoint["model"])
    net.eval()

    # Initialize the face recognition cascade
    face_cascade = cv.CascadeClassifier(
        cv.data.haarcascades + "haarcascade_frontalface_default.xml"
    )

    # Create a video capture device
    cap = cv.VideoCapture(0)

    if not cap.isOpened():
        print("Error: Could not open video capture device.")
        exit()

    while True:
        ret, frame = cap.read()
        if not ret:
            break

        gray = cv.cvtColor(frame, cv.COLOR_BGR2GRAY)
        faces = face_cascade.detectMultiScale(
            gray, scaleFactor=1.1, minNeighbors=5, minSize=(30, 30)
        )

        for x, y, w, h in faces:
            border_size = int(min(w, h) * args.border)
            x1, y1 = max(x - border_size, 0), max(y - border_size, 0)
            x2, y2 = (
                min(x + w + border_size, frame.shape[1]),
                min(y + h + border_size, frame.shape[0]),
            )

            face_img = frame[y1:y2, x1:x2]
            face_pil = Image.fromarray(cv.cvtColor(face_img, cv.COLOR_BGR2RGB))
            face_tensor = ValidationTransform(face_pil).unsqueeze(0)

            with torch.no_grad():
                outputs = net(face_tensor)
                _, predicted = torch.max(outputs, 1)
                label = checkpoint["classes"][predicted.item()]

            cv.rectangle(frame, (x1, y1), (x2, y2), (255, 0, 0), 2)
            cv.putText(
                frame,
                label,
                (x1, y1 - 10),
                cv.FONT_HERSHEY_SIMPLEX,
                0.9,
                (255, 0, 0),
                2,
            )

        cv.imshow("Live Face Recognition", frame)
        if cv.waitKey(1) & 0xFF == ord("q"):
            break

    cap.release()
    cv.destroyAllWindows()


parser = argparse.ArgumentParser()
parser.add_argument(
    "--border", type=float, required=True, help="Border value for face detection"
)
args = parser.parse_args()

live(args)

import cv2
import logging
import argparse
import warnings
import numpy as np

import torch
import torch.nn.functional as F
from torchvision import transforms

from models import SCRFD
from config import data_config
from utils.helpers import get_model, draw_bbox_gaze, draw_bbox

import pandas as pd

warnings.filterwarnings("ignore")
logging.basicConfig(level=logging.INFO, format='%(message)s')

dict = {
        "pitch":[],
        "yaw":[],
        "emotion":[]
    }

def draw_text_ui(frame, pitch, yaw):
    # Convert tensors to scalars (Python floats)
    pitch_value = pitch.item() if isinstance(pitch, torch.Tensor) else pitch
    yaw_value = yaw.item() if isinstance(yaw, torch.Tensor) else yaw

    # UI box dimensions
    x, y, w, h = 10, 10, 200, 70

    # Semi-transparent background
    overlay = frame.copy()
    cv2.rectangle(overlay, (x, y), (x + w, y + h), (0, 0, 0), -1)
    alpha = 0.5  # Transparency factor
    frame = cv2.addWeighted(overlay, alpha, frame, 1 - alpha, 0)

    # Text settings
    font = cv2.FONT_HERSHEY_SIMPLEX
    font_scale = 0.6
    font_color = (255, 255, 255)
    thickness = 2
    line_type = cv2.LINE_AA

    # Text positions
    pitch_text = f"Pitch: {pitch_value:.2f}"
    yaw_text = f"Yaw: {yaw_value:.2f}"
    cv2.putText(frame, pitch_text, (x + 10, y + 25), font, font_scale, font_color, thickness, line_type)
    cv2.putText(frame, yaw_text, (x + 10, y + 55), font, font_scale, font_color, thickness, line_type)
    def map_gaze_to_emotion(pitch, yaw):
            norm_pitch = pitch
            norm_yaw = yaw
            # Emotion classification based on gaze
            if norm_pitch > 0.5 and abs(norm_yaw) < 0.2:
                return "Thinking, Curiosity"
            elif norm_pitch < -0.5 and abs(norm_yaw) < 0.2:
                return "Sadness, Submission"
            elif norm_pitch < -0.5 and norm_yaw > 0.2:
                return "Guilt, Avoidance"
            elif norm_pitch < -0.5 and norm_yaw < -0.2:
                return "Reflection, Shame"
            elif abs(norm_pitch) < 0.2 and norm_yaw > 0.2:
                return "Distraction, Interest"
            elif abs(norm_pitch) < 0.2 and norm_yaw < -0.2:
                return "Skepticism, Doubt"
            elif norm_pitch > 0.5 and norm_yaw > 0.2:
                return "Imagination, Excitement"
            elif norm_pitch > 0.5 and norm_yaw < -0.2:
                return "Daydreaming"
            else:
                return "Neutral"

    emotion = map_gaze_to_emotion(pitch=float(pitch_value), yaw=float(yaw_value))
    cv2.putText(frame, emotion, (x + 10, y + 85), font, font_scale, font_color, thickness, line_type)
    
    dict["pitch"].append(pitch_value)
    dict["yaw"].append(yaw_value)
    dict["emotion"].append(emotion)
    

    return frame


class Args:
    def __init__(self, arch, gaze_weights, face_weights, view, input, output, bins, binwidth, angle):
        self.arch = arch
        self.gaze_weights = gaze_weights
        self.face_weights = face_weights
        self.view = view
        self.input = input
        self.output = output
        self.bins = bins
        self.binwidth = binwidth
        self.angle = angle

def pre_process(image):
    image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    transform = transforms.Compose([
        transforms.ToPILImage(),
        transforms.Resize(448),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])

    image = transform(image)
    image_batch = image.unsqueeze(0)
    return image_batch


def main(params):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    idx_tensor = torch.arange(params.bins, device=device, dtype=torch.float32)

    try:
        face_detector = SCRFD(model_path=params.face_weights)
        logging.info("Face Detection model weights loaded.")
    except Exception as e:
        logging.info(f"Exception occured while loading pre-trained weights of face detection model. Exception: {e}")

    try:
        gaze_detector = get_model(params.arch, params.bins, inference_mode=True)
        state_dict = torch.load(params.gaze_weights, map_location=device)
        gaze_detector.load_state_dict(state_dict)
        logging.info("Gaze Estimation model weights loaded.")
    except Exception as e:
        logging.info(f"Exception occured while loading pre-trained weights of gaze estimation model. Exception: {e}")

    gaze_detector.to(device)
    gaze_detector.eval()

    cap = cv2.VideoCapture(0)

    with torch.no_grad():
        while True:
            success, frame = cap.read()

            if not success:
                logging.info("Failed to get live camera data")
                break

            bboxes, keypoints = face_detector.detect(frame)
            for bbox, keypoint in zip(bboxes, keypoints):
                x_min, y_min, x_max, y_max = map(int, bbox[:4])

                image = frame[y_min:y_max, x_min:x_max]
                image = pre_process(image)
                image = image.to(device)

                pitch, yaw = gaze_detector(image)

                pitch_predicted, yaw_predicted = F.softmax(pitch, dim=1), F.softmax(yaw, dim=1)

                # Mapping from binned (0 to 90) to angles (-180 to 180) or (0 to 28) to angles (-42, 42)
                pitch_predicted = torch.sum(pitch_predicted * idx_tensor, dim=1) * params.binwidth - params.angle
                yaw_predicted = torch.sum(yaw_predicted * idx_tensor, dim=1) * params.binwidth - params.angle

                # Degrees to Radians
                pitch_predicted = np.radians(pitch_predicted.cpu())
                yaw_predicted = np.radians(yaw_predicted.cpu())

                # draw box and gaze direction
                draw_bbox_gaze(frame, bbox, pitch_predicted, yaw_predicted)
                #draw_bbox(frame, bbox)
                # Display the resulting frame
                # Draw styled UI with text on the frame
            frame = draw_text_ui(frame, pitch_predicted, yaw_predicted)
            cv2.imshow('Camera Feed', frame)

            # Exit the loop when 'q' is pressed
            if cv2.waitKey(1) & 0xFF == ord('q'):
                # Convert the dictionary to a DataFrame
                df = pd.DataFrame(dict)
            
                # Save the DataFrame to a CSV file
                df.to_csv("output.csv", index=False)
                break

    cv2.destroyAllWindows()


if __name__ == "__main__":
    args = Args("resnet34","weights/resnet34.pt","weights/det_10g.onnx",False,"assets/in_video.mp4","assets/resnet34_out_our_code.mp4",90,4,180)
    main(args)


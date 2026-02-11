import cv2
import os
import torch
from tqdm import tqdm
from ultralytics import YOLO
from transformers import TrOCRProcessor, VisionEncoderDecoderModel
from PIL import Image

class InterestFilter:
    def __init__(self):
        print("Initializing AI Models... please wait.")
        
        # 1. Load YOLOv10 (The Eye) - CPU optimized
        self.detector = YOLO("yolov10n.pt")
        
        # 2. Load VLM-ALPR (The Brain)
        # This implements the "Vision and Language" approach for layout-independent OCR
        self.processor = TrOCRProcessor.from_pretrained("microsoft/trocr-small-printed")
        self.model = VisionEncoderDecoderModel.from_pretrained("microsoft/trocr-small-printed")
        
        print("System Ready.")

    def get_windshield_roi(self, car_box, frame):
        """Extracts the upper portion of the car where the driver is visible."""
        x1, y1, x2, y2 = map(int, car_box)
        h = y2 - y1
        # Take the top 45% of the car bounding box (Standard cabin location)
        windshield_crop = frame[y1 : y1 + int(h * 0.45), x1 : x2]
        return windshield_crop

    def apply_vlm_ocr(self, plate_crop):
        """Integrated Vision and Language Recognition"""
        try:
            # Convert OpenCV (BGR) to PIL (RGB) for the Transformer
            image = Image.fromarray(cv2.cvtColor(plate_crop, cv2.COLOR_BGR2RGB)).convert("RGB")
            
            # Pre-process image for the Vision Encoder
            pixel_values = self.processor(images=image, return_tensors="pt").pixel_values
            
            # Generate text using the Language Decoder (The 'Language' part of the VLM)
            generated_ids = self.model.generate(pixel_values)
            generated_text = self.processor.batch_decode(generated_ids, skip_special_tokens=True)[0]
            
            return generated_text
        except Exception as e:
            return f"OCR Error: {e}"

    def process_frame(self, frame):
        """Processes a single frame to find cars, plates, and windshields."""
        # Inference with YOLOv10
        results = self.detector(frame, verbose=False)[0]
        candidates = []

        for box in results.boxes:
            # Class 2 is 'car' in the standard COCO dataset
            if int(box.cls[0]) == 2:
                coords = box.xyxy[0].tolist()
                x1, y1, x2, y2 = map(int, coords)
                
                # 1. Capture Windshield for Stage 2 (Behavioral Analysis)
                windshield_img = self.get_windshield_roi(coords, frame)
                
                # 2. Capture Plate for Stage 1 (Identification)
                # Heuristic: License plates are usually in the lower 30% of a car's box
                plate_roi = frame[y1 + int((y2-y1)*0.7):y2, x1:x2]
                
                plate_text = "None"
                if plate_roi.size > 0:
                    plate_text = self.apply_vlm_ocr(plate_roi)
                
                candidates.append({
                    "windshield": windshield_img,
                    "plate": plate_text,
                    "box": [x1, y1, x2, y2],
                    "confidence": float(box.conf[0])
                })
        return candidates

if __name__ == "__main__":
    filter_sys = InterestFilter()
    
    # 1. Path Setup
    video_path = r"D:\downloads\ALHAMZAH\Collage\years\Fourth Year\Second Semester\Graduation Project\Model\Driver_Distraction_Project\data\raw\Street footage\license-plate-detection-test.mp4"
    
    if not os.path.exists(video_path):
        print(f"!!! ERROR: Video not found at {video_path}")
    else:
        cap = cv2.VideoCapture(video_path)
        
        # --- PROGRESS BAR SETUP LINES ---
        total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        # This creates the bar in your terminal
        pbar = tqdm(total=total_frames, desc="AI Analysis Progress") 
        # --------------------------------

        while cap.isOpened():
            ret, frame = cap.read()
            if not ret:
                break

            # AI Logic
            found_candidates = filter_sys.process_frame(frame)

            # --- PROGRESS BAR UPDATE LINE ---
            pbar.update(1) 
            # --------------------------------

            # Optional: Show the frame (Note: This makes it slower)
            cv2.imshow("CCTV Stage 1", frame)
            if cv2.waitKey(1) & 0xFF == ord('q'):
                break

        # --- CLOSE THE BAR ---
        pbar.close()
        # ---------------------
        
        cap.release()
        cv2.destroyAllWindows()
        print("Processing Finished.")
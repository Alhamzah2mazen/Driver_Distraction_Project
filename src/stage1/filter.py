import cv2
import os
import torch
from tqdm import tqdm
from ultralytics import YOLO
from transformers import TrOCRProcessor, VisionEncoderDecoderModel
from PIL import Image

class InterestFilter:
    def __init__(self):
        print("--- Initializing AI Models on RTX 3050 ---")
        # Check for NVIDIA GPU
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        
        # 1. Load YOLOv10
        self.detector = YOLO("yolov10n.pt")
        self.detector.to(self.device)
        
        # 2. Load VLM-ALPR (TrOCR)
        self.processor = TrOCRProcessor.from_pretrained("microsoft/trocr-small-printed")
        self.model = VisionEncoderDecoderModel.from_pretrained("microsoft/trocr-small-printed")
        self.model.to(self.device)
        
        if self.device == "cuda":
            print(f"STATUS: GPU Accelerated ({torch.cuda.get_device_name(0)})")
        else:
            print("STATUS: Falling back to CPU. Ensure CUDA is installed for speed.")

    def apply_vlm_ocr(self, plate_crop):
        """Processes the license plate ROI using the Vision-Language Model."""
        try:
            image = Image.fromarray(cv2.cvtColor(plate_crop, cv2.COLOR_BGR2RGB)).convert("RGB")
            pixel_values = self.processor(images=image, return_tensors="pt").pixel_values.to(self.device)
            
            with torch.no_grad():
                generated_ids = self.model.generate(pixel_values)
            
            return self.processor.batch_decode(generated_ids, skip_special_tokens=True)[0]
        except Exception:
            return "N/A"

    def process_frame(self, frame):
        """Detects cars and extracts ROI data."""
        results = self.detector(frame, verbose=False)[0]
        candidates = []

        for box in results.boxes:
            if int(box.cls[0]) == 2:  # 'car' class
                coords = box.xyxy[0].tolist()
                x1, y1, x2, y2 = map(int, coords)
                
                # ROI Definitions
                h_car = y2 - y1
                # Windshield: Top 45%
                windshield_roi = frame[max(0, y1) : y1 + int(h_car * 0.45), x1 : x2]
                # Plate: Bottom 30%
                plate_roi = frame[y1 + int(h_car * 0.7):y2, x1:x2]
                
                plate_text = "N/A"
                if plate_roi.size > 0:
                    plate_text = self.apply_vlm_ocr(plate_roi)
                
                candidates.append({
                    "plate": plate_text,
                    "box": [x1, y1, x2, y2],
                    "windshield_crop": windshield_roi
                })
        return candidates

if __name__ == "__main__":
    filter_sys = InterestFilter()
    
    # Path configuration
    video_path = r"C:\Users\alhas\Downloads\Alhamzah\Univ\Graduation Project\Driver_Distraction_Project\data\raw\Street footage\license-plate-detection-test.mp4"
    output_dir = "output/detections"
    log_path = "output/results_log.txt"
    
    os.makedirs(output_dir, exist_ok=True)
    
    if not os.path.exists(video_path):
        print(f"!!! ERROR: Video not found at {video_path}")
    else:
        cap = cv2.VideoCapture(video_path)
        total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        pbar = tqdm(total=total_frames, desc="RTX 3050 Analysis") 
        
        cv2.namedWindow("Stage 1: Interest Filter", cv2.WINDOW_NORMAL)

        with open(log_path, "a") as log_file:
            while cap.isOpened():
                ret, frame = cap.read()
                if not ret: break

                frame_idx = int(cap.get(cv2.CAP_PROP_POS_FRAMES))
                found_candidates = filter_sys.process_frame(frame)

                for i, c in enumerate(found_candidates):
                    x1, y1, x2, y2 = c["box"]
                    h_car = y2 - y1

                    # --- Visual Feedback ---
                    # Green Box: Windshield (for Stage 2)
                    cv2.rectangle(frame, (x1, y1), (x2, y1 + int(h_car * 0.45)), (0, 255, 0), 2)
                    # Blue Box: License Plate (for Stage 1)
                    cv2.rectangle(frame, (x1, y1 + int(h_car * 0.7)), (x2, y2), (255, 0, 0), 2)
                    
                    cv2.putText(frame, f"Plate: {c['plate']}", (x1, y1 - 10), 
                                cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 2)

                    # --- Saving Results ---
                    if c['plate'] != "N/A":
                        save_name = f"frame_{frame_idx}_car_{i}.jpg"
                        cv2.imwrite(os.path.join(output_dir, save_name), frame)
                        log_file.write(f"Frame: {frame_idx} | File: {save_name} | Plate: {c['plate']}\n")

                pbar.update(1)
                cv2.imshow("Stage 1: Interest Filter", frame)
                if cv2.waitKey(1) & 0xFF == ord('q'): break

        cap.release()
        cv2.destroyAllWindows()
        print(f"Done! Results saved in: {os.path.abspath('output')}")
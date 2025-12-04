import os
from ultralytics import YOLO
import cv2
import numpy as np

# put here your model path
model = YOLO(r"D:\Rickshaw Detector\Model\model_test\runs\detect\train\weights\best.pt") 

# put here your input video path
input_video = r"D:\Rickshaw Detector\Model\model_test\test_video\video3.mp4" 

# Output folder
output_folder = "count_output"

# Create the folder if it doesn't exist
if not os.path.exists(output_folder):
    os.mkdir(output_folder)

output_video = os.path.join(output_folder, "entry_exit_counted.mp4")


cap = cv2.VideoCapture(input_video)
fps = int(cap.get(cv2.CAP_PROP_FPS))
width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))

fourcc = cv2.VideoWriter_fourcc(*"mp4v")
out = cv2.VideoWriter(output_video, fourcc, fps, (width, height))

# Counting logic
entry_count = 0
exit_count = 0
unique_ids = {}  # Dictionary to store each rickshaw's counted status

# Virtual line (y-coordinate)
line_y = height // 2

while True:
    ret, frame = cap.read()
    if not ret:
        break

    # Draw virtual line
    cv2.line(frame, (0, line_y), (width, line_y), (0, 0, 255), 2)

    # Model prediction with tracking
    results = model.track(frame, conf=0.3, iou=0.5, persist=True, classes=[0], vid_stride=3)

    # Extract detections
    if results[0].boxes.id is not None:
        boxes = results[0].boxes.xyxy.cpu().numpy()    # [x1, y1, x2, y2]
        ids = results[0].boxes.id.cpu().numpy()       # tracker IDs

        for box, obj_id in zip(boxes, ids):
            x1, y1, x2, y2 = map(int, box)
            center_y = (y1 + y2) // 2

            # Initialize ID if not exists
            if int(obj_id) not in unique_ids:
                unique_ids[int(obj_id)] = {"counted": False, "direction": None}

            # Check crossing line
            if not unique_ids[int(obj_id)]["counted"]:
                if center_y > line_y + 5:  # crossed downwards => entry
                    entry_count += 1
                    unique_ids[int(obj_id)]["counted"] = True
                    unique_ids[int(obj_id)]["direction"] = "entry"
                elif center_y < line_y - 5:  # crossed upwards => exit
                    exit_count += 1
                    unique_ids[int(obj_id)]["counted"] = True
                    unique_ids[int(obj_id)]["direction"] = "exit"

            # Draw rectangle and ID
            cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 2)
            cv2.putText(frame, f"ID:{int(obj_id)}", (x1, y1 - 10),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)

    # Draw counts on top-right corner
    count_text = f"Entry: {entry_count}  Exit: {exit_count}"
    cv2.putText(frame, count_text, (width - 400, 40), cv2.FONT_HERSHEY_SIMPLEX,
                1, (0, 255, 0), 2, cv2.LINE_AA)

    # Write frame and show
    out.write(frame)
    cv2.imshow("Rickshaw Entry/Exit Counting", frame)
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
out.release()
cv2.destroyAllWindows()

print("Processing complete!")
print(f"Total Entries: {entry_count}")
print(f"Total Exits: {exit_count}")
print(f"Saved output as: {output_video}")

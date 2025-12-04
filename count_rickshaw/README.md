# Rickshaw Entry/Exit Counting

Real-time detection and counting of rickshaws in video using **YOLO** and **OpenCV**.  
This script tracks individual rickshaws and counts how many cross a virtual line (entry or exit) in the frame.  

---

## Features

- Detect rickshaws in videos with a YOLO-based model.
- Assign unique IDs to each detected rickshaw for accurate tracking.
- Count entries and exits across a defined virtual line.
- Draw bounding boxes, IDs, and counts on the video.
- Save the processed video in a dedicated output folder.

---

## Requirements

- Python 3.8+
- [Ultralytics YOLO](https://github.com/ultralytics/ultralytics)
- OpenCV
- NumPy

Install dependencies using pip:

```bash
pip install ultralytics opencv-python numpy
```

---

## Usage Instructions

1. **Clone or download the repository** and place your YOLO model in the project folder.  

2. **Update the script paths**:

```python
# Path to your trained YOLO model
model = YOLO(r"YOUR_MODEL_PATH/best.pt") 

# Path to your input video
input_video = r"YOUR_VIDEO_PATH/video.mp4" 

# Output folder (will be created automatically if it doesn't exist)
output_folder = "count_output"
```

3. **Run the script**:

```bash
python rickshaw_count.py
```

4. **Controls during execution**:  
   - Press **`q`** to exit the video display window before the video ends.

5. **Output**:  
   - Processed video with bounding boxes, IDs, and counts will be saved as:

```
count_output/entry_exit_counted.mp4
```

6. **Console Output**:  
   - The script prints total entries and exits after processing:

```
Processing complete!
Total Entries: X
Total Exits: Y
Saved output as: count_output/entry_exit_counted.mp4
```

---

## Notes

- `classes=[0]` in `model.track()` assumes rickshaw is class 0 in your YOLO model. Adjust if necessary.
- The virtual line is set at `height // 2`. Change `line_y` to reposition it.
- You can increase `vid_stride` for faster processing, but it may skip fast-moving objects.

---

## License

This project is licensed under the MIT License.
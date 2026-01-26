Here is a professional **README.md** file for your project. You can copy and paste this directly into a file named `README.md` on GitHub or your project folder.

---

# 🚗 Basic ADAS Prototype (Lane Detection & Collision Warning)

A Python-based **Advanced Driver-Assistance System (ADAS)** prototype that leverages **OpenCV** for lane detection/safety monitoring and **YOLOv8** for AI-powered vehicle tracking.

This project processes a dashboard camera video feed to perform three key tasks in real-time:

1. **Lane Line Visualization:** Detects road markings.
2. **Forward Collision Warning:** Monitors a safety zone for sudden obstructions.
3. **Vehicle Counting:** Tracks and counts vehicles entering the driver's lane.

## 🌟 Features

* **Hybrid Detection Pipeline:** Combines traditional Computer Vision (Canny Edge/Hough Transform) with Deep Learning (YOLOv8).
* **Real-time Object Tracking:** Uses YOLOv8's `track()` mode to assign persistent IDs to vehicles.
* **Dynamic Overlay:** Visualizes lane lines, bounding boxes, confidence scores, and safety warnings.
* **Zone-Based Logic:** Distinguishes between vehicles *inside* the ego-lane vs. those in adjacent lanes.
* **Motion-Based Safety Trigger:** Uses Background Subtraction to detect sudden movement in the immediate path of the vehicle.

## 🛠️ Tech Stack

* **Language:** Python 3.8+
* **Computer Vision:** OpenCV (`cv2`)
* **Deep Learning:** Ultralytics YOLOv8
* **Matrix Operations:** NumPy

## ⚙️ Installation

1. **Clone the repository:**
```bash
git clone https://github.com/yourusername/adas-prototype.git
cd adas-prototype

```


2. **Install dependencies:**
It is recommended to use a virtual environment.
```bash
pip install opencv-python numpy ultralytics

```


3. **Prepare Directory Structure:**
Ensure your project folder looks like this:
```
├── video/
│   └── eg_1.mp4          # Your input video
├── output_video/         # Output will be saved here
├── main.py               # The script provided
└── README.md

```



## 🚀 Usage

1. **Configure Input Source:**
Open `main.py` and modify the `INPUT_VIDEO` variable to point to your video file.
```python
INPUT_VIDEO = "video/my_driving_footage.mp4"

```


2. **Run the script:**
```bash
python main.py

```


3. **Output:**
The processed video will be saved to `output_video/final_full_adas.mp4`. A window will also pop up showing the real-time processing (Press `q` to quit).

## ⚠️ Configuration & Calibration

**Crucial Note:** This script uses **hardcoded Region of Interest (ROI)** coordinates optimized for a specific video resolution and camera angle.

If you use your own video, you **must** adjust the following coordinates in the code to match your camera perspective:

* **`lane_triangle`**: The triangular mask used for lane line detection.
* **`safety_roi_points`**: The polygon used for the collision warning zone.

*Tip: You can use `cv2.imshow` to display the mask temporarily while adjusting these coordinate points.*

## 🧠 How It Works

### 1. Lane Detection (OpenCV)

* Converts frame to Grayscale and applies Gaussian Blur.
* Uses **Canny Edge Detection** to find boundaries.
* Masks the image to a "Triangle of Interest" (the road ahead).
* Applies **Hough Line Transform** to detect straight lines within the mask.

### 2. Safety Warning (Motion Detection)

* Uses `cv2.createBackgroundSubtractorMOG2` to detect moving pixels in the "Safety ROI".
* If the area of motion contours exceeds `2000` pixels, a **RED WARNING** overlay is triggered.

### 3. Vehicle Tracking (YOLOv8)

* The model detects vehicles (classes: Car, Motorcycle, Bus, Truck).
* It calculates the center point of every bounding box.
* **`cv2.pointPolygonTest`** checks if the center point is inside the `lane_triangle`.
* **Green Box:** Vehicle is in your lane (Counted).
* **Red Box:** Vehicle is in adjacent lane (Ignored).



## 🔮 Future Improvements

* [ ] **Dynamic Calibration:** Auto-detect vanishing point to set ROIs automatically.
* [ ] **Curved Lane Support:** Replace Hough Transform with Sliding Window Polyfit for curved roads.
* [ ] **Distance Estimation:** Use bounding box size to estimate distance to the car ahead.


---

*Author: [Swayam]*

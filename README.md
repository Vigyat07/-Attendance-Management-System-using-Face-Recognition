# -Attendance-Management-System-using-Face-Recognition by Vigyat Bhat
This repository contains face recognition project for the AICTE internship on AI: TechSaksham 

This project, **Attendance-Management-System-using-Face-Recognition**, leverages face recognition technology to automate attendance marking. Developed in Python with OpenCV and Flask, the system ensures efficiency, security, and real-time operation.

---

## **Features**
- Real-time facial recognition using LBPH (Local Binary Patterns Histogram) algorithm.
- Automated attendance marking with timestamped logs.
- Attendance data stored in a CSV file for easy access.
- User-friendly interface for image capture and attendance marking.

---

## **Setup and Installation**

1. **Clone the Repository:**
   ```bash
   git clone https://github.com/Vigyat07/Attendance-Management-System-using-Face-Recognition.git
2.Navigate to the Project Directory:
cd Attendance-Management-System-using-Face-Recognition

3.Install Dependencies: Ensure Python is installed, then install the required libraries:
pip install -r requirements.txt


4.Run the Application: Start the Flask server:
python app.py
Open your browser and go to http://127.0.0.1:5000/.

---
## Usage Guide
---
## ** Capturing Images for Training**
- Click "Capture Images" on the homepage.
- Important Instructions:
1)Look directly at the webcam and wait for a prompt (e.g., blink or confirmation sound).
2)Ensure your face is fully visible and remain steady.
3)The system captures multiple images, so maintain your position.
---
  
## ** Training the Model**
   After capturing images, click "Train Model".
- The system will process images and train the LBPH model.
- Wait until a success message appears.
---
## ** Marking Attendance**
- Click "Mark Attendance".
- Look at the webcam for recognition.
- Important Instructions:
- Ensure proper lighting and position your face directly in front of the camera.
- The system identifies you and marks attendance automatically with a timestamp.
---
## System Design
**Workflow Steps**
- Capture Images: Collect facial images using a webcam.
- Train Model: Use the LBPH algorithm to train the system.
- Mark Attendance: Real-time facial recognition for attendance logging.
---
## **Requirements**
## **Hardware:**
- A computer or laptop with:
- Webcam
- Minimum 4GB RAM
- 1GHz processor or higher
## **Software:**
- Python 3.6 or later
- Flask Framework
- OpenCV Library
---
## **Future Enhancements**
- Cloud-based storage for attendance data.
- Mobile app integration for marking attendance.
- Enhanced facial recognition algorithms for diverse conditions.
---
## **References**
- K. Mridha and N. T. Yousef, "Study and Analysis of Implementing a Smart Attendance Management System Based on Face Recognition Technique using OpenCV and Machine Learning," 2021 IEEE International Conference, doi: 10.1109/CSNT51715.2021.9509614.
- A. Kumar, S. Samal, M. S. Saluja and A. Tiwari, "Automated Attendance System Based on Face Recognition Using OpenCV," 2023 IEEE International Conference, doi: 10.1109/ICACCS57279.2023.10112665.
- N. Stekas and D. Van Den Heuvel, "Face Recognition Using Local Binary Patterns Histograms (LBPH) on an FPGA-Based System on Chip (SoC)," 2016 IEEE International Parallel and Distributed Processing Symposium Workshops, doi: 10.1109/IPDPSW.2016.67.
- M. Khan, S. Chakraborty, R. Astya and S. Khepra, "Face Detection and Recognition Using OpenCV," 2019 International Conference, doi: 10.1109/ICCCIS48478.2019.8974493.
- ArXiv: Face Recognition and Detection, "Face Recognition: State of the Art," arXiv, doi: 10.48550/arXiv.1907.12739.
---
## **License**
- This project is licensed under the MIT License - see the LICENSE file for details.

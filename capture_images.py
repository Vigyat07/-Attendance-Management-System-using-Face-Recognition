import cv2
import os

# Directory to save images
SAVE_DIR = "dataset/user_1"  # Customize the path
os.makedirs(SAVE_DIR, exist_ok=True)

# Initialize camera
cap = cv2.VideoCapture(0)

if not cap.isOpened():
    print("Error: Could not open camera.")
    exit()

print("Press 'q' to quit and save images.")
count = 0

while True:
    ret, frame = cap.read()
    if not ret:
        print("Failed to capture frame.")
        break

    # Display the video feed
    cv2.imshow("Capture Images", frame)

    # Save frames when 's' is pressed
    key = cv2.waitKey(1) & 0xFF
    if key == ord('s'):  # Press 's' to save the image
        count += 1
        img_path = os.path.join(SAVE_DIR, f"image_{count}.jpg")
        cv2.imwrite(img_path, frame)
        print(f"Saved: {img_path}")

    elif key == ord('q'):  # Press 'q' to quit
        break

cap.release()
cv2.destroyAllWindows()
print(f"Captured {count} images. Stored in {SAVE_DIR}.")

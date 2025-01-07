import os
import cv2
from mtcnn import MTCNN
from mtcnn.utils.images import load_image
from mtcnn.utils.plotting import plot
import matplotlib.pyplot as plt

def save_faces_from_folder(images_folder_path, output_folder="face_img"):
    # Ensure output folder exists
    os.makedirs(output_folder, exist_ok=True)

    # Initialize MTCNN for face detection
    mtcnn = MTCNN()

    # Iterate over all files in the given folder
    for filename in os.listdir(images_folder_path):
        file_path = os.path.join(images_folder_path, filename)

        # Check if it's a valid image file
        if os.path.isfile(file_path) and filename.lower().endswith((".jpg", ".jpeg", ".png")):
            try:
                # Read and convert the image
                image = cv2.imread(file_path)
                img_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

                # Detect faces in the image
                results = mtcnn.detect_faces(img_rgb)

                # Process each detected face
                for i, result in enumerate(results):
                    x, y, w, h = result['box']

                    # Ensure bounding box values are positive
                    x, y = max(0, x), max(0, y)

                    # Crop and resize the face
                    face = img_rgb[y:y+h, x:x+w]
                    face_resized = cv2.resize(face, (160, 160))

                    # Save the face image
                    output_path = os.path.join(output_folder, f"face_{os.path.splitext(filename)[0]}_{i+1}.jpg")
                    plt.imsave(output_path, face_resized)
                    print(f"Saved: {output_path}")

            except Exception as e:
                print(f"Error processing {file_path}: {e}")

# Example usage
images_folder_path = r"E:\face recognition\single_image of each people"  # Change to your folder path
save_faces_from_folder(images_folder_path)
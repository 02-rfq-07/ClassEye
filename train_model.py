import os
import face_recognition
import pickle
from PIL import Image
import numpy as np

def train_face_recognition_model():
    """
    Train the face recognition model by encoding faces from the dataset
    """
    dataset_path = r"C:\Users\navee\Downloads\dataset"
    models_path = r"C:\Users\navee\Downloads\attendance_system\models"
    
    # Create models directory if it doesn't exist
    os.makedirs(models_path, exist_ok=True)
    
    known_face_encodings = []
    known_face_names = []
    
    print("Training face recognition model...")
    
    # Iterate through each person's folder
    for person_name in os.listdir(dataset_path):
        person_folder = os.path.join(dataset_path, person_name)
        
        if not os.path.isdir(person_folder):
            continue
            
        print(f"Processing images for {person_name}...")
        
        # Process each image in the person's folder
        for image_file in os.listdir(person_folder):
            if image_file.lower().endswith(('.jpg', '.jpeg', '.png')):
                image_path = os.path.join(person_folder, image_file)
                
                try:
                    # Load image
                    image = face_recognition.load_image_file(image_path)
                    
                    # Get face encodings
                    face_encodings = face_recognition.face_encodings(image)
                    
                    if len(face_encodings) > 0:
                        # Use the first face found in the image
                        face_encoding = face_encodings[0]
                        known_face_encodings.append(face_encoding)
                        known_face_names.append(person_name)
                        print(f"  ✓ Processed {image_file}")
                    else:
                        print(f"  ⚠ No face found in {image_file}")
                        
                except Exception as e:
                    print(f"  ✗ Error processing {image_file}: {str(e)}")
    
    # Save the encodings and names
    data = {
        'encodings': known_face_encodings,
        'names': known_face_names
    }
    
    model_file = os.path.join(models_path, "new_face_encodings.pkl")
    with open(model_file, 'wb') as f:
        pickle.dump(data, f)
    
    print(f"\nTraining completed!")
    print(f"Total faces encoded: {len(known_face_encodings)}")
    print(f"Model saved to: {model_file}")
    
    # Print summary
    unique_names = set(known_face_names)
    print(f"People in database: {len(unique_names)}")
    for name in unique_names:
        count = known_face_names.count(name)
        print(f"  - {name}: {count} images")

if __name__ == "__main__":
    train_face_recognition_model()
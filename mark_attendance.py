import os
import face_recognition
import pickle
import cv2
import numpy as np
import csv
from datetime import datetime
from PIL import Image

# TUNE THIS: lower => stricter (more unknowns), higher => looser (fewer unknowns)
DEFAULT_TOLERANCE = 0.45
# Set to True to print face-distance arrays for debugging
DEBUG = True

def load_known_faces():
    """
    Load the trained face encodings from the model file
    """
    model_path = r"C:\Users\navee\Downloads\attendance_system\models\face_encodings2.pkl"
    
    if not os.path.exists(model_path):
        print("❌ Model file not found! Please run train_model.py first.")
        return None, None
    
    with open(model_path, 'rb') as f:
        data = pickle.load(f)
    
    return data.get('encodings', []), data.get('names', [])

def mark_attendance_from_image(image_path, tolerance=DEFAULT_TOLERANCE, debug=DEBUG):
    known_encodings, known_names = load_known_faces()
    if known_encodings is None:
        return [], 0
    print(f"📸 Processing image: {image_path}")
    try:
        test_image = face_recognition.load_image_file(image_path)
    except Exception as e:
        print(f"❌ Error loading image: {str(e)}")
        return [], 0

    face_locations = face_recognition.face_locations(test_image)
    face_encodings = face_recognition.face_encodings(test_image, face_locations)
    
    print(f"👤 Found {len(face_locations)} face(s) in the image")
    
    present_people = []
    unknown_count = 0
    
    # Create folder for saving unknown faces
    unknown_dir = "unknown_faces"
    os.makedirs(unknown_dir, exist_ok=True)
    
    for i, (face_encoding, face_location) in enumerate(zip(face_encodings, face_locations), start=1):
        # If we have known encodings, compute distances and decide by min distance
        if len(known_encodings) > 0:
            face_distances = face_recognition.face_distance(known_encodings, face_encoding)
            best_match_index = int(np.argmin(face_distances))
            best_distance = float(face_distances[best_match_index])
            
            if debug:
                # Print a concise summary of distances
                d_list = ", ".join([f"{d:.3f}" for d in face_distances])
                print(f"   Face #{i} distances: [{d_list}]")
                print(f"   → Best distance: {best_distance:.3f} (threshold: {tolerance})")
            
            # Accept the match only if the minimum distance is <= tolerance
            if best_distance <= tolerance:
                name = known_names[best_match_index]
                confidence = max(0.0, 1.0 - best_distance)  # approximate confidence
                if name not in present_people:
                    present_people.append(name)
                    print(f"✅ Recognized: {name} (confidence: {confidence:.2f})")
                else:
                    print(f"🔄 {name} already marked present")
            else:
                # Treat as unknown
                unknown_count += 1
                print(f"❓ Unknown person detected (Face #{i}) — min distance {best_distance:.3f} > {tolerance}")
                
                # Crop and save unknown face
                top, right, bottom, left = face_location
                face_image = test_image[top:bottom, left:right]
                pil_image = Image.fromarray(face_image)
                
                timestamp = datetime.now().strftime("%Y%m%d_%H%M%S_%f")
                base = os.path.splitext(os.path.basename(image_path))[0]
                filename = f"unknown_{base}_{timestamp}_{unknown_count}.jpg"
                save_path = os.path.join(unknown_dir, filename)
                pil_image.save(save_path)
                print(f"💾 Unknown face saved to: {save_path}")
        else:
            # No known encodings at all — treat every face as unknown
            unknown_count += 1
            print(f"❓ Unknown person detected (Face #{i}) — no known encodings loaded")
            top, right, bottom, left = face_location
            face_image = test_image[top:bottom, left:right]
            pil_image = Image.fromarray(face_image)
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S_%f")
            base = os.path.splitext(os.path.basename(image_path))[0]
            filename = f"unknown_{base}_{timestamp}_{unknown_count}.jpg"
            save_path = os.path.join(unknown_dir, filename)
            pil_image.save(save_path)
            print(f"💾 Unknown face saved to: {save_path}")
    
    return present_people, unknown_count

def save_attendance(present_people, session_name="", unknown_count=0):
    """
    Save attendance to CSV file
    """
    logs_dir = "attendance_logs"
    os.makedirs(logs_dir, exist_ok=True)
    
    current_date = datetime.now().strftime("%Y-%m-%d")
    filename = f"attendance_{current_date}.csv"
    if session_name:
        filename = f"attendance_{current_date}_{session_name}.csv"
    
    filepath = os.path.join(logs_dir, filename)
    file_exists = os.path.exists(filepath)
    
    with open(filepath, 'a', newline='') as csvfile:
        writer = csv.writer(csvfile)
        
        if not file_exists:
            writer.writerow(['Date', 'Time', 'Name', 'Status'])
        
        current_time = datetime.now().strftime("%H:%M:%S")
        current_date = datetime.now().strftime("%Y-%m-%d")
        
        # Save recognized people
        for person in present_people:
            writer.writerow([current_date, current_time, person, 'Present'])
        
        # Save unknowns
        for i in range(1, unknown_count + 1):
            writer.writerow([current_date, current_time, f"Unknown #{i}", 'Unknown'])
    
    print(f"📄 Attendance saved to: {filepath}")

def process_multiple_images():
    test_dir = r"C:\Users\navee\Downloads\attendance_system\test_images"
    
    if not os.path.exists(test_dir):
        print(f"❌ Test images directory '{test_dir}' not found!")
        return
    
    image_files = [f for f in os.listdir(test_dir) if f.lower().endswith(('.jpg', '.jpeg', '.png'))]
    if not image_files:
        print(f"❌ No image files found in '{test_dir}'!")
        return
    
    all_present = set()
    total_unknowns = 0
    
    print(f"🔄 Processing {len(image_files)} image(s)...")
    print("="*50)
    
    for image_file in image_files:
        image_path = os.path.join(test_dir, image_file)
        present_in_image, unknown_count = mark_attendance_from_image(image_path, tolerance=DEFAULT_TOLERANCE, debug=DEBUG)
        all_present.update(present_in_image)
        total_unknowns += unknown_count
        print("-"*30)
    
    print("="*50)
    print("📋 ATTENDANCE SUMMARY")
    print("="*50)
    
    if all_present or total_unknowns > 0:
        if all_present:
            print(f"✅ Total people present: {len(all_present)}")
            for person in sorted(all_present):
                print(f"   • {person}")
        if total_unknowns > 0:
            print(f"❓ Unknown faces detected: {total_unknowns-1}")
        
        save_attendance(list(all_present), unknown_count=total_unknowns)
    else:
        print("❌ No faces detected in any image.")

def mark_single_image():
    image_path = input("Enter the path to the image file: ").strip()
    
    if not os.path.exists(image_path):
        print("❌ Image file not found!")
        return
    
    present_people, unknown_count = mark_attendance_from_image(image_path, tolerance=DEFAULT_TOLERANCE, debug=DEBUG)
    
    if present_people or unknown_count > 0:
        if present_people:
            print(f"\n✅ People present: {', '.join(present_people)}")
        if unknown_count > 0:
            print(f"❓ Unknown faces detected: {unknown_count}")
        
        save_choice = input("Save attendance to file? (y/n): ").strip().lower()
        if save_choice == 'y':
            session_name = input("Enter session name (optional): ").strip()
            save_attendance(present_people, session_name, unknown_count)
    else:
        print("\n❌ No faces detected.")

def main():
    print("="*50)
    print("🎯 FACE RECOGNITION ATTENDANCE SYSTEM")
    print("="*50)
    
    print("\nChoose an option:")
    print("1. Process all images in test_images folder")
    print("2. Process a single image")
    print("3. Exit")
    
    choice = input("\nEnter your choice (1-3): ").strip()
    
    if choice == '1':
        process_multiple_images()
    elif choice == '2':
        mark_single_image()
    elif choice == '3':
        print("👋 Goodbye!")
        return
    else:
        print("❌ Invalid choice!")

if __name__ == "__main__":
    main()

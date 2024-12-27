import streamlit as st  # type: ignore
import cv2
import os
import csv
from datetime import datetime
import pandas as pd  # Import pandas for data manipulation
import numpy as np

# Set the title for Streamlit
st.title("Face Recognition System with Check-in/Check-out Logging and Photo Capture")

# Set output folder paths
output_folder = r"E:\face_database9"
model_file = os.path.join(output_folder, r'E:\face_database9\face_recognizer.yml')
label_map_file = os.path.join(output_folder, r'E:\face_database9\label_map.csv')
check_in_log_file = os.path.join(output_folder, r'check_in_log.csv')  # Log check-in/out data
photos_folder = os.path.join(output_folder, 'photos')  # Folder to save captured photos

# Ensure the photos folder exists
if not os.path.exists(photos_folder):
    os.makedirs(photos_folder)

# Load the pre-trained Haarcascade XML classifier for face detection
face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')

# Load the trained model
def load_model():
    recognizer = cv2.face.LBPHFaceRecognizer_create()
    try:
        recognizer.read(model_file)
        st.success(f"Model loaded from '{model_file}'.")
        return recognizer
    except cv2.error as e:
        st.error(f"Error loading model: {e}")
        return None

# Load the label map
def load_label_map():
    id_to_label = {}
    try:
        with open(label_map_file, 'r') as f:
            reader = csv.reader(f)
            for row in reader:
                label, person_name = row
                id_to_label[int(label)] = person_name
        st.success(f"Label map loaded from '{label_map_file}'.")
    except Exception as e:
        st.error(f"Error loading label map: {e}")
    
    return id_to_label

# Log the check-in/out details into the CSV file
def log_check_in_out(person_name):
    timestamp = datetime.now().strftime('%Y-%m-%d %H:%M:%S')
    check_in_out_status = "Check-in" if is_check_in(person_name) else "Check-out"

    # If the log file doesn't exist, create it and write the header
    file_exists = os.path.isfile(check_in_log_file)
    
    with open(check_in_log_file, 'a', newline='') as f:
        writer = csv.writer(f)
        if not file_exists:
            writer.writerow(['Person', 'Timestamp', 'Status'])
        writer.writerow([person_name, timestamp, check_in_out_status])
    
    st.success(f"{check_in_out_status} logged for {person_name} at {timestamp}")

# Check if the person is performing a check-in or check-out
def is_check_in(person_name):
    if not os.path.exists(check_in_log_file):
        return True  # If the log doesn't exist, assume check-in
    
    # Read the log data
    df = pd.read_csv(check_in_log_file)
    
    # Get the last record for the person
    last_record = df[df['Person'] == person_name].tail(1)
    
    if last_record.empty or last_record['Status'].values[0] == "Check-out":
        return True  # If no previous record or last record is check-out, it's a check-in
    return False  # Otherwise, it's a check-out

# Save the recognized person's photo and announce their name
def capture_photo_and_announce(person_name, frame, x, y, w, h):
    # Save the photo
    timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
    photo_filename = os.path.join(photos_folder, f"{person_name}_{timestamp}.jpg")
    cv2.imwrite(photo_filename, frame[y:y+h, x:x+w])

    # Announce the name and display photo in Streamlit
    st.write(f"Recognized: {person_name}")
    st.image(photo_filename, caption=f"{person_name}'s photo", use_column_width=True)
    st.success(f"Photo captured and saved as {photo_filename}")

# Face recognition logic
def recognize_face():
    recognizer = load_model()
    if recognizer is None:
        return
    
    id_to_label = load_label_map()
    cap = cv2.VideoCapture(0)
    
    while True:
        ret, frame = cap.read()
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        faces = face_cascade.detectMultiScale(gray, scaleFactor=1.1, minNeighbors=5, minSize=(30, 30))

        for (x, y, w, h) in faces:
            cv2.rectangle(frame, (x, y), (x + w, y + h), (255, 0, 0), 2)
            face = gray[y:y + h, x:x + w]
            label, confidence = recognizer.predict(face)
            person_name = id_to_label.get(label, "Unknown")

            cv2.putText(frame, f'{person_name}', (x, y - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 255, 255), 2)

            # Log check-in or check-out and capture the photo only if the person is recognized
            if person_name != "Unknown":
                log_check_in_out(person_name)
                capture_photo_and_announce(person_name, frame, x, y, w, h)

        # Show the frame
        cv2.imshow('Face Recognition', frame)

        # Break the loop with 'q'
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    cap.release()
    cv2.destroyAllWindows()

# Generate attendance and analytics report
def generate_attendance_and_analytics_report():
    if os.path.exists(check_in_log_file):
        df = pd.read_csv(check_in_log_file)
        df['Timestamp'] = pd.to_datetime(df['Timestamp'])  # Ensure the Timestamp column is datetime

        # Group the data by person and calculate total work hours per person
        grouped = df.groupby('Person')
        report = pd.DataFrame()

        for person, group in grouped:
            check_in_times = group[group['Status'] == 'Check-in']['Timestamp'].reset_index(drop=True)
            check_out_times = group[group['Status'] == 'Check-out']['Timestamp'].reset_index(drop=True)

            # Calculate work hours by pairing check-ins and check-outs
            work_hours = (check_out_times - check_in_times).dt.total_seconds() / 3600
            summary = pd.DataFrame({
                'Person': person,
                'Date': check_in_times.dt.date,
                'Work Hours': work_hours
            })
            report = pd.concat([report, summary], ignore_index=True)

        # Analytics part

        # Total attendance (total number of check-ins)
        total_check_ins = len(df[df['Status'] == 'Check-in'])

        # Average work hours per person
        average_work_hours = report.groupby('Person')['Work Hours'].mean()

        # Most active users (users with the most check-ins)
        most_active_users = df[df['Status'] == 'Check-in']['Person'].value_counts().head(3)

        # Daily attendance (people who checked in per day)
        daily_attendance = df[df['Status'] == 'Check-in'].groupby(df['Timestamp'].dt.date)['Person'].nunique()

        # Display report and analytics in Streamlit
        st.subheader("Attendance Report")
        st.write(report)

        st.subheader("Analytics Summary")
        st.write(f"Total Check-ins: {total_check_ins}")
        st.write("Average Work Hours per Person:")
        st.table(average_work_hours)

        st.write("Most Active Users:")
        st.table(most_active_users)

        st.write("Daily Attendance:")
        st.line_chart(daily_attendance)  # Display a line chart for daily attendance

        # Allow users to download the attendance report
        csv_data = report.to_csv(index=False)
        st.download_button(label="Download Attendance Report", data=csv_data, file_name="attendance_report.csv", mime="text/csv")

    else:
        st.warning("No check-in logs available.")

# Streamlit app UI
st.header("Face Recognition Attendance System")

# Button to start recognition
if st.button("Start Face Recognition"):
    recognize_face()

# Button to generate attendance report with analytics
if st.button("Generate Attendance and Analytics Report"):
    generate_attendance_and_analytics_report()

# Display check-in logs in a table
if os.path.exists(check_in_log_file):
    # Read the CSV file into a DataFrame
    df = pd.read_csv(check_in_log_file)
    
    # Add a serial number column
    df.insert(0, 'Serial Number', range(1, len(df) + 1))
    
    # Display the DataFrame as a table in Streamlit
    st.subheader("Check-in Log")
    st.table(df)
else:
    st.warning("No check-in logs available.")


import cv2
import os
from flask import Flask, request, render_template
from datetime import date, datetime
import numpy as np
from sklearn.neighbors import KNeighborsClassifier
import pandas as pd
import joblib

# Flask App
app = Flask(__name__)

nimgs = 10

# Date
datetoday = date.today().strftime("%m_%d_%y")
datetoday2 = date.today().strftime("%d-%B-%Y")

# Face Detector
face_detector = cv2.CascadeClassifier('haarcascade_frontalface_default.xml')

# Create folders if not exist
os.makedirs('Attendance', exist_ok=True)
os.makedirs('static/faces', exist_ok=True)

# Create today's attendance file
attendance_file = f'Attendance/Attendance-{datetoday}.csv'
if not os.path.isfile(attendance_file):
    with open(attendance_file, 'w') as f:
        f.write('Name,Roll,Time')

# Total users
def totalreg():
    return len(os.listdir('static/faces'))

# Face extraction
def extract_faces(img):
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    return face_detector.detectMultiScale(gray, 1.2, 5, minSize=(20, 20))

# Identify face
def identify_face(facearray):
    model = joblib.load('static/face_recognition_model.pkl')
    return model.predict(facearray)

# Train model
def train_model():
    faces = []
    labels = []
    userlist = os.listdir('static/faces')

    for user in userlist:
        for imgname in os.listdir(f'static/faces/{user}'):
            img = cv2.imread(f'static/faces/{user}/{imgname}')
            resized_face = cv2.resize(img, (50, 50))
            faces.append(resized_face.ravel())
            labels.append(user)

    faces = np.array(faces)
    knn = KNeighborsClassifier(n_neighbors=5)
    knn.fit(faces, labels)
    joblib.dump(knn, 'static/face_recognition_model.pkl')

# Extract attendance
def extract_attendance():
    df = pd.read_csv(attendance_file)
    return df['Name'], df['Roll'], df['Time'], len(df)

# ✅ UPDATED: Add attendance (NO DUPLICATES)
def add_attendance(name):
    username = name.split('_')[0]
    userid = int(name.split('_')[1])
    current_time = datetime.now().strftime("%H:%M:%S")

    df = pd.read_csv(attendance_file)

    if not ((df['Name'] == username) & (df['Roll'] == userid)).any():
        with open(attendance_file, 'a') as f:
            f.write(f'\n{username},{userid},{current_time}')
        return True
    else:
        return False

# Get all users
def getallusers():
    userlist = os.listdir('static/faces')
    names, rolls = [], []

    for user in userlist:
        name, roll = user.split('_')
        names.append(name)
        rolls.append(roll)

    return userlist, names, rolls, len(userlist)

# Delete user
def deletefolder(duser):
    for file in os.listdir(duser):
        os.remove(duser + '/' + file)
    os.rmdir(duser)

# Routes

@app.route('/')
def home():
    names, rolls, times, l = extract_attendance()
    return render_template('home.html', names=names, rolls=rolls,
                           times=times, l=l, totalreg=totalreg(),
                           datetoday2=datetoday2)

@app.route('/listusers')
def listusers():
    userlist, names, rolls, l = getallusers()
    return render_template('listusers.html', userlist=userlist,
                           names=names, rolls=rolls, l=l,
                           totalreg=totalreg(), datetoday2=datetoday2)

@app.route('/deleteuser')
def deleteuser():
    duser = request.args.get('user')
    deletefolder('static/faces/' + duser)

    if os.listdir('static/faces/') == []:
        os.remove('static/face_recognition_model.pkl')

    try:
        train_model()
    except:
        pass

    return listusers()

# 🚀 UPDATED START FUNCTION
@app.route('/start')
def start():

    if 'face_recognition_model.pkl' not in os.listdir('static'):
        return "Please add a user first!"

    cap = cv2.VideoCapture(0)

    while True:
        ret, frame = cap.read()

        faces = extract_faces(frame)

        for (x, y, w, h) in faces:
            face = cv2.resize(frame[y:y+h, x:x+w], (50, 50))
            identified_person = identify_face(face.reshape(1, -1))[0]

            # ✅ Check attendance
            is_new = add_attendance(identified_person)

            if is_new:
                display_text = identified_person
                color = (0, 255, 0)  # Green
            else:
                display_text = identified_person + " (Already Marked)"
                color = (0, 0, 255)  # Red

            # Draw
            cv2.rectangle(frame, (x, y), (x+w, y+h), color, 2)
            cv2.putText(frame, display_text, (x, y-10),
                        cv2.FONT_HERSHEY_SIMPLEX, 1, color, 2)

        cv2.imshow('Attendance', frame)

        if cv2.waitKey(1) == 27:
            break

    cap.release()
    cv2.destroyAllWindows()

    return home()

# Add new user
@app.route('/add', methods=['POST'])
def add():
    username = request.form['newusername']
    userid = request.form['newuserid']

    path = f'static/faces/{username}_{userid}'
    os.makedirs(path, exist_ok=True)

    cap = cv2.VideoCapture(0)
    i, j = 0, 0

    while True:
        _, frame = cap.read()
        faces = extract_faces(frame)

        for (x, y, w, h) in faces:
            if j % 5 == 0:
                cv2.imwrite(f'{path}/{username}_{i}.jpg', frame[y:y+h, x:x+w])
                i += 1
            j += 1

        cv2.imshow('Adding User', frame)

        if j >= nimgs * 5 or cv2.waitKey(1) == 27:
            break

    cap.release()
    cv2.destroyAllWindows()

    train_model()
    return home()

# Run app
if __name__ == '__main__':
    app.run(debug=True)


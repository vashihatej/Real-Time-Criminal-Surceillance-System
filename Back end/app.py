import multiprocessing
import pymongo
from pymongo import MongoClient
from flask import Flask, request, jsonify
import cv2
import numpy as np
from PIL import Image
import os
import json
import smtplib, ssl
from flask_cors import CORS
import psutil
import subprocess
import time
from binascii import a2b_base64
import atexit
app = Flask(__name__)
CORS(app)
sender_email = "rajlohith2@gmail.com"
receiver_email = "bharadwajkarthik7@gmail.com"
port = 465  # For SSL
password = "virendersehwag"
cameraIPList = [0]
jobList = []
context = ssl.create_default_context()
currentNewUser = 0
crim_list = {}


def trainData():
    path = 'dataset'
    recognizer = cv2.face.LBPHFaceRecognizer_create()
    detector = cv2.CascadeClassifier("haarcascade_frontalface_default.xml")
    print ("\n [INFO] Training faces. It will take a few seconds. Wait ...")
    faces,ids = getImagesAndLabels(path)
    recognizer.train(faces, np.array(ids))

    # Save the model into trainer/trainer.yml
    recognizer.write('trainer/trainer.yml') # recognizer.save() worked on Mac, but not on Pi

    # Print the numer of faces trained and end program
    print("\n [INFO] {0} faces trained. Exiting Program".format(len(np.unique(ids))))


def trainDataAndRecognize(c_list,pid):
    path = 'dataset'
    recognizer = cv2.face.LBPHFaceRecognizer_create()
    detector = cv2.CascadeClassifier("haarcascade_frontalface_default.xml")
    print ("\n [INFO] Training faces. It will take a few seconds. Wait ...")
    faces,ids = getImagesAndLabels(path)
    recognizer.train(faces, np.array(ids))

    # Save the model into trainer/trainer.yml
    recognizer.write('trainer/trainer.yml') # recognizer.save() worked on Mac, but not on Pi

    # Print the numer of faces trained and end program
    print("\n [INFO] {0} faces trained. Exiting Program".format(len(np.unique(ids))))
    p = multiprocessing.Process(target = performPrediction,args = (" ",0, c_list, pid))
    p.start()


def recordCriminalFaceData(filename, cameraIP, face_id):
    if filename == " ":
        cam = cv2.VideoCapture(0)
    else:
        cam = cv2.VideoCapture(filename)

    cam.set(3, 640) # set video width
    cam.set(4, 480) # set video height

    face_detector = cv2.CascadeClassifier('haarcascade_frontalface_default.xml')
    print("\n [INFO] Initializing face capture. Look the camera and wait ...")
    # Initialize individual sampling face count
    count = 0

    while(True):
        time.sleep(0.1)
        ret, img = cam.read()
        #img = cv2.flip(img, -1) # flip video image vertically
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        faces = face_detector.detectMultiScale(gray, 1.3, 5)
        for (x,y,w,h) in faces:
            cv2.rectangle(img, (x,y), (x+w,y+h), (255,0,0), 2)     
            count += 1

            # Save the captured image into the datasets folder
            cv2.imwrite("dataset/User." + str(face_id) + '.' + str(count) + ".jpg", gray[y:y+h,x:x+w])
        if filename == " ":
            if count >= 60:
                break

    # Do a bit of cleanup
    print("\n [INFO] Exiting Program and cleanup stuff")
    cam.release()

def performPredictionFromVideo(filepath):
    recognizer = cv2.face.LBPHFaceRecognizer_create()
    recognizer.read('trainer/trainer.yml')
    cascadePath = "haarcascade_frontalface_default.xml"
    faceCascade = cv2.CascadeClassifier(cascadePath)
    #iniciate id counter
    id = 0

    # names related to ids: example ==> Marcelo: id=1,  etc
    names = ['None', 'veer', 'Lohith', 'Ganapati', 'Z', 'W']
    cam = cv2.VideoCapture(filepath)
    # Initialize and start realtime video capture
    cam.set(3, 640) # set video widht
    cam.set(4, 480) # set video height

    # Define min window size to be recognized as a face
    minW = 0.1*cam.get(3)
    minH = 0.1*cam.get(4)

    while True:

        #img = cv2.flip(img, -1) # Flip vertically
        try:
            ret, img =cam.read()
            gray = cv2.cvtColor(img,cv2.COLOR_BGR2GRAY)
        except:
                return(crim_list)

        faces = faceCascade.detectMultiScale( 
            gray,
            scaleFactor = 1.2,
            minNeighbors = 5,
            minSize = (int(minW), int(minH)),
        )

        for(x,y,w,h) in faces:

            cv2.rectangle(img, (x,y), (x+w,y+h), (0,255,0), 2)

            id, confidence = recognizer.predict(gray[y:y+h,x:x+w])

            # Check if confidence is less them 100 ==> "0" is perfect match 
            if (confidence < 100):
                confidence = int(round(100 - confidence))
                if int(confidence) >= 50:
                    print(id)
                    print(confidence)
                    if id in c_list:
                        if crim_list[id]<confidence:
                            crim_list[id]=confidence
                            print(crim_list[id])
                    else:
                        crim_list[id]=confidence
            else:
                id = "unknown"
                confidence = "  {0}%".format(round(100 - confidence))
            
    print("\n--------List of Criminals Identified--------\n")
    for name,val in c_list.items():
        print(name,":-Identified with",val,"accuracy")
    print("\n [INFO] Exiting Program and cleanup stuff")
    cam.release()
    return(crim_list)

def performPrediction(filename, cameraIP, c_list, pid):
    process_id = os.getpid()
    pid[0] = process_id
    print(pid[0])
    recognizer = cv2.face.LBPHFaceRecognizer_create()
    recognizer.read('trainer/trainer.yml')
    cascadePath = "haarcascade_frontalface_default.xml"
    faceCascade = cv2.CascadeClassifier(cascadePath)
    #iniciate id counter
    id = 0

    # names related to ids: example ==> Marcelo: id=1,  etc
    names = ['None', 'Lohith', 'Lohith', 'Ganapati', 'Z', 'W']
    if filename == " ":
        cam = cv2.VideoCapture(0)
    else:
        cam = cv2.VideoCapture(filename)
    # Initialize and start realtime video capture
    cam.set(3, 640) # set video widht
    cam.set(4, 480) # set video height

    # Define min window size to be recognized as a face
    minW = 0.1*cam.get(3)
    minH = 0.1*cam.get(4)

    while True:
        time.sleep(0.1)
        ret, img =cam.read()
        #img = cv2.flip(img, -1) # Flip vertically

        gray = cv2.cvtColor(img,cv2.COLOR_BGR2GRAY)

        faces = faceCascade.detectMultiScale( 
            gray,
            scaleFactor = 1.2,
            minNeighbors = 5,
            minSize = (int(minW), int(minH)),
        )

        for(x,y,w,h) in faces:

            cv2.rectangle(img, (x,y), (x+w,y+h), (0,255,0), 2)
            id, confidence = recognizer.predict(gray[y:y+h,x:x+w])

            # Check if confidence is less them 100 ==> "0" is perfect match 
            if (confidence < 100 & id != 4):
                confidence = int(round(100 - confidence))
                if int(confidence) >= 25:
                    if id in c_list:
                        if c_list[id]<confidence:
                            c_list[id]=confidence
                            print(c_list[id])
                    else:
                        print("ID", id)
                        print(confidence)
                        c_list[id]=confidence
                        client = MongoClient('mongodb+srv://rajlohith2:bit123@criminal-database-drxaw.mongodb.net/test?retryWrites=true&w=majority')
                        db = client.CriminalDB
                        collection = db["Crime_list"]
                        collection1 = db["trial_new"]

                        res = collection1.find_one({"ID" : int(id)},{ "_id": 0 })
                        ts = time.time()
                        collection.insert_one({"ID": id, "Name": res["First_Name"]+" "+res["Last_Name"],"Recorded_Time": ts,"Location":"K R Puram", "Image": res['Image']})
            else:
                id = "unknown"
                confidence = "  {0}%".format(round(100 - confidence))
            
    print("\n--------List of Criminals Identified--------\n")
    for name,val in c_list.items():
        print(name,":-Identified with",val,"accuracy")
    print("\n [INFO] Exiting Program and cleanup stuff")
    cam.release()
# function to get the images and label data

def getImagesAndLabels(path):

    imagePaths = [os.path.join(path,f) for f in os.listdir(path)]     
    faceSamples=[]
    ids = []

    for imagePath in imagePaths:

        PIL_img = Image.open(imagePath).convert('L') # convert it to grayscale
        img_numpy = np.array(PIL_img,'uint8')
        detector = cv2.CascadeClassifier("haarcascade_frontalface_default.xml")
        id = int(os.path.split(imagePath)[-1].split(".")[1])
        faces = detector.detectMultiScale(img_numpy)

        for (x,y,w,h) in faces:
            faceSamples.append(img_numpy[y:y+h,x:x+w])
            ids.append(id)

    return faceSamples,ids

def exit_handler():
    client = MongoClient('mongodb+srv://rajlohith2:bit123@criminal-database-drxaw.mongodb.net/test?retryWrites=true&w=majority')
    db = client.CriminalDB
    collection = db["Crime_list"]
    res = collection.find({'Notified': True})
    # collection.delete_many({})
    collection2 = db["timeline"]
    collection2.insert_many(res, {'Notified': False })



#atexit.register(exit_handler)

@app.route('/block-camera')
def block():
    os.popen('TASKKILL /PID '+str(pid[0])+' /F 2>NUL')
    return jsonify({})

@app.route('/release-camera')
def release():
    p = multiprocessing.Process(target = performPrediction,args = (" ",0, c_list, pid))
    p.start()
    return " "
@app.route('/recording')
def recordRealTime():
    os.popen('TASKKILL /PID '+str(pid[0])+' /F 2>NUL')
    filter = {} 
    collection = db["trial_new"]
    doc_count = collection.count_documents(filter)
    currentNewUser = doc_count + 1
    recordCriminalFaceData(" ", 0, currentNewUser)
    p2 = multiprocessing.Process(target = trainDataAndRecognize, args = (c_list, pid))
    p2.start()
    return jsonify({})

@app.route('/recording-from-footage')
def recordingFromFootage():
    face_id = request.args['userid']
    recordCriminalFaceData("C:\\Users\\KS1\\Desktop\\sample.mp4",0, face_id)


@app.route('/recognize-from-footage')
def recognizeFromFootage():
    image_path = request.args['image-path']
    print(image_path)
    searchResult = performPredictionFromVideo("C:\\Users\\KS1\\Desktop\\"+image_path)
    detectedList = []
    collection = db['trial_new']
    for x in searchResult.keys():
        print(x)
        res = collection.find_one({"ID" : int(x)},{ "_id": 0 })
        detectedList.append(res)
    print(detectedList)    
    return jsonify(detectedList)


@app.route('/full-list')
def fullList():
    response = []
    collection = db['trial_new']
    res = collection.find({},{ "_id": 0 ,'Image': 0})
    for i in res:
        response.append(i)
    return jsonify(response)
    

@app.route('/recognition')
def recognizeRealTime():
    criminalsDetected = []
    collection = db["Crime_list"]
    collection1 = db["timeline"]
    res = collection.find({},{"_id":0,"ID": 1, "Name": 1, "Recorded_Time": 1})
    for i in res:
        message = "The following criminal has been discovered in your area at camera 0:"
        message += i['Name']
        criminalsDetected.append(i)
        res1 = collection.find_one_and_delete({"ID" : i["ID"]})
        if collection1.find_one({"ID" : i["ID"]}):
            collection1.find_one_and_update({"ID" : i["ID"]}, {'$set': {"Recorded_Time": i["Recorded_Time"]}})
        else:
            collection1.insert(res1)


        # client = Client("AC005dc1deb4ae1c3efd6265aeacf69997", "c4a9eb554443c3a39b8d8ba673f5a3a4")
        # client.messages.create(to="+918553587952", 
        #                 from_="+12058439615", 
        #                 body=message)
        # with smtplib.SMTP_SSL("smtp.gmail.com", port, context=context) as server:
        #     server.login("rajlohith2@gmail.com", password)
        #     # TODO: Send email here
        #     server.sendmail(sender_email, receiver_email, "The following criminal has been discovered in your area at camera 0:"+ i['Name'])
    print(criminalsDetected)
    return jsonify(criminalsDetected)

@app.route('/search-by-id')
def searchById():
    detectedList = []
    id = request.args['id']
    collection = db["trial_new"]
    res = collection.find_one({"ID" : int(id)},{ "_id": 0 })
    detectedList.append(res)
    return jsonify(detectedList)


@app.route('/search-by-name')
def searchByName():
    detectedList = []
    fn = request.args['fn']
    ln = request.args['ln']
    collection = db["trial_new"]
    res = collection.find_one({"First_Name" : fn, "Last_Name" : ln},{ "_id": 0 })
    detectedList.append(res)
    return jsonify(detectedList)

@app.route('/admin-list')
def adminList():
    admins = []
    collection = db["admin_table"]
    cursor = collection.find({},{ "_id": 0 })
    print(cursor)
    for document in cursor:
          print(document)
          admins.append(document)
    return jsonify(admins)


@app.route('/add-criminal',methods = ['POST'])
def addCriminal():
    data = request.get_data()
    filter = {} 
    collection = db["trial_new"]
    doc_count = collection.count_documents(filter)
    currentNewUser = doc_count + 1
    print(type(data))
    data = json.loads(data)
    img = data['Image']

    # with open("C:\\Users\\rajlo\\front_end\\Images\\"+ str(currentNewUser) +".png", "wb") as fh:
    #     fh.write(img.decode('base64'))
    data.update({"ID" : currentNewUser})
    x = collection.insert_one(data)
    print(x.inserted_id)
    return jsonify({})

@app.route('/login')
def login():
    user = request.args['user']
    password = request.args['pass']
    print(user)
    print(password)
    collection = db["admin_table"]
    res = collection.find_one({"User_id" : user, "password" : password},{ "_id": 0 })
    print(res)
    if res== None:
        return jsonify({"message":"failure"})
    return jsonify({"message":"success"})


@app.route('/add-user')
def addUser():
    user = request.args['user']
    password = request.args['pass']
    print(user)
    print(password)
    collection = db["admin_table"]
    doc_count = collection.count_documents({})
    currentNewUser = doc_count + 1
    res = collection.insert_one({"ID": currentNewUser,"User_id" : user, "password" : password})
    print(res)
    return jsonify({"message":"success"})
    
@app.route('/remove-user')
def removeUser():
    user = request.args['user']
    password = request.args['pass']
    print(user)
    print(password)
    collection = db["admin_table"]
    res = collection.delete_one({"User_id" : user, "password" : password})
    print(res)
    return jsonify({"message":"success"})

@app.route('/get-timeline')
def getTimeline():
    timelineList = []
    collection = db["timeline"]
    res = collection.find({},{'_id':0})
    for i in res:
        timelineList.append(i)
    return jsonify(timelineList)


@app.route('/get-home')
def getHome():
    timelineList = []
    collection = db["Crime_list"]
    res = collection.count_documents({})
    return jsonify({'count': res})



if __name__ == '__main__':
    manager = multiprocessing.Manager()
    c_list=manager.dict()
    pid = manager.list([0])
    path = 'dataset'

    recognizer = cv2.face.LBPHFaceRecognizer_create()
    detector = cv2.CascadeClassifier("haarcascade_frontalface_default.xml")
    print ("\n [INFO] Training faces. It will take a few seconds. Wait ...")
    faces,ids = getImagesAndLabels(path)
    recognizer.train(faces, np.array(ids))

    # Save the model into trainer/trainer.yml
    recognizer.write('trainer/trainer.yml') # recognizer.save() worked on Mac, but not on Pi

    # Print the numer of faces trained and end program
    print("\n [INFO] {0} faces trained. Exiting Program".format(len(np.unique(ids))))
    p = multiprocessing.Process(target = performPrediction,args = (" ",0, c_list, pid))
    p.start()
    client = MongoClient('mongodb+srv://rajlohith2:bit123@criminal-database-drxaw.mongodb.net/test?retryWrites=true&w=majority')
    db = client.CriminalDB
    app.run()
from django.http import HttpResponseRedirect
from django.shortcuts import render, redirect, HttpResponse
from django.contrib import messages
import cv2, os, shutil, csv, datetime, time
import numpy as np
from PIL import Image, ImageTk
import pandas as pd
from django.urls import reverse


def home(request):
    return render(request, 'home.html')


def admin_login(request):
    if request.method == "POST":
        username = request.POST['username']
        password = request.POST['password']
        if username == 'Admin@':
            if password == 'Admin@123':
                return redirect('./')
            else:
                messages.warning(request, 'wrong password ...!')
                return redirect('login')
        else:
            messages.warning(request, 'wrong username ...!')
            return redirect('login')
    else:
        return render(request, 'login.html')


def getImagesAndLabels(path):
    # get the path of all the files in the folder
    imagePaths = [os.path.join(path, f) for f in os.listdir(path)]
    # print(imagePaths)

    # create empth face list
    faces = []
    # create empty ID list
    Ids = []
    # now looping through all the image paths and loading the Ids and the images
    for imagePath in imagePaths:
        # loading the image and converting it to gray scale
        pilImage = Image.open(imagePath).convert('L')
        # Now we are converting the PIL image into numpy array
        imageNp = np.array(pilImage, 'uint8')
        # getting the Id from the image
        Id = int(os.path.split(imagePath)[-1].split(".")[1])
        # extract the face from the training image sample
        faces.append(imageNp)
        Ids.append(Id)
    return faces, Ids


def watch_live(request):

    # trace image
    recognizer = cv2.face.LBPHFaceRecognizer_create()
    recognizer.read("TrainingImageLabel\Trainner.yml")
    harcascadePath = "haarcascade_frontalface_default.xml"
    faceCascade = cv2.CascadeClassifier(harcascadePath);
    df = pd.read_csv("ResidenceDetails\ResidenceDetails.csv")
    df2 = pd.read_csv("VisitorDetails\VisitorDetails.csv")
    cam = cv2.VideoCapture(0)
    font = cv2.FONT_HERSHEY_SIMPLEX
    while True:
        ret, im = cam.read()
        gray = cv2.cvtColor(im, cv2.COLOR_BGR2GRAY)
        faces = faceCascade.detectMultiScale(gray, 1.2, 5)
        recognizer.read("TrainingImageLabel\Trainner.yml")
        for (x, y, w, h) in faces:
            Id, conf = recognizer.predict(gray[y:y + h, x:x + w])
            recognizer.read("TrainingImageLabel\Trainner2.yml")
            Id2, conf2 = recognizer.predict(gray[y:y + h, x:x + w])
            if (conf < 50):
                cv2.rectangle(im, (x, y), (x + w, y + h), (255, 0, 0), 2)
                aa = df.loc[df['Flat_No'] == Id]['Name'].values
                tt=aa
            elif (conf2 < 50):
                cv2.rectangle(im, (x, y), (x + w, y + h), (0, 0, 0), 2)

                tt = "bro"
            else:
                cv2.rectangle(im, (x, y), (x + w, y + h), (0, 0, 255), 2)
                tt = 'Unknown'
            cv2.putText(im, str(tt), (x, y + h), font, 1, (255, 255, 255), 2)
        cv2.imshow('Live Camera', im)
        if (cv2.waitKey(1) == ord('q')):
            break
    cam.release()
    cv2.destroyAllWindows()

    return redirect("./")


def add_visitor(request):
    if request.method == "POST":
        name = request.POST['name']
        contact = request.POST['contact']
        gender = request.POST['gender']
        flat_no = request.POST['flat_no']
        reason = request.POST['reason']

        # image processing
        cam = cv2.VideoCapture(0)
        harcascadePath = "haarcascade_frontalface_default.xml"
        detector = cv2.CascadeClassifier(harcascadePath)
        sampleNum = 0
        while (True):
            ret, img = cam.read()
            gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
            faces = detector.detectMultiScale(gray, 1.3, 5)
            for (x, y, w, h) in faces:
                cv2.rectangle(img, (x, y), (x + w, y + h), (255, 0, 0), 2)
                # incrementing sample number
                sampleNum = sampleNum + 1
                # saving the captured face in the dataset folder TrainingImage
                cv2.imwrite("VisitorImage\ " + name + "." + contact + '.' + str(sampleNum) + ".jpg",
                            gray[y:y + h, x:x + w])
                # display the frame
                cv2.imshow('frame', img)
            # wait for 100 miliseconds
            if cv2.waitKey(100) & 0xFF == ord('q'):
                break
            # break if the sample number is morethan 100
            elif sampleNum > 30:
                break
        cam.release()
        cv2.destroyAllWindows()
        row = [name, contact, gender, flat_no, reason]
        with open('VisitorDetails\VisitorDetails.csv', 'a+') as csvFile:
            writer = csv.writer(csvFile)
            writer.writerow(row)
        csvFile.close()

        # train the system
        recognizer = cv2.face_LBPHFaceRecognizer.create()
        harcascadePath = "haarcascade_frontalface_default.xml"
        detector = cv2.CascadeClassifier(harcascadePath)
        faces, Id = getImagesAndLabels("VisitorImage")
        recognizer.train(faces, np.array(Id))
        recognizer.write("TrainingImageLabel\Trainner2.yml")
        messages.success(request, 'Visitor added successfully ...')
        return HttpResponseRedirect('./add_visitor')
    else:
        return render(request, 'add_visitor.html')


def add_residence(request):
    if request.method == "POST":
        name = request.POST['name']
        flat_no = request.POST['flat_no']
        contact = request.POST['contact']
        gender = request.POST['gender']

        # image processing
        cam = cv2.VideoCapture(0)
        harcascadePath = "haarcascade_frontalface_default.xml"
        detector = cv2.CascadeClassifier(harcascadePath)
        sampleNum = 0
        while (True):
            ret, img = cam.read()
            gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
            faces = detector.detectMultiScale(gray, 1.3, 5)
            for (x, y, w, h) in faces:
                cv2.rectangle(img, (x, y), (x + w, y + h), (255, 0, 0), 2)
                # incrementing sample number
                sampleNum = sampleNum + 1
                # saving the captured face in the dataset folder TrainingImage
                cv2.imwrite("ResidenceImage\ " + name + "." + flat_no + '.' + str(sampleNum) + ".jpg",
                            gray[y:y + h, x:x + w])
                # display the frame
                cv2.imshow('Take Photo', img)
            # wait for 100 miliseconds
            if cv2.waitKey(100) & 0xFF == ord('q'):
                break
            # break if the sample number is morethan 100
            elif sampleNum > 30:
                break
        cam.release()
        cv2.destroyAllWindows()
        row = [3, name, flat_no, contact, gender]
        with open('ResidenceDetails\ResidenceDetails.csv', 'a+') as csvFile:
            writer = csv.writer(csvFile)
            writer.writerow(row)
        csvFile.close()

        #train the system
        recognizer = cv2.face_LBPHFaceRecognizer.create()
        harcascadePath = "haarcascade_frontalface_default.xml"
        detector = cv2.CascadeClassifier(harcascadePath)
        faces, Id = getImagesAndLabels("ResidenceImage")
        recognizer.train(faces, np.array(Id))
        recognizer.write("TrainingImageLabel\Trainner.yml")
        messages.success(request, 'Residence added successfully ...')
        return HttpResponseRedirect('./add_residence')
    else:
        return render(request, 'add_residence.html')

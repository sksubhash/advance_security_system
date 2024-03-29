from django.http import HttpResponseRedirect
from django.shortcuts import render, redirect
from django.contrib import messages
import cv2, os, datetime, time
import numpy as np
from PIL import Image
import winsound
from assapp.models import tbldata, tblvdetails
from django.contrib.auth.decorators import login_required
from django.contrib.auth import authenticate, login, logout
from django.contrib.auth.models import User


@login_required(login_url='./login')
def home(request):
    return render(request, 'home.html')


def admin_login(request):
    if request.method == "POST":
        username = request.POST['username']
        password = request.POST['password']
        try:
            if User.objects.get(username=username):
                user = User.objects.get(username=username)
            else:
                messages.warning(request, 'Username is not matched !')
                return redirect('./login')
            pwd_valid = user.check_password(password)
            if pwd_valid:
                login(request, user)
                return redirect('./')
            else:
                messages.warning(request, 'Invalid Password !')
                return redirect('./login')
        except User.DoesNotExist:
            messages.warning(request, 'Username is not matched !')
            return redirect('./login')
    else:
        return render(request, 'login.html')


def getImagesAndLabels(path):
    # get the path of all the files in the folder
    imagePaths = [os.path.join(path, f) for f in os.listdir(path)]

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
        Id = int(os.path.split(imagePath)[-1].split(".")[0])
        # extract the face from the training image sample
        faces.append(imageNp)
        Ids.append(Id)
    return faces, Ids


@login_required(login_url='./login')
def watch_live(request):
    # trace image
    recognizer = cv2.face.LBPHFaceRecognizer_create()
    recognizer.read("TrainingImageLabel\Trainner.yml")
    harcascadePath = "haarcascade_frontalface_default.xml"
    faceCascade = cv2.CascadeClassifier(harcascadePath)
    cam = cv2.VideoCapture(0)
    font = cv2.FONT_HERSHEY_SIMPLEX
    while True:
        ret = cam.set(3, 1024)
        ret = cam.set(4, 600)
        ret, im = cam.read()
        gray = cv2.cvtColor(im, cv2.COLOR_BGR2GRAY)
        faces = faceCascade.detectMultiScale(gray, 1.2, 5)
        recognizer.read("TrainingImageLabel\Trainner.yml")
        lblfortime = 0
        for (x, y, w, h) in faces:
            Id, conf = recognizer.predict(gray[y:y + h, x:x + w])
            if (conf < 50):
                tasks_obj = tbldata.objects.get(id=Id)
                type1 = tasks_obj.type
                if type1 == 'Residence':
                    cv2.rectangle(im, (x, y), (x + w, y + h), (255, 0, 0), 2)
                else:
                    cv2.rectangle(im, (x, y), (x + w, y + h), (0, 255, 0), 2)
                tt = str(type1)
                lblfortime = 1
            else:
                cv2.rectangle(im, (x, y), (x + w, y + h), (0, 0, 255), 2)
                frequency = 2500  # Set Frequency To 2500 Hertz
                duration = 2000  # 1000 ms == 1  second
                winsound.Beep(frequency, duration)
                tt = 'Unknown'

            cv2.putText(im, str(tt), (x, y + h), font, 1, (255, 255, 255), 2)
        cv2.imshow('Live Camera', im)
        if int(lblfortime) == 1:
            ts = time.time()
            date = datetime.datetime.fromtimestamp(ts).strftime('%Y-%m-%d')
            time1 = datetime.datetime.fromtimestamp(ts).strftime('%H:%M:%S')

            tasks_obj = tblvdetails.objects.latest('date')
            date2 = tasks_obj.date
            tblvdetails_id = tasks_obj.id

            if date == date2:
                try:
                    if tblvdetails.objects.get(date=date, tbldata_id=Id):
                        pass
                    else:
                        a = tblvdetails(tbldata_id=Id, date=date, no_of_time=time1)
                        a.save()
                except:
                    a = tblvdetails(tbldata_id=Id, date=date, no_of_time=time1)
                    a.save()
            else:
                a = tblvdetails(tbldata_id=Id, date=date, no_of_time=time1)
                a.save()
        else:
            pass
        if (cv2.waitKey(1) == ord('q')):
            break

    cam.release()
    cv2.destroyAllWindows()

    return redirect("./")


@login_required(login_url='./login')
def vdetails(request):
    if request.method == "POST":
        id = request.POST['id']
        tasks_obj2 = tbldata.objects.raw('SELECT * FROM tbldata WHERE id = %s', [id])
        tasks_obj = tbldata.objects.raw('SELECT * FROM tblvdetails WHERE tbldata_id = %s', [id])
        return render(request, 'view_more_details.html', {"alltasks": tasks_obj, "alltasks2": tasks_obj2})

    else:
        # trace image
        recognizer = cv2.face.LBPHFaceRecognizer_create()
        recognizer.read("TrainingImageLabel\Trainner.yml")
        harcascadePath = "haarcascade_frontalface_default.xml"
        faceCascade = cv2.CascadeClassifier(harcascadePath)
        cam = cv2.VideoCapture(0)
        font = cv2.FONT_HERSHEY_SIMPLEX
        while True:
            ret, im = cam.read()
            gray = cv2.cvtColor(im, cv2.COLOR_BGR2GRAY)
            faces = faceCascade.detectMultiScale(gray, 1.2, 5)
            recognizer.read("TrainingImageLabel\Trainner.yml")
            for (x, y, w, h) in faces:
                Id, conf = recognizer.predict(gray[y:y + h, x:x + w])
                if (conf < 50):
                    frequency = 2500  # Set Frequency To 2500 Hertz
                    duration = 1000  # 1000 ms == 1 second
                    winsound.Beep(frequency, duration)
                    tasks_obj = tbldata.objects.raw('SELECT * FROM tbldata WHERE id = %s', [Id])
                    return render(request, 'view_details.html', {"alltasks": tasks_obj})
                else:
                    messages.warning(request, 'Your Are Not Identified ! Please Try Again ...')
                    return redirect("./")
            cv2.imshow('View Detail', im)
            if (cv2.waitKey(1) == ord('q')):
                break
        cam.release()
        cv2.destroyAllWindows()


@login_required(login_url='./login')
def add_visitor(request):
    if request.method == "POST":
        name = request.POST['name']
        contact = request.POST['contact']
        gender = request.POST['gender']
        flat_no = request.POST['flat_no']
        reason = request.POST['reason']
        try:
            if tbldata.objects.latest('id'):
                tasks_obj = tbldata.objects.latest('id')
                last_id = tasks_obj.id
                last_id = last_id + 1
        except:
            messages.warning(request, 'Please Try Again ...')
            return HttpResponseRedirect('./add_visitor')

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
                cv2.imwrite("Images\ " + str(last_id) + "." + str(sampleNum) + ".jpg",
                            gray[y:y + h, x:x + w])
                # display the frame
                cv2.imshow('frame', img)
            # wait for 100 miliseconds
            if cv2.waitKey(100) & 0xFF == ord('q'):
                break
            # break if the sample number is morethan 30
            elif sampleNum > 30:
                frequency = 2500  # Set Frequency To 2500 Hertz
                duration = 1000  # 1000 ms == 1 second
                winsound.Beep(frequency, duration)
                break
        cam.release()
        cv2.destroyAllWindows()
        a = tbldata(name=name, mobile_number=contact, gender=gender, flat_no=flat_no, type='Visitor', reason=reason)
        a.save()

        # train the system
        recognizer = cv2.face_LBPHFaceRecognizer.create()
        harcascadePath = "haarcascade_frontalface_default.xml"
        detector = cv2.CascadeClassifier(harcascadePath)
        faces, Id = getImagesAndLabels("Images")
        recognizer.train(faces, np.array(Id))
        recognizer.write("TrainingImageLabel\Trainner.yml")
        messages.success(request, 'Visitor added successfully ...')
        return HttpResponseRedirect('./add_visitor')
    else:
        try:
            if tbldata.objects.latest('id'):
                tasks_obj = tbldata.objects.latest('id')
        except:
            a = tbldata(name="test", mobile_number="test", gender="test", flat_no=101,
                        type='test', reason="test")
            a.save()
        return render(request, 'add_visitor.html')


@login_required(login_url='./login')
def add_residence(request):
    if request.method == "POST":
        name = request.POST['name']
        flat_no = request.POST['flat_no']
        contact = request.POST['contact']
        gender = request.POST['gender']
        try:
            if tbldata.objects.latest('id'):
                tasks_obj = tbldata.objects.latest('id')
                last_id = tasks_obj.id
                last_id = last_id + 1
        except:
            messages.warning(request, 'Please Try Again ...')
            return HttpResponseRedirect('./add_visitor')

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
                cv2.imwrite("Images\ " + str(last_id) + "." + str(sampleNum) + ".jpg",
                            gray[y:y + h, x:x + w])
                # display the frame
                cv2.imshow('Take Photo', img)
            # wait for 100 miliseconds
            if cv2.waitKey(100) & 0xFF == ord('q'):
                break
            # break if the sample number is morethan 100
            elif sampleNum > 30:
                frequency = 2500  # Set Frequency To 2500 Hertz
                duration = 1000  # 1000 ms == 1 second
                winsound.Beep(frequency, duration)
                break
        cam.release()
        cv2.destroyAllWindows()
        a = tbldata(name=name, mobile_number=contact, gender=gender, flat_no=flat_no, type='Residence',
                    reason='Null')
        a.save()

        # train the system
        recognizer = cv2.face_LBPHFaceRecognizer.create()
        harcascadePath = "haarcascade_frontalface_default.xml"
        detector = cv2.CascadeClassifier(harcascadePath)
        faces, Id = getImagesAndLabels("Images")
        recognizer.train(faces, np.array(Id))
        recognizer.write("TrainingImageLabel\Trainner.yml")
        messages.success(request, 'Residence added successfully ...')
        return HttpResponseRedirect('./add_residence')
    else:
        try:
            if tbldata.objects.latest('id'):
                tasks_obj = tbldata.objects.latest('id')
        except:
            a = tbldata(name="test", mobile_number="test", gender="test", flat_no=101,
                        type='test', reason="test")
            a.save()
        return render(request, 'add_residence.html')


def user_logout(request):
    logout(request)
    return render(request, 'logout.html')
import numpy as np
import cv2
import pickle
from datetime import datetime, timedelta
import pyrebase
from firebase_admin import db
import time
import pandas as pd
import matplotlib
import matplotlib.pyplot as plt
from pandas import DataFrame
import os
plt.style.use('ggplot')
import tkinter as tk
from tkinter import filedialog
import arcfelismerotanito as at
#  adatbazis eleres
config = {
    "apiKey": "AIzaSyDllZJXw0kWxz-YnhClSodt-copuaNaxVI",
    "authDomain": "arcfelismeres-3546a.firebaseapp.com",
    "databaseURL": "https://arcfelismeres-3546a.firebaseio.com",
    "projectId": "arcfelismeres-3546a",
    "storageBucket": "arcfelismeres-3546a.appspot.com",
    "messagingSenderId": "536626976107",

}
firebase = pyrebase.initialize_app(config)
db = firebase.database()

#arcdetektáló
face_cascade = cv2.CascadeClassifier(
    'cascades/haarcascade_frontalface_default.xml')

#felismero létrehozása és kiolvasás
recognizer = cv2.face.LBPHFaceRecognizer_create()
recognizer.read("trainner.yml")

# egy tomb a nevek id-nek
labels = {"person_name": 1}
with open("labels.pickle", 'rb') as f:
    og_labels = pickle.load(f)
    labels = {v: k for k, v in og_labels.items()}

# pdf generáló
def pdf(kurzus):
    from firebase import firebase
    firebase=firebase.FirebaseApplication("https://arcfelismeres-3546a.firebaseio.com/",None)
    results=firebase.get('/Datum',None)
    nev=[]
    datumok=[]
    orak=[]
    kurzusok=[]
    for i in results:
        kurzusok.append(i)
        for a in results[i]:
            for j in results[i][a]:
                if i==kurzus:
                    temp1 = results[i][a][j]['date']
                    temp1 = temp1.split(' ')
                    datum = temp1[0]
                    datumok.append(datum)
                    ora = temp1[1]
                    orak.append(ora)
                    azon = results[i][a][j]['id']
                    n=results[i][a][j]['name']
                    nev.append(n)

    matplotlib.style.use('ggplot')
    matplotlib.pyplot.rcParams['figure.figsize'] = (10, 10)
    matplotlib.pyplot.rcParams['font.family'] = 'sans-serif'

    df = pd.DataFrame(
        {  
            'Nev': nev,
            'Datum': datumok,
            'Ora': orak,
        }
    )
    fig, ax = plt.subplots()
    fig.patch.set_visible(False)
    ax.axis('off')
    ax.axis('tight')
    plt.title(kurzus)
    ax.table(cellText=df.values, colLabels=df.columns, loc='center')
    fig.tight_layout()
    plt.savefig(os.path.join(kurzus+"Jelenleti ív" + '.pdf'))
    print(kurzus+"hoz kigenerálta a jelenlétet!")
  
#ablak kurzus adat kitoltés
def ablak():
    def letrehozCallback():
        global tanar, kurzus,diakokszam
        tanar=tanarneve.get()
        kurzus=kurzusneve.get()
        diakokszam=diakszam.get()
        if ((not tanarneve.get() )|( not kurzusneve.get())| (not diakszam.get())):
            print("Sikertelen! Töltse ki a mezőket")
        else:
            print("Létrehozva!")
            
    
    def befejezCallback():
        if ((not tanarneve.get() )|( not kurzusneve.get())| (not diakszam.get())):
            print("Töltse ki a mezőket!")
        else:
            root.destroy()

    def donothing():
      pass

    def tanitCallback():
        print("Tanitás elkezdődött!")
        at.tanito()
        print("Tanitás befejezdőtt!")
    
    def keptesztbezarCallback():
        global diakokszam, kurzus
        diakokszam=0
        kurzus=''
        root.destroy()
    def fileDialog():
        root.filename = filedialog.askopenfilename(initialdir =  "/", title = "Select A File", filetype =
        (("jpeg files","*.jpg"),("all files","*.*")))
        print(root.filename)
        return root.filename
    # tesztelés arcfelismerő működése képeken
    def arcfelismereskepeken():
        def faceDetection(test_img):
            gray_img=cv2.cvtColor(test_img,cv2.COLOR_BGR2GRAY)
            face_haar_cascade=cv2.CascadeClassifier('E:/Allamvizsga/Allamvizsga/arcfelismero/cascades/haarcascade_frontalface_default.xml')
            faces=face_haar_cascade.detectMultiScale(gray_img,scaleFactor=1.3,minNeighbors=5)

            return faces,gray_img
        test_img=cv2.imread(fileDialog())
        recognizer = cv2.face.LBPHFaceRecognizer_create()
        recognizer.read("trainner.yml")
        labels = {"person_name": 1}
        with open("labels.pickle", 'rb') as f:
            og_labels = pickle.load(f)
            labels = {v: k for k, v in og_labels.items()}

        faces_detected,gray_img=faceDetection(test_img)
        print("Felismert arc: ", faces_detected)

        for (x,y,w,h) in faces_detected:
            roi_gray = gray_img[y:y+h, x:x+w]  
            id_, conf = recognizer.predict(roi_gray)
        
            print("Becsült érték", conf);
            print("Azonosító",id_);
            print("Név:", labels[id_]);
            font=cv2.FONT_HERSHEY_SIMPLEX
            color=(255,255,255)
            stroke=2
            if conf<=25:
                cv2.putText(gray_img,labels[id_],(x,y),cv2.FONT_HERSHEY_COMPLEX,3,(255,255,255),2)
            cv2.rectangle(gray_img,(x,y),(x+w,y+h),(255,0,0),thickness=5) 
        cv2.namedWindow('custom window', cv2.WINDOW_KEEPRATIO)
        cv2.imshow('custom window', gray_img)
        cv2.resizeWindow('custom window', 450, 450) 
        cv2.waitKey(0)
        cv2.destroyAllWindows
    def keptesztCallback():
        print("Arcfelismerés kép segitségével:")
        arcfelismereskepeken()

    root= tk.Tk()
    root.title("Arcfelismerő")
    root.protocol('WM_DELETE_WINDOW',donothing)

    canvas1 = tk.Canvas(root, width =400, height = 400)
    
    canvas1.pack()

    button3 = tk.Button(text='Felismerő tanítása!', command=tanitCallback)
    canvas1.create_window(200, 20, window=button3)

    test = tk.Label(root, text="Képen való tesztelés:")
    canvas1.create_window(175, 45, window=test)

    button4 = tk.Button(text='Kép alapú tesztelés!', command=keptesztCallback)
    canvas1.create_window(150, 80, window=button4)
    button5 = tk.Button(text='Befejez', command=keptesztbezarCallback)
    canvas1.create_window(250,80, window=button5)

    elokep = tk.Label(root, text="Élőkép elötti arcfelismerés:")
    canvas1.create_window(175, 120, window=elokep)

    tanar = tk.Label(root, text="Tanár neve:")
    canvas1.create_window(200, 140, window=tanar)
    tanarneve = tk.Entry (root) 
    canvas1.create_window(200, 160, window=tanarneve)

    kurzus = tk.Label(root, text="Kurzus neve:")
    canvas1.create_window(200, 200, window=kurzus)
    kurzusneve = tk.Entry (root) 
    canvas1.create_window(200, 220, window=kurzusneve)

    diak = tk.Label(root, text="Diákok száma:")
    canvas1.create_window(200, 240, window=diak)
    diakszam= tk.Entry (root) 
    canvas1.create_window(200, 260, window=diakszam)
        
    button1 = tk.Button(text='Adat elmentése', command=letrehozCallback)
    canvas1.create_window(200, 300, window=button1)

    button2 = tk.Button(text='Arcfelismerő inditása', command=befejezCallback)
    canvas1.create_window(200, 340, window=button2)

    root.mainloop()
ablak()
#felismerő élőkép előtt.
def felismerovideo():
    diaksz= int(diakokszam)  
    if diaksz==0:
        if kurzus is '':
            return
    cap = cv2.VideoCapture(0,cv2.CAP_DSHOW)
    # lista megoldja az ismétlödések elkerülését
    lista=[] 
    while(True):
        ret,frame=cap.read()
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        faces = face_cascade.detectMultiScale(gray, scaleFactor=1.3, minNeighbors=5)
        for(x,y,w,h) in faces:
            roi_gray = gray[y:y+h, x:x+w]
            roi_color = frame[y:y+h, x:x+w]
            id_, conf = recognizer.predict(roi_gray)
            if conf >= 4 and conf <= 55:
                name = labels[id_]
                if name not in lista:
                    lista.append(name)
                    print("Becsült érték", conf);
                    print(id_, name.capitalize())
                    print(name)
                    now = datetime.now()
                    dt_string = now.strftime("%d/%m/%Y %H:%M:%S")
                    t=dt_string.split(" ")
                    day=t[0]
                    day=day.replace("/","-")
                    # adatbazisba ment
                    db.child("Datum").child(kurzus).child(day).child(labels[id_].capitalize()).set(
                        {"id": id_, "name": labels[id_].capitalize(), "date": dt_string})
                    db.child("Kurzusok").child(kurzus).child(tanar).set(
                        {"Tanar": tanar, "Kurzus": kurzus})
                    # csokenti a felismeres szamat
                    diaksz=diaksz-1;
                    font = cv2.FONT_HERSHEY_SIMPLEX
                    color = (255, 255, 255)
                    stroke = 2
                    cv2.putText(frame, name, (x, y), font, 1, color, stroke, cv2.LINE_AA)
                else:
                    continue
            cv2.rectangle(frame, (x, y), (x + w,  y + h), (0, 0, 0), thickness=2)
        cv2.imshow("Arcfelismeres", frame)
        if cv2.waitKey(20) & 0xFF == ord('q'):
            break
        if diaksz == 0:
            pdf(kurzus)
            print("Nincs több diák!")
            cv2.destroyAllWindows()
            break
    cap.release()
    cv2.destroyAllWindows()
felismerovideo()
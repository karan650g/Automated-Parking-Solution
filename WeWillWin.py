from kivy.app import App
from kivy.uix.boxlayout import BoxLayout
from kivy.uix.screenmanager import ScreenManager, Screen
from kivy.core.window import Window
from kivy.uix.button import Button
from kivy.uix.label import Label
from kivy.uix.textinput import TextInput
from kivy.uix.popup import Popup
import sqlite3
from kivy.lang import Builder
import os
import cv2
import numpy as np
import tensorflow as tf
import sys
import pytesseract
import pyttsx3
from PIL import Image

from utils import label_map_util
from utils import visualization_utils as vis_util
sys.path.append("..")

pytesseract.pytesseract.tesseract_cmd ='C:/Program Files (x86)/Tesseract-OCR/tesseract.exe'
MODEL_NAME = 'inference_graph'
VIDEO_NAME = 'park.mp4'
CWD_PATH = os.getcwd()
PATH_TO_CKPT = os.path.join(CWD_PATH,MODEL_NAME,'frozen_inference_graph.pb')


PATH_TO_LABELS = os.path.join(CWD_PATH,'training','labelmap.pbtxt')


PATH_TO_VIDEO = os.path.join(CWD_PATH,VIDEO_NAME)


NUM_CLASSES = 1


label_map = label_map_util.load_labelmap(PATH_TO_LABELS)
categories = label_map_util.convert_label_map_to_categories(label_map, max_num_classes=NUM_CLASSES, use_display_name=True)
category_index = label_map_util.create_category_index(categories)


detection_graph = tf.Graph()
with detection_graph.as_default():
    od_graph_def = tf.GraphDef()
    with tf.gfile.GFile(PATH_TO_CKPT, 'rb') as fid:
        serialized_graph = fid.read()
        od_graph_def.ParseFromString(serialized_graph)
        tf.import_graph_def(od_graph_def, name='')

    sess = tf.Session(graph=detection_graph)


image_tensor = detection_graph.get_tensor_by_name('image_tensor:0')


detection_boxes = detection_graph.get_tensor_by_name('detection_boxes:0')


detection_scores = detection_graph.get_tensor_by_name('detection_scores:0')
detection_classes = detection_graph.get_tensor_by_name('detection_classes:0')

num_detections = detection_graph.get_tensor_by_name('num_detections:0')
Window.clearcolor = (0.3, 0.3, 0.3, 0.3)
Builder.load_string('''
<MainWindow>:
    id: main_win
    orientation: "vertical"
    spacing: 10
    space_x: self.size[0]/3


    canvas.before:
        Color:
            rgba: (1,1,1,1)
        Rectangle:
            source:'work.jpg'
            size: root.width-350,root.height
            pos: self.pos

    BoxLayout:
        size_hint_y: None
        height: 50
        canvas.before:
            Color:
                rgba: (.9, .5,.4, 1)
            Rectangle:
                size: self.size
                pos: self.pos
        Label:
            text: "3RD EYE SECURITY SYSTEM"
            bold: True
            size_hint_x: .9
    BoxLayout:
        orientation: 'vertical'
        padding: main_win.space_x, 10
        #spacing: 20
        Label:
            id: sp21


        Button:
            text: "Login"
            pos_hint:{'center_x': 1.25, 'center_y': 1}
            size_hint_y: None
            height: 40
            background_color: (.9, .5,.4, 1)
            background_normal: ''
            on_release:root.change_login()


        Label:
            id: sp2
            spacing:10

        Button:
            text :"Sign In"
            pos_hint:{'center_x': 1.25, 'center_y': 1}
            size_hint_y: None
            height: 40
            background_color:(.9, .5,.4, 1)
            background_normal: ''
            on_release:root.change_signup()
        Label:
            id :sp4
        Label:
            id :sp5
        Button:
            text :"EXIT"
            pos_hint:{'center_x': 1.25, 'center_y': 1}
            size_hint_y: None
            height: 40
            background_color: (.9, .5,.4, 1)
            background_normal: ''
            on_press:app.stop()
<Login>:
    id: roll
    orientation: "vertical"
    spacing: 10
    space_x: self.size[0]/3
    emaill:emaill
    pwdl:pwdl
    canvas.before:
        Color:
            rgba: (1,1,1,1)
        Rectangle:
            source:'work.jpg'
            size: root.width-350,root.height
            pos: self.pos
    BoxLayout:
        size_hint_y: None
        height: 50
        canvas.before:
            Color:
                rgba: (.9, .5,.4, 1)
            Rectangle:
                size: self.size
                pos: self.pos
        Label:
            text: "LOG IN"
            bold: True
            size_hint_x: .9
    BoxLayout:
        orientation: 'vertical'
        padding: roll.space_x, 10
        #spacing: 20
        BoxLayout:
            orientation: "vertical"
            spacing: 1
            size_hint_y: None
            height: 100
            Label:
                id:ws
            Label:
                id:ws1

            TextInput:
                id: emaill
                hint_text: "EMail ID"
                size_hint_y:None
                height:40
                focus:True
                multiline: False
                pos_hint:{'center_x': 1.25, 'center_y': 0}
            Label:
                id: info
                text: ''
                markup: True
                size_hint_y: None
                height: 20
            TextInput:
                id: pwdl
                hint_text: "Password"
                size_hint_y:None
                height:40
                focus:True
                multiline: False
                password:True
                pos_hint:{'center_x': 1.25, 'center_y': 0.3}

        Label:
            id: sp
            size_hint_y: None
            height: 40
        Button:
            text: "Done"
            size_hint_y: None
            height: 40
            background_color: (.9, .5,.4, 1)
            background_normal: ''
            pos_hint:{'center_x': 1.25, 'center_y': 0.3}
            on_release:root.change_mainmenu()


        Label:
            id: sp2
        Label:
            id :sp3
        Label:
            id :sp3
        Label:
            id :sp3
        Button:
            text :"Go back"
            size_hint_y: None
            height: 40
            background_color: (.9, .5,.4, 1)
            background_normal: ''
            pos_hint:{'center_x': 1.25, 'center_y': 0.3}
            on_release:root.change_main()
        Label:
            id :sp4
        Label:
            id :sp5
<Signup>:
    id : logged
    orientation: "vertical"
    spacing: 10
    space_x: self.size[0]/3
    name:name
    email:email
    pwd:pwd
    phone:phone
    idd:idd
    spz:spz
    canvas.before:
        Color:
            rgba: (1,1,1,1)
        Rectangle:
            source:'work.jpg'
            size: root.width-350,root.height
            pos: self.pos
    BoxLayout:
        size_hint_y: None
        height: 50
        canvas.before:
            Color:
                rgba: (.9, .5,.4, 1)
            Rectangle:
                size: self.size
                pos: self.pos
        Label:
            text: "SIGN IN"
            bold: True
            size_hint_x: .9
    BoxLayout:
        orientation: 'vertical'
        padding: logged.space_x, 5
        #spacing: 20
        TextInput:
            id: name
            hint_text: "Full Name"
            pos_hint:{'center_x': 1.25, 'center_y': 1}
            multiline: False
        Label:
            id: sp2
            size_hint_y:None
            height:25
        TextInput:
            id: email
            hint_text: "EMail ID"
            pos_hint:{'center_x': 1.25, 'center_y': 0.9}
            multiline: False
        Label:
            id: sp2
            size_hint_y:None
            height:25
        TextInput:
            id: pwd
            hint_text: "Password"
            pos_hint:{'center_x': 1.25, 'center_y': 0.7}
            multiline: False
            password:True
        Label:
            id: sp2
            size_hint_y:None
            height:25
        TextInput:
            id: phone
            hint_text: "Phone Number"
            pos_hint:{'center_x': 1.25, 'center_y': 0.5}
            multiline: False
        Label:
            id: sp2
            size_hint_y:None
            height:25
        TextInput:
            id: idd
            hint_text: "ID-Number"
            pos_hint:{'center_x': 1.25, 'center_y': 0.3}
            multiline: False

        Label:
            id: sp2
            size_hint_y:None
            height:50
        Button:
            text: "Continue"
            size_hint_y: None
            height: 40
            background_color: (.06,.45,.45, 1)
            background_normal: ''
            pos_hint:{'center_x': 1.25, 'center_y': 0.3}
            on_release:root.change_mainmenu()
        Label:
            id: sp2
            size_hint_y:None
            height:50
        Button:
            text: "GO BACK"
            size_hint_y: None
            height: 40
            background_color: (.9, .5,.4, 1)
            background_normal: ''
            on_release:root.change_main()
            pos_hint:{'center_x': 1.25, 'center_y': 0.3}
        Label:
            id: spz
            size_hint_y:None
            height:50
            pos_hint:{'center_x': 1.25, 'center_y': 1}

        Label:
            id: sp2
            size_hint_y:None
            height:50
<MainMenu>:
    id: roll
    orientation: "vertical"
    spacing: 10
    space_x: self.size[0]/3
    canvas.before:
        Color:
            rgba: (1,1,1, 1)
        Rectangle:
            source:'security.jpg'
            size: root.width,root.height
            pos: self.pos
    BoxLayout:
        size_hint_y: None
        height: 50
        canvas.before:
            Color:
                rgba: (.9, .5,.4, 1)
            Rectangle:
                size: self.size
                pos: self.pos
        Label:
            text: "MAIN MENU"
            bold: True
            size_hint_x: .9
    BoxLayout:
        orientation: 'vertical'
        padding: roll.space_x, 10
        #spacing: 20
        Button:
            text :"Register Vehicle"
            size_hint_y: None
            height: 40
            background_color: (.9, .5,.4, 1)
            background_normal: ''
            on_release:root.register()
        Label:
            id :sp4
        Label:
            id :sp5
        Button:
            text :"Delete Registered Vehicle"
            size_hint_y: None
            height: 40
            background_color: (.9, .5,.4, 1)
            background_normal: ''
            on_release:root.delete()
        Label:
            id :sp4
        Label:
            id :sp5
        Button:
            text :"View Known Cars in lot"
            size_hint_y: None
            height: 40
            background_color: (.9, .5,.4, 1)
            background_normal: ''
            on_release:root.moni()
        Label:
            id :sp4
        Label:
            id :sp5
        Button:
            text :"View Unknown Cars in lot"
            size_hint_y: None
            height: 40
            background_color: (.9, .5,.4, 1)
            background_normal: ''
            on_release:root.moni()
        Label:
            id :sp4
        Label:
            id :sp5
        Button:
            text :"Detection on Video"
            size_hint_y: None
            height: 40
            background_color: (.9, .5,.4, 1)
            background_normal: ''
            on_release:root.change_video()
        Label:
            id :sp4
        Label:
            id :sp5
        Button:
            text :"Dectection Of Images"
            size_hint_y: None
            height: 40
            background_color: (.9, .5,.4, 1)
            background_normal: ''
            on_release:root.change_picture()
        Label:
            id :sp4
        Label:
            id :sp5
        Button:
            text :"LOGOUT"
            size_hint_y: None
            height: 40
            background_color: (.9, .5,.4, 1)
            background_normal: ''
            on_release:root.change_main()

<playvideo>:

    id: roll
    orientation: "vertical"
    spacing: 10
    space_x: self.size[0]/3
    canvas.before:
        Color:
            rgba: (1,1,1, 1)
        Rectangle:
            source:'park.jpg'
            size: root.width,root.height
            pos: self.pos
    BoxLayout:
        size_hint_y: None
        height: 50
        canvas.before:
            Color:
                rgba: (.9, .5,.4, 1)
            Rectangle:
                size: self.size
                pos: self.pos
        Label:
            text: "PLAY THE VIDEO"
            bold: True
            size_hint_x: .9
    BoxLayout:
        orientation: 'vertical'
        padding: roll.space_x, 10
        #spacing: 20
        Button:
            text :"Open Video"
            size_hint_y: None
            height: 40
            background_color: (.9, .5,.4, 1)
            background_normal: ''
            on_release:root.play()
        Label:
            id: sp2
            size_hint_y:None
            height:50
        Button:
            text :"Go Back"
            size_hint_y: None
            height: 40
            background_color: (.9, .5,.4, 1)
            background_normal: ''
            on_release:root.back()

<pic>:

    id: roll
    orientation: "vertical"
    spacing: 10
    space_x: self.size[0]/3
    canvas.before:
        Color:
            rgba: (1,1,1, 1)
        Rectangle:
            source:'security.jpg'
            size: root.width,root.height
            pos: self.pos
    BoxLayout:
        size_hint_y: None
        height: 50
        canvas.before:
            Color:
                rgba: (.9, .5,.4, 1)
            Rectangle:
                size: self.size
                pos: self.pos
        Label:
            text: "IMAGE RECOGNITION"
            bold: True
            size_hint_x: .9
    BoxLayout:
        orientation: 'vertical'
        padding: roll.space_x, 10
        #spacing: 20
        Button:
            text :"Open Image 1"
            size_hint_y: None
            height: 40
            background_color: (.9, .5,.4, 1)
            background_normal: ''
            on_release:root.play1()
        Label:
            id :sp4
        Label:
            id :sp5
        Button:
            text :"Open Image 2"
            size_hint_y: None
            height: 40
            background_color: (.9, .5,.4, 1)
            background_normal: ''
            on_release:root.play2()
        Label:
            id :sp4
        Label:
            id :sp5
        Button:
            text :"Open Image 3"
            size_hint_y: None
            height: 40
            background_color: (.9, .5,.4, 1)
            background_normal: ''
            on_release:root.play3()
        Label:
            id :sp4
        Label:
            id :sp5
        Button:
            text :"Open Image 4"
            size_hint_y: None
            height: 40
            background_color: (.9, .5,.4, 1)
            background_normal: ''
            on_release:root.play4()
        Label:
            id :sp4
        Label:
            id :sp5
        Button:
            text :"Go Back"
            size_hint_y: None
            height: 40
            background_color: (.9, .5,.4, 1)
            background_normal: ''
            on_release:root.back()
''')


class MainWindow(BoxLayout):
    def __init__(self, **kwargs):
        super().__init__(**kwargs)

    def change_login(self):
        recognizer = cv2.face.LBPHFaceRecognizer_create()
        recognizer.read('train/trainningData.yml')
        faceCascade = cv2.CascadeClassifier('haarcascade_frontalface_default.xml');
        id = -1
        cam = cv2.VideoCapture(0)
        cam.set(3, 640)
        cam.set(4, 480)
        minW = 0.1*cam.get(3)
        minH = 0.1*cam.get(4)
        while True:
            ret, img =cam.read()
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
                if (confidence < 75):
                    id =id
                else:
                    id ="UNKNOWN"
            cv2.imshow('camera',img)
            k = cv2.waitKey(30) & 0xff
            if k == 27:
                break
        cam.release()
        cv2.destroyAllWindows()
        comd=sqlite3.connect("security.db")
        print("Opened database successfully")
        cmd="SELECT * from Security_g"
        cursor=comd.execute(cmd)
        isrecord=0
        print(id)
        for row in cursor:
            if str(id) in row :
                isrecord=1
                studdd = 1
                engine = pyttsx3.init()
                engine.say("Login Succesful")
                engine.runAndWait()
                sa.screen_manager.current = 'four'
        
        if isrecord==0:
            engine = pyttsx3.init()
            engine.say("Sorry , You are not recognized")
            engine.runAndWait()
            sa.screen_manager.current = 'two'

    def change_signup(self):
        sa.screen_manager.current = 'three'

class pic(BoxLayout):
    def __init__(self, **kwargs):
        super().__init__(**kwargs)



    def play1(self):
        IMAGE_NAME = 'example1.jpg'
        PATH_TO_IMAGE = os.path.join(CWD_PATH,IMAGE_NAME)
        image = cv2.imread(PATH_TO_IMAGE)
        image_expanded = np.expand_dims(image, axis=0)
        (boxes, scores, classes, num) = sess.run(
            [detection_boxes, detection_scores, detection_classes, num_detections],
            feed_dict={image_tensor: image_expanded})

        vis_util.visualize_boxes_and_labels_on_image_array(
            image,
            np.squeeze(boxes),
            np.squeeze(classes).astype(np.int32),
            np.squeeze(scores),
            category_index,
            use_normalized_coordinates=True,
            line_thickness=8,
            min_score_thresh=0.60)
        cv2.imshow('Object detector', image)
        cv2.waitKey(0)
        cv2.destroyAllWindows()
        engine = pyttsx3.init()
        engine.say("Returning back to the main menu")
        engine.runAndWait()
        sa.screen_manager.current = 'four'

    def play2(self):
        IMAGE_NAME = 'example2.jpg'
        PATH_TO_IMAGE = os.path.join(CWD_PATH,IMAGE_NAME)
        image = cv2.imread(PATH_TO_IMAGE)
        image_expanded = np.expand_dims(image, axis=0)
        (boxes, scores, classes, num) = sess.run(
            [detection_boxes, detection_scores, detection_classes, num_detections],
            feed_dict={image_tensor: image_expanded})

        vis_util.visualize_boxes_and_labels_on_image_array(
            image,
            np.squeeze(boxes),
            np.squeeze(classes).astype(np.int32),
            np.squeeze(scores),
            category_index,
            use_normalized_coordinates=True,
            line_thickness=8,
            min_score_thresh=0.60)
        cv2.imshow('Object detector', image)
        cv2.waitKey(0)
        cv2.destroyAllWindows()
        sa.screen_manager.current = 'four'

    def play3(self):
        IMAGE_NAME = 'example3.jpg'
        PATH_TO_IMAGE = os.path.join(CWD_PATH,IMAGE_NAME)
        image = cv2.imread(PATH_TO_IMAGE)
        image_expanded = np.expand_dims(image, axis=0)
        (boxes, scores, classes, num) = sess.run(
            [detection_boxes, detection_scores, detection_classes, num_detections],
            feed_dict={image_tensor: image_expanded})

        vis_util.visualize_boxes_and_labels_on_image_array(
            image,
            np.squeeze(boxes),
            np.squeeze(classes).astype(np.int32),
            np.squeeze(scores),
            category_index,
            use_normalized_coordinates=True,
            line_thickness=8,
            min_score_thresh=0.60)
        cv2.imshow('Object detector', image)
        cv2.waitKey(0)
        cv2.destroyAllWindows()
        sa.screen_manager.current = 'four'
    def back(self):
        sa.screen_manager.current = 'four'
        
class Login(BoxLayout):
    def __init__(self, **kwargs):
        super().__init__(**kwargs)

    def change_main(self):
        sa.screen_manager.current = 'one'

    def change_mainmenu(self):
        sa.screen_manager.current = 'four'


class Signup(BoxLayout):
    def __init__(self, **kwargs):
        super().__init__(**kwargs)

    def change_main(self):
        sa.screen_manager.current = 'one'

    def change_mainmenu(self):
        l = []
        l.append(self.name.text)
        l.append(self.email.text)
        l.append(self.pwd.text)
        l.append(self.phone.text)
        l.append(self.idd.text)
        connection = sqlite3.connect("security.db")
        crsr = connection.cursor()
        crsr.execute("INSERT INTO Security_g VALUES (?,?,?,?,?)", l)
        connection.commit()
        connection.close()
        faceDetect=cv2.CascadeClassifier('haarcascade_frontalface_default.xml')
        cam=cv2.VideoCapture(0)
        id=self.idd.text
        sampleno=0
        while(True):
            ret, img = cam.read()
            gray=cv2.cvtColor(img,cv2.COLOR_BGR2GRAY)
            faces = faceDetect.detectMultiScale(gray, 1.3, 5)
            for(x,y,w,h) in faces:
                sampleno+=1
                cv2.imwrite('dataset/user.'+str(id)+'.'+str(sampleno)+'.jpg',gray[y:y+h,x:x+w])
                cv2.rectangle(img,(x,y),(x+w,y+h),(0,0,255),2)
            cv2.imshow("face",img)
            cv2.waitKey(1)
            if sampleno>15:
                break
        cam.release() 
        cv2.destroyAllWindows()
        recognizer=cv2.face.LBPHFaceRecognizer_create();
        path="C:/Tensorflow/models/research/object_detection/dataset"

        def getImagesWithID(path):
            imagePaths=[os.path.join(path,f) for f in os.listdir(path)]
            faces=[]
            IDs=[]
            for imagePath in imagePaths:
                faceImg=Image.open(imagePath).convert('L')
                faceNp=np.array(faceImg,'uint8')
                a=imagePath.split('user')
                idd=int(a[1].split('.')[1])
                faces.append(faceNp)
                IDs.append(idd)
                cv2.imshow("training",faceNp)
                cv2.waitKey(10)
            return np.array(IDs), faces
        Ids, faces = getImagesWithID(path)
        recognizer.train(faces, np.array(Ids))
        recognizer.save('train/trainningData.yml')
        cam.release() 
        cv2.destroyAllWindows()  
        sa.screen_manager.current = 'four'
        


class MainMenu(BoxLayout):
    def __init__(self, **kwargs):
        super().__init__(**kwargs)

    def change_main(self):
        engine = pyttsx3.init()
        engine.say("Logout Successful")
        engine.runAndWait()
        sa.screen_manager.current = 'one'
    def change_video(self):
        engine = pyttsx3.init()
        engine.say("Number plate detection on video.")
        engine.runAndWait()
        sa.screen_manager.current='five'

    def register(self):
        layout = BoxLayout(orientation='vertical')
        popupLabel = Label(text="Enter Car Number")
        x = Label(text="")
        self.text1 = TextInput(hint_text="Car Number")
        popupLabel2 = Label(text="Enter Owner Name")
        self.text2 = TextInput(hint_text="Owner Name")
        popupLabel3 = Label(text="Enter Model Type")
        self.text3 = TextInput(hint_text="Model Type")
        popupLabel4 = Label(text="Enter Color Of the Car")
        self.text4 = TextInput(hint_text="Color")
        closeButton = Button(text="Register")
        layout.add_widget(popupLabel)
        layout.add_widget(self.text1)
        layout.add_widget(popupLabel2)
        layout.add_widget(self.text2)
        layout.add_widget(popupLabel3)
        layout.add_widget(self.text3)
        layout.add_widget(popupLabel4)
        layout.add_widget(self.text4)
        layout.add_widget(x)
        layout.add_widget(closeButton)
        popup = Popup(title='Register Vehicle', content=layout, size_hint=(None, None), size=(400, 400))
        popup.open()
        closeButton.bind(on_press=self.regis)

    def regis(self, a):
        l = []
        l.append(self.text1.text)
        l.append(self.text2.text)
        l.append(self.text3.text)
        l.append(self.text4.text)
        connection = sqlite3.connect("security.db")
        crsr = connection.cursor()
        crsr.execute("INSERT INTO Details VALUES (?,?,?,?)", l)
        connection.commit()
        connection.close()
        engine = pyttsx3.init()
        engine.say("Vehicle is sucessfully registered!")
        engine.runAndWait()

    def delete(self):
        layout = BoxLayout(orientation='vertical')
        w = Label(text="Enter Car Number")
        x = Label(text="")
        x1 = Label(text="")
        x2 = Label(text="")
        self.m = TextInput(hint_text="Car Number")
        closeButton = Button(text="Close the pop-up")
        layout.add_widget(w)
        layout.add_widget(self.m)
        layout.add_widget(x)
        layout.add_widget(x1)
        layout.add_widget(x2)
        layout.add_widget(closeButton)
        popup = Popup(title='Delete Registered Vehicle', content=layout, size_hint=(None, None), size=(350, 350))
        popup.open()
        closeButton.bind(on_press=self.dell)

    def dell(self, a):
        connection = sqlite3.connect("security.db")
        crsr = connection.cursor()
        a = """Delete from Details where Car_no = """ + self.m.text
        print(a)
        crsr.execute(a)
        connection.commit()
        connection.close()
        engine = pyttsx3.init()
        engine.say("Vehicle is removed from database")
        engine.runAndWait()

    def change_picture(self):
        engine = pyttsx3.init()
        engine.say("Number plate detection on images.")
        engine.runAndWait()
        sa.screen_manager.current = 'six'
class playvideo(BoxLayout):

    def __init__(self, **kwargs):
        super().__init__(**kwargs)

    def play(self):
        
        video = cv2.VideoCapture(PATH_TO_VIDEO)

        while(video.isOpened()):
            ret, frame = video.read()
            frame_expanded = np.expand_dims(frame, axis=0)
            (boxes, scores, classes, num) = sess.run(
                [detection_boxes, detection_scores, detection_classes, num_detections],
                feed_dict={image_tensor: frame_expanded})
            vis_util.visualize_boxes_and_labels_on_image_array(
                frame,
                np.squeeze(boxes),
                np.squeeze(classes).astype(np.int32),
                np.squeeze(scores),
                category_index,
                use_normalized_coordinates=True,
                line_thickness=8,
                min_score_thresh=0.90)
            cv2.imshow('Object detector', frame)
            if (cv2.waitKey(1) & 0xFF == ord('q')):
                break
        video.release()
        cv2.destroyAllWindows()
    def back(self):
        sa.screen_manager.current = 'four'
        

class MyApp(App):

    def build(self):
        self.screen_manager = ScreenManager()

        self.one = MainWindow()
        screen = Screen(name="one")
        screen.add_widget(self.one)
        self.screen_manager.add_widget(screen)

        self.two = Login()
        screen = Screen(name="two")
        screen.add_widget(self.two)
        self.screen_manager.add_widget(screen)

        self.three = Signup()
        screen = Screen(name="three")
        screen.add_widget(self.three)
        self.screen_manager.add_widget(screen)

        self.four = MainMenu()
        screen = Screen(name="four")
        screen.add_widget(self.four)
        self.screen_manager.add_widget(screen)

        self.five = playvideo()
        screen = Screen(name='five')
        screen.add_widget(self.five)
        self.screen_manager.add_widget(screen)

        self.six = pic()
        screen = Screen(name='six')
        screen.add_widget(self.six)
        self.screen_manager.add_widget(screen)

        return self.screen_manager


if __name__ == "__main__":
    sa = MyApp()
    sa.run()

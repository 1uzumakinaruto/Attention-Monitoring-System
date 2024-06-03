import cv2
import os
import time
import numpy as np
import time
import pandas as pd
import cv2
import numpy as np
from keras.models import model_from_json

def Attendence():
    emotion_dict = {0: "Angry", 1: "Disgusted", 2: "Fearful", 3: "Happy", 4: "Neutral", 5: "Sad", 6: "Surprised"}

    # load json and create model
    json_file = open('models/emotion_model.json', 'r')
    loaded_model_json = json_file.read()
    json_file.close()
    emotion_model = model_from_json(loaded_model_json)

    # load weights into new model
    emotion_model.load_weights("models/emotion_model.h5")
    print("Loaded model from disk")

    recognizer = cv2.face.LBPHFaceRecognizer_create()

    recognizer.read('trainer/trainer.yml')

    cascadePath = "haarcascade_frontalface_default.xml"

    faceCascade = cv2.CascadeClassifier(cascadePath)

    font = cv2.FONT_HERSHEY_SIMPLEX
    cam = cv2.VideoCapture(0)
    df=pd.read_csv('StudentDetails.csv')
    ncount = []
    emotions = []
    counts = 0
    while True:
            ret, im =cam.read()
            gray = cv2.cvtColor(im,cv2.COLOR_BGR2GRAY)
            faces = faceCascade.detectMultiScale(gray, 1.2,5)
            for(x,y,w,h) in faces:
                counts += 1
                roi_gray_frame = gray[y:y + h, x:x + w]
                cropped_img = np.expand_dims(np.expand_dims(cv2.resize(roi_gray_frame, (48, 48)), -1), 0)

                # predict the emotions
                emotion_prediction = emotion_model.predict(cropped_img)
                maxindex = int(np.argmax(emotion_prediction))

                cv2.rectangle(im, (x,y), (x+w,y+h), (0,255,0), 1)
                Id,i = recognizer.predict(gray[y:y+h,x:x+w])
                if i < 60:
                    name=df.loc[(df['Id']==Id)]['Name'].values[0]
                    cv2.putText(im, name, (x+5,y-10), font, 0.5, (255,255,255), 1)
                    cv2.putText(im, emotion_dict[maxindex], (x+5,y+15), font, 0.5, (255,255,255), 1)
                    emotions.append(emotion_dict[maxindex])
                    if not name in ncount: 
                        ncount.append(name)
                else:
                    cv2.putText(im, "unknown", (x,y-10), font, 0.5, (255,255,255), 1)

            cv2.imshow('im',im)
            if cv2.waitKey(1) & 0xFF == ord('q'): 
                break
            elif counts>26:
                break
    cam.release()
    cv2.destroyAllWindows()

    if len(ncount) > 0:
        
        Angry = emotions.count('Angry')
        print(Angry)
        if Angry > 0:
            Angry = Angry / len(emotions) * 100
        
        Disgusted = emotions.count('Disgusted')
        if Disgusted > 0:
            Disgusted = Disgusted / len(emotions) * 100

        Fearful = emotions.count('Fearful')
        if Fearful > 0:
            Fearful = Fearful / len(emotions) * 100

        Happy = emotions.count('Happy')
        if Happy > 0:
            Happy = Happy / len(emotions) * 100

        Neutral = emotions.count('Neutral')
        if Neutral > 0:
            Neutral = Neutral / len(emotions) * 100
        
        Sad = emotions.count('Sad')
        if Sad > 0:
            Sad = Sad / len(emotions) * 100

        Surprised = emotions.count('Surprised')
        if Surprised > 0:
            Surprised = Surprised / len(emotions) * 100

        data = [Angry, Disgusted, Fearful, Happy, Neutral, Sad, Surprised]

        def most_frequent(List):
            counter = 0
            num = List[0]
            
            for i in List:
                curr_frequency = List.count(i)
                if(curr_frequency> counter):
                    counter = curr_frequency
                    num = i
        
            return num

        name = most_frequent(ncount)
        return name, data
    else:
        return 'unknown', [0, 0, 0, 0, 0, 0, 0]

from flask import Flask, render_template, request
import os
import pandas as pd
import datetime
import time
import cv2
import numpy as np
import csv
from PIL import Image, ImageTk
from datasets import create_dataset
from training import Train
from recognition import Attendence
import csv
import matplotlib.pyplot as plt 
import random
from aiproctoring import checkProctoring

app = Flask(__name__)

def save_csv(data1):
   print(data1)
   if not os.path.exists('emotions.csv'):
      f = open('emotions.csv', 'w', newline = '')
      writer = csv.writer(f)
      writer.writerow(['Name','Angry', 'Disgusted', 'Fearful', 'Happy', 'Neutral', 'Sad', 'Surprised'] )
      f.close()

   f = open('emotions.csv', 'a', newline = '')
   writer = csv.writer(f)
   writer.writerow(data1)
   f.close()

def getid():
   if not os.path.exists('StudentDetails.csv'):
      return 1
   else:
      df = pd.read_csv('StudentDetails.csv')
      ids = df['Id'].values
      ids = list(set(ids))
      return int(ids[-1])+1

def getnames():
   if not os.path.exists('StudentDetails.csv'):
      return []
   else:
      df = pd.read_csv('StudentDetails.csv')
      names = df['Name'].values
      print(names)
      return names

@app.route('/')
def index():
   return render_template("index.html",ID=getid())

@app.route('/perform')
def perform():
   return render_template("perform.html", Names = getnames())

@app.route('/create_datsets',  methods=['POST','GET'])
def create_datsets():
   if request.method == 'POST':
      Id = request.form['Id']
      Name = request.form['Name']
      Phone = request.form['Phone']
      Email = request.form['Email']
      Sem = request.form['Sem']
      Cource = request.form['Cource']
      Branch = request.form['Branch']

      print(Id+' '+Name+' '+Phone+' '+Email+' '+Sem+' '+Cource+' '+Branch)

      create_dataset(Id, Name)
      
      msg = ['Images Saved for',
            'ID : ' + Id,
            'Name : ' + Name,
            'Phone : ' + Phone,
            'Email : ' + Email,
            'Semester : ' + Sem,
            'Cource : ' + Cource,
            'Branch : ' + Branch]
      
      row = [Id, Name, Phone, Email, Sem, Cource, Branch]

      if not os.path.exists('StudentDetails.csv'):
         row1 = ['Id', 'Name', 'Phone', 'Email', 'Sem', 'Cource', 'Branch']
         with open('StudentDetails.csv','w',newline='') as csvFile:
            writer = csv.writer(csvFile)
            writer.writerow(row1)
         csvFile.close()

      with open('StudentDetails.csv','a', newline='') as csvFile:
         writer = csv.writer(csvFile)
         writer.writerow(row)
      csvFile.close()

      Train()

      return render_template("index.html", msg=msg, ID=getid())
   return render_template("index.html", ID=getid())

@app.route('/emotions')
def emotions():
   name, data= Attendence()
   if name == 'unknown':
      return render_template("perform.html", Names = getnames(), msg = ['Unknown person'])
   else:
      data1 = [name]
      x = ['Angry', 'Disgusted', 'Fearful', 'Happy', 'Neutral', 'Sad', 'Surprised'] 
      y = data
      data1.extend(data)
      save_csv(data1)
      fig = plt.figure(figsize = (10, 5))
      plt.bar(x, y, color ='maroon', width = 0.4) 
      plt.xlabel('Emotions') 
      plt.ylabel('Percentage') 
      plt.title("{}'s emotion visualize".format(name)) 
      filename = random.randint(1000, 9999)
      for i in os.listdir('static/output_graph'):
         os.remove('static/output_graph/'+i)

      plt.savefig('static/output_graph/'+str(filename)+'.jpg')

      return render_template("perform.html", Names = getnames(), image = 'http://127.0.0.1:5000/static/output_graph/'+str(filename)+'.jpg')

@app.route('/Search',  methods=['POST','GET'])
def Search():
   if request.method == 'POST':
      name = request.form['name']
      df = pd.read_csv('emotions.csv')
      dd = df.loc[df['Name'] == name]
      if not dd.empty:
         x = ['Angry', 'Disgusted', 'Fearful', 'Happy', 'Neutral', 'Sad', 'Surprised'] 
         y = dd.values[-1][1:].tolist()
         fig = plt.figure(figsize = (10, 5))
         plt.bar(x, y, color ='maroon', width = 0.4) 
         plt.xlabel('Emotions') 
         plt.ylabel('Percentage') 
         plt.title("{}'s emotion visualize".format(name)) 
         filename = random.randint(1000, 9999)
         for i in os.listdir('static/output_graph'):
            os.remove('static/output_graph/'+i)

         plt.savefig('static/output_graph/'+str(filename)+'.jpg')
         return render_template("perform.html", Names = getnames(), image = 'http://127.0.0.1:5000/static/output_graph/'+str(filename)+'.jpg')
      else:
         return render_template("perform.html",  msg=['details not found of {}'.format(name)], Names = getnames())
   return render_template("perform.html", Names = getnames())


@app.route('/proctoring')
def proctoring():
   cell_counts, headup, headdown, headleft, headright, headcenter = checkProctoring()
   x = ['cell phone', 'headup', 'headdown', 'headleft', 'headright', 'headcenter'] 
   y = [cell_counts, headup, headdown, headleft, headright, headcenter]
   fig = plt.figure(figsize = (10, 5))
   plt.bar(x, y, color ='maroon', width = 0.4) 
   plt.xlabel('Proctrings') 
   plt.ylabel('Percentage') 
   plt.title("Classroom attention and behaviour visualize") 
   filename = random.randint(1000, 9999)
   for i in os.listdir('static/output_graph'):
      os.remove('static/output_graph/'+i)

   plt.savefig('static/output_graph/'+str(filename)+'.jpg')
   return render_template("perform.html", Names = getnames(), image = 'http://127.0.0.1:5000/static/output_graph/'+str(filename)+'.jpg')

@app.route('/attendance')
def attendance():
   return render_template("attendance.html")

@app.route('/viewattendance')
def viewattendance():
   from serial_test import saveFile
   saveFile()
   List = os.listdir('attendance')
   path = 'attendance/'+List[-1]
   print(path)
   result = []
   f = open(path, 'r')
   reader = csv.reader(f)
   for row in reader:
      result.append(row)
   f.close()
   print(result)
   return render_template("attendance.html", result = result)

if __name__ == "__main__":
   app.run(debug=True, use_reloader=False)

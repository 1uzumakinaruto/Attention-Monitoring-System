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

app = Flask(__name__)

def getid():
   if not os.path.exists('StudentDetails.csv'):
      return 1
   else:
      df = pd.read_csv('StudentDetails.csv')
      names1 = df['Id'].values
      names1 = list(set(names1))
      return int(names1[-1])+1
   
@app.route('/')
def index():
   return render_template("index.html", ID=getid())

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
   return render_template("index.html")

@app.route('/attendence',  methods=['POST','GET'])
def attendence():
   if request.method == 'POST':
      Subject = request.form['Subject']
      names= Attendence()
      if names == 'unknown':
         return render_template("index.html", ID=getid(), msg = ['Unknown person'])
      else:
         df = pd.read_csv('StudentDetails.csv')
         names1 = df['Name'].values
         names1 = list(set(names1))
         
         col_names =  ['Name','Date','Time', 'Status']
         ts = time.time()      
         date = datetime.datetime.fromtimestamp(ts).strftime('%Y-%m-%d')
         timeStamp = datetime.datetime.fromtimestamp(ts).strftime('%H:%M:%S')
         Hour,Minute,Second=timeStamp.split(":")
         fileName="StudentAttendence/"+str(Subject)+"/Attendence_"+date+"_"+Hour+"-"+Minute+"-"+Second+".csv"

         attendence_info = []
         f = open(fileName, 'w', newline='')
         writer = csv.writer(f)
         writer.writerow(col_names)
         
         for name in names1:
            if name in names:
               writer.writerow([name, date, timeStamp, 'Present'])
               attendence_info.append(name+' is Present')
            else:
               writer.writerow([name, date, timeStamp, 'Absent'])
               attendence_info.append(name+' is Absent')
         f.close()
         from serial_test import sendFile
         sendFile(fileName)
         return render_template("index.html", ID=getid(), List=attendence_info,  subject=Subject, date=date, time=timeStamp)
   return render_template("index.html")

if __name__ == "__main__":
   app.run(debug=True, use_reloader=False)

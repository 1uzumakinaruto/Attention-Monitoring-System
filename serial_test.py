import serial
import time
import datetime

data = serial.Serial(
                  'COM7',
                  baudrate = 9600,
                  parity=serial.PARITY_NONE,
                  stopbits=serial.STOPBITS_ONE,
                  bytesize=serial.EIGHTBITS,                  
                  timeout=1
                  )

def saveFile():
    while True:
        d = data.readline()
        if d:
            ts = time.time()      
            date = datetime.datetime.fromtimestamp(ts).strftime('%Y%m%d')
            timeStamp = datetime.datetime.fromtimestamp(ts).strftime('%H%M%S')
            fileName="attendance/"+date+timeStamp+".csv"
            f = open(fileName, 'wb')
            f.write(d)
            f.close()
            break
        time.sleep(1)
    

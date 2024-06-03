import serial
import time
import datetime

data = serial.Serial(
                  '/dev/ttyS0',
                  baudrate = 9600,
                  parity=serial.PARITY_NONE,
                  stopbits=serial.STOPBITS_ONE,
                  bytesize=serial.EIGHTBITS,                  
                  timeout=1
                  )

def sendFile(filename):
    f = open(filename, 'rb')
    a = f.read()
    f.close()
    data.write(a)
    
from time import sleep
import time
import RPi.GPIO as GPIO
from twilio.rest import Client

DIR = 23   # Direction GPIO Pin
STEP = 24  # Step GPIO Pin
CW = 1     # Clockwise Rotation
CCW = 0    # Counterclockwise Rotation
SPR = 200   # Steps per Revolution (360 / 7.5)
KEY_IN = 27
MODE = (14, 15, 18)
GPIO.setmode(GPIO.BCM)
GPIO.setup(MODE, GPIO.OUT)
GPIO.setmode(GPIO.BCM)
GPIO.setup(DIR, GPIO.OUT)
GPIO.setup(STEP, GPIO.OUT)
direction = 1
GPIO.output(DIR, direction)
GPIO.setup(KEY_IN, GPIO.IN)
GPIO.output(MODE, (1,1,0))

consecutive = False

def turn():
    global direction
    consecutive = True
    delay = .00005
    for x in range(3200):
        GPIO.output(STEP, GPIO.HIGH)
        current_time = time.time() * 1000
        sleep(delay)
        GPIO.output(STEP, GPIO.LOW)
        sleep(delay)
    if direction == 1:
        client = Client(account_sid, auth_token)

        message = client.messages.create(
          from_='+1010010010',
          body='SOMEONE HAS COLLAPSED OF CARDIAC ARREST. PLEASE SEND IMMEDIATE HELP',
          to='+821011111111'
        )

        print(message.sid)
    direction = 1-direction
    GPIO.output(DIR, direction)
count = 0
not_turned = True
while True:
    val = GPIO.input(KEY_IN)
    print(val)
    if not val:
        not_turned = True
        count = 0
    else:
        count += 1
    if count > 2 and not_turned:
        turn()
        not_turned = False

GPIO.cleanup()

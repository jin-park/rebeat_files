import time
import board
import busio
from tflite_runtime.interpreter import Interpreter
import numpy as np
#===========================================
from adafruit_bno08x.i2c import BNO08X_I2C
from adafruit_bno08x import BNO_REPORT_LINEAR_ACCELERATION
#===========================================
# Analog Digital Converter
import adafruit_ads1x15.ads1115 as ADS
from adafruit_ads1x15.analog_in import AnalogIn
import pygame
from pygame import mixer
from math import floor
from multiprocessing import Pool, Process
import pusher
import csv
import datetime

i2c = busio.I2C(board.SCL, board.SDA)
ads = ADS.ADS1115(i2c, data_rate=860)
chan = AnalogIn(ads, ADS.P0)

bno = BNO08X_I2C(i2c)
bno.enable_feature(BNO_REPORT_LINEAR_ACCELERATION)
prev_time = 0.0; current_time = 0.0
disp = 0.0; vel = 0.0; prev_vel = 0.0; accel_z = 0.0; force = 0
count = 1
sensor_values = (0.0, 0)
interpreter = Interpreter("saved_model/2024-04-26 22:47:42.678452.tflite")
input_details = interpreter.get_input_details()
output_details = interpreter.get_output_details()
window = 16
datapoints = np.array([[
    [0.0,0.0],[0.0,0.0],[0.0,0.0],[0.0,0.0],[0.0,0.0],
    [0.0,0.0],[0.0,0.0],[0.0,0.0],[0.0,0.0],[0.0,0.0],
    [0.0,0.0],[0.0,0.0],[0.0,0.0],[0.0,0.0],[0.0,0.0],
    [0.0,0.0]
]], dtype="float32")
outputs = [0.0]
actuals = [0.0]
forces = [0]

alerts = [0, 0, 0, 0, 0] # too deep, too shallow, too slow, too fast, recoil/leaning
                         # assume that there probably won't be too many of too deep/shallow mixed together? or too fast/slow
deepest = 0.0
values = [0.0, 0.0, 0.0, 0.0]
warnings = [0, 0, 0, 0, 0] # too deep, deeper, fast, slow, lean
warning_not_updated = False
deepest_time = 0; prev_deepest_time = 0
display_warning_bools = [False, False, False, False, False]
pusher_client = pusher.Pusher()
csvfile = open("./data/" + str(datetime.datetime.now()) + "_slope.csv", "w", newline="")
writer = csv.writer(csvfile)
def init_screen():
    pygame.init()
    mixer.init()
    screen = pygame.display.set_mode((320, 240), pygame.FULLSCREEN)
    clock = pygame.time.Clock()
    return screen, clock

screen, clock = init_screen()
player_pos = pygame.Vector2(0, 0)

def sensor_measurement_task():
    _, _, accel_z = bno.linear_acceleration
    force = chan.value
    accel_z /= 20
    force /= 5100
    return accel_z, force

weights = [19.460941, 0.001522, 1.952678]

def update_datapoints(accel, f):
    for i in range(window-2):
        datapoints[0,i,0] = datapoints[0,i+1,0]
        datapoints[0,i,1] = datapoints[0,i+1,1]
    datapoints[0,window-1,0] = accel / 20
    datapoints[0, window-1, 1] = f / 12000 / 2

def inference_task():
    interpreter.allocate_tensors()
    interpreter.set_tensor(input_details[0]['index'], datapoints)
    interpreter.invoke()
    output = interpreter.get_tensor(output_details[0]['index'])
    return output[0,0] * 7

parity = True
show_direction = True

text_info = ""
last_deepest = -1
ratio = 0
last_depth = 0.0
vocal_count = [0, 0, 0, 0] 
tooslow = pygame.mixer.Sound("tooslow3.ogg")
tooslow.set_volume(1.0)
toofast = pygame.mixer.Sound("toofast.ogg")
toofast.set_volume(1.0)
bpm = -1
last_slope = 1
slope = 1
def update_warnings(new_depth, new_force):

    global slope, last_slope, bpm, ratio, last_deepest, text_info, warnings, deepest, warning_not_updated, deepest_time, prev_deepest_time
    font = pygame.font.SysFont("Arial", 60)
    if new_depth > deepest:
        deepest = new_depth
        deepest_time = time.time()
    deepest = max(new_depth, deepest)
    font5 = pygame.font.SysFont("Roboto", 40)
    text5 = "Good"
    text5 = font5.render(text5, True, (255, 255, 255))
    text5 = pygame.transform.rotate(text5, 90)
    text_rect5 = text5.get_rect(center=(247, 120))
    screen.blit(text5, text_rect5)


    font6 = pygame.font.SysFont("Roboto", 40)
    text6 = "Deeper"
    text6 = font6.render(text6, True, (255, 255, 255))
    text6 = pygame.transform.rotate(text6, 90)
    text_rect6 = text6.get_rect(center=(112, 120))
    screen.blit(text6, text_rect6)

    font7 = pygame.font.SysFont("Robot", 40)
    text7 = "Too Deep"
    text7 = font7.render(text7, True, (255, 255, 255))
    text7 = pygame.transform.rotate(text7, 90)
    text_rect7 = text7.get_rect(center=(292, 120))
    screen.blit(text7, text_rect7)


    if deepest_time == prev_deepest_time:
        return
    bpm = 60 / (deepest_time - prev_deepest_time)
    if new_force > 1000 and not warning_not_updated and bpm < 160:
        warning_not_updated = True
    last_slope = slope
    slope = (new_depth - last_depth) / 0.02
    slopeslope = (slope - last_slope) / 0.02
    writer.writerow((slope,new_depth))
    if slope < 3 and warning_not_updated and new_depth < 1.5 and bpm < 160:
        if slopeslope < 400:
            return

        print("sec. deriv.:", (slope - last_slope)/ 0.02)
        warning_not_updated = False
        ratio = floor(deepest / 7 * 320)
        prev_deepest_time = deepest_time
        last_deepest = floor(deepest*10)/10
        deepest = 0
        if bpm > 120:
            warnings[2] += 1
            warnings[3] = max(warnings[3]-1, 0)
            vocal_count[1] += 1
            vocal_count[0] = 0
        elif bpm < 100:
            warnings[2] = max(warnings[2]-1, 0)
            warnings[3] += 1
            vocal_count[0] += 1
            vocal_count[1] = 0
        else:
            vocal_count[0] = 0
            vocal_count[1] = 0
            warnings[2] = 0
            warnings[3] = 0
        if vocal_count[0] > 2:
            tooslow.play()
            vocal_count[0] = 0
        if vocal_count[1] > 2:
            toofast.play()
            vocal_count[1] = 0



    pygame.draw.rect(screen, (255, 0, 0), pygame.Rect(pygame.Vector2(ratio, 0), pygame.Vector2(10, 240)))
    pygame.draw.polygon(screen, (255, 0, 0), ((ratio, 170), (ratio, 240), (ratio-30, 240), (ratio-30, 190)))
    font8 = pygame.font.SysFont("Roboto", 30)
    text8 = "YOU"
    text8 = font8.render(text8, True, (255, 255, 255))
    text8 = pygame.transform.rotate(text8, 90)
    text_rect8 = text8.get_rect(center=(ratio-10,210))
    screen.blit(text8, text_rect8)

    font5 = pygame.font.SysFont("Roboto", 40)
    text5 = "Good"
    text5 = font5.render(text5, True, (255, 255, 255))
    text5 = pygame.transform.rotate(text5, 90)
    text_rect5 = text5.get_rect(center=(247, 120))
    screen.blit(text5, text_rect5)


    font6 = pygame.font.SysFont("Roboto", 40)
    text6 = "Deeper"
    text6 = font6.render(text6, True, (255, 255, 255))
    text6 = pygame.transform.rotate(text6, 90)
    text_rect6 = text6.get_rect(center=(112, 120))
    screen.blit(text6, text_rect6)

    font7 = pygame.font.SysFont("Robot", 40)
    text7 = "Too Deep"
    text7 = font7.render(text7, True, (255, 255, 255))
    text7 = pygame.transform.rotate(text7, 90)
    text_rect7 = text7.get_rect(center=(292, 120))
    screen.blit(text7, text_rect7)



screen, clock = init_screen()
player_pos = pygame.Vector2(0, 0)

new_value = 0.0; max_value = 0.0
avg = 0
run = True
current_time_warning = 0
prev_time_warning = 0
deepest_time = 0; last_deepest_time = 0
speed_warning_updated = True

def play_sound():
    mixer.music.load("metronome.mp3")
    mixer.music.play(1)
mixer.music.load("metronome.mp3")
mixer.music.play(-1)
counter = 0
while run:
    current_time = time.time() * 1000
    for event in pygame.event.get():
        if event.type == pygame.QUIT:
            run = False
        elif event.type == pygame.KEYDOWN:
            if event.key == pygame.K_ESCAPE:
                run = False
                mixer.music.stop()
    if current_time_warning - prev_time_warning > 545:
        parity = not parity
        prev_time_warning = current_time_warning
    current_time_warning = time.time() * 1000
    if current_time - prev_time > 20:
        counter += 1
        prev_time = current_time
        sensor_values = sensor_measurement_task()
        update_datapoints(sensor_values[0], sensor_values[1])
        screen.fill((0,0,0))
        update_warnings(new_value, sensor_values[1]) 
        last_depth = new_value
        new_value = inference_task()
        if counter % 2 == 0:
            data = {
                'score': max(floor(last_deepest/5*50)+floor(bpm/110*50), 0),
                'depth': max(round(new_value, 2), 0),
                'pressure': -1,
                'elapsed_time': floor(time.time()),
                'cycle': -1
            }
            pusher_client.trigger('my-channel', 'my-event', data)
        if new_value > max_value:
            deepest_time = time.time() * 1000
            max_value = new_value
        pygame.display.update()
csvfile.close()
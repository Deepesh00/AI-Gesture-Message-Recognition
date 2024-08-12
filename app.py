import cv2
import numpy as np
from collections import deque
from flask import Flask,render_template,Response, redirect, url_for
import time
from gtts import gTTS 
import os

app=Flask(__name__)

def setValues(x):
    print("")
    
def generate_frames(colorIndex,colors,kernel,bpoints,gpoints,rpoints,ypoints,bkpoints, blue_index,green_index,red_index,yellow_index, black_index):
    cap = cv2.VideoCapture(0)
    canvas = None
    while True:    
        ret, frame = cap.read()
    
        frame = cv2.flip(frame, 1)
        if canvas is None:
            canvas = np.zeros_like(frame)
        hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)

        u_hue = cv2.getTrackbarPos("Upper Hue", "Color detectors")
        u_saturation = cv2.getTrackbarPos("Upper Saturation", "Color detectors")
        u_value = cv2.getTrackbarPos("Upper Value", "Color detectors")
        l_hue = cv2.getTrackbarPos("Lower Hue", "Color detectors")
        l_saturation = cv2.getTrackbarPos("Lower Saturation", "Color detectors")
        l_value = cv2.getTrackbarPos("Lower Value", "Color detectors")
        Upper_hsv = np.array([u_hue,u_saturation,u_value])
        Lower_hsv = np.array([l_hue,l_saturation,l_value])
    
        frame = cv2.rectangle(frame, (40,1), (140,65), (122,122,122), -1)
        frame = cv2.rectangle(frame, (160,1), (255,65), colors[0], -1)
        frame = cv2.rectangle(frame, (275,1), (370,65), colors[1], -1)
        frame = cv2.rectangle(frame, (390,1), (485,65), colors[2], -1)
        frame = cv2.rectangle(frame, (505,1), (600,65), colors[3], -1)
        frame = cv2.rectangle(frame, (570,422), (640,480), (122,122,122), -1)
        frame = cv2.rectangle(frame, (0,422), (70,480), (122,122,122), -1)
        cv2.putText(frame, "CLEAR ALL", (49, 33), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 2, cv2.LINE_AA)
        cv2.putText(frame, "BLUE", (185, 33), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 2, cv2.LINE_AA)
        cv2.putText(frame, "GREEN", (298, 33), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 2, cv2.LINE_AA)
        cv2.putText(frame, "RED", (420, 33), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 2, cv2.LINE_AA)
        cv2.putText(frame, "YELLOW", (520, 33), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (150,150,150), 2, cv2.LINE_AA)
        cv2.putText(frame, "PAUSE", (582, 455), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 2, cv2.LINE_AA)
        cv2.putText(frame, "ERASE", (10, 455), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 2, cv2.LINE_AA)
  
        Mask = cv2.inRange(hsv, Lower_hsv, Upper_hsv)
        Mask = cv2.erode(Mask, kernel, iterations=1)
        Mask = cv2.morphologyEx(Mask, cv2.MORPH_OPEN, kernel)
        Mask = cv2.dilate(Mask, kernel, iterations=1)
        
        cnts,_ = cv2.findContours(Mask.copy(), cv2.RETR_EXTERNAL,cv2.CHAIN_APPROX_SIMPLE)
        center = None
        
        if len(cnts) > 0:
            
            cnt = sorted(cnts, key = cv2.contourArea, reverse = True)[0]        
            ((x, y), radius) = cv2.minEnclosingCircle(cnt)
            cv2.circle(frame, (int(x), int(y)), int(radius), (0, 255, 255), 2)
            M = cv2.moments(cnt)
            center = (int(M['m10'] / M['m00']), int(M['m01'] / M['m00']))
            if center[1] >= 422 and center[0] >= 570:
                break
            if center[1] >= 422 and center[0] <= 70:
                colorIndex = 4
            if center[1] <= 65:
                if 40 <= center[0] <= 140: 
                    bpoints = [deque(maxlen=512)]
                    gpoints = [deque(maxlen=512)]
                    rpoints = [deque(maxlen=512)]
                    ypoints = [deque(maxlen=512)]
                    bkpoints = [deque(maxlen=512)]

                    blue_index = 0
                    green_index = 0
                    red_index = 0
                    yellow_index = 0

                    canvas[67:,:,:] = 0
                elif 160 <= center[0] <= 255:
                        colorIndex = 0 
                elif 275 <= center[0] <= 370:
                        colorIndex = 1 
                elif 390 <= center[0] <= 485:
                        colorIndex = 2 
                elif 505 <= center[0] <= 600:
                        colorIndex = 3    
            else :
                if colorIndex == 0:
                    bpoints[blue_index].appendleft(center)
                elif colorIndex == 1:
                    gpoints[green_index].appendleft(center)
                elif colorIndex == 2:
                    rpoints[red_index].appendleft(center)
                elif colorIndex == 3:
                    ypoints[yellow_index].appendleft(center)
        else:
            bpoints.append(deque(maxlen=512))
            blue_index += 1
            gpoints.append(deque(maxlen=512))
            green_index += 1
            rpoints.append(deque(maxlen=512))
            red_index += 1
            ypoints.append(deque(maxlen=512))
            yellow_index += 1

        points = [bpoints, gpoints, rpoints, ypoints]
        for i in range(len(points)):
            for j in range(len(points[i])):
                for k in range(1, len(points[i][j])):
                    if points[i][j][k - 1] is None or points[i][j][k] is None:
                        continue
                    cv2.line(canvas, points[i][j][k - 1], points[i][j][k], colors[i], 2)

        _ , mask = cv2.threshold(cv2.cvtColor (canvas, cv2.COLOR_BGR2GRAY), 20, 
        255, cv2.THRESH_BINARY)
        foreground = cv2.bitwise_and(canvas, canvas, mask = mask)
        background = cv2.bitwise_and(frame, frame,
        mask = cv2.bitwise_not(mask))
        frame = cv2.add(foreground,background)

        # cv2.imshow("Tracking", canvas)
        # cv2.imshow("Paint", paintWindow)
        cv2.imshow("mask",Mask)
        ret, buffer = cv2.imencode('.jpg', frame)
        frame = buffer.tobytes()
        #yield (b'--frame\r\n'
                #b'Content-Type: image/jpeg\r\n\r\n' + frame + b'\r\n')  # concat frame one by one and show result
        if cv2.waitKey(1) & 0xFF == ord("q"):
            break
    cap.release()
    cv2.destroyAllWindows()

#def setValues(x):
 #  print("")

def updated_generate_frames():
    load_from_disk = True
    if load_from_disk:
        penval = np.load('penval.npy')

    colors = [(255, 0, 0), (0, 255, 0), (0, 0, 255), (0, 255, 255)]
    paintWindow = np.zeros((471,636,3)) + 255
    cv2.namedWindow('Paint', cv2.WINDOW_AUTOSIZE)


    cap = cv2.VideoCapture(0)

    # Load these 2 images and resize them to the same size.
    # pen_img = cv2.resize(cv2.imread('air canvas files/pen.png',1), (50, 50))
    # eraser_img = cv2.resize(cv2.imread('air canvas files/eraser.jpg',1), (50, 50))

    kernel = np.ones((5,5),np.uint8)

    # Making window size adjustable
    #cv2.namedWindow('image', cv2.WINDOW_NORMAL)

    # This is the canvas on which we will draw upon
    canvas = None

    # Create a background subtractor Object
    backgroundobject = cv2.createBackgroundSubtractorMOG2(detectShadows = False)

    # This threshold determines the amount of disruption in the background.
    background_threshold = 600

    # A variable which tells you if you're using a pen or an eraser.
    switch = 'Pen'

    # With this variable we will monitor the time between previous switch.
    last_switch = time.time()

    # Initilize x1,y1 points
    x1,y1=0,0

    # Threshold for noise
    noiseth = 800

    # Threshold for wiper, the size of the contour must be bigger than this for # us to clear the canvas
    wiper_thresh = 40000

    # A variable which tells when to clear canvas
    clear = False

    colorIndex=0  
    while(1):
        _, frame = cap.read()
        frame = cv2.flip( frame, 1 )
        
        # Initilize the canvas as a black image
        if canvas is None:
            canvas = np.zeros_like(frame)
        
        colors = [(255, 0, 0), (0, 255, 0), (0, 0, 255), (0, 255, 255), (0,0,0)]      

        # Take the top left of the frame and apply the background subtractor
        # there    
        # top_left = frame[422: 480, 0: 70]
        # fgmask = backgroundobject.apply(top_left)

        # # Note the number of pixels that are white, this is the level of 
        # # disruption.
        # switch_thresh = np.sum(fgmask==255)
        
        # # If the disruption is greater than background threshold and there has 
        # # been some time after the previous switch then you. can change the 
        # # object type.
        # if switch_thresh>background_threshold and (time.time()-last_switch) > 1:

        #     # Save the time of the switch. 
        #     last_switch = time.time()
            
        #     if switch == 'Pen':
        #         switch = 'Eraser'
        #     else:
        #         switch = 'Pen'

        # Convert BGR to HSV
        hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)
        
        # If you're reading from memory then load the upper and lower ranges 
        # from there
        if load_from_disk:
                lower_range = penval[0]
                upper_range = penval[1]
                
        # Otherwise define your own custom values for upper and lower range.
        else:             
            lower_range  = np.array([26,80,147])
            upper_range = np.array([81,255,255])
        
        mask = cv2.inRange(hsv, lower_range, upper_range)
        
        # Perform morphological operations to get rid of the noise
        mask = cv2.erode(mask,kernel,iterations = 1)
        mask = cv2.dilate(mask,kernel,iterations = 2)
        
        # Find Contours
        contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, 
        cv2.CHAIN_APPROX_SIMPLE)
        
        # Make sure there is a contour present and also it size is bigger than 
        # noise threshold.
        if contours and cv2.contourArea(max(contours,
                                        key = cv2.contourArea)) > noiseth:
                    
            c = max(contours, key = cv2.contourArea)    
            x2,y2,w,h = cv2.boundingRect(c)
            
            # Get the area of the contour
            area = cv2.contourArea(c)
            
            # If there were no previous points then save the detected x2,y2 
            # coordinates as x1,y1. 
            center=[x2, y2]
            if center[1] >= 422 and center[0] >= 570:
                break
            elif (center[1] >= 422 and center[0] <= 70) and (time.time()-last_switch) > 1:
                last_switch = time.time()
                if switch == 'Pen':
                    switch = 'Eraser'
                else:
                    switch = 'Pen'
            if center[1] <= 65:
                if 40 <= center[0] <= 140: 
                    canvas[:,:,:] = 0
                    paintWindow[:,:,:] = 255
                elif 160 <= center[0] <= 255:
                        colorIndex = 0 
                elif 275 <= center[0] <= 370:
                        colorIndex = 1 
                elif 390 <= center[0] <= 485:
                        colorIndex = 2 
                elif 505 <= center[0] <= 600:
                        colorIndex = 3 

            if x1 == 0 and y1 == 0:
                x1,y1= x2,y2
            else:
                if switch == 'Pen':
                    # Draw the line on the canvas
                    canvas = cv2.line(canvas, (x1,y1), (x2,y2), colors[colorIndex], 5)
                    paintWindow = cv2.line(paintWindow, (x1,y1), (x2,y2), colors[colorIndex], 5)
                else:
                    cv2.circle(canvas, (x2, y2), 20, (0,0,0), -1)
                    cv2.circle(paintWindow, (x2, y2), 20, (0,0,0), -1)
                
            # After the line is drawn the new points become the previous points.
            x1,y1 = x2,y2
        else:
            # If there were no contours detected then make x1,y1 = 0
            x1,y1 = 0,0
    
        # Now this piece of code is just for smooth drawing. (Optional)
        _ , mask = cv2.threshold(cv2.cvtColor (canvas, cv2.COLOR_BGR2GRAY), 20, 
        255, cv2.THRESH_BINARY)
        foreground = cv2.bitwise_and(canvas, canvas, mask = mask)
        background = cv2.bitwise_and(frame, frame,
        mask = cv2.bitwise_not(mask))
        frame = cv2.add(foreground,background)

        frame = cv2.rectangle(frame, (40,1), (140,65), (122,122,122), -1)
        frame = cv2.rectangle(frame, (160,1), (255,65), colors[0], -1)
        frame = cv2.rectangle(frame, (275,1), (370,65), colors[1], -1)
        frame = cv2.rectangle(frame, (390,1), (485,65), colors[2], -1)
        frame = cv2.rectangle(frame, (505,1), (600,65), colors[3], -1)
        frame = cv2.rectangle(frame, (570,422), (640,480), (122,122,122), -1)
        frame = cv2.rectangle(frame, (0,422), (70,480), (122,122,122), -1)
        cv2.putText(frame, "CLEAR ALL", (49, 33), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 2, cv2.LINE_AA)
        cv2.putText(frame, "BLUE", (185, 33), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 2, cv2.LINE_AA)
        cv2.putText(frame, "GREEN", (298, 33), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 2, cv2.LINE_AA)
        cv2.putText(frame, "RED", (420, 33), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 2, cv2.LINE_AA)
        cv2.putText(frame, "YELLOW", (520, 33), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (150,150,150), 2, cv2.LINE_AA)
        cv2.putText(frame, "PAUSE", (582, 455), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 2, cv2.LINE_AA)
        if(switch == 'Pen'):
            cv2.putText(frame, "ERASE", (10, 455), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 2, cv2.LINE_AA)
        else:
            cv2.putText(frame, "PEN", (10, 455), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 2, cv2.LINE_AA)

        if switch != 'Pen':
            cv2.circle(frame, (x1, y1), 20, (255,255,255), -1)
            cv2.circle(paintWindow, (x1, y1), 20, (255,255,255), -1)
        
        cv2.imshow("Paint", paintWindow)

        ret, buffer = cv2.imencode('.jpg', frame)
        frame = buffer.tobytes()
        yield (b'--frame\r\n'
                b'Content-Type: image/jpeg\r\n\r\n' + frame + b'\r\n')  # concat frame one by one and show result

        if cv2.waitKey(1) & 0xFF == ord("q"):
            break
        
        # Clear the canvas after 1 second, if the clear variable is true
        if clear == True: 
            time.sleep(1)
            canvas = None
            
            # And then set clear to false
            clear = False
            
    def close_vdo_download_img(fingertip_current_trace):
        file_name = "Trace_Image_" + time.strftime('%Y%m%d%H%M%S') + ".jpg"

        # free up memory
        cap.release()
        cv2.destroyAllWindows()

        cv2.imwrite(file_name, fingertip_current_trace)
        print("Image saved successfully!!!")
    close_vdo_download_img(paintWindow)

@app.route('/')
def index():
    return render_template('index.html')    

@app.route('/video')
def video():
    return render_template('video.html')    

@app.route('/video_feed')
def video_feed():
    print("inside video --------------")
    # cv2.namedWindow("Color detectors")
    # cv2.createTrackbar("Upper Hue", "Color detectors", 162, 180,setValues)
    # cv2.createTrackbar("Upper Saturation", "Color detectors", 255, 255,setValues)
    # cv2.createTrackbar("Upper Value", "Color detectors", 255, 255,setValues)
    # cv2.createTrackbar("Lower Hue", "Color detectors", 88, 180,setValues)
    # cv2.createTrackbar("Lower Saturation", "Color detectors", 108, 255,setValues)
    # cv2.createTrackbar("Lower Value", "Color detectors", 76, 255,setValues)

    # bpoints = [deque(maxlen=1024)]
    # gpoints = [deque(maxlen=1024)]
    # rpoints = [deque(maxlen=1024)]
    # ypoints = [deque(maxlen=1024)]
    # bkpoints = [deque(maxlen=1024)]

    # blue_index = 0
    # green_index = 0
    # red_index = 0
    # yellow_index = 0
    # black_index = 0

    # kernel = np.ones((5,5),np.uint8)

    # colors = [(255, 0, 0), (0, 255, 0), (0, 0, 255), (0, 255, 255), (0,0,0)]
    # colorIndex = 0

    # paintWindow = np.zeros((471,636,3)) + 255
    # paintWindow = cv2.rectangle(paintWindow, (40,1), (140,65), (0,0,0), 2)
    # paintWindow = cv2.rectangle(paintWindow, (160,1), (255,65), colors[0], -1)
    # paintWindow = cv2.rectangle(paintWindow, (275,1), (370,65), colors[1], -1)
    # paintWindow = cv2.rectangle(paintWindow, (390,1), (485,65), colors[2], -1)
    # paintWindow = cv2.rectangle(paintWindow, (505,1), (600,65), colors[3], -1)

    # cv2.putText(paintWindow, "CLEAR", (49, 33), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 0), 2, cv2.LINE_AA)
    # cv2.putText(paintWindow, "BLUE", (185, 33), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 2, cv2.LINE_AA)
    # cv2.putText(paintWindow, "GREEN", (298, 33), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 2, cv2.LINE_AA)
    # cv2.putText(paintWindow, "RED", (420, 33), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 2, cv2.LINE_AA)
    # cv2.putText(paintWindow, "YELLOW", (520, 33), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (150,150,150), 2, cv2.LINE_AA)
    # cv2.namedWindow('Paint', cv2.WINDOW_AUTOSIZE)

    # cap = cv2.VideoCapture(0)
    print("going to generate frames function")
    # generate_frames(colorIndex,paintWindow,colors,kernel,bpoints,gpoints,rpoints,ypoints,blue_index,green_index,red_index,yellow_index)
    # return redirect(url_for('index'))
    # return Response(generate_frames(colorIndex,colors,kernel,bpoints,gpoints,rpoints,ypoints,bkpoints,blue_index,green_index,red_index,yellow_index, black_index),mimetype='multipart/x-mixed-replace; boundary=frame')
    return Response(updated_generate_frames(),mimetype='multipart/x-mixed-replace; boundary=frame')

  
def audioFile():
    fh = open("sample.txt", "r")
    myText = fh.read().replace("\n", " ")

    # Language we want to use 
    language = 'en'

    output = gTTS(text=myText, lang=language, slow=False)

    filename="output"+time.strftime('%Y%m%d%H%M%S')+".mp3"

    output.save(filename)
    fh.close()

    # Play the converted file 
    os.system(filename)

@app.route('/audio')
def audio():
    return render_template('audio.html')    

@app.route('/audio_feed')
def audio_feed():
    return Response(audioFile(),mimetype='multipart/x-mixed-replace; boundary=frame')


if __name__=="__main__":
    print("inside main")
    app.run(debug=True,threaded=False)

import cv2
import numpy as np
import math
"""
Python 3.7.3 + OpenCV 4.0.0
"""
# initialize weight for running average
aWeight = 0.05

# region of interest (ROI) coordinates
top, right, bottom, left = 100, 400, 300, 600
    
# initialize num of frames
num_frames = 0
value = (35, 35)

cap = cv2.VideoCapture(0)
_,init = cap.read()
init = cv2.flip(init, 1)
bg = init[top:bottom, right:left].copy().astype("float")
cv2.imshow('bg',init)
threshold=25

nocount = 0
lastno = False
"""
mhi = np.zeros((bottom-top, left-right), np.float32)
mask = None
orientation = None
retval = 0
"""
cX0, cY0 = None, None
con_mat = [0, 0, 0, 0, 0]

while(cap.isOpened()):
    crtno = False
    # ret returns True if camera is running, frame grabs each frame of the video feed
    ret, frame = cap.read()

    # mirror the frame
    frame = cv2.flip(frame, 1)
    
    # Recognizing skin color
    for i in range(top, bottom):
        for j in range(right, left):
            b,g,r = frame[i,j]
            if(r>95 and g>40 and b>20 and int(max(r,g,b))-int(min(r,g,b))>15 and abs(int(r)-int(g))>15 and r>g and r>b):
                frame[i,j] = [0,0,0] #black
    roi = frame[top:bottom, right:left]

    # convert to grayscale and apply gaussian blur
    gray = cv2.cvtColor(roi, cv2.COLOR_BGR2GRAY)
    blurred = cv2.GaussianBlur(gray, value, 0)

    # calibrate the static background at the beginning
    if num_frames < 30:
        cv2.accumulateWeighted(roi, bg, aWeight)
    elif num_frames == 30:
        print("Initialization Ready!")
    else:
        # display number of frames
        if num_frames % 100 == 30:
            print("Frames: ", (num_frames // 100) * 100)
        if sum(con_mat) % 100 == 0:
            print(con_mat)
        # isolate the hand in region of interest
        # find the absolute difference between background and current frame
        graybg = cv2.cvtColor(init[top:bottom, right:left].copy(), cv2.COLOR_BGR2GRAY)
        blrdbg = cv2.GaussianBlur(graybg, value, 0)
        diff = cv2.absdiff(blrdbg.astype("uint8"), blurred)

        # apply a threshold filter/mask on diff image to get foreground
        thresholded = cv2.threshold(diff, threshold, 255, cv2.THRESH_BINARY)[1]
        # get the contours in the thresholded frame
        cnts, hierarchy = cv2.findContours(thresholded.copy(), cv2.RETR_TREE, cv2.CHAIN_APPROX_NONE)

        if len(cnts) != 0:
            # get the maximum contour which is the hand
            segmented = max(cnts, key=cv2.contourArea)

            # finding convex hull
            hull = cv2.convexHull(segmented)

            # define area of hull and area of hand
            areahull = cv2.contourArea(hull)
            areacnt = cv2.contourArea(segmented)
        
            # find the percentage of area not covered by hand in convex hull
            arearatio=((areahull-areacnt)/areacnt)*100

            # draw contours
            drawing = np.zeros(roi.shape,np.uint8)
            cv2.drawContours(drawing, [segmented], 0, (0, 255, 0), 0)
            cv2.drawContours(drawing, [hull], 0,(0, 0, 255), 0)

            # find convex hull
            hull = cv2.convexHull(segmented, returnPoints=False)

            # find convexity defects
            defects = cv2.convexityDefects(segmented, hull)
            count_defects = 0
            cv2.drawContours(thresholded, cnts, -1, (0, 255, 0), 3)
            thumb_angle = 0

            # find angle between fingers using trig formulas
            for i in range(defects.shape[0]):
                s,e,f,d = defects[i,0]

                start = tuple(segmented[s][0])
                end = tuple(segmented[e][0])
                far = tuple(segmented[f][0])

                # find length for each side of triangle
                a = math.sqrt((end[0] - start[0])**2 + (end[1] - start[1])**2)
                b = math.sqrt((far[0] - start[0])**2 + (far[1] - start[1])**2)
                c = math.sqrt((end[0] - far[0])**2 + (end[1] - far[1])**2)

                # apply cosine rule to find angle
                angle = math.acos((b**2 + c**2 - a**2)/(2*b*c)) * 57

                # ignore angles > 105 and highlight rest with red dots
                if angle <= 110:
                    count_defects += 1
                    cv2.circle(roi, far, 5, [0,0,255], -1)
                #dist = cv2.pointPolygonTest(cnt,far,True)

                # draw lines from start to end i.e. the convex points (finger tips)
                cv2.line(roi,start, end, [0,255,0], 2)
                #cv2.circle(roi,far,5,[0,0,255],-1)

            # update text and display based on gesture detected
            if count_defects == 0 and arearatio<25:
                cv2.putText(frame,"Rock", (right, top-5), cv2.FONT_HERSHEY_SIMPLEX, 1, (0,0,0), 2)
                """con_mat[0] += 1
                if con_mat[0] == 1:
                    cv2.imwrite('trial00a.png',frame[top:bottom, right:left])
                    cv2.imwrite('trial00b.png',thresholded)
                    print("Mark0", arearatio, count_defects)
                """
            elif (count_defects == 2 or count_defects == 1):
                if arearatio > 20:
                    cv2.putText(frame,"L", (right, top-5), cv2.FONT_HERSHEY_SIMPLEX, 1, (0,0,0), 2)
                    """con_mat[1] += 1
                    if con_mat[1] == 1:
                        cv2.imwrite('trial01a.png',frame[top:bottom, right:left])
                        cv2.imwrite('trial01b.png',thresholded)
                        print("Mark1", arearatio, count_defects)
                    """
                else:
                    cv2.putText(frame,"Scissors", (right, top-5), cv2.FONT_HERSHEY_SIMPLEX, 1, (0,0,0), 2)
                    """
                    con_mat[2] += 1
                    if con_mat[2] == 1:
                        cv2.imwrite('trial02a.png',frame[top:bottom, right:left])
                        cv2.imwrite('trial02b.png',thresholded)
                        print("Mark2", arearatio, count_defects)
                    """
            elif count_defects == 4:
                cv2.putText(frame,"Paper", (right, top-5), cv2.FONT_HERSHEY_SIMPLEX, 1, (0,0,0), 2)
                """con_mat[3] += 1
                if con_mat[3] == 1:
                    cv2.imwrite('trial03a.png',frame[top:bottom, right:left])
                    cv2.imwrite('trial03b.png',thresholded)
                    print("Mark3", arearatio, count_defects)
                """
            elif count_defects == 4 and arearatio>18: # dynamic gesture - waving
                crtno = True
                if nocount == 0:
                    nocount += 1

                    #mhi = cv2.motempl.updateMotionHistory(thresholded, mhi, num_frames, 5)
                    #mask, orientation = cv2.motempl.calcMotionGradient(mhi, 0.05, 0.25, apertureSize=5)
                    #retval = cv2.motempl.calcGlobalOrientation(orientation, mask, mhi, num_frames, 5)

                    for c in cnts:
                        # calculate moments for each contour
                        M = cv2.moments(c)
                            
                        # calculate x,y coordinate of center
                        if M["m00"] != 0:
                            cX0 = int(M["m10"] / M["m00"])
                            cY0 = int(M["m01"] / M["m00"])
                            cv2.circle(frame, (cX0, cY0), 5, (255, 255, 255), -1)
                elif nocount < 5:
                    if lastno:
                        nocount += 1
                    else:
                        nocount = 0
                else:
                    for c in cnts:
                        # calculate moments for each contour
                        M = cv2.moments(c)
                            
                        # calculate x,y coordinate of center
                        if M["m00"] != 0:
                            cX1 = int(M["m10"] / M["m00"])
                            cY1 = int(M["m01"] / M["m00"])
                            cv2.circle(frame, (cX1, cY1), 5, (255, 255, 255), -1)
                    if math.sqrt((cX1-cX0)**2 + (cY1-cY0)**2) > 20:
                        cv2.putText(frame,"No", (right, top-5), cv2.FONT_HERSHEY_SIMPLEX, 1, (0,0,0), 2) 
                        #con_mat[4] += 1                     
                    nocount = 0
                    lastno = False            
            else:
                cv2.putText(frame,"Lavi's Cam", (right, top-5), cv2.FONT_HERSHEY_SIMPLEX, 1, (0,0,0), 2)
                """con_mat[4] += 1
                if con_mat[4] == 1:
                    cv2.imwrite('trial04a.png',frame[top:bottom, right:left])
                    cv2.imwrite('trial04b.png',thresholded)
                    print("Mark4", arearatio, count_defects)
                """

            
            #print(nocount, retval)
            #print(arearatio, count_defects)
            # draw the segmented region and display the frame after thresholding
            cv2.drawContours(frame, [segmented + (right, top)], -1, (0, 0, 255))
            cv2.imshow("Thesholded", thresholded)
            #lastno = crtno
            if num_frames <= 35: 
                    cv2.imwrite('trial00a.png',frame[top:bottom, right:left])
                    cv2.imwrite('trial00b.png',thresholded)
                    print("Mark0", arearatio, count_defects)
    # draw the ROI
    cv2.rectangle(frame, (left, top), (right, bottom), (0,255,0), 2)
    cv2.imshow('frame',frame)


    num_frames += 1

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()



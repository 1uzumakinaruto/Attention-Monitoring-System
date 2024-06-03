import numpy as np
import argparse
import imutils
import time
import cv2
import os
import cv2
import numpy as np
import math
from face_detector import get_face_detector, find_faces
from face_landmarks import get_landmark_model, detect_marks
import datetime
import time

def checkProctoring():
    #headpose
    #-----------------------------------------------------------------------------------
    def get_2d_points(img, rotation_vector, translation_vector, camera_matrix, val):
        """Return the 3D points present as 2D for making annotation box"""
        point_3d = []
        dist_coeffs = np.zeros((4,1))
        rear_size = val[0]
        rear_depth = val[1]
        point_3d.append((-rear_size, -rear_size, rear_depth))
        point_3d.append((-rear_size, rear_size, rear_depth))
        point_3d.append((rear_size, rear_size, rear_depth))
        point_3d.append((rear_size, -rear_size, rear_depth))
        point_3d.append((-rear_size, -rear_size, rear_depth))
        
        front_size = val[2]
        front_depth = val[3]
        point_3d.append((-front_size, -front_size, front_depth))
        point_3d.append((-front_size, front_size, front_depth))
        point_3d.append((front_size, front_size, front_depth))
        point_3d.append((front_size, -front_size, front_depth))
        point_3d.append((-front_size, -front_size, front_depth))
        point_3d = np.array(point_3d, dtype=np.float64).reshape(-1, 3)
        
        # Map to 2d img points
        (point_2d, _) = cv2.projectPoints(point_3d,
                                            rotation_vector,
                                            translation_vector,
                                            camera_matrix,
                                            dist_coeffs)
        point_2d = np.int32(point_2d.reshape(-1, 2))
        return point_2d

    def draw_annotation_box(img, rotation_vector, translation_vector, camera_matrix,
                            rear_size=300, rear_depth=0, front_size=500, front_depth=400,
                            color=(255, 255, 0), line_width=2):
        """
        Draw a 3D anotation box on the face for head pose estimation

        Parameters
        ----------
        img : np.unit8
            Original Image.
        rotation_vector : Array of float64
            Rotation Vector obtained from cv2.solvePnP
        translation_vector : Array of float64
            Translation Vector obtained from cv2.solvePnP
        camera_matrix : Array of float64
            The camera matrix
        rear_size : int, optional
            Size of rear box. The default is 300.
        rear_depth : int, optional
            The default is 0.
        front_size : int, optional
            Size of front box. The default is 500.
        front_depth : int, optional
            Front depth. The default is 400.
        color : tuple, optional
            The color with which to draw annotation box. The default is (255, 255, 0).
        line_width : int, optional
            line width of lines drawn. The default is 2.

        Returns
        -------
        None.

        """
        
        rear_size = 1
        rear_depth = 0
        front_size = img.shape[1]
        front_depth = front_size*2
        val = [rear_size, rear_depth, front_size, front_depth]
        point_2d = get_2d_points(img, rotation_vector, translation_vector, camera_matrix, val)
        # # Draw all the lines
        cv2.polylines(img, [point_2d], True, color, line_width, cv2.LINE_AA)
        cv2.line(img, tuple(point_2d[1]), tuple(
            point_2d[6]), color, line_width, cv2.LINE_AA)
        cv2.line(img, tuple(point_2d[2]), tuple(
            point_2d[7]), color, line_width, cv2.LINE_AA)
        cv2.line(img, tuple(point_2d[3]), tuple(
            point_2d[8]), color, line_width, cv2.LINE_AA)
        
        
    def head_pose_points(img, rotation_vector, translation_vector, camera_matrix):
        """
        Get the points to estimate head pose sideways    

        Parameters
        ----------
        img : np.unit8
            Original Image.
        rotation_vector : Array of float64
            Rotation Vector obtained from cv2.solvePnP
        translation_vector : Array of float64
            Translation Vector obtained from cv2.solvePnP
        camera_matrix : Array of float64
            The camera matrix

        Returns
        -------
        (x, y) : tuple
            Coordinates of line to estimate head pose

        """
        rear_size = 1
        rear_depth = 0
        front_size = img.shape[1]
        front_depth = front_size*2
        val = [rear_size, rear_depth, front_size, front_depth]
        point_2d = get_2d_points(img, rotation_vector, translation_vector, camera_matrix, val)
        y = (point_2d[5] + point_2d[8])//2
        x = point_2d[2]
        
        return (x, y)
        
    face_model = get_face_detector()
    landmark_model = get_landmark_model()
    cap = cv2.VideoCapture(0)
    ret, img = cap.read()
    cap.release()
    size = img.shape
    font = cv2.FONT_HERSHEY_SIMPLEX 
    # 3D model points.
    model_points = np.array([
                                (0.0, 0.0, 0.0),             # Nose tip
                                (0.0, -330.0, -65.0),        # Chin
                                (-225.0, 170.0, -135.0),     # Left eye left corner
                                (225.0, 170.0, -135.0),      # Right eye right corne
                                (-150.0, -150.0, -125.0),    # Left Mouth corner
                                (150.0, -150.0, -125.0)      # Right mouth corner
                            ])

    # Camera internals
    focal_length = size[1]
    center = (size[1]/2, size[0]/2)
    camera_matrix = np.array(
                                [[focal_length, 0, center[0]],
                                [0, focal_length, center[1]],
                                [0, 0, 1]], dtype = "double"
                                )


    # cell phone 
    #-------------------------------------------------------------                            
    # construct the argument parse and parse the arguments
    ap = argparse.ArgumentParser()
    ap.add_argument("-c", "--confidence", type=float, default=0.5,
            help="minimum probability to filter weak detections")
    ap.add_argument("-t", "--threshold", type=float, default=0.3,
            help="threshold when applyong non-maxima suppression")
    args = vars(ap.parse_args())

    # load the COCO class labels our YOLO model was trained on
    labelsPath = os.path.sep.join(["models/coco.names"])
    LABELS = open(labelsPath).read().strip().split("\n")

    # initialize a list of colors to represent each possible class label
    np.random.seed(42)
    COLORS = np.random.randint(0, 255, size=(len(LABELS), 3),
            dtype="uint8")

    # derive the paths to the YOLO weights and model configuration
    weightsPath = os.path.sep.join(["models/yolov3.weights"])
    configPath = os.path.sep.join(["models/yolov3.cfg"])

    net = cv2.dnn.readNetFromDarknet(configPath, weightsPath)
    ln = net.getLayerNames()
    ln = [ln[i - 1] for i in net.getUnconnectedOutLayers()]

    (W, H) = (None, None)

    WITDH = 450
    HEIGHT = 350

    print("Streaming started............")
    cap = cv2.VideoCapture(0)

    cell_counts = 0
    headup = 0
    headdown = 0
    headleft = 0
    headright = 0
    headcenter = 0

    while(True):
        ret, frame = cap.read()
        frame = cv2.resize(frame, (WITDH, HEIGHT))
        
        #headpose
        #-----------------------------------------------------------------------------
        img = frame.copy()
        faces = find_faces(img, face_model)
        for face in faces:
            marks = detect_marks(img, landmark_model, face)
            # mark_detector.draw_marks(img, marks, color=(0, 255, 0))
            image_points = np.array([
                                    marks[30],     # Nose tip
                                    marks[8],     # Chin
                                    marks[36],     # Left eye left corner
                                    marks[45],     # Right eye right corne
                                    marks[48],     # Left Mouth corner
                                    marks[54]      # Right mouth corner
                                ], dtype="double")
            dist_coeffs = np.zeros((4,1)) # Assuming no lens distortion
            (success, rotation_vector, translation_vector) = cv2.solvePnP(model_points, image_points, camera_matrix, dist_coeffs, flags=cv2.SOLVEPNP_UPNP)
            
            
            # Project a 3D point (0, 0, 1000.0) onto the image plane.
            # We use this to draw a line sticking out of the nose
            
            (nose_end_point2D, jacobian) = cv2.projectPoints(np.array([(0.0, 0.0, 1000.0)]), rotation_vector, translation_vector, camera_matrix, dist_coeffs)
            
            for p in image_points:
                cv2.circle(img, (int(p[0]), int(p[1])), 3, (0,0,255), -1)
            
            
            p1 = ( int(image_points[0][0]), int(image_points[0][1]))
            p2 = ( int(nose_end_point2D[0][0][0]), int(nose_end_point2D[0][0][1]))
            x1, x2 = head_pose_points(img, rotation_vector, translation_vector, camera_matrix)

            cv2.line(img, p1, p2, (0, 255, 255), 2)
            cv2.line(img, tuple(x1), tuple(x2), (255, 255, 0), 2)
            # for (x, y) in marks:
            #     cv2.circle(img, (x, y), 4, (255, 255, 0), -1)
            # cv2.putText(img, str(p1), p1, font, 1, (0, 255, 255), 1)
            try:
                m = (p2[1] - p1[1])/(p2[0] - p1[0])
                ang1 = int(math.degrees(math.atan(m)))
            except:
                ang1 = 90
                
            try:
                m = (x2[1] - x1[1])/(x2[0] - x1[0])
                ang2 = int(math.degrees(math.atan(-1/m)))
            except:
                ang2 = 90
                
                # print('div by zero error')
            if ang1 >= 48:
                print('Head down')
                headdown += 1
                ts=time.time()
                date=datetime.datetime.fromtimestamp(ts).strftime('%y-%m-%d')
                timeStamp=datetime.datetime.fromtimestamp(ts).strftime('%H:%M:%S')
                Hour,Minute,Second=timeStamp.split(":")
                cv2.imwrite('dataset/head down'+date+'.'+Hour+'.'+Minute+'.'+Second+'.jpg', img)
                cv2.putText(img, 'Head down', (20, 50), font, 0.5, (255, 255, 128), 2)
            elif ang1 <= -48:
                print('Head up')
                headup += 1
                ts=time.time()
                date=datetime.datetime.fromtimestamp(ts).strftime('%y-%m-%d')
                timeStamp=datetime.datetime.fromtimestamp(ts).strftime('%H:%M:%S')
                Hour,Minute,Second=timeStamp.split(":")
                cv2.imwrite('dataset/head up'+date+'.'+Hour+'.'+Minute+'.'+Second+'.jpg', img)
                cv2.putText(img, 'Head up', (20, 50), font, 0.5, (255, 255, 128), 2)
                
            if ang2 >= 48:
                print('Head right')
                headright += 1
                ts=time.time()
                date=datetime.datetime.fromtimestamp(ts).strftime('%y-%m-%d')
                timeStamp=datetime.datetime.fromtimestamp(ts).strftime('%H:%M:%S')
                Hour,Minute,Second=timeStamp.split(":")
                cv2.imwrite('dataset/head right'+date+'.'+Hour+'.'+Minute+'.'+Second+'.jpg', img)
                cv2.putText(img, 'Head right', (20, 50), font, 0.5, (255, 255, 128), 2)
            elif ang2 <= -48:
                print('Head left')
                headleft += 1
                ts=time.time()
                date=datetime.datetime.fromtimestamp(ts).strftime('%y-%m-%d')
                timeStamp=datetime.datetime.fromtimestamp(ts).strftime('%H:%M:%S')
                Hour,Minute,Second=timeStamp.split(":")
                cv2.imwrite('dataset/head left'+date+'.'+Hour+'.'+Minute+'.'+Second+'.jpg', img)
                cv2.putText(img, 'Head left', (20, 50), font, 0.5, (255, 255, 128), 2)
            else:
                print('Head center')
                headcenter += 1
                ts=time.time()
                date=datetime.datetime.fromtimestamp(ts).strftime('%y-%m-%d')
                timeStamp=datetime.datetime.fromtimestamp(ts).strftime('%H:%M:%S')
                Hour,Minute,Second=timeStamp.split(":")
                cv2.imwrite('dataset/head center'+date+'.'+Hour+'.'+Minute+'.'+Second+'.jpg', img)
                cv2.putText(img, 'Head center', (20, 50), font, 0.5, (255, 255, 128), 2)
            
            cv2.putText(img, str(ang1), tuple(p1), font, 2, (128, 255, 255), 2)
            cv2.putText(img, str(ang2), tuple(x1), font, 2, (255, 255, 128), 2)
            
        cv2.putText(img, 'Head pose detection', (20, 20), font, 0.5, (255, 255, 128), 2)

        #person and phone 
        #----------------------------------------------------------------
        img4 = frame.copy()
        img5 = frame.copy()
        
        if W is None or H is None:
                (H, W) = frame.shape[:2]

        # construct a blob from the input frame and then perform a forward
        # pass of the YOLO object detector, giving us our bounding boxes
        # and associated probabilities
        blob = cv2.dnn.blobFromImage(frame, 1 / 255.0, (416, 416),
                swapRB=True, crop=False)
        net.setInput(blob)
        start = time.time()
        layerOutputs = net.forward(ln)
        end = time.time()

        # initialize our lists of detected bounding boxes, confidences,
        # and class IDs, respectively
        boxes = []
        confidences = []
        classIDs = []

        # loop over each of the layer outputs
        for output in layerOutputs:
                # loop over each of the detections
                for detection in output:
                        # extract the class ID and confidence (i.e., probability)
                        # of the current object detection
                        scores = detection[5:]
                        classID = np.argmax(scores)
                        confidence = scores[classID]

                        # filter out weak predictions by ensuring the detected
                        # probability is greater than the minimum probability
                        if confidence > args["confidence"]:
                                # scale the bounding box coordinates back relative to
                                # the size of the image, keeping in mind that YOLO
                                # actually returns the center (x, y)-coordinates of
                                # the bounding box followed by the boxes' width and
                                # height
                                box = detection[0:4] * np.array([W, H, W, H])
                                (centerX, centerY, width, height) = box.astype("int")

                                # use the center (x, y)-coordinates to derive the top
                                # and and left corner of the bounding box
                                x = int(centerX - (width / 2))
                                y = int(centerY - (height / 2))

                                # update our list of bounding box coordinates,
                                # confidences, and class IDs
                                boxes.append([x, y, int(width), int(height)])
                                confidences.append(float(confidence))
                                classIDs.append(classID)

        # apply non-maxima suppression to suppress weak, overlapping
        # bounding boxes
        idxs = cv2.dnn.NMSBoxes(boxes, confidences, args["confidence"],
                args["threshold"])

        # ensure at least one detection exists
        count=0
        if len(idxs) > 0:
                # loop over the indexes we are keeping
                for i in idxs.flatten():
                        # extract the bounding box coordinates
                        (x, y) = (boxes[i][0], boxes[i][1])
                        (w, h) = (boxes[i][2], boxes[i][3])

                        obj = "{}".format(LABELS[classIDs[i]])
                        if obj == 'cell phone':
                            cell_counts += 1
                            # draw a bounding box rectangle and label on the frame
                            ts=time.time()
                            date=datetime.datetime.fromtimestamp(ts).strftime('%y-%m-%d')
                            timeStamp=datetime.datetime.fromtimestamp(ts).strftime('%H:%M:%S')
                            Hour,Minute,Second=timeStamp.split(":")
                            cv2.imwrite('dataset/phone'+date+'.'+Hour+'.'+Minute+'.'+Second+'.jpg', img5)
                            color = [int(c) for c in COLORS[classIDs[i]]]
                            cv2.rectangle(img5, (x, y), (x + w, y + h), color, 2)
                            cv2.putText(img5, obj, (x, y - 5),
                                    font, 0.5, color, 2)

        cv2.putText(img5, 'Cell phone detection', (20, 20), font, 0.5, (255, 255, 128), 2)
        row1_frame = np.concatenate((img, img5), axis=1)
        cv2.imshow('result', row1_frame)
        if cv2.waitKey(1) & 0xFF == ord('q'):
            cap.release()
            cv2.destroyAllWindows()
            break

    return cell_counts, headup, headdown, headleft, headright, headcenter
from flask import Flask,Blueprint, render_template
from flask_socketio import SocketIO
from threading import Thread

import cv2
import mediapipe as mp
import numpy as np
mp_drawing = mp.solutions.drawing_utils
mp_pose = mp.solutions.pose

app = Flask(__name__, template_folder='.')
socketio = SocketIO(app)

# Blueprint for 'js' static files
kineval_bp = Blueprint('kineval', __name__, static_url_path='/kineval', static_folder='kineval')
js_bp = Blueprint('js', __name__, static_url_path='/js', static_folder='js')
robots_bp = Blueprint('robots', __name__, static_url_path='/robots', static_folder='robots')
worlds_bp = Blueprint('worlds', __name__, static_url_path='/worlds', static_folder='worlds')

# Register blueprints with the app
app.register_blueprint(kineval_bp)
app.register_blueprint(js_bp)
app.register_blueprint(robots_bp)
app.register_blueprint(worlds_bp)

@app.route('/')
def index():
    return render_template('home.html')

def pose_detect():
    try:
        cap = cv2.VideoCapture(0)
        ## Setup mediapipe instance
        with mp_pose.Pose(min_detection_confidence=0.5, min_tracking_confidence=0.5) as pose:
            while cap.isOpened():
                ret, frame = cap.read()
                # Uncomment below to make the camera act like a mirror.
                # frame = cv2.flip(frame,1)
                
                # Recolor image to RGB
                image = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                image.flags.writeable = False
            
                # Make detection
                results = pose.process(image)
            
                # Recolor back to BGR
                image.flags.writeable = True
                image = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)
                
                # Extract landmarks
                try:
                    ### Changes by Mohit
                    # Get the landmarks in World Frame.
                    angles = make_angles(results.pose_world_landmarks.landmark)
                    socketio.emit('angles', angles)
                except Exception as e:
                    print(e)
                
                # Render detections
                mp_drawing.draw_landmarks(image, results.pose_landmarks, mp_pose.POSE_CONNECTIONS,
                                        mp_drawing.DrawingSpec(color=(245,117,66), thickness=2, circle_radius=2), 
                                        mp_drawing.DrawingSpec(color=(245,66,230), thickness=2, circle_radius=2) 
                                        )               
                # Set the window size
                cv2.namedWindow('Mediapipe Feed', cv2.WINDOW_NORMAL)
                cv2.resizeWindow('Mediapipe Feed', 620, 340)  # Set the desired size (width, height)
                cv2.imshow('Mediapipe Feed', image)

                if cv2.waitKey(10) & 0xFF == ord('q'):
                    break

            cv2.destroyAllWindows()
            cv2.waitKey(1)
            cap.release()
    
    except Exception as e:
        print(e)

def make_angles(landmarks):
    # Store the result in a numpy array, each row is a point and columns are x,y,z.
    data_points = np.empty((33,3))
    for i,point in enumerate(landmarks):
        data_points[i] = point.x, point.y, point.z

    # Rot Matrix for making the data in proper coordinate system.
    rot_mat_x = np.array([[1,0,0],[0,0,1],[0,-1,0]])
    rot_mat_z = np.array([[-1,0,0],[0,-1,0],[0,0,1]])
    rot_mat = np.matmul(rot_mat_z, rot_mat_x)

    # Rotate all the points in proper coordinate system.
    rotated_data = np.empty((33,3))
    for i,point in enumerate(data_points):
        rotated_data[i] = np.matmul(rot_mat, np.atleast_2d(point).T).T

    # Storing the points which are used in joint angle calculations in a dict.
    points={}
    points['N']  = rotated_data[0]      # Nose
    points['Sl'] = rotated_data[11]     # Shoulder Left
    points['Sr'] = rotated_data[12]     # Shoulder Right
    points['El'] = rotated_data[13]     # Elbow Left
    points['Er'] = rotated_data[14]     # Elbow Right
    points['Wl'] = rotated_data[15]     # Wrist Left
    points['Wr'] = rotated_data[16]     # Wrist Right
    points['Il'] = rotated_data[19]     # Index Left
    points['Ir'] = rotated_data[20]     # Index Right

    # Move the points to origin.

    # Find points s0.
    s0 = np.array([(points['Sl'][0] + points['Sr'][0])/2,
                (points['Sl'][1] + points['Sr'][1])/2,
                (points['Sl'][2] + points['Sr'][2])/2])

    points['S0'] = list(s0)
    # print(points)
    # Make s0 the origin in points i.e., move the points to this center.
    for key, value in points.items():
        points[key] = [value[0]- s0[0], value[1]- s0[1], value[2]- s0[2]]

    # Get the angle between sholder and the know plane.

    # Move the shoulder point to s0 plane.
    Sl = points['Sl']
    # print(Sl)

    # Sl point projected in xy plane.
    Sl_xy = [Sl[0],Sl[1],0]

    # Now find the angle between two vectors.
    # Angle between the vector Sl_S0 and  -x axis is the required angle.
    # angle_sholder_z = np.arccos((np.dot(Sl_xy, [-1,0,0])) / (np.linalg.norm(Sl_xy)*(np.linalg.norm([-1,0,0]))))
    angle_sholder_z = np.arctan2(np.linalg.det([Sl_xy[0:2],[-1,0]]), np.dot(Sl_xy[0:2],[-1,0]))

    # Rotate the points about z axis by this angle.
    # Rotation matrix about z axis: [cos,sin,0], [-sin, cos, 0], [0,0,1]
    rot_mat_z_shoulder =[[np.cos(angle_sholder_z),np.sin(angle_sholder_z),0],
                            [-1*np.sin(angle_sholder_z), np.cos(angle_sholder_z) ,0],
                            [0,0,1]]
    # Rotate the points.
    for key, value in points.items():
        points[key] = np.matmul(rot_mat_z_shoulder, np.atleast_2d(value).T).T[0]
    
    # Get points from dictionary for calculations.
    N = points['N']
    Sl = points['Sl']
    Sr = points['Sr']
    El = points['El']
    Er = points['Er']
    Wl = points['Wl']
    Wr = points['Wr']
    Il = points['Il']
    Ir = points['Ir']

    # Angle Calculations for Right Shoulder joint's movement in horizontal plane i,e. right_s0 joint in baxter.
    # Reference Vector. See notes.
    RSr = [1,1,0]

    # Vector Er_Sr in xy plane.
    Er_Sr_xy = [Er[0]-Sr[0], Er[1]-Sr[1], 0]

    # Angle with Sign. Here det is directional.
    right_s0_angle = np.arctan2(np.linalg.det([RSr[0:2], Er_Sr_xy[0:2]]), np.dot(RSr, Er_Sr_xy))
    # Initialize the dictionary of angles, which will be returned at the end.
    angles={'right_s0' : right_s0_angle}

    # Same calculations for left shoulder i.e., left_s0 joint of baxter.
    # Reference Vector. See notes.
    RSl = [-1,1,0]

    # Vector El_Sl in xy plane.
    El_Sl_xy = [El[0]-Sl[0], El[1]-Sl[1], 0]

    # Angle with Sign. Here det is directional.
    left_s0_angle = np.arctan2(np.linalg.det([RSl[0:2], El_Sl_xy[0:2]]), np.dot(RSl, El_Sl_xy))
    angles['left_s0'] = left_s0_angle

    # Angle Calculations for Right Shoulder joint's movement in Verticle plane i,e. right_s1 joint in baxter.
    ## It is not possible to get directional angle of two 3D points we can give the direction here manually and just get the angle by the points.

    # Vector pointing to Er base at Sr
    Er_Sr = [ Er[0] - Sr[0], Er[1] - Sr[1], Er[2] - Sr[2]]

    # Projection of vector Er_Sr in xy(horizontal) plane.
    Er_Sr_xy = [ Er_Sr[0], Er_Sr[1], 0]

    # Calculate the angle between veotors.
    right_s1_angle = np.arctan2(np.linalg.norm(np.cross(Er_Sr, Er_Sr_xy)), np.dot(Er_Sr, Er_Sr_xy))
    # Correct the direction of angle.
    if Er_Sr[2] >= 0:
        right_s1_angle = -1*right_s1_angle
    angles['right_s1'] = right_s1_angle

    # Same for left side i.e., left_s1 joint in baxter.
    # Vector pointing to El base at Sl.
    El_Sl = [ El[0] - Sl[0], El[1] - Sl[1], El[2] - Sl[2]]

    # Projection of vector El_Sl in xy(horizontal) plane.
    El_Sl_xy = [ El_Sl[0] - Sl[0], El_Sl[1] - Sl[1], 0]

    # Calculate the angle between veotors.
    left_s1_angle = np.arctan2(np.linalg.norm(np.cross(El_Sl, El_Sl_xy)), np.dot(El_Sl, El_Sl_xy))
    if El_Sl[2] >= 0:
        left_s1_angle = -1*left_s1_angle
    angles['left_s1'] = left_s1_angle

    # Angle of Right Elbow joint, i.e., right_e1 joint in baxter.

    # Make the required Vectors.
    Er_Sr = [Er[0]-Sr[0], Er[1]-Sr[1], Er[2]-Sr[2]]
    Wr_Er = [Wr[0]-Er[0], Wr[1]-Er[1], Wr[2]-Er[2]]

    # Angle between the above vectors.
    right_e1_angle = np.arccos(np.dot(Er_Sr, Wr_Er) / (np.linalg.norm(Er_Sr) * np.linalg.norm(Wr_Er)))
    angles['right_e1'] = right_e1_angle

    # Angle of Left Elbow joint, i.e., left_e1 joint in baxter.
    El_Sl = [El[0]-Sl[0], El[1]-Sl[1], El[2]-Sl[2]]
    Wl_El = [Wl[0]-El[0], Wl[1]-El[1], Wl[2]-El[2]]

    left_e1_angle = np.arccos(np.dot(El_Sl, Wl_El) / (np.linalg.norm(El_Sl) * np.linalg.norm(Wl_El)))
    angles['left_e1'] = left_e1_angle

    # Calculate the angle for Head Pan, i.e., headpan joint of baxter.

    # Project the nose point on the xy(Horizontal) Plane.
    # Taking 2D Vector only, as angle calculation is in a plane.
    N_xy = [N[0], N[1]]

    # Find the angle between Vector N_xy and the y-axis will give us the required angle.
    # Angle with Sign. Here det is directional.
    headpan_angle = np.arctan2(np.linalg.det([[0,1],N_xy]), np.dot([0,1], N_xy))
    angles['headpan'] = headpan_angle

    # Calculation of angle of right wrist i.e., right_w1 joint of baxter.

    # Make the required Vectors.
    Wr_Er = [Wr[0]-Er[0], Wr[1]-Er[1], Wr[2]-Er[2]]
    Ir_Wr = [Ir[0]-Wr[0], Ir[1]-Wr[1], Ir[2]-Wr[2]]

    right_w1_angle = np.arccos(np.dot(Wr_Er, Ir_Wr) / (np.linalg.norm(Wr_Er) * np.linalg.norm(Ir_Wr)))
    angles['right_w1'] = right_w1_angle

    # Angle of Left Wrist joint, i.e., left_w1 joint in baxter.
    Wl_El = [Wl[0]-El[0], Wl[1]-El[1], Wl[2]-El[2]]
    Il_Wl = [Il[0]-Wl[0], Il[1]-Wl[1], Il[2]-Wl[2]]

    left_w1_angle = np.arccos(np.dot(Wl_El, Il_Wl) / (np.linalg.norm(Wl_El) * np.linalg.norm(Il_Wl)))
    angles['left_w1'] = left_e1_angle

    # Calculations for rotation of shoulder joint on it's axis i.e., right_e0 joint of baxter.

    # We need a point that is in a plane which passes through Er, Sr and the plane is parellel to Z-axis.
    # Calculate distance between Er and Sr 
    dist_Er_Sr = np.linalg.norm(Er - Sr)
    P_Er_Sr_Z = [(Er[0]+Sr[0])/2, (Er[1]+Sr[1])/2, ((Er[2]+Sr[2])/2) + 2 * dist_Er_Sr]

    # Find the normal vector to the aforementioned plane.
    norm_Er_Sr_Z = np.cross(P_Er_Sr_Z - Sr, Er - Sr)

    # Now we find the normal to the Plane which passes through Sr, Er and Wr.
    norm_Er_Sr_Wr = np.cross(Wr - Sr, Er - Sr)

    # Angle between these two vectors is the angle of joint.
    right_e0_angle = np.arccos(np.dot(norm_Er_Sr_Z, norm_Er_Sr_Wr) / (np.linalg.norm(norm_Er_Sr_Z) * np.linalg.norm(norm_Er_Sr_Wr)))
    angles['right_e0'] = 3.05 - right_e0_angle

    # Doing the same for left side, left_e0 joint.

    # We need a point that is in a plane which passes through El, Sl and the plane is parellel to Z-axis.
    # Calculate distance between El and Sl 
    dist_El_Sl = np.linalg.norm(El - Sl)
    P_El_Sl_Z = [(El[0]+Sl[0])/2, (El[1]+Sl[1])/2, ((El[2]+Sl[2])/2) + 2 * dist_El_Sl]

    # Find the normal vector to the aforementioned plane.
    norm_El_Sl_Z = np.cross(P_El_Sl_Z - Sl, El - Sl)

    # Now we find the normal to the Plane which passes through Sl, El and Wl.
    norm_El_Sl_Wl = np.cross(Wl - Sl, El - Sl)

    # Angle between these two vectors is the angle of joint.
    left_e0_angle = np.arccos(np.dot(norm_El_Sl_Z, norm_El_Sl_Wl) / (np.linalg.norm(norm_El_Sl_Z) * np.linalg.norm(norm_El_Sl_Wl)))
    angles['left_e0'] = left_e0_angle - 3.05

    return angles

@socketio.on('connect')
def handle_connect():
    print('Client connected')

def run_flask():
    socketio.run(app, port=8000, debug=True, use_reloader=False)

if __name__ == '__main__':
    flask_thread = Thread(target=run_flask)
    flask_thread.start()
    pose_detect()

    
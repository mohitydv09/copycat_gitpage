import {
    PoseLandmarker,
    FilesetResolver,
    DrawingUtils
  } from 'https://cdn.skypack.dev/@mediapipe/tasks-vision@0.10.14';
  
const video = document.getElementById('webcam');
const canvas = document.getElementById('mp-canvas');
const canvasCtx = canvas.getContext('2d');
const drawingUtils = new DrawingUtils(canvasCtx);

let poseLandmarker;
let runningMode = "VIDEO";
const videoHeight = "360px";
const videoWidth = "480px";
let firstRun = true;

// add a delay of 2 seconds to this code.

const createPoseLandmarker = async () => {
    const vision = await FilesetResolver.forVisionTasks(
        'https://cdn.jsdelivr.net/npm/@mediapipe/tasks-vision@0.10.14/wasm'
    );
    poseLandmarker = await PoseLandmarker.createFromOptions(vision, {
        baseOptions: { 
            modelAssetPath: 'https://storage.googleapis.com/mediapipe-models/pose_landmarker/pose_landmarker_lite/float16/1/pose_landmarker_lite.task',
            delegate: "GPU",
        },
        runningMode: runningMode,
        numPoses: 1
    });
}
createPoseLandmarker();

let lastVideoTime = -1;
async function predictWebcam() {
    canvas.style.width = videoWidth;
    video.style.width = videoWidth;
    canvas.style.height = videoHeight;
    video.style.height = videoHeight;

    if(lastVideoTime !== video.currentTime) {
        lastVideoTime = video.currentTime;
        poseLandmarker.detectForVideo(video, performance.now(), (result) => {
            if(!result.worldLandmarks || !result.worldLandmarks[0]){
                return;
            }
            kineval.params.pose = makeAngles(result.worldLandmarks[0]); 
            if (firstRun) {
                kineval.params.persist_pd = true;
                kineval.params.persist_pd_dance = true;
                kineval.params.update_pd_dance = true;
                firstRun = false;
            }
            canvasCtx.save();
            canvasCtx.clearRect(0, 0, canvas.width, canvas.height);
            canvasCtx.translate(canvas.width, 0);
            canvasCtx.scale(-1, 1);
            for (const landmark of result.landmarks){
                drawingUtils.drawLandmarks(landmark, {
                    radius: (data) => DrawingUtils.lerp(data.from.z, -0.15, 0.1, 5, 1)
                });
                drawingUtils.drawConnectors(landmark, PoseLandmarker.POSE_CONNECTIONS);
            }
            canvasCtx.restore();
        });
    }
    window.requestAnimationFrame(predictWebcam);
};

const main = async () => {

    navigator.mediaDevices.getUserMedia({ video: true }).then((stream) => {
        video.srcObject = stream;
        video.addEventListener('loadeddata', () => {
            createPoseLandmarker().then(() => {
                predictWebcam();
            });
        });
    }).catch((error) => {
        console.error('Error accessing webcam: ', error);
    });
}

function makeAngles(landmarks) {
    // Convert landmarks to data_points
    let dataPoints = math.matrix(landmarks.map(point => [point.x, point.y, point.z]));

    // Rot Matrix for making the data in proper coordinate system
    let rotMatX = math.matrix([[1, 0, 0], [0, 0, 1], [0, -1, 0]]);
    let rotMatZ = math.matrix([[-1, 0, 0], [0, -1, 0], [0, 0, 1]]);
    let rotMat = math.multiply(rotMatZ, rotMatX);

    // Rotate all the points in proper coordinate system
    let rotatedData = math.multiply(dataPoints, math.transpose(rotMat));

    // Storing the points which are used in joint angle calculations in a dictionary
    let points = {
        'N': rotatedData.subset(math.index(0, math.range(0, 3))),      // Nose
        'Sl': rotatedData.subset(math.index(11, math.range(0, 3))),    // Shoulder Left
        'Sr': rotatedData.subset(math.index(12, math.range(0, 3))),    // Shoulder Right
        'El': rotatedData.subset(math.index(13, math.range(0, 3))),    // Elbow Left
        'Er': rotatedData.subset(math.index(14, math.range(0, 3))),    // Elbow Right
        'Wl': rotatedData.subset(math.index(15, math.range(0, 3))),    // Wrist Left
        'Wr': rotatedData.subset(math.index(16, math.range(0, 3))),    // Wrist Right
        'Il': rotatedData.subset(math.index(19, math.range(0, 3))),    // Index Left
        'Ir': rotatedData.subset(math.index(20, math.range(0, 3)))     // Index Right
    };

    // Move the points to origin
    // Find points s0
    let s0 = math.matrix([[
        (points['Sl'].get([0,0]) + points['Sr'].get([0,0])) / 2,
        (points['Sl'].get([0,1]) + points['Sr'].get([0,1])) / 2,
        (points['Sl'].get([0,2]) + points['Sr'].get([0,2])) / 2
    ]]);

    // Make s0 the origin in points
    for (let key in points) {
        points[key].set([0,0], points[key].get([0,0]) - s0.get([0,0]));
        points[key].set([0,1], points[key].get([0,1]) - s0.get([0,1]));
        points[key].set([0,2], points[key].get([0,2]) - s0.get([0,2]));
    }

    // Get the angle between shoulder and the known plane
    let Sl_ = points['Sl'];

    let angleShoulderZ = math.atan2( -Sl_.subset(math.index(0, 1)), -Sl_.subset(math.index(0, 0)));

    let rotMatZShoulder = math.matrix(
        [[Math.cos(angleShoulderZ), Math.sin(angleShoulderZ), 0],
        [-Math.sin(angleShoulderZ), Math.cos(angleShoulderZ), 0],
        [0, 0, 1]]
    )

    for (let key in points) {
        points[key] = math.transpose(math.multiply(rotMatZShoulder, math.transpose(points[key])));
    }

    // Get points from dictionary for calculations
    let N = points['N'];
    let Sl = points['Sl'];
    let Sr = points['Sr'];
    let El = points['El'];
    let Er = points['Er'];
    let Wl = points['Wl'];
    let Wr = points['Wr'];
    let Il = points['Il'];
    let Ir = points['Ir'];

    // Angle calculations for Right Shoulder joint's movement in horizontal plane
    let RSr = [1, 1, 0];
    let ErSrXY = [Er.get([0,0]) - Sr.get([0,0]), Er.get([0,1]) - Sr.get([0,1]), 0];
    let rightS0Angle = math.atan2(
        math.det([RSr.slice(0, 2), ErSrXY.slice(0, 2)]),
        math.dot(RSr, ErSrXY)
    );

    // Initialize the dictionary of angles
    let angles = { 'right_s0': rightS0Angle };

    // Same calculations for left shoulder
    let RSl = [-1, 1, 0];
    let ElSlXY = [El.get([0,0]) - Sl.get([0,0]), El.get([0,1]) - Sl.get([0,1]), 0];
    let leftS0Angle = math.atan2(
        math.det([RSl.slice(0, 2), ElSlXY.slice(0, 2)]),
        math.dot(RSl, ElSlXY)
    );
    angles['left_s0'] = leftS0Angle;


    // Angle calculations for Right Shoulder joint's movement in vertical plane
    let ErSr = [Er.get([0,0]) - Sr.get([0,0]), Er.get([0,1]) - Sr.get([0,1]), Er.get([0,2]) - Sr.get([0,2])];
    ErSrXY = [ErSr[0], ErSr[1], 0];
    let rightS1Angle = math.atan2(
        math.norm(math.cross(ErSr, ErSrXY)),
        math.dot(ErSr, ErSrXY)
    )
    if (ErSr[2] >= 0) rightS1Angle = -rightS1Angle;
    angles['right_s1'] = rightS1Angle;

    // Same for left side
    let ElSl = [El.get([0,0]) - Sl.get([0,0]), El.get([0,1]) - Sl.get([0,1]), El.get([0,2]) - Sl.get([0,2])];
    ElSlXY = [ElSl[0], ElSl[1], 0];
    let leftS1Angle = math.atan2(
        math.norm(math.cross(ElSl, ElSlXY)),
        math.dot(ElSl, ElSlXY)
    )
    if (ElSl[2] >= 0) leftS1Angle = -leftS1Angle;
    angles['left_s1'] = leftS1Angle;

    // Angle of Right Elbow joint
    ErSr = [Er.get([0,0]) - Sr.get([0,0]), Er.get([0,1]) - Sr.get([0,1]), Er.get([0,2]) - Sr.get([0,2])];
    let WrEr = [Wr.get([0,0]) - Er.get([0,0]), Wr.get([0,1]) - Er.get([0,1]), Wr.get([0,2]) - Er.get([0,2])];
    let rightE1Angle = math.acos(
        math.dot(ErSr, WrEr) / (math.norm(ErSr) * math.norm(WrEr))
    );
    angles['right_e1'] = rightE1Angle;

    // Angle of Left Elbow joint
    ElSl = [El.get([0,0]) - Sl.get([0,0]), El.get([0,1]) - Sl.get([0,1]), El.get([0,2]) - Sl.get([0,2])];
    let WlEl = [Wl.get([0,0]) - El.get([0,0]), Wl.get([0,1]) - El.get([0,1]), Wl.get([0,2]) - El.get([0,2])];
    let leftE1Angle = math.acos(
        math.dot(ElSl, WlEl) / (math.norm(ElSl) * math.norm(WlEl))
    );
    angles['left_e1'] = leftE1Angle;

    // Calculate the angle for Head Pan
    let NXY = [N.get([0,0]), N.get([0,1])];
    let headpanAngle = math.atan2(
        math.det([[0, 1], NXY]),
        math.dot([0, 1], NXY)
    )
    angles['headpan'] = 3*headpanAngle;

    // Calculation of angle of right wrist
    WrEr = [Wr.get([0,0]) - Er.get([0,0]), Wr.get([0,1]) - Er.get([0,1]), Wr.get([0,2]) - Er.get([0,2])];
    let IrWr = [Ir.get([0,0]) - Wr.get([0,0]), Ir.get([0,1]) - Wr.get([0,1]), Ir.get([0,2]) - Wr.get([0,2])];
    let rightW1Angle = math.acos(
        math.dot(WrEr, IrWr) / (math.norm(WrEr) * math.norm(IrWr))
    )
    angles['right_w1'] = rightW1Angle;

    // Angle of Left Wrist joint
    WlEl = [Wl.get([0,0]) - El.get([0,0]), Wl.get([0,1]) - El.get([0,1]), Wl.get([0,2]) - El.get([0,2])];
    let IlWl = [Il.get([0,0]) - Wl.get([0,0]), Il.get([0,1]) - Wl.get([0,1]), Il.get([0,2]) - Wl.get([0,2])];
    let leftW1Angle = math.acos(
        math.dot(WlEl, IlWl) / (math.norm(WlEl) * math.norm(IlWl))
    )
    angles['left_w1'] = leftW1Angle;

    // Calculations for rotation of shoulder joint on its axis
    let distErSr = math.norm(math.subtract(Er, Sr).toArray()[0]);
    let PErSrZ = [
        (Er.get([0,0]) + Sr.get([0,0])) / 2,
        (Er.get([0,1]) + Sr.get([0,1])) / 2,
        (Er.get([0,2]) + Sr.get([0,2])) / 2 + 2 * distErSr
    ];
    let normErSrZ = math.cross(
        math.subtract(PErSrZ, Sr),
        math.subtract(Er, Sr)
    );
    let normErSrWr = math.cross(
        math.subtract(Wr, Sr),
        math.subtract(Er, Sr)
    );
    let rightE0Angle = math.acos(
        math.dot(math.transpose(normErSrZ), math.transpose(normErSrWr)) /
        (math.norm(normErSrZ.toArray()[0]) * math.norm(normErSrWr.toArray()[0]))
    );
    angles['right_e0'] = 3.05 - rightE0Angle;

    // Same for left side
    let distElSl = math.norm(math.subtract(El, Sl).toArray()[0]);
    let PElSlZ = [
        (El.get([0,0]) + Sl.get([0,0])) / 2,
        (El.get([0,1]) + Sl.get([0,1])) / 2,
        (El.get([0,2]) + Sl.get([0,2])) / 2 + 2 * distElSl  
    ];
    let normElSlZ = math.cross(
        math.subtract(PElSlZ, Sl),
        math.subtract(El, Sl)
    );
    let normElSlWl = math.cross(
        math.subtract(Wl, Sl),
        math.subtract(El, Sl)
    );
    let leftE0Angle = math.acos(
        math.dot(math.transpose(normElSlZ), math.transpose(normElSlWl)) /
        (math.norm(normElSlZ.toArray()[0]) * math.norm(normElSlWl.toArray()[0]))
    );
    angles['left_e0'] = leftE0Angle - 3.05;

    return angles; 
}

setTimeout(main, 2000)
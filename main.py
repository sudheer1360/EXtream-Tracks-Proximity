"""
Next-Gen Road Safety - Object Detection Application
A Flask-based web application for real-time object detection using YOLO
"""
from flask import Flask, render_template, Response, request, redirect, url_for, flash, jsonify
from flask_login import LoginManager, login_user, login_required, logout_user, current_user
from werkzeug.security import generate_password_hash, check_password_hash
import cv2
from ultralytics import YOLO
from sqlalchemy import create_engine, func, text
from sqlalchemy.orm import sessionmaker
import os
import sys
import json

# Add the project root directory to Python path
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from app.models.models import User, Detection, Base
from datetime import datetime
import logging
import traceback
import threading
import queue
import time
import os
import sys
from flask_cors import CORS
import numpy as np

# Configure logging
logging.basicConfig(level=logging.INFO,
                   format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
                   handlers=[
                       logging.StreamHandler(sys.stdout),
                       logging.FileHandler('app.log')
                   ])
logger = logging.getLogger(__name__)

# Initialize Flask app
app = Flask(__name__, template_folder='app/templates', static_folder='app/static')
app.config['SECRET_KEY'] = os.environ.get('SECRET_KEY', 'dev-secret-key')  # Use environment variable in production

# Enable CORS with more specific settings
CORS(app, resources={r"/*": {"origins": "*"}}, supports_credentials=True)

# Add a basic error handler
@app.errorhandler(500)
def handle_500(e):
    return jsonify({
        'error': 'Internal Server Error',
        'message': str(e)
    }), 500

@app.errorhandler(404)
def handle_404(e):
    return jsonify({
        'error': 'Not Found',
        'message': 'The requested resource was not found.'
    }), 404

# Database setup
db_path = os.path.abspath('app/database/app.db')
logger.info(f"Using database at: {db_path}")

# Ensure database directory exists
db_dir = os.path.dirname(db_path)
if not os.path.exists(db_dir):
    os.makedirs(db_dir, exist_ok=True)

# Create database engine without pooling parameters
engine = create_engine(
    f'sqlite:///{db_path}',
    connect_args={'check_same_thread': False}
)

# Create session factory
SessionFactory = sessionmaker(bind=engine)

# Create database tables
Base.metadata.create_all(bind=engine)

# Login manager setup
login_manager = LoginManager()
login_manager.init_app(app)
login_manager.login_view = 'login'

# Global variables
camera = None
camera_lock = threading.Lock()
detection_queue = queue.Queue(maxsize=1000)  # Queue for storing detections
detection_thread = None
db_thread = None
stop_event = threading.Event()
processed_count = 0
camera_fps = 30  # Target FPS
active_user_id = None  # Store the ID of the active user for detection storage

# Load YOLO model - do this at startup to avoid delay later
logger.info("Loading YOLO model...")
try:
    model = YOLO('yolov8n.pt')
    logger.info("YOLO model loaded successfully")
except Exception as e:
    logger.error(f"Error loading YOLO model: {str(e)}")
    model = None

# MQTT Configuration
# MQTT_BROKER = "broker.hivemq.com"  # You can use your own broker
# MQTT_PORT = 1883
# MQTT_TOPIC_BUZZER = "road_safety/buzzer"
# MQTT_TOPIC_LED = "road_safety/led"

# Initialize MQTT client
# mqtt_client = mqtt.Client()

# def setup_mqtt():
#     """Initialize MQTT connection"""
#     try:
#         mqtt_client.connect(MQTT_BROKER, MQTT_PORT, 60)
#         mqtt_client.loop_start()
#         logger.info("MQTT client connected successfully")
#     except Exception as e:
#         logger.error(f"MQTT connection failed: {str(e)}")

# def control_buzzer(state):
#     """Control buzzer via MQTT"""
#     try:
#         mqtt_client.publish("road_safety/buzzer", 
#                           json.dumps({"state": state}),
#                           qos=1)
#         print(f"MQTT: Sent buzzer state: {state}")
#     except Exception as e:
#         print(f"MQTT Error (buzzer): {str(e)}")

# def control_led(color, state):
#     """Control LED via MQTT"""
#     try:
#         mqtt_client.publish("road_safety/led",
#                           json.dumps({"led": color, "state": state}),
#                           qos=1)
#         print(f"MQTT: Sent LED state: {color}={state}")
#     except Exception as e:
#         print(f"MQTT Error (LED): {str(e)}")

@login_manager.user_loader
def load_user(user_id):
    """Load user from database"""
    with SessionFactory() as session:
        return session.query(User).get(int(user_id))

def initialize_camera():
    """Initialize camera with 30fps settings"""
    global camera
    try:
        with camera_lock:
            if camera is not None and camera.isOpened():
                camera.release()
                
            # Try camera index 0
            logger.info("Initializing camera with index 0")
            camera = cv2.VideoCapture(0)
            
            # If failed, try index 1
            if not camera.isOpened():
                logger.info("Failed with index 0, trying index 1")
                camera = cv2.VideoCapture(1)
            
            # Check if camera is now open
            if camera.isOpened():
                # Set camera parameters for optimal performance
                camera.set(cv2.CAP_PROP_FPS, camera_fps)  # Set to 30 FPS
                camera.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
                camera.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)
                camera.set(cv2.CAP_PROP_BUFFERSIZE, 2)  # Minimize buffering for real-time
                
                actual_fps = camera.get(cv2.CAP_PROP_FPS)
                logger.info(f"Camera initialized successfully. Actual FPS: {actual_fps}")
                return True
            else:
                logger.error("Failed to initialize camera with any index")
                return False
    except Exception as e:
        logger.error(f"Error initializing camera: {str(e)}")
        logger.error(traceback.format_exc())
        return False

def db_worker():
    """Worker thread for saving detections to database"""
    logger.info("Database worker thread started")
    session = None
    count = 0
    last_log_time = time.time()
    error_count = 0
    
    try:
        logger.info("DB worker ready to process detection queue")
        # Create a fresh session
        session = SessionFactory()
        logger.info("Initial database session created")
        
        while not stop_event.is_set():
            try:
                # Get item from queue with timeout (allows for checking stop_event)
                try:
                    if count % 10 == 0:  # Only log occasionally to avoid spam
                        logger.debug(f"Waiting for item from queue. Current queue size: {detection_queue.qsize()}")
                    
                    item = detection_queue.get(timeout=0.5)
                    logger.info(f"Got item from queue: {item['object_name']} with confidence {item['confidence']:.2f}")
                except queue.Empty:
                    continue
                
                # Create a new session for each transaction for extra safety
                if session is None:
                    session = SessionFactory()
                    logger.info("Created new database session")
                
                # Create Detection object
                logger.info(f"Creating Detection object: {item['object_name']}")
                detection = Detection(
                    user_id=item['user_id'],
                    object_name=item['object_name'],
                    confidence=item['confidence'],
                    timestamp=item['timestamp']
                )
                
                # Use a try-except-finally pattern for safely handling the transaction
                try:
                    # Add and commit in its own transaction
                    logger.info(f"Adding detection to database: {item['object_name']}")
                    session.add(detection)
                    session.commit()
                    logger.info(f"Successfully committed detection to database: {item['object_name']}")
                    
                    # Reset error counter on success
                    error_count = 0
                    
                    # Update counters
                    count += 1
                    global processed_count
                    processed_count += 1
                except Exception as db_error:
                    # Handle database errors
                    error_count += 1
                    logger.error(f"Database error while saving detection: {str(db_error)}")
                    logger.error(traceback.format_exc())
                    
                    try:
                        session.rollback()
                        logger.info("Session rolled back")
                    except:
                        logger.error("Error during rollback")
                    
                    # Create a fresh session if there were errors
                    if error_count > 3:
                        logger.warning("Multiple database errors, refreshing session")
                        try:
                            if session:
                                session.close()
                        except:
                            pass
                        session = SessionFactory()
                        error_count = 0
                finally:
                    # Always mark the task as done
                    detection_queue.task_done()
                
                # Log progress periodically
                current_time = time.time()
                if current_time - last_log_time > 10:  # Log every 10 seconds
                    logger.info(f"DB worker processed {count} detections. Queue size: {detection_queue.qsize()}")
                    last_log_time = current_time
                    count = 0
                
            except Exception as e:
                logger.error(f"Error in DB worker: {str(e)}")
                logger.error(traceback.format_exc())
                
                # Recreate session on general errors
                try:
                    if session:
                        session.close()
                except:
                    pass
                    
                session = SessionFactory()
                logger.info("Recreated database session after error")
                
                # Sleep a bit to avoid hammering if there's a persistent error
                time.sleep(1)
    finally:
        # Close the session
        logger.info("Closing database session")
        if session:
            try:
                session.close()
            except:
                pass
        logger.info("Database worker thread stopped")

def queue_detection(user_id, object_name, confidence):
    """Queue a detection for database storage"""
    logger.info(f"Attempting to queue detection: {object_name} with confidence {confidence:.2f} for user {user_id}")
    
    if user_id is None:
        logger.warning("Cannot queue detection: user_id is None")
        return False
    
    try:
        # Check if queue is too full
        queue_size_before = detection_queue.qsize()
        if queue_size_before > 800:  # 80% of max capacity
            logger.warning(f"Detection queue is getting full: {queue_size_before}/1000")
        
        # Create detection item
        detection = {
            'user_id': user_id,
            'object_name': object_name,
            'confidence': confidence,
            'timestamp': datetime.now()
        }
        
        # Add to queue with a timeout to avoid blocking indefinitely
        logger.info(f"Adding detection to queue: {object_name}")
        detection_queue.put(detection, timeout=1.0)
        
        # Log queue size after adding
        queue_size_after = detection_queue.qsize()
        logger.info(f"Detection queued successfully. Queue size: {queue_size_after}")
        
        return True
    except queue.Full:
        logger.error("Detection queue is full, detection lost")
        return False
    except Exception as e:
        logger.error(f"Error queuing detection: {str(e)}")
        logger.error(traceback.format_exc())
        return False

def detection_worker():
    """Worker thread for processing camera frames and detecting objects"""
    global camera
    logger.info("Detection thread started")
    
    frame_count = 0
    fps_start_time = time.time()
    fps_frame_count = 0
    fps = 0
    last_detection_time = 0
    detection_cooldown = 0.2  # Seconds between saving detections
    detection_log_interval = 50  # Log detection activity every 50 frames
    
    try:
        while not stop_event.is_set():
            try:
                # Check if camera is initialized
                with camera_lock:
                    if camera is None or not camera.isOpened():
                        if not initialize_camera():
                            time.sleep(1)  # Wait before retrying
                            continue
                    
                    # Read frame from camera
                    success, frame = camera.read()
                
                if not success:
                    logger.warning("Failed to read frame from camera")
                    # Try to reinitialize the camera
                    with camera_lock:
                        if not initialize_camera():
                            time.sleep(1)
                            continue
                    continue
                
                # Update frame count and FPS calculation
                frame_count += 1
                fps_frame_count += 1
                current_time = time.time()
                
                # Calculate FPS every second
                if (current_time - fps_start_time) > 1:
                    fps = fps_frame_count / (current_time - fps_start_time)
                    fps_start_time = current_time
                    fps_frame_count = 0
                    logger.debug(f"Detection thread FPS: {fps:.1f}")
                
                # Skip frame processing if no user is logged in (for efficiency)
                user_id = None
                if hasattr(current_user, 'is_authenticated') and current_user.is_authenticated:
                    user_id = current_user.id
                    if frame_count % detection_log_interval == 0:
                        logger.info(f"User is authenticated with ID: {user_id}")
                else:
                    if frame_count % detection_log_interval == 0:
                        logger.info("No authenticated user - skipping detection storage")
                
                # Run detection on every frame with YOLO
                if model is not None:
                    results = model(frame)
                    
                    # Log detection activity periodically
                    if frame_count % detection_log_interval == 0:
                        logger.info(f"Frame {frame_count}: Running detection with YOLO")
                
                    # Process detection results
                    for result in results:
                        boxes = result.boxes.cpu().numpy()
                        
                        if frame_count % detection_log_interval == 0:
                            logger.info(f"Frame {frame_count}: Found {len(boxes)} detections")
                        
                        # If detections found and user is authenticated and cooldown passed
                        if len(boxes) > 0 and user_id is not None and (current_time - last_detection_time) >= detection_cooldown:
                            logger.info(f"Processing {len(boxes)} detections with cooldown passed. User ID: {user_id}")
                            
                            # Process all detections above the threshold
                            valid_detections = []
                            confidence_threshold = 0.35  # Min confidence threshold
                            
                            for box in boxes:
                                confidence = float(box.conf[0])
                                class_id = int(box.cls[0])
                                class_name = model.names[class_id]
                                
                                if frame_count % detection_log_interval == 0:
                                    logger.info(f"Detection: {class_name} with confidence {confidence:.2f}")
                                
                                # Add any detection above threshold to our list
                                if confidence >= confidence_threshold:
                                    valid_detections.append({
                                        'class_name': class_name,
                                        'confidence': confidence
                                    })
                            
                            # Save all valid detections
                            if valid_detections:
                                saved_count = 0
                                for detection in valid_detections:
                                    logger.info(f"Queueing detection: {detection['class_name']} with confidence {detection['confidence']:.2f}")
                                    success = queue_detection(
                                        user_id, 
                                        detection['class_name'], 
                                        detection['confidence']
                                    )
                                    if success:
                                        logger.info(f"Successfully queued detection: {detection['class_name']}")
                                        saved_count += 1
                                    else:
                                        logger.error(f"Failed to queue detection: {detection['class_name']}")
                                
                                if saved_count > 0:
                                    last_detection_time = current_time
                                    logger.info(f"Saved {saved_count} detections to queue")
                
                # Add alert logic based on detections
                if results and len(results[0].boxes) > 0:
                    dangerous_objects = ["person", "car", "motorcycle", "truck"]
                    for detection in results[0].boxes:
                        obj_name = results[0].names[int(detection.cls[0])]  # Get actual class name
                        confidence = detection.conf[0]
                        
                        print(f"Detected: {obj_name} (confidence: {confidence:.2f})")  # Debug print
                        
                        if obj_name in dangerous_objects and confidence > 0.6:
                            print(f"Dangerous object detected: {obj_name}")  # Debug print
                            time.sleep(2)  # Alert duration

                # Slight delay to prevent maxing out CPU
                time.sleep(1/camera_fps)  # Aim for target FPS
                
            except Exception as e:
                logger.error(f"Error in detection worker: {str(e)}")
                logger.error(traceback.format_exc())
                time.sleep(0.5)  # Pause briefly
    finally:
        logger.info("Detection thread stopped")

def generate_frames():
    """Generator function for streaming video frames to web client"""
    global camera, processed_count, active_user_id
    logger.info("Client connected to video stream")
    
    frame_count = 0
    fps_start_time = time.time()
    fps_frame_count = 0
    fps = 0
    last_save_time = time.time()
    save_interval = 2.0  # Only save detections every 2 seconds
    
    try:
        while True:
            # Check if camera is initialized
            with camera_lock:
                if camera is None or not camera.isOpened():
                    if not initialize_camera():
                        # Return a placeholder image
                        placeholder_path = 'app/static/img/no_camera.jpg'
                        if os.path.exists(placeholder_path):
                            placeholder = open(placeholder_path, 'rb').read()
                        else:
                            # Create a blank image with text if no placeholder
                            blank = np.zeros((480, 640, 3), np.uint8)
                            cv2.putText(blank, "Camera not available", (100, 240), 
                                       cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2)
                            _, buffer = cv2.imencode('.jpg', blank)
                            placeholder = buffer.tobytes()
                            
                        yield (b'--frame\r\n'
                              b'Content-Type: image/jpeg\r\n\r\n' + placeholder + b'\r\n')
                        time.sleep(1)
                        continue
                
                # Read frame from camera
                success, frame = camera.read()
            
            if not success:
                logger.warning("Failed to read frame from camera in generate_frames")
                # Try to reinitialize the camera
                with camera_lock:
                    if not initialize_camera():
                        time.sleep(1)
                        continue
                continue
            
            # Update frame count and FPS calculation
            frame_count += 1
            fps_frame_count += 1
            current_time = time.time()
            
            # Calculate FPS every second
            if (current_time - fps_start_time) > 1:
                fps = fps_frame_count / (current_time - fps_start_time)
                fps_start_time = current_time
                fps_frame_count = 0
            
            # Process frame with YOLO for detection and visualization
            if model is not None:
                results = model(frame)
                orig_frame = frame.copy()  # Save a copy of the original frame
                
                # Draw detection boxes on frame
                annotated_frame = results[0].plot()
                
                # Get the user ID - use the global variable instead of current_user
                user_id = active_user_id
                
                # Log the active user ID periodically
                if frame_count % 100 == 0:
                    logger.info(f"Active user ID in generate_frames: {user_id}")
                    
                # Check if we should save detection (based on time and login status)
                should_save = False
                
                # Check if a user is logged in using our global variable
                if user_id is not None:
                    # Only save every few seconds to avoid database spam
                    if current_time - last_save_time >= save_interval:
                        should_save = True
                        last_save_time = current_time
                    
                # Process detections
                saved_detection = False
                if should_save and user_id is not None:
                    # Find all valid detections above the threshold
                    boxes = results[0].boxes.cpu().numpy()
                    
                    if len(boxes) > 0:
                        valid_detections = []
                        confidence_threshold = 0.40  # Set minimum confidence level
                        
                        # First collect all valid detections
                        for box in boxes:
                            confidence = float(box.conf[0])
                            class_id = int(box.cls[0])
                            class_name = model.names[class_id]
                            
                            # Only consider detections with good confidence
                            if confidence >= confidence_threshold:
                                valid_detections.append({
                                    'class_name': class_name,
                                    'confidence': confidence
                                })
                        
                        # Save all valid detections to the database
                        if valid_detections:
                            saved_count = 0
                            save_messages = []
                            
                            # Create a single database session for all detections in this frame
                            with SessionFactory() as session:
                                for detection in valid_detections:
                                    try:
                                        # Create Detection object
                                        db_detection = Detection(
                                            user_id=user_id,
                                            object_name=detection['class_name'],
                                            confidence=detection['confidence'],
                                            timestamp=datetime.now()
                                        )
                                        session.add(db_detection)
                                        
                                        # Add to save messages list
                                        save_messages.append(f"{detection['class_name']} ({detection['confidence']:.2f})")
                                        saved_count += 1
                                    except Exception as e:
                                        logger.error(f"Error creating detection: {str(e)}")
                                
                                try:
                                    # Commit all detections at once
                                    session.commit()
                                    
                                    # Update counter
                                    processed_count += saved_count
                                    saved_detection = True
                                    
                                    # Log successful save
                                    logger.info(f"Successfully saved {saved_count} detections in database: {', '.join(save_messages)}")
                                    
                                    # Add text to indicate saved detections
                                    cv2.putText(
                                        annotated_frame,
                                        f"SAVED: {saved_count} objects",
                                        (10, 70),
                                        cv2.FONT_HERSHEY_SIMPLEX,
                                        0.7,
                                        (0, 0, 255),  # Red
                                        2
                                    )
                                    
                                    # Add each detection name in smaller text
                                    y_pos = 100
                                    for i, msg in enumerate(save_messages[:3]):  # Show at most 3 
                                        cv2.putText(
                                            annotated_frame,
                                            msg,
                                            (15, y_pos),
                                            cv2.FONT_HERSHEY_SIMPLEX,
                                            0.5,
                                            (0, 0, 255),  # Red
                                            1
                                        )
                                        y_pos += 20
                                    
                                    # If there are more than 3 detections, show a count of the rest
                                    if len(save_messages) > 3:
                                        cv2.putText(
                                            annotated_frame,
                                            f"+ {len(save_messages) - 3} more",
                                            (15, y_pos),
                                            cv2.FONT_HERSHEY_SIMPLEX,
                                            0.5,
                                            (0, 0, 255),  # Red
                                            1
                                        )
                                        
                                except Exception as e:
                                    logger.error(f"Error committing detections batch: {str(e)}")
                                    logger.error(traceback.format_exc())
                                    session.rollback()
                
                # Add FPS and counter info
                login_status = "LOGGED IN (User ID: " + str(user_id) + ")" if user_id else "NOT LOGGED IN"
                cv2.putText(
                    annotated_frame,
                    f"FPS: {fps:.1f} | Saved: {processed_count} | {login_status}",
                    (10, 30),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.6,  # Smaller font to fit
                    (0, 255, 0),  # Green
                    2
                )
                
                frame = annotated_frame
            
            # Convert to JPEG
            _, buffer = cv2.imencode('.jpg', frame)
            frame_bytes = buffer.tobytes()
            
            # Yield the frame
            yield (b'--frame\r\n'
                  b'Content-Type: image/jpeg\r\n\r\n' + frame_bytes + b'\r\n')
            
            # Slight delay to not overwhelm the client
            time.sleep(1/45)  # Slightly faster than camera rate to avoid buffering
            
    except Exception as e:
        logger.error(f"Error in generate_frames: {str(e)}")
        logger.error(traceback.format_exc())
    finally:
        logger.info("Client disconnected from video stream")

# Routes
@app.route('/')
def index():
    """Landing page route"""
    return render_template('index.html')

@app.route('/login', methods=['GET', 'POST'])
def login():
    """Login route"""
    if request.method == 'POST':
        username = request.form.get('username')
        password = request.form.get('password')
        
        if not username or not password:
            flash('Please enter both username and password')
            return render_template('login.html')
        
        with SessionFactory() as session:
            user = session.query(User).filter_by(username=username).first()
            
            if user and check_password_hash(user.password, password):
                login_user(user)
                logger.info(f"User {username} logged in successfully")
                return redirect(url_for('dashboard'))
            
            flash('Invalid username or password')
            logger.warning(f"Failed login attempt for username: {username}")
            
    return render_template('login.html')

@app.route('/signup', methods=['GET', 'POST'])
def signup():
    """Signup route"""
    if request.method == 'POST':
        username = request.form.get('username')
        email = request.form.get('email')
        password = request.form.get('password')
        
        if not username or not email or not password:
            flash('Please fill in all fields')
            return render_template('signup.html')
        
        with SessionFactory() as session:
            # Check if username or email already exists
            existing_user = session.query(User).filter(
                (User.username == username) | (User.email == email)
            ).first()
            
            if existing_user:
                flash('Username or email already exists')
                return render_template('signup.html')
            
            # Create new user
            hashed_password = generate_password_hash(password)
            new_user = User(
                username=username,
                email=email,
                password=hashed_password
            )
            
            try:
                session.add(new_user)
                session.commit()
                logger.info(f"New user created: {username}")
                flash('Account created successfully! Please log in.')
                return redirect(url_for('login'))
            except Exception as e:
                session.rollback()
                logger.error(f"Error creating user: {str(e)}")
                flash('An error occurred. Please try again.')
                
    return render_template('signup.html')

@app.route('/dashboard')
@login_required
def dashboard():
    """Dashboard route"""
    try:
        # Store the current user ID for the video feed
        global active_user_id
        active_user_id = current_user.id
        logger.info(f"Setting active_user_id to {active_user_id} for user {current_user.username}")
        
        with SessionFactory() as session:
            # Get recent detections
            detections = session.query(Detection)\
                .filter_by(user_id=current_user.id)\
                .order_by(Detection.timestamp.desc())\
                .limit(10).all()
            
            # Get detection statistics
            detection_count = session.query(func.count(Detection.id))\
                .filter_by(user_id=current_user.id).scalar() or 0
            
            # Get detection by object type (for charts)
            object_counts = {}
            object_query = session.query(
                Detection.object_name, 
                func.count(Detection.id)
            ).filter_by(user_id=current_user.id).group_by(Detection.object_name).all()
            
            for obj_name, count in object_query:
                object_counts[obj_name] = count
            
            # System status info
            system_status = {
                'user_id': current_user.id,
                'username': current_user.username,
                'detection_count': detection_count,
                'queue_size': detection_queue.qsize(),
                'worker_running': not stop_event.is_set(),
                'last_detection_time': detections[0].timestamp.strftime('%Y-%m-%d %H:%M:%S') if detections else 'None',
                'connection_status': 'Connected',
                'object_counts': object_counts
            }
            
            logger.info(f"Dashboard loaded for user {current_user.username} with {len(detections)} recent detections")
            return render_template('dashboard.html', 
                                  detections=detections, 
                                  system_status=system_status,
                                  datetime=datetime)
    except Exception as e:
        logger.error(f"Error loading dashboard: {str(e)}")
        logger.error(traceback.format_exc())
        return render_template('dashboard.html', 
                              detections=[], 
                              system_status={'error': str(e)},
                              datetime=datetime)

@app.route('/video_feed')
@login_required
def video_feed():
    """Video streaming route"""
    global active_user_id
    try:
        # Ensure the active user ID is set
        if active_user_id is None and hasattr(current_user, 'id'):
            active_user_id = current_user.id
            logger.info(f"Setting active_user_id to {active_user_id} in video_feed route")
            
        return Response(generate_frames(),
                      mimetype='multipart/x-mixed-replace; boundary=frame')
    except Exception as e:
        logger.error(f"Error in video_feed: {str(e)}")
        return jsonify({'error': str(e)}), 500

@app.route('/logout')
@login_required
def logout():
    """Logout route"""
    global active_user_id
    active_user_id = None  # Clear the active user ID
    logout_user()
    return redirect(url_for('index'))

@app.route('/api/get_detections')
@login_required
def get_detections():
    """API endpoint for getting detection data"""
    try:
        # Check if threads are running and restart if needed
        global detection_thread, db_thread
        
        # Check detection thread
        detection_thread_running = detection_thread is not None and detection_thread.is_alive()
        if not detection_thread_running:
            logger.warning("Detection thread not running - attempting to restart")
            if detection_thread is not None:
                detection_thread = None
            detection_thread = threading.Thread(target=detection_worker, daemon=True)
            detection_thread.start()
            logger.info("Detection thread restarted")
        
        # Check database worker thread
        db_thread_running = db_thread is not None and db_thread.is_alive()
        if not db_thread_running:
            logger.warning("Database worker thread not running - attempting to restart")
            if db_thread is not None:
                db_thread = None
            db_thread = threading.Thread(target=db_worker, daemon=True)
            db_thread.start()
            logger.info("Database worker thread restarted")
        
        with SessionFactory() as session:
            # Check if user exists and is valid
            user = session.query(User).get(current_user.id)
            if not user:
                return jsonify({
                    'error': 'User not found',
                    'authenticated': current_user.is_authenticated,
                    'user_id': current_user.id if hasattr(current_user, 'id') else None
                }), 404
            
            # Get recent detections with error handling
            try:
                detections = session.query(Detection)\
                    .filter_by(user_id=current_user.id)\
                    .order_by(Detection.timestamp.desc())\
                    .limit(10).all()
                
                detection_list = [{
                    'id': d.id,
                    'object_name': d.object_name,
                    'confidence': f"{d.confidence * 100:.2f}%",
                    'confidence_raw': d.confidence,
                    'timestamp': d.timestamp.strftime('%Y-%m-%d %H:%M:%S')
                } for d in detections]
                
                # Count all detections for this user
                detection_count = session.query(func.count(Detection.id))\
                    .filter_by(user_id=current_user.id).scalar() or 0
                
                # Get detection count by type
                type_counts = {}
                try:
                    object_query = session.query(
                        Detection.object_name, 
                        func.count(Detection.id)
                    ).filter_by(user_id=current_user.id).group_by(Detection.object_name).all()
                    
                    for obj_name, count in object_query:
                        type_counts[obj_name] = count
                except Exception as count_error:
                    logger.error(f"Error getting type counts: {str(count_error)}")
                    type_counts = {'error': str(count_error)}
                
                return jsonify({
                    'success': True,
                    'detections': detection_list,
                    'count': len(detection_list),
                    'total_count': detection_count,
                    'type_counts': type_counts,
                    'user': {
                        'id': current_user.id,
                        'username': current_user.username,
                        'authenticated': current_user.is_authenticated
                    },
                    'status': {
                        'processed_count': processed_count,
                        'queue_size': detection_queue.qsize(),
                        'detection_thread': detection_thread_running,
                        'db_thread': db_thread_running,
                        'worker_running': not stop_event.is_set()
                    },
                    'last_detection_time': detections[0].timestamp.strftime('%Y-%m-%d %H:%M:%S') if detections else 'None'
                })
            except Exception as query_error:
                logger.error(f"Error querying detections: {str(query_error)}")
                logger.error(traceback.format_exc())
                return jsonify({
                    'success': False,
                    'error': f"Database query error: {str(query_error)}",
                    'authenticated': current_user.is_authenticated,
                    'user_id': current_user.id
                }), 500
    except Exception as e:
        logger.error(f"Error in get_detections: {str(e)}")
        logger.error(traceback.format_exc())
        return jsonify({
            'success': False,
            'error': str(e),
            'message': 'An error occurred retrieving detection data.'
        }), 500

@app.route('/api/test_detection', methods=['POST'])
@login_required
def test_detection():
    """Add a test detection directly to the database"""
    try:
        # Create a test detection
        test_obj = 'TEST_OBJECT'
        test_confidence = 0.99
        
        logger.info(f"Creating test detection: {test_obj} for user {current_user.id}")
        
        # Save directly to database
        with SessionFactory() as session:
            # Create Detection object
            detection = Detection(
                user_id=current_user.id,
                object_name=test_obj,
                confidence=test_confidence,
                timestamp=datetime.now()
            )
            
            # Save to database
            session.add(detection)
            session.commit()
            
            # Increment counter
            global processed_count
            processed_count += 1
            
            detection_id = detection.id
            
            logger.info(f"Test detection saved with ID {detection_id}")
            
            return jsonify({
                'success': True,
                'message': f'Test detection added: {test_obj}',
                'detection_id': detection_id, 
                'processed_count': processed_count
            })
    except Exception as e:
        logger.error(f"Error creating test detection: {str(e)}")
        logger.error(traceback.format_exc())
        return jsonify({
            'success': False,
            'message': f'Error: {str(e)}'
        }), 500

@app.route('/api/check-camera')
def check_camera():
    """Check camera status"""
    global camera
    status = {
        'status': 'unknown',
        'message': 'Camera status unknown'
    }
    
    try:
        with camera_lock:
            if camera is None:
                status = {
                    'status': 'not_initialized',
                    'message': 'Camera not initialized'
                }
            elif not camera.isOpened():
                status = {
                    'status': 'closed',
                    'message': 'Camera is closed'
                }
            else:
                # Try to read a frame
                ret, _ = camera.read()
                if ret:
                    status = {
                        'status': 'ok',
                        'message': 'Camera is working properly',
                        'fps': camera.get(cv2.CAP_PROP_FPS)
                    }
                else:
                    status = {
                        'status': 'error',
                        'message': 'Camera is open but cannot read frames'
                    }
    except Exception as e:
        status = {
            'status': 'error',
            'message': f'Error checking camera: {str(e)}'
        }
        logger.error(f"Error checking camera: {str(e)}")
    
    return jsonify(status)

@app.route('/health')
def health_check():
    """Simple health check endpoint that doesn't require authentication"""
    return jsonify({
        'status': 'ok',
        'message': 'Server is running',
        'timestamp': datetime.now().strftime('%Y-%m-%d %H:%M:%S')
    })

@app.route('/get_recent_detections')
@login_required
def get_recent_detections():
    """API endpoint for getting recent detections"""
    try:
        with SessionFactory() as session:
            # Get recent detections for current user
            detections = session.query(Detection)\
                .filter_by(user_id=current_user.id)\
                .order_by(Detection.timestamp.desc())\
                .limit(10).all()
            
            # Format detections for JSON response
            detection_list = [{
                'id': d.id,
                'object_name': d.object_name,
                'confidence': f"{d.confidence * 100:.2f}%",
                'timestamp': d.timestamp.strftime('%Y-%m-%d %H:%M:%S')
            } for d in detections]
            
            return jsonify({
                'success': True,
                'detections': detection_list
            })
    except Exception as e:
        logger.error(f"Error getting recent detections: {str(e)}")
        logger.error(traceback.format_exc())
        return jsonify({
            'success': False,
            'error': str(e)
        }), 500

@app.route('/api/latest_detections', methods=['GET'])
def get_latest_detections():
    """API endpoint for getting latest detections for ESP32"""
    try:
        with SessionFactory() as session:
            # Get the most recent detections
            latest_detections = session.query(Detection)\
                .order_by(Detection.timestamp.desc())\
                .limit(5).all()
            
            # Define dangerous objects
            dangerous_objects = ["person", "car", "motorcycle", "truck", "bus"]
            
            # Process detections
            dangerous_detections = []
            has_dangerous = False
            
            for detection in latest_detections:
                if detection.object_name.lower() in dangerous_objects and detection.confidence > 0.5:
                    has_dangerous = True
                    dangerous_detections.append({
                        "name": detection.object_name,
                        "confidence": detection.confidence,
                        "timestamp": detection.timestamp.strftime('%Y-%m-%d %H:%M:%S')
                    })
            
            response = {
                "success": True,
                "has_dangerous_objects": has_dangerous,
                "total_detections": len(latest_detections),
                "objects": dangerous_detections,
                "timestamp": datetime.now().strftime('%Y-%m-%d %H:%M:%S')
            }
            
            return jsonify(response)
            
    except Exception as e:
        logger.error(f"Error fetching latest detections: {str(e)}")
        return jsonify({
            "success": False,
            "error": str(e),
            "timestamp": datetime.now().strftime('%Y-%m-%d %H:%M:%S')
        }), 500

def start_worker_threads():
    """Start the worker threads for detection and database operations"""
    global detection_thread, db_thread, stop_event
    
    # Reset the stop event
    stop_event.clear()
    
    # Start database worker thread if not running
    if db_thread is None or not db_thread.is_alive():
        db_thread = threading.Thread(target=db_worker, daemon=True)
        db_thread.start()
        logger.info("Database worker thread started")
    
    # Start detection worker thread if not running
    if detection_thread is None or not detection_thread.is_alive():
        detection_thread = threading.Thread(target=detection_worker, daemon=True)
        detection_thread.start()
        logger.info("Detection worker thread started")

def cleanup():
    """Cleanup resources on application shutdown"""
    global camera, stop_event
    
    logger.info("Application shutting down, cleaning up resources...")
    
    # Signal threads to stop
    stop_event.set()
    
    # Release camera
    with camera_lock:
        if camera is not None:
            camera.release()
            camera = None
            logger.info("Camera released")
    
    # Wait for threads to finish (with timeout)
    if db_thread is not None and db_thread.is_alive():
        db_thread.join(timeout=2.0)
        logger.info("Database worker thread joined" if not db_thread.is_alive() else "Database worker thread timeout")
    
    if detection_thread is not None and detection_thread.is_alive():
        detection_thread.join(timeout=2.0)
        logger.info("Detection thread joined" if not detection_thread.is_alive() else "Detection thread timeout")
    
    logger.info("Cleanup completed")

if __name__ == '__main__':
    try:
        # Initialize camera
        initialize_camera()
        
        # Start worker threads
        start_worker_threads()
        
        # Register cleanup on exit
        import atexit
        atexit.register(cleanup)
        
        port = int(os.environ.get('PORT', 3456))
        
        print(f"\n\n=====================================================")
        print(f"  Access the application at: http://127.0.0.1:{port}")
        print(f"=====================================================\n")
        
        logger.info(f"Starting Flask application on port {port}")
        app.run(host='0.0.0.0', port=port, debug=True, threaded=True, use_reloader=False)
    except Exception as e:
        logger.error(f"Error starting application: {str(e)}")
        logger.error(traceback.format_exc()) 


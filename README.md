# Next-Gen Road Safety

![License](https://img.shields.io/badge/license-MIT-blue.svg)

A multi-threaded Flask application providing real-time object detection using YOLO technology for road safety monitoring and analysis.

## Features

- **Real-time Object Detection**: Uses YOLO (You Only Look Once) to detect objects at 30 FPS
- **Multi-threaded Architecture**: 
  - Camera thread for smooth video processing
  - Database thread for non-blocking storage of detections
  - Main thread for responsive web interface
- **User Authentication System**:
  - Secure signup and login functionality
  - User-specific detection history
- **Modern Dashboard**:
  - Live camera feed with bounding boxes
  - Real-time detection statistics
  - Historical data analysis
  - System status monitoring
- **Responsive Design**:
  - Beautiful UI built with Tailwind CSS
  - Mobile-friendly interface
  - Intuitive navigation

## Architecture

The application is built with a focus on performance and reliability:

- **Flask**: Web framework for the backend
- **SQLAlchemy**: ORM for database interactions
- **SQLite**: Lightweight database for storing users and detections
- **OpenCV**: For camera access and frame processing
- **YOLO**: State-of-the-art object detection model
- **Threading**: For non-blocking operations
- **Tailwind CSS**: For modern UI components

## Setup Instructions

1. Clone the repository:
```bash
git clone https://github.com/yourusername/next-gen-road-safety.git
cd next-gen-road-safety
```

2. Create a virtual environment:
```bash
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate
```

3. Install dependencies:
```bash
pip install -r requirements.txt
```

4. Initialize the database:
```bash
python init_db.py
```

5. Run the application:
```bash
python main.py
```

6. Open your browser and navigate to `http://localhost:3456`

## Requirements

- Python 3.8+
- Webcam access
- Modern web browser
- Minimum 4GB RAM (8GB recommended for optimal performance)
- CUDA-compatible GPU (optional, for improved detection speed)

## Usage Guide

1. **Create an Account**: Sign up with a username and password
2. **Login**: Access your personal dashboard
3. **Dashboard**: View the live detection feed and statistics
4. **Detection History**: Analyze past detections with timestamps

## Data Storage

Detection data includes:
- Object class (person, car, bicycle, etc.)
- Confidence score
- Timestamp
- User ID (for association with the authenticated user)

## Performance Optimization

- Detection runs on every frame at 30 FPS
- Batch saving to reduce database load
- Confidence threshold filtering (≥ 0.40)
- Dedicated threads for I/O operations

## License

This project is licensed under the MIT License - see the LICENSE file for details.

## Acknowledgments

- YOLO model developers
- Flask framework
- OpenCV contributors
- Tailwind CSS team 
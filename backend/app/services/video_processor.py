"""
AirType Video Processor

This module provides real-time hand tracking and coordinate extraction
using MediaPipe Hands and OpenCV.
"""

from typing import Optional, Tuple, List, Dict, Any
from dataclasses import dataclass
from collections import deque
import time

import cv2
import numpy as np

try:
    import mediapipe as mp
    MEDIAPIPE_AVAILABLE = True
except ImportError:
    MEDIAPIPE_AVAILABLE = False


@dataclass
class HandLandmark:
    """Represents a single hand landmark point."""
    x: float
    y: float
    z: float
    visibility: float = 1.0


@dataclass
class TrackedPoint:
    """Represents a tracked fingertip position."""
    x: float  # Normalized [0, 1]
    y: float  # Normalized [0, 1]
    timestamp_ms: int
    confidence: float
    is_drawing: bool  # Whether the finger is in drawing position


class KalmanFilter1D:
    """
    Simple 1D Kalman filter for smoothing coordinate tracking.
    
    Uses constant velocity model for prediction.
    """
    
    def __init__(
        self,
        process_noise: float = 0.01,
        measurement_noise: float = 0.1,
        initial_estimate: float = 0.5,
        initial_error: float = 1.0,
    ):
        """
        Initialize the Kalman filter.
        
        Args:
            process_noise: Process noise covariance (Q)
            measurement_noise: Measurement noise covariance (R)
            initial_estimate: Initial state estimate
            initial_error: Initial error covariance
        """
        # State: [position, velocity]
        self.state = np.array([initial_estimate, 0.0])
        self.P = np.eye(2) * initial_error
        
        # Process noise
        self.Q = np.array([
            [process_noise, 0],
            [0, process_noise * 0.1],
        ])
        
        # Measurement noise
        self.R = np.array([[measurement_noise]])
        
        # Measurement matrix (observe position only)
        self.H = np.array([[1, 0]])
        
        # State transition matrix (dt will be set during predict)
        self.dt = 1.0 / 30.0  # Default 30 FPS
    
    def predict(self, dt: Optional[float] = None) -> float:
        """
        Predict next state.
        
        Args:
            dt: Time step (uses default if not provided)
        
        Returns:
            Predicted position
        """
        if dt is not None:
            self.dt = dt
        
        # State transition matrix
        F = np.array([
            [1, self.dt],
            [0, 1],
        ])
        
        # Predict state
        self.state = F @ self.state
        
        # Predict error covariance
        self.P = F @ self.P @ F.T + self.Q
        
        return self.state[0]
    
    def update(self, measurement: float) -> float:
        """
        Update state with new measurement.
        
        Args:
            measurement: Observed position
        
        Returns:
            Updated (filtered) position
        """
        # Innovation
        y = measurement - self.H @ self.state
        
        # Innovation covariance
        S = self.H @ self.P @ self.H.T + self.R
        
        # Kalman gain
        K = self.P @ self.H.T @ np.linalg.inv(S)
        
        # Update state
        self.state = self.state + (K @ y).flatten()
        
        # Update error covariance
        I = np.eye(2)
        self.P = (I - K @ self.H) @ self.P
        
        return self.state[0]


class VideoProcessor:
    """
    Real-time video processor for hand tracking.
    
    Uses MediaPipe Hands to detect and track the index fingertip,
    with Kalman filtering for smooth trajectory tracking.
    """
    
    # MediaPipe hand landmark indices
    INDEX_FINGER_TIP = 8
    INDEX_FINGER_DIP = 7
    THUMB_TIP = 4
    
    def __init__(
        self,
        min_detection_confidence: float = 0.7,
        min_tracking_confidence: float = 0.5,
        max_num_hands: int = 1,
        enable_kalman_filter: bool = True,
        drawing_threshold: float = 0.1,  # Distance threshold for drawing detection
    ):
        """
        Initialize the video processor.
        
        Args:
            min_detection_confidence: Minimum confidence for hand detection
            min_tracking_confidence: Minimum confidence for tracking
            max_num_hands: Maximum number of hands to track
            enable_kalman_filter: Whether to use Kalman filtering
            drawing_threshold: Thumb-index distance threshold for drawing detection
        """
        self.min_detection_confidence = min_detection_confidence
        self.min_tracking_confidence = min_tracking_confidence
        self.max_num_hands = max_num_hands
        self.enable_kalman_filter = enable_kalman_filter
        self.drawing_threshold = drawing_threshold
        
        # Initialize MediaPipe
        if MEDIAPIPE_AVAILABLE:
            self.mp_hands = mp.solutions.hands
            self.mp_drawing = mp.solutions.drawing_utils
            self.hands = self.mp_hands.Hands(
                static_image_mode=False,
                max_num_hands=max_num_hands,
                min_detection_confidence=min_detection_confidence,
                min_tracking_confidence=min_tracking_confidence,
            )
        else:
            self.hands = None
        
        # Kalman filters for x and y coordinates
        self.kalman_x = KalmanFilter1D() if enable_kalman_filter else None
        self.kalman_y = KalmanFilter1D() if enable_kalman_filter else None
        
        # Tracking state
        self.last_timestamp: Optional[int] = None
        self.tracking_history: deque = deque(maxlen=100)
        self.is_tracking: bool = False
        
        # Frame dimensions
        self.frame_width: int = 640
        self.frame_height: int = 480
    
    def process_frame(
        self,
        frame: np.ndarray,
        timestamp_ms: Optional[int] = None,
    ) -> Tuple[Optional[TrackedPoint], np.ndarray]:
        """
        Process a single video frame.
        
        Args:
            frame: BGR image from camera
            timestamp_ms: Frame timestamp in milliseconds
        
        Returns:
            Tuple of (tracked_point, annotated_frame)
            tracked_point is None if no hand is detected
        """
        if timestamp_ms is None:
            timestamp_ms = int(time.time() * 1000)
        
        # Update frame dimensions
        self.frame_height, self.frame_width = frame.shape[:2]
        
        # Convert BGR to RGB for MediaPipe
        rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        
        # Process with MediaPipe
        tracked_point = None
        
        if MEDIAPIPE_AVAILABLE and self.hands is not None:
            results = self.hands.process(rgb_frame)
            
            if results.multi_hand_landmarks:
                # Get first hand
                hand_landmarks = results.multi_hand_landmarks[0]
                
                # Extract index fingertip
                index_tip = hand_landmarks.landmark[self.INDEX_FINGER_TIP]
                thumb_tip = hand_landmarks.landmark[self.THUMB_TIP]
                
                # Normalize coordinates to [0, 1]
                raw_x = index_tip.x
                raw_y = index_tip.y
                
                # Apply Kalman filtering
                if self.enable_kalman_filter:
                    # Calculate time delta
                    if self.last_timestamp is not None:
                        dt = (timestamp_ms - self.last_timestamp) / 1000.0
                        self.kalman_x.predict(dt)
                        self.kalman_y.predict(dt)
                    
                    filtered_x = self.kalman_x.update(raw_x)
                    filtered_y = self.kalman_y.update(raw_y)
                else:
                    filtered_x = raw_x
                    filtered_y = raw_y
                
                # Detect drawing state (pinch gesture)
                thumb_index_dist = np.sqrt(
                    (thumb_tip.x - index_tip.x) ** 2 +
                    (thumb_tip.y - index_tip.y) ** 2
                )
                is_drawing = thumb_index_dist < self.drawing_threshold
                
                # Create tracked point
                tracked_point = TrackedPoint(
                    x=filtered_x,
                    y=filtered_y,
                    timestamp_ms=timestamp_ms,
                    confidence=min(index_tip.visibility, 1.0) if hasattr(index_tip, 'visibility') else 1.0,
                    is_drawing=is_drawing,
                )
                
                # Update tracking history
                self.tracking_history.append(tracked_point)
                self.is_tracking = True
                
                # Draw hand landmarks
                self.mp_drawing.draw_landmarks(
                    frame,
                    hand_landmarks,
                    self.mp_hands.HAND_CONNECTIONS,
                )
                
                # Draw fingertip marker
                tip_x = int(filtered_x * self.frame_width)
                tip_y = int(filtered_y * self.frame_height)
                
                color = (0, 255, 0) if is_drawing else (0, 0, 255)
                cv2.circle(frame, (tip_x, tip_y), 10, color, -1)
                
                # Draw status text
                status = "Drawing" if is_drawing else "Tracking"
                cv2.putText(
                    frame, status,
                    (10, 30),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    1, color, 2,
                )
            else:
                self.is_tracking = False
        
        self.last_timestamp = timestamp_ms
        
        return tracked_point, frame
    
    def get_stroke_points(self, only_drawing: bool = True) -> List[Dict[str, Any]]:
        """
        Get tracked points as stroke data.
        
        Args:
            only_drawing: If True, only return points where is_drawing=True
        
        Returns:
            List of point dictionaries
        """
        points = []
        for point in self.tracking_history:
            if only_drawing and not point.is_drawing:
                continue
            
            points.append({
                "x": point.x,
                "y": point.y,
                "timestamp_ms": point.timestamp_ms,
                "confidence": point.confidence,
            })
        
        return points
    
    def clear_history(self):
        """Clear tracking history."""
        self.tracking_history.clear()
        self.is_tracking = False
    
    def release(self):
        """Release resources."""
        if MEDIAPIPE_AVAILABLE and self.hands is not None:
            self.hands.close()


class VideoCapture:
    """
    Wrapper for OpenCV VideoCapture with additional utilities.
    """
    
    def __init__(
        self,
        source: int = 0,
        width: int = 640,
        height: int = 480,
        fps: int = 30,
    ):
        """
        Initialize video capture.
        
        Args:
            source: Camera index or video file path
            width: Frame width
            height: Frame height
            fps: Target frame rate
        """
        self.cap = cv2.VideoCapture(source)
        
        # Set capture properties
        self.cap.set(cv2.CAP_PROP_FRAME_WIDTH, width)
        self.cap.set(cv2.CAP_PROP_FRAME_HEIGHT, height)
        self.cap.set(cv2.CAP_PROP_FPS, fps)
        
        self.width = int(self.cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        self.height = int(self.cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        self.fps = int(self.cap.get(cv2.CAP_PROP_FPS))
        
        self.frame_count = 0
        self.start_time = time.time()
    
    def read(self) -> Tuple[bool, Optional[np.ndarray]]:
        """
        Read a frame from the capture.
        
        Returns:
            Tuple of (success, frame)
        """
        ret, frame = self.cap.read()
        if ret:
            self.frame_count += 1
        return ret, frame
    
    def get_actual_fps(self) -> float:
        """Calculate actual FPS."""
        elapsed = time.time() - self.start_time
        if elapsed > 0:
            return self.frame_count / elapsed
        return 0.0
    
    def is_opened(self) -> bool:
        """Check if capture is opened."""
        return self.cap.isOpened()
    
    def release(self):
        """Release the capture."""
        self.cap.release()


def process_video_stream(
    source: int = 0,
    on_point: Optional[callable] = None,
    on_stroke_complete: Optional[callable] = None,
    show_preview: bool = True,
):
    """
    Process video stream and track finger movements.
    
    Args:
        source: Camera index
        on_point: Callback for each tracked point
        on_stroke_complete: Callback when stroke is completed
        show_preview: Whether to show preview window
    """
    cap = VideoCapture(source=source)
    processor = VideoProcessor()
    
    stroke_points = []
    was_drawing = False
    
    print("Press 'q' to quit, 'c' to clear current stroke")
    
    try:
        while cap.is_opened():
            ret, frame = cap.read()
            if not ret:
                break
            
            # Mirror frame for natural interaction
            frame = cv2.flip(frame, 1)
            
            # Process frame
            timestamp_ms = int(time.time() * 1000)
            point, annotated = processor.process_frame(frame, timestamp_ms)
            
            if point is not None:
                if point.is_drawing:
                    stroke_points.append({
                        "x": point.x,
                        "y": point.y,
                        "timestamp_ms": point.timestamp_ms,
                    })
                    
                    if on_point:
                        on_point(point)
                    
                    was_drawing = True
                elif was_drawing:
                    # Stroke completed
                    if len(stroke_points) >= 5 and on_stroke_complete:
                        on_stroke_complete(stroke_points)
                    
                    stroke_points = []
                    was_drawing = False
            
            # Display FPS
            fps_text = f"FPS: {cap.get_actual_fps():.1f}"
            cv2.putText(
                annotated, fps_text,
                (10, annotated.shape[0] - 10),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.5, (255, 255, 255), 1,
            )
            
            if show_preview:
                cv2.imshow("AirType - Hand Tracking", annotated)
            
            key = cv2.waitKey(1) & 0xFF
            if key == ord("q"):
                break
            elif key == ord("c"):
                stroke_points = []
                processor.clear_history()
    
    finally:
        cap.release()
        processor.release()
        cv2.destroyAllWindows()


if __name__ == "__main__":
    # Demo mode
    def on_point(p):
        print(f"Point: ({p.x:.3f}, {p.y:.3f}) drawing={p.is_drawing}")
    
    def on_stroke_complete(points):
        print(f"Stroke complete: {len(points)} points")
    
    process_video_stream(
        source=0,
        on_point=on_point,
        on_stroke_complete=on_stroke_complete,
        show_preview=True,
    )

# FaceDetect - Event-based Face Detection Module
# For privacy-preserving face detection in event camera data

from .model import EventFaceDetector, EventVoxelEncoder, create_model

__all__ = [
    'EventFaceDetector',
    'EventVoxelEncoder',
    'create_model',
]

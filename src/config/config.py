"""
Config File
"""

import cv2

# Mapping of classes to clinical severity (1=routine, 5=urgent)
GRAVITE_CLINIQUE: dict[str, int] = {
    "glioma": 5,  # Malignant tumor - urgent
    "meningioma": 3,  # Benign tumor but possible intervention
    "pituitary": 3,  # Benign tumor but hormonal impact
    "notumor": 1,  # No tumor - routine if high confidence
}

# OpenCV est une extension C, pylint peut ne pas résoudre ses membres statiquement.
CV_IMREAD = getattr(cv2, "imread")
CV_IMREAD_COLOR = getattr(cv2, "IMREAD_COLOR")
CV_CVT_COLOR = getattr(cv2, "cvtColor")
CV_COLOR_BGR2RGB = getattr(cv2, "COLOR_BGR2RGB")
CV_RESIZE = getattr(cv2, "resize")
CV_INTER_AREA = getattr(cv2, "INTER_AREA")

"""New page and the no-hallucination fallback hand detector."""
from dataclasses import dataclass

import cv2
import numpy as np
from fastapi.testclient import TestClient

from src.site import app
from src.gesture_runtime import FeatureExtractor


def test_play_page_and_existing_nav():
    c = TestClient(app)
    r = c.get('/play')
    assert r.status_code == 200
    assert 'CHECK MY FINGERS' in r.text
    assert '/static/js/play.js' in r.text
    for route in ('/', '/studio', '/guide'):
        assert 'href="/play"' in c.get(route).text
    assert '/api/analyze/numbers' in c.get('/static/js/play.js').text


def test_no_hand_returns_none_with_attempt_name():
    class Missing:
        def process(self, rgb):
            return type('Result', (), {'multi_hand_landmarks': None})()

    fx = FeatureExtractor.__new__(FeatureExtractor)
    fx.backend = 'mediapipe'
    fx.hands = Missing()
    fx.last_detection_attempt = 'not_attempted'
    image = np.full((260, 340, 3), 55, dtype=np.uint8)
    feat, bbox, pts, blur = fx.extract(image)
    assert feat is None and bbox is None and pts is None
    assert fx.last_detection_attempt == 'none'


def test_padded_detector_inverse_maps_landmarks_to_original_coordinates():
    class Landmark:
        def __init__(self, x, y, z=0):
            self.x, self.y, self.z = x, y, z

    class Detector:
        def __init__(self):
            self.calls = 0
        def process(self, rgb):
            self.calls += 1
            if self.calls < 3:
                return type('Result', (), {'multi_hand_landmarks': None})()
            # The 3rd attempt has padding of int(.17 * 320) = 54 pixels.
            h, w = rgb.shape[:2]
            pts = [Landmark((54+160)/w + (i-9)*.001, (54+100)/h, 0) for i in range(21)]
            return type('Result', (), {
                'multi_hand_landmarks': [type('Hand', (), {'landmark': pts})()]
            })()

    fx = FeatureExtractor.__new__(FeatureExtractor)
    fx.backend = 'mediapipe'
    fx.hands = Detector()
    fx.last_detection_attempt = 'not_attempted'
    image = np.full((200, 320, 3), 90, dtype=np.uint8)
    feature, box, points, blur = fx.extract(image)
    assert fx.hands.calls == 3
    assert fx.last_detection_attempt == 'padded_contrast'
    assert feature is not None and len(feature) > 63
    assert points is not None and len(points) == 21
    assert abs(points[9][0] - 112) <= 1  # middle joint image center -> x=112/224
    assert abs(points[9][1] - 112) <= 1

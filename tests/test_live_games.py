"""Backend and UI tests for live webcam game endpoints.

No fake accuracy is claimed; only request/response contracts are mocked.
"""
import io
import cv2
import numpy as np
from fastapi.testclient import TestClient
from src import site
from src.gesture_runtime import Prediction


def _jpeg():
    image = np.zeros((200, 260, 3), dtype=np.uint8)
    ok, data = cv2.imencode('.jpg', image)
    assert ok
    return data.tobytes()


class _FakeRecognizer:
    def __init__(self):
        self.extractor=type('Extractor', (), {'last_detection_attempt': 'original'})()
    def predict_both(self, frame):
        assert frame.shape[:2] == (200, 260)
        core=Prediction('Peace', .88, (30,40,170,180), [[101,102]]*21, 10.0, 23.0, 'mediapipe-hagrid-six')
        number=Prediction('Number_2', .73, (30,40,170,180), [[101,102]]*21, 10.0, 23.0, 'mediapipe-hagrid-numbers')
        return core, number


def test_live_endpoint_no_full_png_and_includes_landmarks(monkeypatch):
    monkeypatch.setattr(site, 'live_recognizer', lambda: _FakeRecognizer())
    c=TestClient(site.app)
    for mode,label in [('core','Peace'),('numbers','Number_2')]:
        r=c.post(f'/api/live?mode={mode}', files={'image':('frame.jpg', _jpeg(), 'image/jpeg')})
        assert r.status_code==200, r.text
        result=r.json()
        assert 'image' not in result
        assert result['details']['gesture']==label
        assert len(result['details']['points'])==21
        assert result['details']['detection_attempt']=='original'
    invalid=c.post('/api/live?mode=unknown',files={'image':('frame.jpg',_jpeg(),'image/jpeg')})
    assert invalid.status_code==422


def test_play_and_studio_assets_are_live():
    c=TestClient(site.app)
    home=c.get('/').text;play=c.get('/play').text;studio=c.get('/studio').text;guide=c.get('/guide').text
    assert 'href="/play?game=math"' in home
    assert 'data-game="memory"' in play
    assert 'id="repeatPrompt"' in play
    assert 'id="liveCanvas"' in studio
    assert 'data-say-number="1"' in guide
    assert '/api/live?mode=core' in c.get('/static/js/studio.js').text
    assert 'speechSynthesis' in c.get('/static/js/play.js').text
    assert c.get('/static/audio/star_steps_original.wav').status_code==200

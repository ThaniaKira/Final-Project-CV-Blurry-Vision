import numpy as np
from pathlib import Path
from fastapi.testclient import TestClient

from src.dual_models import CORE_CLASSES, NUMBER_CLASSES, DualRecognizer, number_by_geometry
from src.site import app


def test_expected_independent_head_classes():
    assert len(CORE_CLASSES) == 6
    assert NUMBER_CLASSES == ('Number_0','Number_1','Number_2','Number_3','Number_4','Number_5')


def test_numeric_rules_reject_noncanonical_pose():
    points=np.zeros((21,2),float)
    assert number_by_geometry(points) is None


def test_number_route_rejects_invalid_image_without_running_detector():
    r=TestClient(app).post('/api/analyze/numbers',files={'image':('bad.png',b'not an image','image/png')})
    assert r.status_code == 400


def test_status_contains_both_independent_model_states():
    status=TestClient(app).get('/api/status').json()
    assert 'number_model' in status
    assert 'mode' in status['number_model']
    assert status['number_model']['mode'] in ('trained','rules')

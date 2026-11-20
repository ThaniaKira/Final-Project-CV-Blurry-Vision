import numpy as np

from src.geometry_rules import classify_geometry


def test_extended_downward_thumb_is_dislike():
    # Folded fingers; a thumb that extends substantially below the wrist.
    points = np.zeros((21, 2), dtype=float)
    points[0] = (100, 100)
    points[9] = (100, 140)  # palm scale
    points[1], points[2], points[3], points[4] = (100, 100), (80, 130), (70, 160), (60, 190)
    for mcp, pip, tip in ((5, 6, 8), (9, 10, 12), (13, 14, 16), (17, 18, 20)):
        points[mcp], points[pip], points[tip] = (110, 110), (100, 105), (101, 105)
    assert classify_geometry(points)[0] == "Dislike"

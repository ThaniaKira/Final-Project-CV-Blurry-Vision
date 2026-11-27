"""Static regression checks for child-friendly game enhancements (no ML runtime required)."""
from pathlib import Path

PROJECT = Path(__file__).resolve().parents[1]


def test_dedicated_guides_not_predictions():
    guides = PROJECT / 'web/static/img/pose-guides'
    for name in ['fist', 'open-hand', 'peace', 'thumbs-up', 'pointing', 'ok']:
        data = (guides / f'{name}.png').read_bytes()
        assert data.startswith(b'\x89PNG\r\n\x1a\n') and len(data) > 1000
    js = (PROJECT / 'web/static/js/play.js').read_text(encoding='utf-8')
    assert '/static/img/pose-guides/' in js
    assert 'image.src=`/static/samples/' not in js


def test_voice_and_timers_have_real_logic():
    js = (PROJECT / 'web/static/js/play.js').read_text(encoding='utf-8')
    html = (PROJECT / 'web/templates/play.html').read_text(encoding='utf-8')
    assert 'const TIME_LIMITS = {math: 60, memory: 75}' in js
    assert 'First time:' in js and 'Now, one more time!' in js
    assert 'confidence<0.50' in js
    assert 'POSE_NAMES[expected()]' in js
    assert 'shuffle([1,2,3,4,5])' in js
    assert 'id="voiceChoice"' in html
    assert 'id="challengeTimer"' in html
    assert 'id="repeatSequence"' in html

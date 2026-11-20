"""Create the printable gesture reference for Blurry Vision."""
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
from matplotlib.backends.backend_pdf import PdfPages
import matplotlib.pyplot as plt


GESTURES = [
    ("Open Hand", "Hold all four fingers straight and separated."),
    ("Fist", "Curl all four fingers into the palm."),
    ("Peace", "Extend the index and middle fingers only."),
    ("Thumbs Up", "Close the fingers and extend the thumb outward."),
    ("Pointing", "Extend the index finger only."),
    ("OK", "Touch thumb and index fingertips; keep middle and ring fingers raised."),
    ("Finger Heart", "Pinch thumb and curled index fingertip together; keep the other fingers folded."),
    ("Call Me / Shaka", "Extend the thumb and pinky; curl the other fingers."),
    ("I Love You", "Extend thumb, index, and pinky; curl middle and ring fingers."),
    ("Rock", "Extend the index and pinky; keep thumb, middle, and ring folded."),
    ("Three Fingers", "Extend index, middle, and ring; fold pinky."),
]


def add_text_page(pdf, title, subtitle, rows):
    fig = plt.figure(figsize=(8.27, 11.69), facecolor="#100b1a")
    ax = fig.add_axes([0, 0, 1, 1])
    ax.set_axis_off()
    ax.text(.08, .91, title, color="#f7f2ff", fontsize=28, fontweight="bold", va="top")
    ax.text(.08, .85, subtitle, color="#c8b8dc", fontsize=11, va="top", wrap=True)
    y = .76
    for index, (name, instruction) in enumerate(rows, start=1):
        ax.text(.09, y, f"{index:02d}", color="#ba8cff", fontsize=12, fontweight="bold", va="center")
        ax.text(.18, y + .012, name, color="#f2eaff", fontsize=15, fontweight="bold", va="center")
        ax.text(.18, y - .025, instruction, color="#c8bfd2", fontsize=10.5, va="center", wrap=True)
        ax.plot([.08, .92], [y - .06, y - .06], color="#4d3b64", linewidth=.7)
        y -= .115
    ax.text(.08, .055, "Blurry Vision • Gesture reference", color="#9b88b5", fontsize=9)
    pdf.savefig(fig, facecolor=fig.get_facecolor(), bbox_inches="tight")
    plt.close(fig)


def main():
    output = Path(__file__).with_name("gesture_reference.pdf")
    with PdfPages(output) as pdf:
        add_text_page(
            pdf,
            "Gesture reference",
            "Supported poses for Blurry Vision's MediaPipe landmark-rule mode. Keep one hand clearly visible, face the camera, and use even lighting.",
            GESTURES[:6],
        )
        add_text_page(
            pdf,
            "More supported poses",
            "The four new poses are heuristic landmark rules. They are not added to the optional trained HaGRID model, which remains trained only on its original six labels.",
            GESTURES[5:],
        )
    print(output)


if __name__ == "__main__":
    main()

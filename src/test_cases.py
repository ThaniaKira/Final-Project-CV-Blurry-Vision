"""Generate and execute 12 documented CP4 cases on the provided six illustrated test samples.

Transformed stress cases are *not* independent held-out test samples; they are
separate demonstration probes. Failure results are recorded honestly.
"""
import os as _os, sys as _sys
_sys.path.insert(0, _os.path.abspath(_os.path.join(_os.path.dirname(__file__), "..")))

import argparse
import csv
from pathlib import Path
import cv2
import numpy as np
from src.gesture_runtime import GestureRecognizer,draw_result


def make_cases(test_dir="data/test"):
    files={p.parent.name:p for p in Path(test_dir).rglob("*.png")}
    for label in ("Open_Hand","Fist","Peace","Thumbs_Up","Pointing","OK"):
        if label not in files:raise FileNotFoundError(f"Missing test example for {label}")
    cases=[]
    for i,label in enumerate(("Open_Hand","Fist","Peace","Thumbs_Up","Pointing","OK"),start=1):
        cases.append((f"TC{i:02}",str(files[label]),"original held-out illustrated test image",label,lambda x:x))
    cases.extend([
        ("TC07",str(files["Peace"]),"hard: very dim image","Peace",lambda x:cv2.convertScaleAbs(x,alpha=0.25,beta=0)),
        ("TC08",str(files["Fist"]),"hard: heavy motion-like blur","Fist",lambda x:cv2.GaussianBlur(x,(31,31),0)),
        ("TC09",str(files["Pointing"]),"hard: very small gesture","Pointing",lambda x:small_hand(x)),
        ("TC10",str(files["OK"]),"hard: 40-degree tilt","OK",lambda x:cv2.warpAffine(x,cv2.getRotationMatrix2D((112,112),40,0.85),(224,224),borderMode=cv2.BORDER_REFLECT)),
        ("TC11",str(files["Thumbs_Up"]),"hard: large brightness shift","Thumbs_Up",lambda x:cv2.convertScaleAbs(x,alpha=1.6,beta=50)),
        ("TC12","generated","no gesture/empty white field","No hand detected",lambda _:np.full((224,224,3),255,dtype=np.uint8)),
    ])
    return cases


def small_hand(x):
    canvas=np.full_like(x,255)
    miniature=cv2.resize(x,(54,54))
    canvas[85:139,85:139]=miniature
    return canvas


def test_all(weights="models/gesture_contour.npz",output="outputs/test_cases"):
    out=Path(output);out.mkdir(parents=True,exist_ok=True)
    cases=make_cases()
    model=GestureRecognizer(weights)
    rows=[]
    try:
        for id,path,condition,expected,modify in cases:
            frame=cv2.imread(path) if path!="generated" else None
            img=modify(frame)
            pred=model.predict(img)
            saved=out/f"{id}.png"
            cv2.imwrite(str(saved),draw_result(img,pred))
            passed=pred.label==expected
            rows.append({"id":id,"input":path,"condition":condition,"expected":expected,
                         "actual":pred.label,"confidence":f"{pred.confidence:.3f}",
                         "pass_fail":"PASS" if passed else "FAIL","output":str(saved)})
    finally:
        model.close()
    with open(out/"test_cases.csv","w",newline="",encoding="utf-8") as f:
        writer=csv.DictWriter(f,fieldnames=list(rows[0]));writer.writeheader();writer.writerows(rows)
    print("Test cases:",sum(x["pass_fail"]=="PASS" for x in rows),"/",len(rows),"passed")
    print("Saved",out/"test_cases.csv")
    return rows


if __name__=="__main__":
    p=argparse.ArgumentParser()
    p.add_argument("--weights",default="models/gesture_contour.npz")
    p.add_argument("--output",default="outputs/test_cases")
    a=p.parse_args();test_all(a.weights,a.output)

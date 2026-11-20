"""Train a six-class classifier from REAL HaGRID hand landmarks.

The train split alone fits the model; validation is used for checkpoint selection
and reporting. The held-out test split is NEVER inspected or optimized here.

Example:
 python src/train.py --backend mediapipe --data data/hagrid_real --output models/gesture_hagrid_mediapipe.npz
"""
import os as _os, sys as _sys
_sys.path.insert(0, _os.path.abspath(_os.path.join(_os.path.dirname(__file__), "..")))

import argparse
import json
from pathlib import Path
import numpy as np
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, f1_score

from src.data_loader import GestureDataLoader
from src.gesture_runtime import FeatureExtractor, CLASSES


def extract_split(extractor, images, labels, paths, split):
    features, y, valid_paths = [], [], []
    n=len(images)
    for i,(image,label,path) in enumerate(zip(images,labels,paths)):
        try:
            vector,box,_,_=extractor.extract(image)
        except Exception as exc:
            raise RuntimeError(f"Cannot extract {path} in {split}: {exc}") from exc
        if vector is not None:
            features.append(vector); y.append(label); valid_paths.append(path)
        if (i+1)%200==0 or i+1==n:
            print(f"  {split}: processed {i+1}/{n} photographs; detected {len(features)} hands",flush=True)
    if not features:
        raise RuntimeError(f"No real hands detected in {split} images. Verify MediaPipe and Kaggle image cropping")
    return np.asarray(features,dtype=np.float32),np.asarray(y),valid_paths


def portable_save(clf,backend,output,metadata):
    trees=[est.tree_ for est in clf.estimators_]
    count=len(trees);nodes=max(t.node_count for t in trees);nclass=len(clf.classes_)
    feat=np.zeros((count,nodes),dtype=np.int32)
    thresholds=np.zeros((count,nodes),dtype=np.float64)
    left=np.full((count,nodes),-1,dtype=np.int32)
    right=np.full((count,nodes),-1,dtype=np.int32)
    values=np.zeros((count,nodes,nclass),dtype=np.float64)
    for i,t in enumerate(trees):
        n=t.node_count;feat[i,:n]=t.feature;thresholds[i,:n]=t.threshold
        left[i,:n]=t.children_left;right[i,:n]=t.children_right
        values[i,:n,:]=t.value[:,0,:]
    Path(output).parent.mkdir(parents=True,exist_ok=True)
    np.savez_compressed(output,schema="blurry-vision-npz-v1",backend=backend,
                        classes=np.asarray(clf.classes_,dtype="U32"),
                        features=feat,thresholds=thresholds,left=left,right=right,values=values,
                        training_count=metadata["train_detected"],seed=metadata["seed"],
                        training_metadata=json.dumps(metadata))


def train(backend="mediapipe",output=None,root="data/hagrid_real",seed=42,mirror=True):
    if output is None:
        output="models/gesture_hagrid_mediapipe.npz" if backend=="mediapipe" else "models/gesture_contour.npz"
    root=Path(root)
    if backend=="mediapipe" and root.resolve()==Path("data").resolve():
        raise RuntimeError("Do not train a MediaPipe model on the original 20 illustrated images! "
                           "Prepare Kaggle HaGRID photos first (docs/KAGGLE_TRAINING.md)")
    if backend=="mediapipe" and not (root/"data_card.json").exists():
        raise RuntimeError(f"Missing {root/'data_card.json'}; prepare Kaggle data before training")
    data=GestureDataLoader(str(root))
    train_images,train_labels,train_paths=data.load_split("train")
    val_images,val_labels,val_paths=data.load_split("val")
    if not train_images or not val_images:
        raise RuntimeError("train/ and val/ require labeled images")
    extractor=FeatureExtractor(backend)
    try:
        X,y,paths=extract_split(extractor,train_images,train_labels,train_paths,"train")
        V,vy,vpaths=extract_split(extractor,val_images,val_labels,val_paths,"val")
    finally:
        extractor.close()
    if len(set(y))!=len(CLASSES):
        raise RuntimeError(f"Training lacks class(es): {set(CLASSES)-set(y)}")
    if len(X)<36:
        raise RuntimeError(f"Only {len(X)} detected real hands. More Kaggle samples needed.")
    if mirror and backend=="mediapipe":
        # Mirror 3D landmark x in wrist-relative coordinates; train-only augmentation.
        mirrored=X.reshape(-1,21,3).copy()
        mirrored[:,:,0]*=-1
        X=np.vstack((X,mirrored.reshape(-1,63)))
        y=np.hstack((y,y))
    candidate_settings=[(160,2,22),(240,1,22),(240,2,None)]
    candidates=[]
    for n_estimators,min_leaf,max_depth in candidate_settings:
        clf=RandomForestClassifier(n_estimators=n_estimators,min_samples_leaf=min_leaf,
                                   max_depth=max_depth,random_state=seed,n_jobs=-1,
                                   class_weight="balanced_subsample")
        clf.fit(X,y)
        vp=clf.predict(V)
        f1=float(f1_score(vy,vp,labels=list(CLASSES),average="macro",zero_division=0))
        acc=float(accuracy_score(vy,vp))
        print(f"Candidate {n_estimators} trees, leaf {min_leaf}, depth {max_depth}: "
              f"validation accuracy={acc:.3f}, macro-F1={f1:.3f}")
        candidates.append((f1,acc,clf,(n_estimators,min_leaf,max_depth)))
    _,_,best,config=max(candidates,key=lambda item:(item[0],item[1]))
    selection_predictions=best.predict(V)
    # Count failed MediaPipe detections against labels, rather than reporting
    # inflated detection-conditional accuracy.
    success={k:0 for k in CLASSES}
    for cls in vy:success[str(cls)]+=1
    coverage=len(V)/len(val_labels)
    val_correct=sum(a==b for a,b in zip(vy,selection_predictions))
    accuracy_full=val_correct/len(val_labels)
    from sklearn.metrics import f1_score as f1fn
    all_val_preds=[]
    valid_pred_iter=iter(selection_predictions.tolist())
    valid_paths=set(vpaths)
    for path in val_paths:
        all_val_preds.append(next(valid_pred_iter) if path in valid_paths else "No hand detected")
    full_f1=float(f1fn(val_labels,all_val_preds,labels=list(CLASSES),average="macro",zero_division=0))
    metadata={"dataset":str(root),"dataset_source":"https://www.kaggle.com/datasets/innominate817/hagrid-sample-30k-384p",
              "seed":seed,"backend":backend,"train_detected":len(paths),"train_images":len(train_labels),
              "val_detected":len(vpaths),"val_images":len(val_labels),"validation_detection_coverage":coverage,
              "validation_accuracy_all_images":accuracy_full,"validation_macro_f1_all_images":full_f1,
              "mirrored_train_only":bool(mirror),"selected_rf":{"n_estimators":config[0],"min_samples_leaf":config[1],"max_depth":config[2]},
              "warning":"Validation not test; test results require src/evaluate.py. Real webcam performance is separate."}
    portable_save(best,backend,output,metadata)
    Path(output).with_suffix(".training.json").write_text(json.dumps(metadata,indent=2),encoding="utf-8")
    print(f"\nSaved: {output}; validation on ALL photos (no-hand counted wrong): "
          f"accuracy={accuracy_full:.4f}; macro-F1={full_f1:.4f}; detection coverage={coverage:.1%}")
    return metadata

if __name__=="__main__":
    p=argparse.ArgumentParser()
    p.add_argument("--backend",choices=["contour","mediapipe"],default="mediapipe")
    p.add_argument("--data",default="data/hagrid_real")
    p.add_argument("--output",default=None)
    p.add_argument("--seed",type=int,default=42)
    p.add_argument("--no-mirror",action="store_true",help="Disable training-only horizontal reflection")
    a=p.parse_args()
    train(a.backend,a.output,a.data,a.seed,not a.no_mirror)

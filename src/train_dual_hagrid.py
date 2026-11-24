"""Train TWO independent genuine HaGRID classifiers with little disk usage.

Training runs on the computer executing this script (preferably Kaggle/Colab),
not on the child's webcam laptop. Reads images directly from an existing HaGRID
30k ZIP, without extracting/saving all JPGs. It does NOT download HaGRID.

Example after placing the public hagrid-sample-30k-384p.zip in cloud storage:
  python src/train_dual_hagrid.py --archive /kaggle/input/.../hagrid-sample-30k-384p.zip

Output: models/gesture_hagrid_mediapipe.npz and models/numbers_hagrid_mediapipe.npz
        plus a truthful evaluation JSON for each. NEVER claim 100% a priori.
"""
from __future__ import annotations
import os as _os, sys as _sys
_sys.path.insert(0, _os.path.abspath(_os.path.join(_os.path.dirname(__file__), '..')))

import argparse, json, random, time
from collections import Counter, defaultdict
from pathlib import Path

import cv2
import numpy as np
from sklearn.ensemble import ExtraTreesClassifier, RandomForestClassifier
from sklearn.metrics import accuracy_score, f1_score, classification_report, confusion_matrix

from src.gesture_runtime import FeatureExtractor
from src.train import portable_save
from src.prepare_kaggle import ArchiveReader, choose_bbox, crop_hand, stable_split_people

CORE_MAP = {"palm":"Open_Hand", "fist":"Fist", "peace":"Peace", "like":"Thumbs_Up", "one":"Pointing", "ok":"OK"}
NUMBER_MAP = {"fist":"Number_0", "one":"Number_1", "peace":"Number_2", "two_up":"Number_2",
              "three":"Number_3", "three2":"Number_3", "four":"Number_4", "palm":"Number_5"}
SOURCE = "https://huggingface.co/datasets/cj-mills/hagrid-sample-30k-384p"


def choose_records(reader, limit, seed):
    rows = []
    rng = random.Random(seed)
    labels = sorted(set(CORE_MAP)|set(NUMBER_MAP))
    for source_class in labels:
        annotation_path = reader.match(f"/ann_train_val/{source_class}.json") or reader.match(f"ann_train_val/{source_class}.json")
        if not annotation_path:
            raise FileNotFoundError(f"Missing HaGRID annotations for {source_class}; expected ann_train_val/{source_class}.json")
        ann = json.loads(reader.read(annotation_path))
        image_files = {Path(p).stem:p for p in reader.names if f"/train_val_{source_class}/" in p and p.lower().endswith(('.jpg','.jpeg','.png'))}
        items = []
        for image_id, data in ann.items():
            if image_id not in image_files: continue
            box = choose_bbox(data, source_class)
            if box is None: continue
            person = data.get('user_id')
            if not person: continue
            items.append({'id': image_id, 'class': source_class, 'user_id': str(person),
                          'bbox': box, 'source': image_files[image_id]})
        rng.shuffle(items)
        items = items[:limit] if limit else items
        print(f"{source_class}: selected {len(items)} matched photos", flush=True)
        if len(items) < 30: raise ValueError(f"Insufficient photos for {source_class}")
        rows.extend(items)
    assignment = stable_split_people(rows,seed)
    for row in rows: row['split'] = assignment[row['user_id']]
    return rows


def extract_features(reader, rows):
    extractor = FeatureExtractor('mediapipe')
    result = []
    try:
        for idx,row in enumerate(rows,1):
            crop = crop_hand(reader.read(row['source']), row['bbox'], pad_ratio=.35)
            if crop is None: continue
            vec,_,_,_ = extractor.extract(crop)
            if vec is not None:
                result.append((row, vec))
            if idx % 100 == 0 or idx == len(rows):
                print(f'  MediaPipe {idx}/{len(rows)}; hands detected: {len(result)}',flush=True)
    finally:
        extractor.close()
    return result


def evaluate_all(clf, test_rows, features, mapping, classes):
    # Preserve missed detections in the denominator, including zero coverage.
    lookup = {row['id']:vec for row,vec in features if row['class'] in mapping}
    matched_rows = [r for r in test_rows if r['class'] in mapping]
    y = [mapping[r['class']] for r in matched_rows]
    pred = [str(clf.predict(lookup[r['id']][None,:])[0]) if r['id'] in lookup else 'No hand detected' for r in matched_rows]
    return {
        'n_images':len(y), 'hands_detected':sum(r['id'] in lookup for r in matched_rows),
        'accuracy_all_images':float(accuracy_score(y,pred)),
        'macro_f1_all_images':float(f1_score(y,pred,labels=classes,average='macro',zero_division=0)),
        'per_class_report':classification_report(y,pred,labels=classes,output_dict=True,zero_division=0),
        'confusion_labels':classes+['No hand detected'],
        'confusion_matrix':confusion_matrix(y,pred,labels=classes+['No hand detected']).tolist()
    }


def train_head(name, mapping, rows, features, output, seed):
    classes = sorted(set(mapping.values()))
    def xy(split):
        selected = [(v,mapping[r['class']]) for r,v in features if r['split']==split and r['class'] in mapping]
        if not selected: raise RuntimeError(f'{name}: no successful hands in {split}')
        return np.asarray([v for v,_ in selected],np.float32), np.asarray([y for _,y in selected])
    X,y = xy('train'); V,vy = xy('val')
    for split, target in (('train',y),('val',vy)):
        missing=set(classes)-set(target)
        if missing: raise RuntimeError(f'{name}: missing labels in {split}: {missing}')
    candidates = [
        ExtraTreesClassifier(n_estimators=240,random_state=seed,n_jobs=-1,class_weight='balanced'),
        RandomForestClassifier(n_estimators=240,random_state=seed,n_jobs=-1,class_weight='balanced_subsample'),
    ]
    history=[]
    for clf in candidates:
        clf.fit(X,y)
        vpred=clf.predict(V)
        score=float(f1_score(vy,vpred,labels=classes,average='macro',zero_division=0))
        history.append((score, float(accuracy_score(vy,vpred)),clf))
        print(f'{name}: {type(clf).__name__} validation macro-F1 {score:.3f}',flush=True)
    _,_,best=max(history,key=lambda record:(record[0],record[1]))
    val_rows=[r for r in rows if r['split']=='val']
    test_rows=[r for r in rows if r['split']=='test']
    validation=evaluate_all(best,val_rows,features,mapping,classes)
    test=evaluate_all(best,test_rows,features,mapping,classes)
    model_info={'dataset_source':SOURCE,'dataset_version':'HaGRID 30K 384p sample',
                'gesture_head':name,'feature_schema':'landmark-v2',
                'split_by':'original HaGRID user_id, globally subject-disjoint',
                'class_source_map':mapping,'seed':seed,'train_detected':int(len(X)),
                'train_images':sum(r['split']=='train' and r['class'] in mapping for r in rows),
                'selected_algorithm':type(best).__name__,
                'validation_accuracy_all_images':validation['accuracy_all_images'],
                'validation_macro_f1_all_images':validation['macro_f1_all_images'],
                'test_accuracy_all_images':test['accuracy_all_images'],
                'test_macro_f1_all_images':test['macro_f1_all_images'],
                'caution':'Only measured results; children webcam performance requires separate testing.'}
    portable_save(best,'mediapipe',output,model_info)
    report={'metadata':model_info,'validation':validation,'test':test}
    Path(output).with_suffix('.evaluation.json').write_text(json.dumps(report,indent=2),encoding='utf-8')
    print(f'{name}: HELD-OUT TEST accuracy={test["accuracy_all_images"]:.3f}, macro-F1={test["macro_f1_all_images"]:.3f} (missed detections counted wrong)',flush=True)
    return report


def run(archive=None, source_dir=None, max_per_class=250, output_dir='models', seed=42):
    if bool(archive)==bool(source_dir): raise ValueError('Supply one of --archive OR --source-dir')
    reader=ArchiveReader(Path(archive) if archive else None, Path(source_dir) if source_dir else None)
    try:
        rows=choose_records(reader,max_per_class,seed)
        features=extract_features(reader,rows)
    finally:
        reader.close()
    output=Path(output_dir);output.mkdir(parents=True,exist_ok=True)
    heads={
      'core-six':train_head('core-six',CORE_MAP,rows,features,output/'gesture_hagrid_mediapipe.npz',seed),
      'numbers':train_head('numbers',NUMBER_MAP,rows,features,output/'numbers_hagrid_mediapipe.npz',seed)
    }
    (output/'dual_models_report.json').write_text(json.dumps(heads,indent=2),encoding='utf-8')
    return heads


if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__)
    g=p.add_mutually_exclusive_group(required=True)
    g.add_argument('--archive',help='Path to HaGRID sample ZIP in cloud storage')
    g.add_argument('--source-dir',help='Extracted HaGRID sample directory in cloud storage')
    p.add_argument('--max-per-class',type=int,default=250)
    p.add_argument('--output-dir',default='models')
    p.add_argument('--seed',type=int,default=42)
    args=p.parse_args()
    run(args.archive,args.source_dir,args.max_per_class,args.output_dir,args.seed)

"""Offline test of the real HaGRID ZIP importer using fake images and annotations.

These mock images only exercise file layout/splitting/cropping. They are never
used to claim model accuracy, and are not packaged as training photographs.
"""
from pathlib import Path
import tempfile
import zipfile
import json
import csv
import cv2
import numpy as np
from src.prepare_kaggle import prepare, CLASS_MAP


def test_kaggle_zip_ingestion_and_subject_split():
    with tempfile.TemporaryDirectory() as tmp:
        tmp=Path(tmp)
        archive=tmp/'fake_hagrid.zip'
        prefix='hagrid-sample-30k-384p'
        with zipfile.ZipFile(archive,'w',compression=zipfile.ZIP_DEFLATED) as z:
            for clazz in CLASS_MAP:
                annotations={}
                for i in range(60):
                    name=f'{clazz}_{i:03d}'
                    annotations[name]={"bboxes":[[0.2,0.2,0.6,0.6]],"labels":[clazz],
                                       "user_id":f'subject_{i:03d}'}
                    im=np.ones((64,96,3),dtype=np.uint8)*210
                    cv2.circle(im,(40,30),17,(20,50,130),-1)
                    ok,data=cv2.imencode('.jpg',im)
                    assert ok
                    z.writestr(f'{prefix}/hagrid_30k/train_val_{clazz}/{name}.jpg',data.tobytes())
                z.writestr(f'{prefix}/ann_train_val/{clazz}.json',json.dumps(annotations))
        output=tmp/'real'
        info=prepare(archive=archive,output=output,max_per_class=60)
        assert info['prepared_n']==360
        with open(output/'manifest.csv',newline='') as f:
            records=list(csv.DictReader(f))
        groups={split:{row['user_id'] for row in records if row['split']==split} for split in ('train','val','test')}
        assert not groups['train'].intersection(groups['val'])
        assert not groups['train'].intersection(groups['test'])
        assert not groups['val'].intersection(groups['test'])
        assert sum(len(list(output.rglob('*.jpg'))) for _ in range(1))==360
        assert all(info['counts'][split][cl]>0 for split in ('train','val','test') for cl in CLASS_MAP.values())

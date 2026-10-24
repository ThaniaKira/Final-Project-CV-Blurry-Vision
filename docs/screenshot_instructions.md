# Screenshot Instructions & Output Evidence Guide

## 1. Automated Output Screenshot Generation
To process all 20 sample gesture images and update the `outputs/` folder with annotated screenshots:

```bash
python -m src.utils
```

This runs the prototype prediction pipeline across `data/samples/` and outputs files named:
- `result_sample_01_Open_Hand.jpg`
- `result_sample_02_Fist.jpg`
- ... up to `result_sample_20_Fist.jpg`

## 2. Individual Prototype CLI Testing
To run inference on an individual image:

```bash
python src/predict.py --input data/samples/sample_01_Open_Hand.png --output outputs/result_sample_01.jpg
```

## 3. Terminal Execution Screenshot (Windows Snipping Tool)
1. Run `python src/baseline.py` in your terminal.
2. Press **`Win + Shift + S`** on Windows.
3. Capture the terminal window showing the execution command and the validation classification report.
4. Save the image inside `outputs/` as `outputs/terminal_baseline_eval.png`.

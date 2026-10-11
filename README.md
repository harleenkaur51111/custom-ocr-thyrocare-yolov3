# Custom-Object Character Recognition (OCR) on AWS (Google Drive / Cloud Storage)

**Author:** Harleen Kaur

A custom OCR system that reads the key fields of Thyrocare lab reports and saves them as an editable CSV / Excel table. A **YOLOv3** model finds the fields, **OpenCV** cleans each crop and **Tesseract** reads the text. Everything ran on Google Colab (free T4 GPU) and Google Drive, so the cost was **$0**.

## Pipeline

1. **Detect**: YOLOv3 finds 5 field types: `patient_info`, `test_name`, `value`, `units`, `ref_range`
2. **Crop**: each box is cut from the original full-size image
3. **Preprocess**: 3x resize, grayscale, Gaussian blur, Otsu threshold, invert
4. **Read**: Tesseract with stricter settings for numbers and units
5. **Clean up**: duplicate-box removal, matching units and test names to known spellings
6. **Save**: `ocr_output.csv`, `report.csv`, `report.xlsx`

## Results (10 unseen test images)

| Measure | Result |
|:--|:--|
| Precision | 0.968 |
| Recall | 0.956 |
| mAP50 | 0.975 |
| mAP50-95 | 0.612 |
| OCR similarity (overall) | 0.938 |

OCR similarity per field: `patient_info` 0.932, `ref_range` 0.925, `test_name` 0.994, `units` 0.888, `value` 0.950.

## Repository contents

| File / folder | Description |
|:--|:--|
| `Custom_OCR_YOLO_Tesseract.ipynb` | Colab notebook with the full workflow |
| `Custom_OCR.py` | Final pipeline script |
| `report.csv`, `report.xlsx` | Final editable report tables |
| `ocr_output_clean.csv` | Cleaned OCR output on the test images |
| `evaluation.csv`, `evaluation_clean.csv` | OCR scores before and after cleanup |
| `ground_truth_final.csv` | Correct answers for the 10 test images |
| `crops/`, `demo/`, `validation_boxes/` | Cropped fields, one-image demo run and detection examples |
| `screenshots/` | Screenshots from each stage |
| `Harleen_Kaur_OCR_Project_Report.pdf` | Full project report |

## Dataset

100 Thyrocare lab-report photos, labelled by hand in Roboflow (free plan), split 70 / 20 / 10 (train / validation / test), resized to 416 x 416, no flip or rotate augmentation. Classes: `patient_info`, `ref_range`, `test_name`, `units`, `value`.

## Trained model

The weights (`best.pt`, about 199 MB) are too big for GitHub. Download them here: **[PASTE YOUR GOOGLE DRIVE LINK HERE]**

## How to run (Google Colab)

```python
from google.colab import drive
drive.mount('/content/drive')
```

```bash
!apt-get -qq install tesseract-ocr
!pip install -q ultralytics pytesseract openpyxl

# one image
!python Custom_OCR.py --image yourimage.jpg --out results/demo

# a whole folder
!python Custom_OCR.py --folder test_images --out results
```

Put `best.pt` where the script expects it (`models/yolov3_ocr/weights/best.pt`) or edit the path at the top of `Custom_OCR.py`.

## Limitations

- Small test set (10 images) from one lab and a few layouts
- Cleanup only fixes tests and units that are in the known list
- Small digits are sometimes misread (for example 551 read as 351)

## Tools

Google Colab, Google Drive, Roboflow, Ultralytics YOLOv3, Tesseract / pytesseract, OpenCV, Python

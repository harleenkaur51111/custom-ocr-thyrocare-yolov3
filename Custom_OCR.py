import argparse, os, glob, csv
import cv2, pytesseract
from ultralytics import YOLO

MODEL = '/content/drive/MyDrive/Custom_OCR/models/yolov3_ocr/weights/best.pt'
OUT = '/content/drive/MyDrive/Custom_OCR/results'

def preprocess(crop):
    crop = cv2.resize(crop, None, fx=3, fy=3, interpolation=cv2.INTER_CUBIC)   # blow up 3x
    gray = cv2.cvtColor(crop, cv2.COLOR_BGR2GRAY)                               # grayscale
    blur = cv2.GaussianBlur(gray, (3, 3), 0)                                    # small blur
    thresh = cv2.threshold(blur, 0, 255, cv2.THRESH_BINARY_INV | cv2.THRESH_OTSU)[1]  # Otsu: white text on black
    return cv2.bitwise_not(thresh)                                              # black text on white

def process(path, model, writer):
    img = cv2.imread(path)
    H, W = img.shape[:2]
    small = cv2.resize(img, (416, 416))          # same stretch used in training
    res = model.predict(small, imgsz=416, conf=0.25, verbose=False)[0]
    name = os.path.splitext(os.path.basename(path))[0]
    os.makedirs(f'{OUT}/crops', exist_ok=True)
    boxes = sorted(res.boxes, key=lambda b: (int(b.cls), float(b.xyxyn[0][1])))
    for i, b in enumerate(boxes):
        cls = model.names[int(b.cls)]
        x1, y1, x2, y2 = b.xyxyn[0].tolist()
        x1, x2 = int(x1 * W), int(x2 * W)
        y1, y2 = int(y1 * H), int(y2 * H)
        crop = img[max(y1-2, 0):y2+2, max(x1-2, 0):x2+2]
        if crop.size == 0:
            continue
        clean = preprocess(crop)
        cv2.imwrite(f'{OUT}/crops/{name}_{i}_{cls}.png', clean)
        text = pytesseract.image_to_string(clean, config='--psm 6').strip()
        text = ' | '.join(l.strip() for l in text.splitlines() if l.strip())
        writer.writerow([name, cls, x1, y1, x2, y2, round(float(b.conf), 2), text])

if __name__ == '__main__':
    ap = argparse.ArgumentParser()
    ap.add_argument('--image'); ap.add_argument('--folder')
    ap.add_argument('--out', default=f'{OUT}/ocr_output.csv')
    a = ap.parse_args()
    files = [a.image] if a.image else sorted(glob.glob(a.folder + '/*.jpg'))
    model = YOLO(MODEL)
    os.makedirs(OUT, exist_ok=True)
    with open(a.out, 'w', newline='') as f:
        w = csv.writer(f)
        w.writerow(['image', 'class', 'x1', 'y1', 'x2', 'y2', 'conf', 'text'])
        for p in files:
            process(p, model, w)
    print('Saved', a.out)

from fastapi import FastAPI, File, UploadFile, HTTPException
from paddleocr import PaddleOCR
import numpy as np
import cv2
import re

app = FastAPI(title="OCR API")

# ---- Load OCR once (startup) ----
ocr = PaddleOCR(
    use_angle_cls=False,
    lang="en",
    rec_algorithm="SVTR_LCNet",
    det=True,
    rec=True,
    use_gpu=True,
    show_log=True
)

# ---- Utility: clean text ----
def clean_text(text: str) -> str:
    # Keep only A-Z and 0-9, uppercase, remove everything else
    text = text.upper()
    text = re.sub(r'[^A-Z0-9]', '', text)
    return text

def preprocess(image):
    result = image.copy()
    result = cv2.cvtColor(result, cv2.COLOR_BGR2RGB)
    result = np.clip(result, 0, 255).astype(np.uint8)
    result = cv2.convertScaleAbs(result, alpha=2.59, beta=-80)
    ksize = int(4 * 0.9 + 1)
    if ksize % 2 == 0:
        ksize += 1
    result = cv2.GaussianBlur(result, (ksize, ksize), 0.9)
    return result

# ---- Endpoint ----
@app.post("/ocr")
async def ocr_endpoint(file: UploadFile = File(...)):
    try:
        contents = await file.read()
        np_arr = np.frombuffer(contents, np.uint8)
        image = cv2.imdecode(np_arr, cv2.IMREAD_COLOR)
        

        if image is None:
            raise HTTPException(status_code=400, detail="Invalid image")
        
        processed = preprocess(image)

        result = ocr.ocr(processed, cls=True)
        print('RESULT:', result)

        texts = []
        for line in result:
            for word_info in line:
                raw_text = word_info[1][0]
                texts.append(raw_text)

        # Join all text, then clean
        joined_text = "".join(texts)
        cleaned = clean_text(joined_text)

        return {
            "raw_text": joined_text,
            "cleaned_text": cleaned
        }

    except Exception as e:
        print('ERROR--------', e)
        raise HTTPException(status_code=500, detail=str(e))
    
if __name__ == "__main__":
    import uvicorn

    uvicorn.run(
        "ocr:app",      # module:app
        host="0.0.0.0",
        port=9000,
        reload=False,   # IMPORTANT for PM2 (avoid multiple workers)
        workers=1       # keep single worker unless you handle concurrency
    )
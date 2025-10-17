import fitz
import pytesseract
import cv2
import camelot
import pandas as pd
import json, os
from docx import Document
import pdfplumber

# ---- Helper Functions ----
def extract_pdf_content(path):
    print(f"[PDF] Processing: {path}")
    result = {"text": [], "ocr_text": [], "tables": []}
    doc = fitz.open(path)
    for pno, page in enumerate(doc, 1):
        result["text"].append({"page": pno, "content": page.get_text("text")})
        for i, img in enumerate(page.get_images(full=True)):
            xref = img[0]
            pix = fitz.Pixmap(doc, xref)
            img_path = f"page{pno}_img{i+1}.png"
            (pix if pix.n < 5 else fitz.Pixmap(fitz.csRGB, pix)).save(img_path)
            img_cv = cv2.imread(img_path)
            gray = cv2.cvtColor(img_cv, cv2.COLOR_BGR2GRAY)
            text = pytesseract.image_to_string(gray)
            result["ocr_text"].append({"page": pno, "image": img_path, "content": text})
    doc.close()

    try:
        tables = camelot.read_pdf(path, pages="all", flavor="stream")
        for i, t in enumerate(tables):
            result["tables"].append({"page": t.page, "data": t.df.to_dict(orient="records")})
    except Exception as e:
        print("Table extraction failed:", e)
    return result


def extract_docx(path):
    print(f"[DOCX] Processing: {path}")
    doc = Document(path)
    text = "\n".join([p.text for p in doc.paragraphs])
    return {"text": text}


def extract_txt(path):
    print(f"[TXT] Processing: {path}")
    with open(path, "r", encoding="utf-8") as f:
        text = f.read()
    return {"text": text}


def extract_xlsx(path):
    print(f"[XLSX] Processing: {path}")
    xls = pd.ExcelFile(path)
    sheets = {sheet: pd.read_excel(xls, sheet).to_dict(orient="records") for sheet in xls.sheet_names}
    return {"tables": sheets}


def extract_image(path):
    print(f"[IMG] Processing: {path}")
    img = cv2.imread(path)
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    text = pytesseract.image_to_string(gray)
    return {"ocr_text": text}


# ---- Master Function ----
def process_file(file_path):
    ext = os.path.splitext(file_path)[1].lower()
    if ext in [".pdf"]:
        return extract_pdf_content(file_path)
    elif ext in [".docx"]:
        return extract_docx(file_path)
    elif ext in [".txt"]:
        return extract_txt(file_path)
    elif ext in [".xlsx"]:
        return extract_xlsx(file_path)
    elif ext in [".png", ".jpg", ".jpeg"]:
        return extract_image(file_path)
    else:
        raise ValueError("Unsupported file format.")


# ---- Runner ----
if __name__ == "__main__":
    input_path = input("Enter file path (PDF/DOCX/TXT/XLSX/IMG): ").strip()
    data = process_file(input_path)
    os.makedirs("output", exist_ok=True)
    with open("output/structured_output.json", "w", encoding="utf-8") as f:
        json.dump(data, f, ensure_ascii=False, indent=2)
    print("✅ Data saved to output/structured_output.json")

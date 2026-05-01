import os
import re
import cv2
import fitz
import pytesseract
import tempfile
import traceback
import google.generativeai as genai
import numpy as np
from PIL import Image
from flask import Flask, request, jsonify


# -------------------------------
# SET TESSERACT PATH
# -------------------------------
# pytesseract.pytesseract.tesseract_cmd = r"C:\Program Files\Tesseract-OCR\tesseract.exe"

# -------------------------------
# FITZ / PYMUPDF VALIDATION
# -------------------------------
if not hasattr(fitz, "open"):
    raise ImportError(
        "Invalid 'fitz' module loaded. Uninstall wrong package with:\n"
        "pip uninstall fitz\n"
        "pip uninstall pymupdf PyMuPDF\n"
        "pip install --upgrade PyMuPDF"
    )

# -------------------------------
# GEMINI CLIENT
# -------------------------------
GEMINI_API_KEY = os.getenv("GOOGLE_API_KEY")
genai.configure(api_key=GEMINI_API_KEY)
gemini_model = genai.GenerativeModel("gemini-2.5-flash")

# -------------------------------
# PARAMETERS (Reference Ranges)
# -------------------------------
PARAMETERS = [
    {"name": "Hemoglobin", "pattern": r'(hemoglobin|hb)\s*[:\-]?\s*(\d+\.?\d*)', "low": 13.0, "high": 17.0, "unit": "g/dL", "category": "Blood"},
    {"name": "WBC", "pattern": r'(wbc|white blood cells?|total leukocyte count|tlc)\s*[:\-]?\s*(\d+\.?\d*)', "low": 4000, "high": 11000, "unit": "cells/μL", "category": "Blood"},
    {"name": "Platelets", "pattern": r'(platelets?|plt|platelet count)\s*[:\-]?\s*(\d+\.?\d*)', "low": 150000, "high": 450000, "unit": "cells/μL", "category": "Blood"},
    {"name": "RBC", "pattern": r'(rbc|red blood cells?)\s*[:\-]?\s*(\d+\.?\d*)', "low": 4.5, "high": 5.9, "unit": "mill/μL", "category": "Blood"},
    {"name": "Hematocrit (PCV)", "pattern": r'(hematocrit|pcv|hct)\s*[:\-]?\s*(\d+\.?\d*)', "low": 41.0, "high": 50.0, "unit": "%", "category": "Blood"},
    {"name": "MCV", "pattern": r'(mcv|mean corpuscular volume)\s*[:\-]?\s*(\d+\.?\d*)', "low": 80.0, "high": 100.0, "unit": "fL", "category": "Blood"},
    {"name": "MCH", "pattern": r'(mch|mean corpuscular hemoglobin)\s*[:\-]?\s*(\d+\.?\d*)', "low": 27.0, "high": 32.0, "unit": "pg", "category": "Blood"},
    {"name": "MCHC", "pattern": r'(mchc)\s*[:\-]?\s*(\d+\.?\d*)', "low": 32.0, "high": 36.0, "unit": "g/dL", "category": "Blood"},
    {"name": "RDW", "pattern": r'(rdw|red cell distribution width)\s*[:\-]?\s*(\d+\.?\d*)', "low": 11.5, "high": 14.5, "unit": "%", "category": "Blood"},
    {"name": "Neutrophils", "pattern": r'(neutrophils?|polymorphs)\s*[:\-]?\s*(\d+\.?\d*)', "low": 40.0, "high": 75.0, "unit": "%", "category": "Blood"},
    {"name": "Lymphocytes", "pattern": r'(lymphocytes?|lymphs)\s*[:\-]?\s*(\d+\.?\d*)', "low": 20.0, "high": 40.0, "unit": "%", "category": "Blood"},
    {"name": "Monocytes", "pattern": r'(monocytes?|monos)\s*[:\-]?\s*(\d+\.?\d*)', "low": 2.0, "high": 10.0, "unit": "%", "category": "Blood"},
    {"name": "Eosinophils", "pattern": r'(eosinophils?|eos)\s*[:\-]?\s*(\d+\.?\d*)', "low": 1.0, "high": 6.0, "unit": "%", "category": "Blood"},
    {"name": "Basophils", "pattern": r'(basophils?|baso)\s*[:\-]?\s*(\d+\.?\d*)', "low": 0.0, "high": 1.0, "unit": "%", "category": "Blood"},
    {"name": "MPV", "pattern": r'(mpv|mean platelet volume)\s*[:\-]?\s*(\d+\.?\d*)', "low": 7.5, "high": 11.5, "unit": "fL", "category": "Blood"},
    {"name": "Total Cholesterol", "pattern": r'(total cholesterol|cholesterol total|cholesterol)\s*[:\-]?\s*(\d+\.?\d*)', "low": 0, "high": 200, "unit": "mg/dL", "category": "Heart"},
    {"name": "Triglycerides", "pattern": r'(triglycerides?|tg)\s*[:\-]?\s*(\d+\.?\d*)', "low": 0, "high": 150, "unit": "mg/dL", "category": "Heart"},
    {"name": "HDL Cholesterol", "pattern": r'(hdl|high density lipoprotein)\s*[:\-]?\s*(\d+\.?\d*)', "low": 40, "high": 60, "unit": "mg/dL", "category": "Heart"},
    {"name": "LDL Cholesterol", "pattern": r'(ldl|low density lipoprotein)\s*[:\-]?\s*(\d+\.?\d*)', "low": 0, "high": 100, "unit": "mg/dL", "category": "Heart"},
    {"name": "VLDL Cholesterol", "pattern": r'(vldl)\s*[:\-]?\s*(\d+\.?\d*)', "low": 2.0, "high": 30.0, "unit": "mg/dL", "category": "Heart"},
    {"name": "Non-HDL Cholesterol", "pattern": r'(non-hdl|non hdl)\s*[:\-]?\s*(\d+\.?\d*)', "low": 0, "high": 130, "unit": "mg/dL", "category": "Heart"},
    {"name": "Cholesterol/HDL Ratio", "pattern": r'(chol/hdl|cholesterol/hdl ratio)\s*[:\-]?\s*(\d+\.?\d*)', "low": 0.0, "high": 5.0, "unit": "Ratio", "category": "Heart"},
    {"name": "Fasting Blood Sugar", "pattern": r'(fasting blood sugar|fbs|glucose fasting)\s*[:\-]?\s*(\d+\.?\d*)', "low": 70, "high": 100, "unit": "mg/dL", "category": "Diabetes"},
    {"name": "Post Prandial Sugar", "pattern": r'(ppbs|post prandial|glucose pp)\s*[:\-]?\s*(\d+\.?\d*)', "low": 70, "high": 140, "unit": "mg/dL", "category": "Diabetes"},
    {"name": "Random Blood Sugar", "pattern": r'(rbs|random blood sugar|glucose random)\s*[:\-]?\s*(\d+\.?\d*)', "low": 70, "high": 140, "unit": "mg/dL", "category": "Diabetes"},
    {"name": "HbA1c", "pattern": r'(hba1c|glycosylated hb|glycohemoglobin)\s*[:\-]?\s*(\d+\.?\d*)', "low": 4.0, "high": 5.6, "unit": "%", "category": "Diabetes"},
    {"name": "Fasting Insulin", "pattern": r'(insulin fasting|fasting insulin)\s*[:\-]?\s*(\d+\.?\d*)', "low": 2.6, "high": 24.9, "unit": "μIU/mL", "category": "Diabetes"},
    {"name": "C-Peptide", "pattern": r'(c-peptide|c peptide)\s*[:\-]?\s*(\d+\.?\d*)', "low": 0.5, "high": 2.0, "unit": "ng/mL", "category": "Diabetes"},
    {"name": "HOMA-IR", "pattern": r'(homa-ir|homa ir)\s*[:\-]?\s*(\d+\.?\d*)', "low": 0.5, "high": 1.4, "unit": "Index", "category": "Diabetes"},
    {"name": "Total Bilirubin", "pattern": r'(total bilirubin|bilirubin total)\s*[:\-]?\s*(\d+\.?\d*)', "low": 0.1, "high": 1.2, "unit": "mg/dL", "category": "Liver"},
    {"name": "Direct Bilirubin", "pattern": r'(direct bilirubin|bilirubin direct)\s*[:\-]?\s*(\d+\.?\d*)', "low": 0.0, "high": 0.3, "unit": "mg/dL", "category": "Liver"},
    {"name": "Indirect Bilirubin", "pattern": r'(indirect bilirubin|bilirubin indirect)\s*[:\-]?\s*(\d+\.?\d*)', "low": 0.2, "high": 0.8, "unit": "mg/dL", "category": "Liver"},
    {"name": "SGOT (AST)", "pattern": r'(sgot|ast|aspartate aminotransferase)\s*[:\-]?\s*(\d+\.?\d*)', "low": 8.0, "high": 40.0, "unit": "U/L", "category": "Liver"},
    {"name": "SGPT (ALT)", "pattern": r'(sgpt|alt|alanine aminotransferase)\s*[:\-]?\s*(\d+\.?\d*)', "low": 7.0, "high": 56.0, "unit": "U/L", "category": "Liver"},
    {"name": "Alkaline Phosphatase", "pattern": r'(alkaline phosphatase|alp)\s*[:\-]?\s*(\d+\.?\d*)', "low": 44.0, "high": 147.0, "unit": "U/L", "category": "Liver"},
    {"name": "GGT", "pattern": r'(ggt|gamma glutamyl transferase)\s*[:\-]?\s*(\d+\.?\d*)', "low": 9.0, "high": 48.0, "unit": "U/L", "category": "Liver"},
    {"name": "Total Protein", "pattern": r'(total protein|protein total)\s*[:\-]?\s*(\d+\.?\d*)', "low": 6.0, "high": 8.3, "unit": "g/dL", "category": "Liver"},
    {"name": "Albumin", "pattern": r'(albumin|serum albumin)\s*[:\-]?\s*(\d+\.?\d*)', "low": 3.4, "high": 5.4, "unit": "g/dL", "category": "Liver"},
    {"name": "Globulin", "pattern": r'(globulin|serum globulin)\s*[:\-]?\s*(\d+\.?\d*)', "low": 2.0, "high": 3.5, "unit": "g/dL", "category": "Liver"},
    {"name": "A/G Ratio", "pattern": r'(a/g ratio|albumin/globulin ratio)\s*[:\-]?\s*(\d+\.?\d*)', "low": 1.2, "high": 2.2, "unit": "Ratio", "category": "Liver"},
    {"name": "Creatinine", "pattern": r'(creatinine|serum creatinine)\s*[:\-]?\s*(\d+\.?\d*)', "low": 0.6, "high": 1.2, "unit": "mg/dL", "category": "Kidney"},
    {"name": "Blood Urea", "pattern": r'(blood urea|urea)\s*[:\-]?\s*(\d+\.?\d*)', "low": 15.0, "high": 40.0, "unit": "mg/dL", "category": "Kidney"},
    {"name": "BUN", "pattern": r'(bun|blood urea nitrogen)\s*[:\-]?\s*(\d+\.?\d*)', "low": 7.0, "high": 20.0, "unit": "mg/dL", "category": "Kidney"},
    {"name": "Uric Acid", "pattern": r'(uric acid)\s*[:\-]?\s*(\d+\.?\d*)', "low": 3.5, "high": 7.2, "unit": "mg/dL", "category": "Kidney"},
    {"name": "eGFR", "pattern": r'(egfr|glomerular filtration rate)\s*[:\-]?\s*(\d+\.?\d*)', "low": 90.0, "high": 120.0, "unit": "mL/min", "category": "Kidney"},
    {"name": "BUN/Creatinine Ratio", "pattern": r'(bun/creatinine ratio)\s*[:\-]?\s*(\d+\.?\d*)', "low": 10.0, "high": 20.0, "unit": "Ratio", "category": "Kidney"},
    {"name": "Sodium", "pattern": r'(sodium|na\+?)\s*[:\-]?\s*(\d+\.?\d*)', "low": 135.0, "high": 145.0, "unit": "mEq/L", "category": "Electrolytes"},
    {"name": "Potassium", "pattern": r'(potassium|k\+?)\s*[:\-]?\s*(\d+\.?\d*)', "low": 3.5, "high": 5.2, "unit": "mEq/L", "category": "Electrolytes"},
    {"name": "Chloride", "pattern": r'(chloride|cl\-?)\s*[:\-]?\s*(\d+\.?\d*)', "low": 96.0, "high": 106.0, "unit": "mEq/L", "category": "Electrolytes"},
    {"name": "Calcium", "pattern": r'(calcium|total calcium|ca\+?)\s*[:\-]?\s*(\d+\.?\d*)', "low": 8.5, "high": 10.5, "unit": "mg/dL", "category": "Electrolytes"},
    {"name": "Ionized Calcium", "pattern": r'(ionized calcium)\s*[:\-]?\s*(\d+\.?\d*)', "low": 4.5, "high": 5.6, "unit": "mg/dL", "category": "Electrolytes"},
    {"name": "Phosphorus", "pattern": r'(phosphorus|phosphate)\s*[:\-]?\s*(\d+\.?\d*)', "low": 2.5, "high": 4.5, "unit": "mg/dL", "category": "Electrolytes"},
    {"name": "Magnesium", "pattern": r'(magnesium|mg\+?)\s*[:\-]?\s*(\d+\.?\d*)', "low": 1.7, "high": 2.2, "unit": "mg/dL", "category": "Electrolytes"},
    {"name": "TSH", "pattern": r'(tsh|thyroid stimulating hormone)\s*[:\-]?\s*(\d+\.?\d*)', "low": 0.4, "high": 4.0, "unit": "mIU/L", "category": "Thyroid"},
    {"name": "Free T3", "pattern": r'(ft3|free t3|free triiodothyronine)\s*[:\-]?\s*(\d+\.?\d*)', "low": 2.3, "high": 4.1, "unit": "pg/mL", "category": "Thyroid"},
    {"name": "Free T4", "pattern": r'(ft4|free t4|free thyroxine)\s*[:\-]?\s*(\d+\.?\d*)', "low": 0.9, "high": 1.7, "unit": "ng/dL", "category": "Thyroid"},
    {"name": "Total T3", "pattern": r'(t3|total t3)\s*[:\-]?\s*(\d+\.?\d*)', "low": 80.0, "high": 200.0, "unit": "ng/dL", "category": "Thyroid"},
    {"name": "Total T4", "pattern": r'(t4|total t4)\s*[:\-]?\s*(\d+\.?\d*)', "low": 5.0, "high": 12.0, "unit": "μg/dL", "category": "Thyroid"},
    {"name": "Vitamin D", "pattern": r'(vitamin d|vit d|25-hydroxy vitamin d)\s*[:\-]?\s*(\d+\.?\d*)', "low": 30.0, "high": 100.0, "unit": "ng/mL", "category": "Vitamins"},
    {"name": "Vitamin B12", "pattern": r'(vitamin b12|vit b12|cobalamin)\s*[:\-]?\s*(\d+\.?\d*)', "low": 200.0, "high": 900.0, "unit": "pg/mL", "category": "Vitamins"},
    {"name": "Serum Iron", "pattern": r'(serum iron|iron)\s*[:\-]?\s*(\d+\.?\d*)', "low": 60.0, "high": 170.0, "unit": "μg/dL", "category": "Iron Profile"},
    {"name": "TIBC", "pattern": r'(tibc|total iron binding capacity)\s*[:\-]?\s*(\d+\.?\d*)', "low": 240.0, "high": 450.0, "unit": "μg/dL", "category": "Iron Profile"},
    {"name": "Ferritin", "pattern": r'(ferritin)\s*[:\-]?\s*(\d+\.?\d*)', "low": 12.0, "high": 300.0, "unit": "ng/mL", "category": "Iron Profile"},
    {"name": "Transferrin Saturation", "pattern": r'(transferrin saturation|iron saturation)\s*[:\-]?\s*(\d+\.?\d*)', "low": 20.0, "high": 50.0, "unit": "%", "category": "Iron Profile"},
    {"name": "Folate", "pattern": r'(folate|folic acid)\s*[:\-]?\s*(\d+\.?\d*)', "low": 2.7, "high": 17.0, "unit": "ng/mL", "category": "Vitamins"},
    {"name": "Zinc", "pattern": r'(zinc)\s*[:\-]?\s*(\d+\.?\d*)', "low": 60.0, "high": 120.0, "unit": "μg/dL", "category": "Vitamins"},
    {"name": "Copper", "pattern": r'(copper)\s*[:\-]?\s*(\d+\.?\d*)', "low": 70.0, "high": 140.0, "unit": "μg/dL", "category": "Vitamins"},
    {"name": "CRP", "pattern": r'(crp|c-reactive protein|c reactive protein)\s*[:\-]?\s*(\d+\.?\d*)', "low": 0.0, "high": 5.0, "unit": "mg/L", "category": "Immunology"},
    {"name": "hs-CRP", "pattern": r'(hs-crp|hscrp|high sensitivity crp)\s*[:\-]?\s*(\d+\.?\d*)', "low": 0.0, "high": 3.0, "unit": "mg/L", "category": "Heart"},
    {"name": "ESR", "pattern": r'(esr|erythrocyte sedimentation rate)\s*[:\-]?\s*(\d+\.?\d*)', "low": 0.0, "high": 20.0, "unit": "mm/hr", "category": "Immunology"},
    {"name": "Rheumatoid Factor", "pattern": r'(ra factor|rheumatoid factor|rf)\s*[:\-]?\s*(\d+\.?\d*)', "low": 0.0, "high": 14.0, "unit": "IU/mL", "category": "Immunology"},
    {"name": "ANA", "pattern": r'(ana|antinuclear antibody)\s*[:\-]?\s*(\d+\.?\d*)', "low": 0.0, "high": 1.0, "unit": "Index", "category": "Immunology"},
    {"name": "IgE Total", "pattern": r'(ige|immunoglobulin e)\s*[:\-]?\s*(\d+\.?\d*)', "low": 0.0, "high": 100.0, "unit": "IU/mL", "category": "Immunology"},
    {"name": "IgG Total", "pattern": r'(igg|immunoglobulin g)\s*[:\-]?\s*(\d+\.?\d*)', "low": 700.0, "high": 1600.0, "unit": "mg/dL", "category": "Immunology"},
    {"name": "IgA Total", "pattern": r'(iga|immunoglobulin a)\s*[:\-]?\s*(\d+\.?\d*)', "low": 70.0, "high": 400.0, "unit": "mg/dL", "category": "Immunology"},
    {"name": "IgM Total", "pattern": r'(igm|immunoglobulin m)\s*[:\-]?\s*(\d+\.?\d*)', "low": 40.0, "high": 230.0, "unit": "mg/dL", "category": "Immunology"},
    {"name": "Total Testosterone", "pattern": r'(total testosterone|testosterone total)\s*[:\-]?\s*(\d+\.?\d*)', "low": 250.0, "high": 900.0, "unit": "ng/dL", "category": "Hormones"},
    {"name": "Free Testosterone", "pattern": r'(free testosterone)\s*[:\-]?\s*(\d+\.?\d*)', "low": 35.0, "high": 155.0, "unit": "pg/mL", "category": "Hormones"},
    {"name": "Estradiol (E2)", "pattern": r'(estradiol|e2)\s*[:\-]?\s*(\d+\.?\d*)', "low": 10.0, "high": 40.0, "unit": "pg/mL", "category": "Hormones"},
    {"name": "Progesterone", "pattern": r'(progesterone)\s*[:\-]?\s*(\d+\.?\d*)', "low": 0.1, "high": 0.9, "unit": "ng/mL", "category": "Hormones"},
    {"name": "LH", "pattern": r'(lh|luteinizing hormone)\s*[:\-]?\s*(\d+\.?\d*)', "low": 1.5, "high": 9.3, "unit": "mIU/mL", "category": "Hormones"},
    {"name": "FSH", "pattern": r'(fsh|follicle stimulating hormone)\s*[:\-]?\s*(\d+\.?\d*)', "low": 1.4, "high": 18.1, "unit": "mIU/mL", "category": "Hormones"},
    {"name": "Prolactin", "pattern": r'(prolactin)\s*[:\-]?\s*(\d+\.?\d*)', "low": 2.0, "high": 18.0, "unit": "ng/mL", "category": "Hormones"},
    {"name": "Cortisol (Morning)", "pattern": r'(cortisol)\s*[:\-]?\s*(\d+\.?\d*)', "low": 5.0, "high": 25.0, "unit": "μg/dL", "category": "Hormones"},
    {"name": "DHEA-S", "pattern": r'(dhea-s|dheas)\s*[:\-]?\s*(\d+\.?\d*)', "low": 80.0, "high": 560.0, "unit": "μg/dL", "category": "Hormones"},
    {"name": "SHBG", "pattern": r'(shbg|sex hormone binding globulin)\s*[:\-]?\s*(\d+\.?\d*)', "low": 10.0, "high": 57.0, "unit": "nmol/L", "category": "Hormones"},
    {"name": "AMH", "pattern": r'(amh|anti-mullerian hormone)\s*[:\-]?\s*(\d+\.?\d*)', "low": 1.0, "high": 4.0, "unit": "ng/mL", "category": "Hormones"},
    {"name": "Troponin I", "pattern": r'(troponin i|trop i)\s*[:\-]?\s*(\d+\.?\d*)', "low": 0.0, "high": 0.04, "unit": "ng/mL", "category": "Heart"},
    {"name": "Troponin T", "pattern": r'(troponin t|trop t)\s*[:\-]?\s*(\d+\.?\d*)', "low": 0.0, "high": 0.01, "unit": "ng/mL", "category": "Heart"},
    {"name": "CK-MB", "pattern": r'(ck-mb|ck mb)\s*[:\-]?\s*(\d+\.?\d*)', "low": 0.0, "high": 5.0, "unit": "ng/mL", "category": "Heart"},
    {"name": "Prothrombin Time (PT)", "pattern": r'(prothrombin time|pt)\s*[:\-]?\s*(\d+\.?\d*)', "low": 11.0, "high": 13.5, "unit": "sec", "category": "Coagulation"},
    {"name": "INR", "pattern": r'(inr)\s*[:\-]?\s*(\d+\.?\d*)', "low": 0.8, "high": 1.1, "unit": "Ratio", "category": "Coagulation"},
    {"name": "aPTT", "pattern": r'(aptt|ptt)\s*[:\-]?\s*(\d+\.?\d*)', "low": 25.0, "high": 35.0, "unit": "sec", "category": "Coagulation"},
    {"name": "Fibrinogen", "pattern": r'(fibrinogen)\s*[:\-]?\s*(\d+\.?\d*)', "low": 200.0, "high": 400.0, "unit": "mg/dL", "category": "Coagulation"},
    {"name": "D-Dimer", "pattern": r'(d-dimer|d dimer)\s*[:\-]?\s*(\d+\.?\d*)', "low": 0.0, "high": 0.5, "unit": "μg/mL", "category": "Coagulation"},
    {"name": "BNP", "pattern": r'(bnp|brain natriuretic peptide)\s*[:\-]?\s*(\d+\.?\d*)', "low": 0.0, "high": 100.0, "unit": "pg/mL", "category": "Heart"},
    {"name": "PSA", "pattern": r'(psa|prostate specific antigen)\s*[:\-]?\s*(\d+\.?\d*)', "low": 0.0, "high": 4.0, "unit": "ng/mL", "category": "Tumor Marker"},
    {"name": "Free PSA", "pattern": r'(free psa)\s*[:\-]?\s*(\d+\.?\d*)', "low": 0.0, "high": 1.0, "unit": "ng/mL", "category": "Tumor Marker"},
    {"name": "CEA", "pattern": r'(cea|carcinoembryonic antigen)\s*[:\-]?\s*(\d+\.?\d*)', "low": 0.0, "high": 2.5, "unit": "ng/mL", "category": "Tumor Marker"},
    {"name": "CA 125", "pattern": r'(ca 125|ca-125)\s*[:\-]?\s*(\d+\.?\d*)', "low": 0.0, "high": 35.0, "unit": "U/mL", "category": "Tumor Marker"},
    {"name": "CA 15-3", "pattern": r'(ca 15-3|ca 15\.3)\s*[:\-]?\s*(\d+\.?\d*)', "low": 0.0, "high": 30.0, "unit": "U/mL", "category": "Tumor Marker"},
    {"name": "CA 19-9", "pattern": r'(ca 19-9|ca 19\.9)\s*[:\-]?\s*(\d+\.?\d*)', "low": 0.0, "high": 37.0, "unit": "U/mL", "category": "Tumor Marker"},
    {"name": "AFP", "pattern": r'(afp|alpha fetoprotein)\s*[:\-]?\s*(\d+\.?\d*)', "low": 0.0, "high": 40.0, "unit": "ng/mL", "category": "Tumor Marker"},
    {"name": "Urine pH", "pattern": r'(urine ph|ph)\s*[:\-]?\s*(\d+\.?\d*)', "low": 4.5, "high": 8.0, "unit": "pH", "category": "Urine Analysis"},
    {"name": "Specific Gravity", "pattern": r'(specific gravity|sg)\s*[:\-]?\s*(\d+\.?\d*)', "low": 1.005, "high": 1.030, "unit": "SG", "category": "Urine Analysis"},
    {"name": "Urine Protein", "pattern": r'(urine protein|protein urine)\s*[:\-]?\s*(\d+\.?\d*)', "low": 0.0, "high": 14.0, "unit": "mg/dL", "category": "Urine Analysis"},
    {"name": "Microalbumin", "pattern": r'(microalbumin|urine albumin)\s*[:\-]?\s*(\d+\.?\d*)', "low": 0.0, "high": 30.0, "unit": "mg/g", "category": "Urine Analysis"},
    {"name": "Urine Creatinine", "pattern": r'(urine creatinine)\s*[:\-]?\s*(\d+\.?\d*)', "low": 20.0, "high": 275.0, "unit": "mg/dL", "category": "Urine Analysis"},
    {"name": "Urine Urea", "pattern": r'(urine urea)\s*[:\-]?\s*(\d+\.?\d*)', "low": 12.0, "high": 20.0, "unit": "g/24h", "category": "Urine Analysis"},
    {"name": "Blood pH", "pattern": r'(blood ph|ph \(abg\))\s*[:\-]?\s*(\d+\.?\d*)', "low": 7.35, "high": 7.45, "unit": "pH", "category": "ABG"},
    {"name": "pCO2", "pattern": r'(pco2|paco2)\s*[:\-]?\s*(\d+\.?\d*)', "low": 35.0, "high": 45.0, "unit": "mmHg", "category": "ABG"},
    {"name": "pO2", "pattern": r'(po2|pao2)\s*[:\-]?\s*(\d+\.?\d*)', "low": 75.0, "high": 100.0, "unit": "mmHg", "category": "ABG"},
    {"name": "HCO3", "pattern": r'(hco3|bicarbonate)\s*[:\-]?\s*(\d+\.?\d*)', "low": 22.0, "high": 28.0, "unit": "mEq/L", "category": "ABG"},
    {"name": "O2 Saturation", "pattern": r'(o2 sat|o2 saturation|sao2)\s*[:\-]?\s*(\d+\.?\d*)', "low": 95.0, "high": 100.0, "unit": "%", "category": "ABG"},
    {"name": "Base Excess", "pattern": r'(base excess|be)\s*[:\-]?\s*(\-?\d+\.?\d*)', "low": -2.0, "high": 2.0, "unit": "mEq/L", "category": "ABG"},
]

# -------------------------------
# HELPERS
# -------------------------------
def _save_upload_to_temp(file_obj, suffix):
    with tempfile.NamedTemporaryFile(delete=False, suffix=suffix) as temp:
        file_obj.save(temp.name)
        return temp.name


def _read_image_text(image_path):
    image = Image.open(image_path).convert("RGB")
    image_np = np.array(image)
    gray = cv2.cvtColor(image_np, cv2.COLOR_RGB2GRAY)
    gray = cv2.threshold(gray, 150, 255, cv2.THRESH_BINARY)[1]
    return pytesseract.image_to_string(gray)


# -------------------------------
# OCR EXTRACTION
# -------------------------------
def extract_text(file_obj):
    filename = getattr(file_obj, "filename", "").lower()

    if not filename:
        raise ValueError("Uploaded file has no filename")

    if filename.endswith(".pdf"):
        suffix = ".pdf"
    elif filename.endswith((".png", ".jpg", ".jpeg", ".webp", ".bmp", ".tiff")):
        suffix = os.path.splitext(filename)[1]
    else:
        raise ValueError("Unsupported file type")

    temp_path = _save_upload_to_temp(file_obj, suffix)
    text_output = ""

    try:
        if filename.endswith(".pdf"):
            print(f"📄 Processing PDF: {temp_path}")
            doc = fitz.open(temp_path)

            for page in doc:
                text_output += page.get_text("text") + "\n"

            if len(text_output.strip()) < 100:
                print("⚠️ OCR fallback triggered for PDF")
                text_output = ""
                for page in doc:
                    pix = page.get_pixmap(dpi=200)
                    img = Image.frombytes("RGB", [pix.width, pix.height], pix.samples)
                    img_np = np.array(img)
                    gray = cv2.cvtColor(img_np, cv2.COLOR_RGB2GRAY)
                    gray = cv2.threshold(gray, 150, 255, cv2.THRESH_BINARY)[1]
                    text_output += pytesseract.image_to_string(gray) + "\n"

            doc.close()
        else:
            print(f"🖼️ Processing image: {temp_path}")
            text_output = _read_image_text(temp_path)

        print("🔍 EXTRACTED TEXT PREVIEW:")
        print(text_output[:500])
        return text_output.lower()

    finally:
        if os.path.exists(temp_path):
            os.remove(temp_path)


# -------------------------------
# DATA ANALYSIS
# -------------------------------
def analyze(text):
    results = []
    seen = set()

    for param in PARAMETERS:
        matches = re.findall(param["pattern"], text, flags=re.IGNORECASE)

        for match in matches:
            try:
                value = float(match[-1])

                if param["name"] in seen:
                    continue

                if value < param["low"]:
                    status = "LOW"
                    advice = "Below normal range"
                elif value > param["high"]:
                    status = "HIGH"
                    advice = "Above normal range"
                else:
                    status = "NORMAL"
                    advice = "Within healthy range"

                results.append({
                    "name": param["name"],
                    "value": value,
                    "unit": param["unit"],
                    "status": status,
                    "normal_range": f"{param['low']} - {param['high']}",
                    "category": param["category"],
                    "advice": advice
                })

                seen.add(param["name"])

            except Exception:
                continue

    return results


# -------------------------------
# SUMMARY GENERATION
# -------------------------------
def generate_summary(results):
    if not results:
        return {
            "risk": "Unknown",
            "message": "No recognized medical parameters found. Please ensure the report is clear and readable."
        }

    abnormal = [r for r in results if r["status"] != "NORMAL"]

    if len(abnormal) == 0:
        return {
            "risk": "Low",
            "message": "Great news! All detected parameters are within the normal range."
        }
    elif len(abnormal) <= 2:
        return {
            "risk": "Moderate",
            "message": "A few parameters are outside the normal range. Please monitor them."
        }
    else:
        return {
            "risk": "High",
            "message": "Multiple abnormalities were detected. A medical consultation is recommended."
        }
    print("🔥 OCR GEMINI CALLED")
    print("API KEY:", GEMINI_API_KEY)


# -------------------------------
# AI SUMMARY (GEMINI - ~100 WORDS)
# -------------------------------
def generate_ai_summary(results, summary_data):
    """
    Uses Gemini 1.5 Flash to generate a ~100-word plain-language summary of the lab report,
    covering: what the report shows, what is abnormal and why it might have occurred,
    and brief health implications. Always disclaims to consult a doctor.
    """
    if not results:
        return "No parameters were detected in the uploaded report. Please ensure the document is clear, properly scanned, and contains standard lab values. If the issue persists, try uploading a higher-quality image or a text-based PDF. Always consult a qualified healthcare professional for interpretation of your medical reports."

    abnormal = [r for r in results if r["status"] != "NORMAL"]
    normal_count = len(results) - len(abnormal)

    param_lines = []
    for r in results:
        flag = f"[{r['status']}]" if r["status"] != "NORMAL" else "[NORMAL]"
        param_lines.append(f"- {r['name']}: {r['value']} {r['unit']} {flag} (ref: {r['normal_range']})")

    param_text = "\n".join(param_lines)
    risk = summary_data.get("risk", "Unknown")

    prompt = f"""You are a compassionate medical assistant. Based on the following lab report data, write a clear, empathetic, plain-language summary in EXACTLY around 100 words.

The summary must cover:
1. What type of report this appears to be and overall status
2. Which specific values are abnormal and what they suggest (e.g., low hemoglobin suggests anemia)
3. Likely causes or health implications in simple language
4. A brief note to consult a doctor for professional advice

Overall Risk Level: {risk}
Total Parameters: {len(results)} ({normal_count} normal, {len(abnormal)} abnormal)

Lab Values:
{param_text}

Write ONLY the summary paragraph. No headings, no bullet points. Keep it warm, clear, and approximately 100 words."""

    try:
        response = gemini_model.generate_content(
            prompt,
            generation_config=genai.types.GenerationConfig(
                max_output_tokens=800,
                temperature=0.4,
            )
        )
        return response.text.strip()

    except Exception as e:
        print(f"❌ Gemini AI Summary generation failed: {e}")
        # Graceful fallback
        if abnormal:
            names = ", ".join([r["name"] for r in abnormal[:3]])
            return (
                f"Your lab report shows {len(results)} parameters analyzed, with {len(abnormal)} value(s) outside "
                f"the normal range, including {names}. This indicates a {risk.lower()} health risk level. "
                f"Abnormal values may result from various factors including diet, stress, or underlying conditions. "
                f"Please consult a qualified healthcare professional for a thorough evaluation and personalized advice."
            )
        return (
            "Your lab report has been analyzed and all detected parameters appear to be within normal reference ranges. "
            "This is an encouraging result. Continue maintaining a balanced diet, regular exercise, and adequate sleep "
            "to support your health. Regular check-ups are still recommended. Please consult a doctor for a complete "
            "clinical evaluation and personalized health guidance."
        )


# -------------------------------
# LIFESTYLE RECOMMENDATIONS
# -------------------------------
def get_recommendations(results):
    rec = []

    has_diabetes_issue = any(r["category"] == "Diabetes" and r["status"] != "NORMAL" for r in results)
    has_heart_issue = any(r["category"] == "Heart" and r["status"] != "NORMAL" for r in results)
    has_kidney_issue = any(r["category"] == "Kidney" and r["status"] != "NORMAL" for r in results)
    has_liver_issue = any(r["category"] == "Liver" and r["status"] != "NORMAL" for r in results)
    has_thyroid_issue = any(r["category"] == "Thyroid" and r["status"] != "NORMAL" for r in results)
    has_blood_issue = any(r["category"] == "Blood" and r["status"] != "NORMAL" for r in results)
    has_vitamin_issue = any(r["category"] == "Vitamins" and r["status"] != "NORMAL" for r in results)

    if has_diabetes_issue:
        rec.append("Reduce refined sugar and high-glycemic carbohydrates. Monitor blood glucose regularly.")
    if has_heart_issue:
        rec.append("Include 30 minutes of cardio or brisk walking daily. Limit saturated fats and processed foods.")
    if has_kidney_issue:
        rec.append("Stay well hydrated (8-10 glasses of water daily) and reduce excess salt and protein intake.")
    if has_liver_issue:
        rec.append("Avoid alcohol and processed foods. Consider a liver-friendly diet rich in leafy greens.")
    if has_thyroid_issue:
        rec.append("Ensure adequate iodine and selenium intake. Manage stress and maintain consistent sleep cycles.")
    if has_blood_issue:
        rec.append("Boost iron and vitamin intake through leafy greens, lean meats, and legumes.")
    if has_vitamin_issue:
        rec.append("Consider supplementing deficient vitamins after consulting your doctor. Get sunlight daily for Vitamin D.")

    if not rec:
        rec.append("Maintain your current balanced diet and regular exercise routine — your results look good!")
        rec.append("Schedule routine check-ups every 6–12 months to track your health proactively.")

    return rec


# -------------------------------
# MAIN PROCESSOR
# -------------------------------
def process_report(file_obj):
    try:
        raw_text = extract_text(file_obj)
        analysis_results = analyze(raw_text)
        summary_data = generate_summary(analysis_results)
        recommendation_list = get_recommendations(analysis_results)
        ai_summary = generate_ai_summary(analysis_results, summary_data)

        return {
            "analysis": analysis_results,
            "summary": summary_data,
            "recommendations": recommendation_list,
            "ai_summary": ai_summary
        }

    except Exception as e:
        print("❌ PROCESS REPORT ERROR:", str(e))
        print(traceback.format_exc())
        return {
            "error": str(e),
            "analysis": [],
            "summary": {},
            "recommendations": [],
            "ai_summary": ""
        }


# -------------------------------
# FLASK ROUTE
# -------------------------------
def analyze_report():
    try:
        if "file" not in request.files:
            return jsonify({"error": "No file uploaded"}), 400

        file = request.files["file"]

        if file.filename == "":
            return jsonify({"error": "Empty filename"}), 400

        result = process_report(file)
        return jsonify(result), 200

    except Exception as e:
        print("❌ ROUTE ERROR:", str(e))
        print(traceback.format_exc())
        return jsonify({"error": str(e)}), 500



# -*- coding: utf-8 -*-
import translation_with_ollama as translator
import os
from pathlib import Path
import pytesseract
from pdf2image import convert_from_bytes, pdfinfo_from_path
import json
import datetime
import platform
import sys
import shutil
import traceback
import math
import tabula
import pandas as pd
import subprocess
import re
import string
import tempfile
from flask import Flask, request, jsonify, send_file
from flask_cors import CORS
from werkzeug.utils import secure_filename
import uuid
import time
from PIL import Image
import io

app = Flask(__name__)
CORS(app)

# *** IMPORTANT: List all Tesseract language codes you installed ***
LANGUAGES_FOR_OCR = "eng+hin+tam+tel+kan+mal+mar+pan+urd+guj+ben"
os.environ["JAVA_HOME"] = r"C:\Program Files\Java\jdk-21"  # Replace with your actual JDK path
os.environ["PATH"] = os.environ["JAVA_HOME"] + r"\bin;" + os.environ["PATH"]
print(f"JAVA_HOME set to: {os.environ['JAVA_HOME']}")
print(f"PATH updated to include: {os.environ['JAVA_HOME']}\bin")

# --- Configuration ---
if platform.system() == "Windows":
    # Update these paths if Tesseract/Poppler are installed elsewhere
    pytesseract.pytesseract.tesseract_cmd = r'C:\Program Files\Tesseract-OCR\tesseract.exe'
    POPPLER_PATH = r"C:\poppler-24.08.0\Library\bin" # Example path, update if needed
else:
    # Linux/macOS: Attempt to find in PATH
    try:
        tess_path = shutil.which("tesseract")
        if tess_path: pytesseract.pytesseract.tesseract_cmd = tess_path
    except Exception as e: 
        print(f"Error checking Tesseract PATH: {e}")
    POPPLER_PATH = None # Assumes poppler binaries are in PATH on Linux/macOS

# Ollama configuration for the FINAL ANALYSIS step
DEFAULT_OLLAMA_URL = "http://localhost:11434/api/generate"
# *** ANALYSIS MODEL ***
DEFAULT_ANALYSIS_MODEL_NAME = "gemma3:4b" # Model for the final comparison step

OUTPUT_DIR = "processed_texts"
# Threshold for non-English words to trigger translation
NON_ENGLISH_WORD_THRESHOLD = 50

# --- Helper Functions ---
def ensure_dir(directory):
    path = Path(directory)
    if not path.exists():
        try:
            path.mkdir(parents=True, exist_ok=True)
            print(f"Created directory: {directory}")
        except OSError as e:
            print(f"Failed to create directory {directory}: {e}")
            return False
    return True

def check_poppler(poppler_path_config):
    """Checks if Poppler seems accessible using pdfinfo_from_path."""
    try:
        _ = pdfinfo_from_path(Path("dummy_nonexistent_check.pdf"), poppler_path=poppler_path_config, timeout=10)
        print("Poppler check: pdfinfo command executed (ignoring 'file not found' message). Poppler seems OK.")
        return True
    except Exception as e:
        err_str = str(e).lower()
        poppler_not_found_indicators = ["pdfinfo not found", "poppler", "'pdfinfo'", "no such file or directory", "command not found", "cannot run program"]
        if any(indicator in err_str for indicator in poppler_not_found_indicators):
             print(f"Poppler Check Failed: Could not execute Poppler utility (pdfinfo). Error details: {e}")
             return False
        else:
            print(f"Poppler check passed (Ignored non-Poppler specific error on dummy file check: '{e}')")
            return True

def pdf_bytes_to_images(pdf_bytes, pdf_filename, poppler_path_config, dpi=300):
    """Converts PDF bytes to a list of PIL Image objects using Poppler."""
    images = []
    try:
        print(f"Converting '{pdf_filename}' to images (using Poppler)...")
        images = convert_from_bytes(
            pdf_bytes,
            dpi=dpi,
            poppler_path=poppler_path_config,
            thread_count=os.cpu_count() or 1, # Use multiple cores if available, default 1
            fmt='jpeg', # Use jpeg for potentially smaller size
            timeout=600 # 10 minutes timeout per PDF
            )
        print(f"Converted {len(images)} pages from '{pdf_filename}'.")
        if not images:
             print(f"Poppler converted '{pdf_filename}' but returned no images. The PDF might be empty or have an unusual format.")
             return None
    except Exception as e:
        print(f"Error converting PDF '{pdf_filename}' to images: {e}")
        err_str = str(e).lower()
        if any(indicator in err_str for indicator in ["poppler", "pdfinfo", "pdftoppm"]):
             print("This seems like a Poppler issue. Check Poppler installation, PATH, and ensure the PDF is valid and not password-protected.")
        elif "timed out" in err_str:
             print("PDF conversion timed out (10 min). The PDF might be very large or complex.")
        else:
            print("The PDF might be corrupted, password-protected without handling, or have an unsupported structure.")
        return None
    return images

def ocr_image(image, page_num, lang_codes):
    """Performs OCR on a single PIL Image using specified Tesseract languages."""
    try:
        # Standard configuration for page segmentation mode and OCR engine mode
        custom_config = r'--oem 3 --psm 3'
        # Timeout for Tesseract processing per page
        text = pytesseract.image_to_string(image, lang=lang_codes, config=custom_config, timeout=300) # 5 min OCR timeout
        return text.strip()
    except pytesseract.TesseractNotFoundError:
        print(f"FATAL: Tesseract executable not found or not configured correctly (Path checking: '{pytesseract.pytesseract.tesseract_cmd}'). Cannot perform OCR.")
        return "[Tesseract Not Found Error]"
    except pytesseract.TesseractError as e:
        err_str = str(e).lower()
        print(f"Tesseract error encountered on page {page_num}: {e}")
        if "failed loading language" in err_str:
             try: avail_langs = pytesseract.get_languages(config='')
             except: avail_langs = ["(could not fetch language list)"]
             print(f"Ensure Tesseract language packs for '{lang_codes}' are installed correctly. Available languages Tesseract found: {avail_langs}")
        elif "empty page" in err_str or "empty image" in err_str:
            print(f"Tesseract reported empty page {page_num}.")
            return "" # Return empty string for empty pages
        elif "timed out" in err_str:
            print(f"Tesseract OCR timed out processing page {page_num}.")
            return "[OCR Timeout]" # Marker for timeout
        else:
            print(f"Non-language/timeout Tesseract error on page {page_num}. Might indicate poor image quality or other issue: {e}") # Show error details
        return "[OCR Error]" # Generic OCR error marker
    except Exception as e:
        print(f"Unexpected error during OCR on page {page_num}: {e}")
        return "[Unexpected OCR Error]" # Marker for other errors

def extract_text_from_pdf(file_path):
    """Extracts text from a PDF file if it contains parsable text."""
    try:
        from PyPDF2 import PdfReader
        reader = PdfReader(file_path)
        
        # More robust text extraction
        text = ""
        for page_num, page in enumerate(reader.pages):
            try:
                page_text = page.extract_text()
                if page_text:
                    text += page_text + "\n\n"
            except Exception as page_err:
                print(f"Error extracting text from page {page_num+1}: {page_err}")
                continue  # Skip problematic pages but continue with others
        
        return text.strip() if text.strip() else None
    except ImportError as e:
        print(f"PyPDF2 library not available: {e}. Install with 'pip install PyPDF2'")
        return None
    except AttributeError as e:
        # Handle the specific '_VirtualList' object has no attribute 'index' error
        print(f"PDF structure error during text extraction: {e}")
        print("Falling back to OCR for text extraction.")
        return None
    except Exception as e:
        print(f"Error extracting text from PDF: {e}")
        return None

def extract_tables_with_tabula(pdf_path, filename):
    """
    Extracts tables using Tabula and returns embedded HTML and table data.
    Tables are extracted as-is without merging/splitting.
    """
    embedded_html_tables = ""
    all_tables_with_pages = []

    print(f"-> Starting Tabula table extraction for {filename}...")
    try:
        # Read PDF info once
        pdf_info_for_tabula = pdfinfo_from_path(pdf_path, poppler_path=POPPLER_PATH)
        total_pages_for_tabula = pdf_info_for_tabula.get("Pages", 0)
        if total_pages_for_tabula == 0:
            print(f"Tabula: Cannot get page count or PDF is empty for '{filename}'.")
            return "", []

        print(f"  Tabula: Processing {total_pages_for_tabula} pages...")
        for page_num in range(1, total_pages_for_tabula + 1):
            print(f"  Tabula: Reading page {page_num}/{total_pages_for_tabula}...")
            try:
                # Use both lattice and stream methods to extract tables
                page_tables = tabula.read_pdf(
                    pdf_path,
                    pages=str(page_num),
                    multiple_tables=True,
                    silent=True,
                    lattice=True,
                    stream=True,
                    encoding='ISO-8859-1'
                )
                if page_tables:
                    for df in page_tables:
                        # Check if df is valid DataFrame and not empty
                        if isinstance(df, pd.DataFrame) and not df.empty:
                            # Basic cleaning
                            df.dropna(axis=0, how='all', inplace=True)
                            df.dropna(axis=1, how='all', inplace=True)
                            # Only add if it still has rows/columns after cleaning
                            if not df.empty and df.shape[0] > 0 and df.shape[1] > 0:
                                # Replace NaN with empty strings for cleaner HTML/Markdown
                                all_tables_with_pages.append((df.fillna(''), page_num))
            except Exception as page_e:
                print(f"  Tabula: Error processing page {page_num}: {page_e}. Skipping page.")

        if not all_tables_with_pages:
            print(f"  Tabula: No tables detected in '{filename}'.")
            return "", []

        print(f"  Tabula: Found {len(all_tables_with_pages)} table(s).")

        # Format tables as HTML without merging/splitting
        if all_tables_with_pages:
            embedded_html_tables += "\n\n<!-- Tabula Extracted Tables Start -->\n"
            print(f"  Tabula: Formatting {len(all_tables_with_pages)} table(s) as HTML...")
            for i, (df, page_num) in enumerate(all_tables_with_pages):
                # Skip if DataFrame is empty
                if df.empty:
                    continue
                embedded_html_tables += f"<!-- HTML Table Start: Table {i+1} (Page: {page_num}) -->\n"
                try:
                    # Convert to markdown table for mermaid compatibility
                    table_md = df.to_markdown(index=False)
                    embedded_html_tables += f"````markdown\n{table_md}\n```\n\n"
                except Exception as md_err:
                    print(f"Could not convert table {i+1} to markdown: {md_err}")
                    try:
                        # Fallback to HTML if markdown conversion fails
                        table_html = df.to_html(index=False, border=1, classes='tabula-table', na_rep='')
                        embedded_html_tables += f"```html\n{table_html}\n```\n\n"
                    except Exception as html_err:
                        print(f"Could not convert table {i+1} to HTML: {html_err}")
                        embedded_html_tables += f"<!-- Error converting table {i+1} to markdown/HTML -->\n"
                embedded_html_tables += f"<!-- HTML Table End: Table {i+1} -->\n\n"
            embedded_html_tables += "<!-- Tabula Extracted Tables End -->\n\n"
        else:
            print(f"  Tabula: No tables remaining after processing for '{filename}'.")

        return embedded_html_tables, all_tables_with_pages  # Return embedded HTML text and the list

    except subprocess.CalledProcessError as e:
        print(f"Tabula Java process failed for '{filename}'. Check Java installation and PDF validity. Error: {e.stderr}")
        return f"<!-- Error: Tabula Java process failed for '{filename}'. -->\n", []
    except Exception as e:
        print(f"An unexpected error occurred during Tabula processing for '{filename}': {e}")
        print(traceback.format_exc())
        return f"<!-- Error: Tabula extraction failed for '{filename}'. -->\n", []

def process_pdf_file(file_bytes, filename, poppler_path_config, ocr_langs):
    """
    Improved PDF processing:
    1. First tries to extract parsable text
    2. Only uses OCR if no parsable text is found
    3. Identifies language and translates only if needed
    4. Extracts tables without merging/splitting
    """
    print(f"--- Starting PDF Processing for: {filename} ---")

    # Save the file temporarily to disk for text extraction and Tabula
    temp_dir = tempfile.mkdtemp()
    temp_pdf_path = Path(temp_dir) / filename
    with open(temp_pdf_path, "wb") as temp_file:
        temp_file.write(file_bytes)

    try:
        # Step 1: Attempt to extract parsable text
        extracted_text = extract_text_from_pdf(temp_pdf_path)
        
        if extracted_text:
            print(f"Parsable text detected in '{filename}'. Skipping OCR.")
            
            # Step 2: Identify language and translate if needed
            non_english_count = translator.count_non_english_words(extracted_text)
            print(f"Detected {non_english_count} non-English words in parsable text.")
            
            lang_info = translator.identify_language(extracted_text)
            print(f"Detected language: {lang_info['name']} ({lang_info['code']})")
            
            # Translate only if significant non-English content or non-English language detected
            if lang_info["code"] != "en" or non_english_count > NON_ENGLISH_WORD_THRESHOLD:
                print(f"Translating text to English...")
                translated_text = translator.translate_to_english(extracted_text, lang_info)
            else:
                print("Text is already in English. No translation needed.")
                translated_text = extracted_text
                
            # Step 3: Extract tables using Tabula
            print(f"Extracting tables from '{filename}' using Tabula...")
            embedded_html_tables, all_tables = extract_tables_with_tabula(temp_pdf_path, filename)
            
            # Step 4: Combine results
            combined_text = f"# Extracted Text from {filename}\n\n"
            combined_text += f"## Parsable Text (Translated to English if needed):\n\n{translated_text}\n\n"
            combined_text += f"## Extracted Tables (Tabula):\n\n{embedded_html_tables}\n"
            
            # Save combined results to a file
            safe_stem = "".join(c if c.isalnum() or c in ('_', '-') else '_' for c in Path(filename).stem)
            save_filename = Path(OUTPUT_DIR) / f"processed_{safe_stem}_en.txt"
            with open(save_filename, "w", encoding="utf-8") as f:
                f.write(combined_text)
            print(f"Saved combined text and table extraction to: `{save_filename}`")
            
            return {
                "original_text": extracted_text,
                "detected_language": lang_info,
                "translation": translated_text,
                "tables": [{"page": page, "table": df.to_markdown(index=False)} for df, page in all_tables],
                "embedded_html_tables": embedded_html_tables,
                "notes": ["Parsable text detected and processed. Tables extracted using Tabula."]
            }
        else:
            print(f"No parsable text found in '{filename}'. Proceeding with OCR.")
            
        # Step 2: Fallback to OCR if no parsable text is found
        images = pdf_bytes_to_images(file_bytes, filename, poppler_path_config)
        if images is None:
            error_msg = f"Error: Failed to convert PDF '{filename}' to images."
            print(error_msg)
            return {"original_text": error_msg, "detected_language": {"code": "error", "name": "PDF Conversion Failed"}, 
                    "translation": error_msg, "notes": ["PDF to Image conversion failed."], 
                    "error": "PDF to Image conversion failed."}

        full_original_text = ""
        full_translated_text = ""
        page_processing_notes = []
        detected_languages = {}
        any_page_processed_successfully = False
        any_ocr_content_found = False
        total_pages = len(images)

        print(f"Processing {total_pages} pages for '{filename}' (OCR Langs: '{ocr_langs}')...")

        for i, img in enumerate(images):
            page_num = i + 1
            print(f"Processing Page {page_num}/{total_pages}")
            page_marker = f"\n--- Page {page_num} ---\n"
            
            try:
                # OCR the page
                page_ocr_text = ocr_image(img, page_num=page_num, lang_codes=ocr_langs)
                ocr_text_len = len(page_ocr_text) if page_ocr_text else 0
                print(f"Page {page_num}: OCR finished. Text length: {ocr_text_len} chars.")

                # Store original OCR text
                full_original_text += page_marker + page_ocr_text + "\n" if page_ocr_text else page_marker + "[No text extracted from page or OCR error]\n"

                # Process and translate if OCR was successful
                if page_ocr_text and page_ocr_text not in ["[OCR Timeout]", "[OCR Error]", "[Unexpected OCR Error]"]:
                    any_ocr_content_found = True
                    
                    # Process the entire page text at once (no chunking)
                    print(f"Page {page_num}: Found text ({ocr_text_len} chars). Processing...")
                    
                    # Identify language and translate if needed
                    try:
                        # Count non-English words to determine if translation is needed
                        non_english_count = translator.count_non_english_words(page_ocr_text)
                        
                        # Identify language
                        lang_info = translator.identify_language(page_ocr_text)
                        detected_languages[page_num] = lang_info
                        
                        # Translate only if significant non-English content or non-English language detected
                        if lang_info["code"] != "en" or non_english_count > NON_ENGLISH_WORD_THRESHOLD:
                            print(f"Page {page_num}: Detected {lang_info['name']} or {non_english_count} non-English words. Translating...")
                            page_translated_text = translator.translate_to_english(page_ocr_text, lang_info)
                        else:
                            print(f"Page {page_num}: Text is in English. No translation needed.")
                            page_translated_text = page_ocr_text
                            
                        # Add the translated text
                        full_translated_text += page_marker + page_translated_text + "\n"
                        any_page_processed_successfully = True
                        
                    except Exception as e:
                        print(f"Error processing page {page_num}: {e}")
                        page_processing_notes.append(f"Page {page_num}: Processing error: {e}")
                        full_translated_text += page_marker + f"[Error processing page {page_num}: {e}]\n"
                        
                elif page_ocr_text in ["[OCR Timeout]", "[OCR Error]", "[Unexpected OCR Error]"]:
                    print(f"Page {page_num}: OCR failed or timed out ({page_ocr_text}). Skipping translation.")
                    full_translated_text += page_marker + page_ocr_text + "\n"
                    page_processing_notes.append(f"Page {page_num}: {page_ocr_text}")
                    detected_languages[page_num] = {"code": "error", "name": page_ocr_text}
                    
                else:  # No text extracted by OCR
                    print(f"Page {page_num}: No text found by OCR.")
                    full_translated_text += page_marker + "[No text extracted from this page via OCR]\n"
                    detected_languages[page_num] = {"code": "none", "name": "No Text OCR"}

            except Exception as e:
                print(f"Critical error during processing for page {page_num}: {e}")
                print(traceback.format_exc())
                full_original_text += page_marker + f"[Error processing page {page_num}: {e}]\n"
                full_translated_text += page_marker + f"[Error processing page {page_num}: {e}]\n"
                page_processing_notes.append(f"Page {page_num}: Critical processing error: {e}")
                detected_languages[page_num] = {"code": "error", "name": "Page Processing Error"}

            finally:
                try: img.close()
                except Exception: pass

        print(f"Finished processing all {total_pages} pages of {filename}.")

        # Extract tables using Tabula
        print(f"Extracting tables from '{filename}' using Tabula...")
        embedded_html_tables, all_tables = extract_tables_with_tabula(temp_pdf_path, filename)

        # Combine OCR results with tables
        combined_text = f"# Extracted Text from {filename}\n\n"
        combined_text += f"## OCR Text (Translated to English if needed):\n\n{full_translated_text}\n\n"
        combined_text += f"## Extracted Tables (Tabula):\n\n{embedded_html_tables}\n"
        
        # Save combined results to a file
        safe_stem = "".join(c if c.isalnum() or c in ('_', '-') else '_' for c in Path(filename).stem)
        save_filename = Path(OUTPUT_DIR) / f"processed_{safe_stem}_en.txt"
        with open(save_filename, "w", encoding="utf-8") as f:
            f.write(combined_text)
        print(f"Saved combined OCR text and table extraction to: `{save_filename}`")

        # Determine final status
        if not any_ocr_content_found:
            print(f"No text was extracted from any page via OCR for '{filename}'.")
            final_status = {
                "original_text": full_original_text, 
                "detected_language": detected_languages, 
                "translation": full_translated_text,
                "tables": [{"page": page, "table": df.to_markdown(index=False)} for df, page in all_tables],
                "embedded_html_tables": embedded_html_tables,
                "notes": page_processing_notes + ["No text found during OCR process for any page."], 
                "warning": "No text content found in the document via OCR."
            }
        elif not any_page_processed_successfully:
            print(f"OCR found text but processing failed for all pages in '{filename}'.")
            final_status = {
                "original_text": full_original_text, 
                "detected_language": detected_languages, 
                "translation": full_translated_text,
                "tables": [{"page": page, "table": df.to_markdown(index=False)} for df, page in all_tables],
                "embedded_html_tables": embedded_html_tables,
                "notes": page_processing_notes + ["Processing failed for all pages containing OCR'd text."], 
                "error": "All page processing failed."
            }
        else:
            print(f"Successfully processed '{filename}' with OCR and table extraction.")
            final_status = {
                "original_text": full_original_text, 
                "detected_language": detected_languages, 
                "translation": full_translated_text,
                "tables": [{"page": page, "table": df.to_markdown(index=False)} for df, page in all_tables],
                "embedded_html_tables": embedded_html_tables,
                "notes": page_processing_notes
            }

        return final_status

    except Exception as e:
        print(f"Error processing PDF file '{filename}': {e}")
        print(traceback.format_exc())
        return {
            "error": f"Error processing PDF file: {e}",
            "translation": f"[Error processing PDF: {e}]",
            "notes": [f"Critical error: {e}"],
            "original_text": ""
        }
    finally:
        # Clean up temporary file
        try:
            shutil.rmtree(temp_dir)
        except Exception as cleanup_error:
            print(f"Could not delete temporary directory '{temp_dir}': {cleanup_error}")

def process_txt_file(file_bytes, filename):
    """
    Improved TXT file processing:
    1. Reads text from TXT file bytes
    2. Identifies language
    3. Translates only if significant non-English content is detected
    """
    original_text_with_markers = ""
    raw_text = ""
    prefix = f"--- Start of Text from {filename} ---\n\n"
    suffix = f"\n\n--- End of Text from {filename} ---"
    decoded_ok = False
    notes = []
    error_msg = None

    try:
        common_encodings = ['utf-8', 'latin-1', 'cp1252']
        for encoding in common_encodings:
            try:
                decoded_text = file_bytes.decode(encoding)
                raw_text = decoded_text
                original_text_with_markers = prefix + decoded_text + suffix
                decoded_ok = True
                print(f"Decoded '{filename}' using {encoding}.")
                break
            except UnicodeDecodeError:
                continue

        if not decoded_ok:
             try:
                 decoded_text = file_bytes.decode('utf-8', errors='ignore')
                 warn_msg = f"Could not decode '{filename}' cleanly with common encodings. Decoded using UTF-8 ignoring errors (some characters might be lost)."
                 print(warn_msg); notes.append(warn_msg)
                 raw_text = decoded_text
                 original_text_with_markers = prefix + decoded_text + suffix
                 decoded_ok = True
             except Exception as decode_err:
                 warn_msg = f"Could not decode '{filename}' even ignoring errors. Using raw byte representation preview. Error: {decode_err}"
                 print(warn_msg); notes.append(warn_msg)
                 raw_bytes_preview = str(file_bytes[:200]).encode('unicode_escape').decode('ascii')
                 raw_text = f"[Could not decode file. Raw bytes preview: {raw_bytes_preview}...]"
                 original_text_with_markers = prefix + raw_text + suffix
                 decoded_ok = True # Proceed with error message as text

    except Exception as e:
        print(f"Unexpected error reading/decoding text file '{filename}': {e}")
        error_msg = f"File read/decode error: {e}"; notes.append(error_msg)
        return {
            "error": f"Error processing TXT file: {e}",
            "original_text": f"Error reading/decoding file: {e}",
            "detected_language": {"code": "error", "name": "File Read Error"},
            "translation": f"Error reading/decoding file: {e}",
            "notes": notes, "error": error_msg
        }

    if not raw_text.strip():
         print(f"File '{filename}' is empty or contains only whitespace.")
         return {
            "original_text": original_text_with_markers,
            "detected_language": {"code": "none", "name": "Empty File"},
            "translation": "[File is empty]",
            "notes": notes + ["File is empty"],
            "warning": "Input TXT file is empty."
         }

    print(f"Processing text from '{filename}'...")
    try:
        # Count non-English words to determine if translation is needed
        non_english_count = translator.count_non_english_words(raw_text)
        print(f"Detected {non_english_count} non-English words in '{filename}'.")
        
        # Identify language
        lang_info = translator.identify_language(raw_text)
        lang_name = lang_info.get('name', 'Unknown')
        lang_code = lang_info.get('code', 'unknown')
        print(f"Detected language for '{filename}': {lang_name} ({lang_code})")
        
        # Translate only if significant non-English content or non-English language detected
        if lang_code != "en" or non_english_count > NON_ENGLISH_WORD_THRESHOLD:
            print(f"Translating text from '{filename}' to English...")
            translated_text = translator.translate_to_english(raw_text, lang_info)
            notes.append(f"Text translated from {lang_name} to English.")
        else:
            print(f"Text in '{filename}' is in English. No translation needed.")
            translated_text = raw_text
            notes.append("Text is in English. No translation performed.")
            
        final_result = {
            "original_text": original_text_with_markers,
            "translation": translated_text,
            "detected_language": lang_info,
            "notes": notes
        }
        
        return final_result

    except Exception as e:
        print(f"Error processing text file '{filename}': {e}")
        print(traceback.format_exc())
        notes.append(f"Processing Error: {e}")
        error_msg = str(e)
        return {
            "error": f"Error processing TXT file: {e}",
            "original_text": original_text_with_markers,
            "detected_language": {"code": "error", "name": "Processing Error"},
            "translation": f"Error: {e}",
            "notes": notes,
            "error": error_msg
        }

def create_comparison_prompt(doc_texts_dict):
    num_docs = len(doc_texts_dict)
    doc_list = "\n".join([f"- {filename}" for filename in doc_texts_dict.keys()])

    prompt = f"""You are an expert Property Document Analyst AI. Your task is to extract and compare key details from the following {num_docs} documents to determine if they refer to the **same property**.

The text provided has been processed as follows:
- PDFs were first checked for parsable text. If found, it was used directly.
- If no parsable text was found, OCR was used with Tesseract (langs: {LANGUAGES_FOR_OCR}).
- Text was analyzed for language. If significant non-English content was detected, it was translated to English.
- Tables were extracted as-is using Tabula.
- The text includes `--- Page X ---` markers and may contain inline error markers.

Focus on extracting and comparing the following details from the **English text**. Acknowledge potential OCR inaccuracies and AI translation limitations. Pay close attention to page markers and any error markers.

**Documents Provided:**
{doc_list}

**Processed Text from Documents (Extracted & Translated to English if needed):**
"""
    for i, (filename, text) in enumerate(doc_texts_dict.items()):
        prompt += f"\n--- Document {i+1}: {filename} ---\n"
        limit = 12000
        truncated_text = text[:limit]
        prompt += truncated_text
        if len(text) > limit:
            prompt += f"\n[... text truncated for Document {i+1} due to length ...]"
        prompt += f"\n--- End of Document {i+1} ---\n"

    prompt += f"""
**Analysis Instructions:**

Extract and compare the following details across ALL provided documents ({', '.join(doc_texts_dict.keys())}) based on the **English text**. Present the results in a **table format** with color-coded cells for mismatches and a final **match percentage** column.

1. **Boundary Information:**
   - Extract the boundaries for all four directions (North, South, East, West) from each document with all the names.
   - Include a final column showing the **match percentage** for boundary information.
   - In technical documents take the boundary information as per the site only not as per the document.
2. **Owner Name:**
   - Extract the owner's name from each document.
   - Compare the names across all documents. If the names do not match, mark the cell as **red**.

3. **Property Address:**
   - Extract the property address from each document.
   - Compare the addresses across all documents. If the addresses do not match, mark the cell as **red**.

4. **Square Feet:**
   - Extract the total square feet (or square yards) from each document.
   - Compare the values across all documents. If the values do not match, mark the cell as **red**.

**Output Format:**

Present the results in a **table format** with the following structure:

| Document Name | North Boundary | South Boundary | East Boundary | West Boundary | Owner Name | Property Address | Square Feet | Match % |
|---------------|----------------|----------------|---------------|---------------|------------|------------------|-------------|---------|
| Doc1          | [Boundary]     | [Boundary]     | [Boundary]    | [Boundary]    | [Name]     | [Address]        | [Sq. Ft.]   | [XX%]   |
| Doc2          | [Boundary]     | [Boundary]     | [Boundary]    | [Boundary]    | [Name]     | [Address]        | [Sq. Ft.]   | [XX%]   |
| ...           | ...            | ...            | ...           | ...           | ...        | ...              | ...         | ...     |

**Final Analysis:**
1. Provide a summary of the overall match percentage for all documents.
2. Highlight key discrepancies in boundaries, owner names, property addresses, and square feet.
3. State whether the documents likely refer to the **same property**, **different properties**, or if the result is **uncertain** (provide specific reasons).

**Important:** Stick strictly to the provided **English text**. State when information is missing/unclear due to processing errors. Do not invent information.

Format your response using Markdown, including Mermaid diagrams where appropriate to visualize relationships or comparisons.
"""
    debug_prompt_path = Path(OUTPUT_DIR) / "comparison_prompt.txt"
    try:
        with open(debug_prompt_path, "w", encoding="utf-8") as f:
            f.write(prompt)
        print(f"Comparison prompt saved to: `{debug_prompt_path}`")
    except Exception as e:
        print(f"Could not save comparison prompt to file: {e}")
    return prompt

def create_follow_up_prompt(processed_texts_dict, initial_analysis, chat_history, user_query, model_name):
    context = f"""You are the Property Document Analyst AI who previously provided an analysis based on processed **English texts**.
Remember that the text was generated by extracting parsable text or using OCR, and translating to English if needed. The text includes page markers (`--- Page X ---`) and potential error markers.
A user has a follow-up question regarding the documents you analyzed.

**Original Documents Summary:**
You analyzed {len(processed_texts_dict)} documents: {', '.join(processed_texts_dict.keys())}.
(The full processed **English** texts were provided. Refer back to your understanding of them).

**Your Initial Analysis Summary (Truncated):**
{initial_analysis[:2500]} {'[... analysis truncated ...]' if len(initial_analysis) > 2500 else ''}

**Previous Conversation History:**
"""
    history_limit = 6
    start_index = max(0, len(chat_history) - history_limit * 2)
    for msg in chat_history[start_index:]:
        role = msg.get("role", "unknown").capitalize()
        content = msg.get("content", "")
        context += f"{role}: {content}\n"

    context += f"""
**User's New Question:**
{user_query}

**Your Task:**
Answer the user's question based *only* on:
1. The information contained in the *processed English document texts* you were given initially.
2. Your *initial analysis* summary provided above.
3. The *conversation history*.

Stick strictly to the information available in the provided **English texts**. If the answer isn't present, is unclear due to processing limitations, or requires external knowledge, **state that clearly**. Keep answers concise. Do not invent information. If referring to a specific document, use its filename.

Format your response using Markdown, including Mermaid diagrams where appropriate to visualize relationships or comparisons.
"""
    debug_context_path = Path(OUTPUT_DIR) / "context_prompt.txt"
    try:
        with open(debug_context_path, "w", encoding="utf-8") as f:
            f.write(context)
        print(f"Context prompt saved to: `{debug_context_path}`")
    except Exception as e:
        print(f"Could not save context prompt to file: {e}")
    return context

def call_ollama_api(prompt, ollama_url, model_name):
    payload = {
        "model": model_name,
        "prompt": prompt,
        "stream": False,
        "options": {"num_ctx": 100000, "max_tokens": 8000}
    }
    full_response_text = ""
    analysis_api_url = ollama_url if ollama_url else DEFAULT_OLLAMA_URL

    try:
        response = requests.post(analysis_api_url, json=payload, timeout=300)
        response.raise_for_status()
        data = response.json()

        if 'response' in data and isinstance(data['response'], str):
            full_response_text = data['response']
        elif 'message' in data and isinstance(data.get('message'), dict) and 'content' in data['message'] and isinstance(data['message']['content'], str):
            full_response_text = data['message']['content']
        elif isinstance(data, dict) and 'error' in data:
            print(f"Ollama API returned an error during analysis: {data['error']}")
            return None
        else:
            print(f"Could not find standard 'response'/'message.content' in Ollama output for analysis.")
            possible_responses = [v for v in data.values() if isinstance(v, str) and len(v) > 50]
            if possible_responses:
                full_response_text = possible_responses[0]
                print(f"Used fallback heuristic for analysis response.")
            else:
                print("Couldn't extract analysis content from Ollama response.")
                return None

        return full_response_text.strip()

    except requests.exceptions.ConnectionError:
        print(f"Connection Error: Could not connect to Ollama at {analysis_api_url} for analysis.")
        return None
    except requests.exceptions.Timeout:
        print(f"Analysis request timed out connecting to Ollama ({analysis_api_url}).")
        return None
    except requests.exceptions.HTTPError as e:
        print(f"HTTP Error calling Ollama API for analysis: {e.response.status_code} {e.response.reason}")
        try:
             error_detail = e.response.json().get('error', e.response.text)
             print(f"Ollama Error Detail: {error_detail}")
        except json.JSONDecodeError:
             print(f"Response Body: {e.response.text}")
        return None
    except requests.exceptions.RequestException as e:
        print(f"Generic Network/Request Error calling Ollama API for analysis: {e}")
        return None
    except json.JSONDecodeError as e:
         print(f"Failed to decode JSON response from Ollama during analysis: {e}")
         return None
    except Exception as e:
        print(f"An unexpected error occurred during Ollama API analysis call: {e}")
        print(f"Traceback: {traceback.format_exc()}")
        return None

ensure_dir(OUTPUT_DIR)
poppler_ok = check_poppler(POPPLER_PATH)
try:
    tesseract_ver = pytesseract.get_tesseract_version()
    tesseract_ok = True
    print(f"Tesseract v{tesseract_ver} detected.")
except pytesseract.TesseractNotFoundError:
    print(f"Tesseract not found at: '{pytesseract.pytesseract.tesseract_cmd}'. Fix path.")
    tesseract_ok = False
except Exception as e:
     print(f"Could not verify Tesseract version: {e}")
     tesseract_ok = True

@app.route('/api/process', methods=['POST'])
def process_documents():
    if not request.files:
        return jsonify({"error": "No files uploaded"}), 400
    
    if not tesseract_ok:
        return jsonify({"error": "Tesseract is not configured correctly"}), 500
        
    files = request.files.getlist('files')
    if len(files) < 2:
        return jsonify({"error": "Please upload at least two documents for comparison"}), 400
    
    session_id = str(uuid.uuid4())
    session_dir = Path(OUTPUT_DIR) / session_id
    ensure_dir(session_dir)
    
    all_results = {}
    
    for uploaded_file in files:
        filename = secure_filename(uploaded_file.filename)
        filetype = uploaded_file.content_type
        
        try:
            file_bytes = uploaded_file.read()
            
            if filetype == "application/pdf":
                if not poppler_ok:
                    return jsonify({"error": "Poppler is not configured correctly for PDF processing"}), 500
                processed_result_dict = process_pdf_file(file_bytes, filename, POPPLER_PATH, LANGUAGES_FOR_OCR)
            elif filetype == "text/plain":
                processed_result_dict = process_txt_file(file_bytes, filename)
            else:
                processed_result_dict = {
                    "error": f"Unsupported file type: {filetype}",
                    "translation": "Error: Unsupported file type",
                    "notes": [f"Unsupported file type: {filetype}"],
                    "original_text": ""
                }
                
            all_results[filename] = processed_result_dict
            
        except Exception as e:
            all_results[filename] = {
                "error": f"Error processing file: {str(e)}",
                "translation": f"Error: {str(e)}",
                "notes": [f"Processing error: {str(e)}"],
                "original_text": ""
            }
    
    docs_for_llm = {}
    for filename, res_dict in all_results.items():
        t_content = res_dict.get("translation", "")
        has_error = res_dict.get("error")
        
        is_unusable = not t_content or \
                      t_content in ["[No text extracted or translated]", "[Translation not returned or empty]", "[File is empty]"] or \
                      t_content.strip().startswith("[Could not decode") or \
                      t_content.strip().startswith("[Error") or \
                      t_content.strip().startswith("[OCR") or \
                      t_content.strip() == "[No text extracted from any page via OCR]" or \
                      has_error
                      
        if not is_unusable:
            docs_for_llm[filename] = t_content
    
    analysis_result = None
    if len(docs_for_llm) >= 2:
        ollama_url = request.form.get('ollama_url', DEFAULT_OLLAMA_URL)
        model_name = request.form.get('model_name', DEFAULT_ANALYSIS_MODEL_NAME)
        
        comparison_prompt = create_comparison_prompt(docs_for_llm)
        analysis_result = call_ollama_api(comparison_prompt, ollama_url, model_name)
    
    response = {
        "session_id": session_id,
        "processed_files": all_results,
        "analysis_result": analysis_result,
        "usable_files_count": len(docs_for_llm),
        "total_files_count": len(all_results)
    }
    
    with open(session_dir / "session_data.json", "w", encoding="utf-8") as f:
        json.dump(response, f, ensure_ascii=False, indent=2)
    
    return jsonify(response)

@app.route('/api/chat', methods=['POST'])
def chat():
    data = request.json
    if not data:
        return jsonify({"error": "No data provided"}), 400
    
    session_id = data.get('session_id')
    if not session_id:
        return jsonify({"error": "No session_id provided"}), 400
    
    session_dir = Path(OUTPUT_DIR) / session_id
    if not session_dir.exists():
        return jsonify({"error": "Invalid session_id"}), 404
    
    session_data_file = session_dir / "session_data.json"
    if not session_data_file.exists():
        return jsonify({"error": "Session data not found"}), 404
    
    try:
        with open(session_data_file, "r", encoding="utf-8") as f:
            session_data = json.load(f)
    except Exception as e:
        return jsonify({"error": f"Error loading session data: {str(e)}"}), 500
    
    user_query = data.get('query')
    if not user_query:
        return jsonify({"error": "No query provided"}), 400
    
    chat_history = data.get('chat_history', [])
    
    docs_for_llm = {}
    for filename, res_dict in session_data.get('processed_files', {}).items():
        t_content = res_dict.get("translation", "")
        has_error = res_dict.get("error")
        
        is_unusable = not t_content or \
                      t_content in ["[No text extracted or translated]", "[Translation not returned or empty]", "[File is empty]"] or \
                      t_content.strip().startswith("[Could not decode") or \
                      t_content.strip().startswith("[Error") or \
                      t_content.strip().startswith("[OCR") or \
                      t_content.strip() == "[No text extracted from any page via OCR]" or \
                      has_error
                      
        if not is_unusable:
            docs_for_llm[filename] = t_content
    
    if len(docs_for_llm) < 2:
        return jsonify({"error": "Not enough usable documents for chat"}), 400
    
    initial_analysis = session_data.get('analysis_result', '')
    if not initial_analysis:
        return jsonify({"error": "No initial analysis available"}), 400
    
    ollama_url = data.get('ollama_url', DEFAULT_OLLAMA_URL)
    model_name = data.get('model_name', DEFAULT_ANALYSIS_MODEL_NAME)
    
    follow_up_prompt = create_follow_up_prompt(
        docs_for_llm,
        initial_analysis,
        chat_history,
        user_query,
        model_name
    )
    
    response = call_ollama_api(follow_up_prompt, ollama_url, model_name)
    
    if not response:
        return jsonify({"error": "Failed to get response from AI model"}), 500
    
    chat_history.append({"role": "user", "content": user_query})
    chat_history.append({"role": "assistant", "content": response})
    
    # Update session data with new chat history
    chat_history_file = session_dir / "chat_history.json"
    try:
        with open(chat_history_file, "w", encoding="utf-8") as f:
            json.dump(chat_history, f, ensure_ascii=False, indent=2)
    except Exception as e:
        print(f"Error saving chat history: {e}")
    
    return jsonify({
        "response": response,
        "chat_history": chat_history
    })

@app.route('/api/download/<session_id>/<filename>', methods=['GET'])
def download_processed_file(session_id, filename):
    session_dir = Path(OUTPUT_DIR) / session_id
    if not session_dir.exists():
        return jsonify({"error": "Invalid session_id"}), 404
    
    safe_filename = secure_filename(filename)
    file_path = session_dir / safe_filename
    
    if not file_path.exists():
        return jsonify({"error": "File not found"}), 404
    
    return send_file(file_path, as_attachment=True)

@app.route('/api/status', methods=['GET'])
def get_status():
    status = {
        "tesseract_ok": tesseract_ok,
        "poppler_ok": poppler_ok,
        "languages_for_ocr": LANGUAGES_FOR_OCR,
        "default_analysis_model": DEFAULT_ANALYSIS_MODEL_NAME,
        "default_ollama_url": DEFAULT_OLLAMA_URL,
        "system_info": {
            "platform": platform.system(),
            "python_version": sys.version.split()[0],
            "non_english_word_threshold": NON_ENGLISH_WORD_THRESHOLD
        }
    }
    
    # Check Ollama reachability
    try:
        if hasattr(translator, 'OLLAMA_API'):
            base_ollama_url = translator.OLLAMA_API
            if not base_ollama_url:
                status["ollama_reachable"] = False
                status["ollama_error"] = "OLLAMA_API is empty"
            else:
                # Strip common API endpoints to get base URL for checking server status
                if base_ollama_url.endswith("/api/generate"): 
                    base_ollama_url = base_ollama_url[:-len("/api/generate")]
                elif base_ollama_url.endswith("/api/chat"): 
                    base_ollama_url = base_ollama_url[:-len("/api/chat")]
                if base_ollama_url.endswith("/"): 
                    base_ollama_url = base_ollama_url[:-1]
                
                response = requests.get(base_ollama_url, timeout=10)
                response.raise_for_status()
                status["ollama_reachable"] = True
                status["ollama_url"] = base_ollama_url
        else:
            status["ollama_reachable"] = False
            status["ollama_error"] = "OLLAMA_API not found in translator module"
    except Exception as e:
        status["ollama_reachable"] = False
        status["ollama_error"] = str(e)
    
    # Add translator module info
    if hasattr(translator, 'TRANSLATION_MODEL'):
        status["translation_model"] = translator.TRANSLATION_MODEL
    if hasattr(translator, 'IDENTIFICATION_MODEL'):
        status["identification_model"] = translator.IDENTIFICATION_MODEL
    
    return jsonify(status)

@app.route('/api/sessions', methods=['GET'])
def list_sessions():
    output_dir = Path(OUTPUT_DIR)
    if not output_dir.exists():
        return jsonify({"sessions": []}), 200
    
    sessions = []
    for session_dir in output_dir.iterdir():
        if session_dir.is_dir():
            session_data_file = session_dir / "session_data.json"
            if session_data_file.exists():
                try:
                    with open(session_data_file, "r", encoding="utf-8") as f:
                        session_data = json.load(f)
                    
                    session_info = {
                        "session_id": session_dir.name,
                        "created_at": datetime.datetime.fromtimestamp(session_dir.stat().st_ctime).isoformat(),
                        "file_count": session_data.get("total_files_count", 0),
                        "usable_file_count": session_data.get("usable_files_count", 0),
                        "has_analysis": bool(session_data.get("analysis_result"))
                    }
                    
                    # Get file names
                    session_info["files"] = list(session_data.get("processed_files", {}).keys())
                    
                    sessions.append(session_info)
                except Exception as e:
                    print(f"Error reading session data for {session_dir.name}: {e}")
    
    # Sort sessions by creation time (newest first)
    sessions.sort(key=lambda x: x["created_at"], reverse=True)
    
    return jsonify({"sessions": sessions})

@app.route('/api/session/<session_id>', methods=['GET'])
def get_session(session_id):
    session_dir = Path(OUTPUT_DIR) / session_id
    if not session_dir.exists():
        return jsonify({"error": "Invalid session_id"}), 404
    
    session_data_file = session_dir / "session_data.json"
    if not session_data_file.exists():
        return jsonify({"error": "Session data not found"}), 404
    
    try:
        with open(session_data_file, "r", encoding="utf-8") as f:
            session_data = json.load(f)
    except Exception as e:
        return jsonify({"error": f"Error loading session data: {str(e)}"}), 500
    
    # Load chat history if available
    chat_history_file = session_dir / "chat_history.json"
    chat_history = []
    if chat_history_file.exists():
        try:
            with open(chat_history_file, "r", encoding="utf-8") as f:
                chat_history = json.load(f)
        except Exception as e:
            print(f"Error loading chat history: {e}")
    
    session_data["chat_history"] = chat_history
    
    return jsonify(session_data)

@app.route('/api/session/<session_id>', methods=['DELETE'])
def delete_session(session_id):
    session_dir = Path(OUTPUT_DIR) / session_id
    if not session_dir.exists():
        return jsonify({"error": "Invalid session_id"}), 404
    
    try:
        shutil.rmtree(session_dir)
        return jsonify({"success": True, "message": f"Session {session_id} deleted successfully"})
    except Exception as e:
        return jsonify({"error": f"Error deleting session: {str(e)}"}), 500

if __name__ == '__main__':
    app.run(debug=True, host='0.0.0.0', port=5000)


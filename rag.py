```python
from loader import *
st.set_page_config(
        page_title="Legal Document Analysis",
        page_icon="🔍",
        layout="wide",  # This ensures full width
        initial_sidebar_state="collapsed"
    )
import faiss
import copy
from Chatbot_Process import retrieve_top_k
from reportlab.lib.pagesizes import letter
from reportlab.lib.styles import getSampleStyleSheet, ParagraphStyle
from reportlab.lib.units import inch
from reportlab.lib import colors
from reportlab.platypus import SimpleDocTemplate, Paragraph, Spacer, Table, TableStyle, PageBreak
import io # Make sure io is imported for BytesIO
import re
import json

# Update the ACCESS_KEYWORDS list with more comprehensive terms
ACCESS_KEYWORDS = [
    'road', 'street', 'path', 'pathway', 'lane', 'highway', 'avenue',
    'boulevard', 'drive', 'alley', 'circle', 'court', 'expressway',
    'freeway', 'terrace', 'place', 'square', 'walk', 'rasta', 'way',
    'access', 'approach', 'entry', 'thoroughfare', 'route','gali','track',
    'thoroughfare', 'passage', 'entryway', 'driveway', 'gate', 'ingress',
    'egress', 'corridor', 'pass', 'bypass', 'artery', 'byway', 'lane',
    'boulevard', 'causeway', 'crossing', 'frontage', 'highroad', 'parkway',
    'promenade', 'walkway', 'alleyway', 'esplanade', 'footpath', 'trail',
    'bridge', 'underpass', 'overpass'
]

# 1. Enhanced Address Extraction (Updated)
ADDRESS_KEYWORDS = [
    "survey no.", "patta no.", "door no.", "village", "district", "street",
    "road", "avenue", "town", "city", "state", "pin code", "situate",
    "location", "property at", "schedule", "boundaries of", "property details",
    "khasra no.","gata no.", "plot no.", "situated at", "address", "location"
]

# (Updated)
COMPONENT_PATTERNS = {
    "survey_no": r"(?:Survey|S\.?[rR]\.?|Plot|Khasra)[\s\.,]*[Nn]o\.?\s*([\w\d/.-]+)",
    "patta_no": r"Patta[\s\.,]*[Nn]o\.?\s*([\w\d/.-]+)",
    "door_no": r"Door[\s\.,]*[Nn]o\.?\s*([\w\d/.-]+)",
    "village": r"Village:\s*([\w\s]+)|([\w\s]+Village)",
    "district": r"District:\s*([\w\s]+)|([\w\s]+District)",
    "state": r"State:\s*([\w\s]+)|([\w\s]+State)",
    "pin_code": r"Pin\s*[Cc]ode:\s*(\d{6})|\b(\d{6})\b",
    "area": r"admeasuring\s*([\d.,]+\s*(?:Sq\.?\s*Yard|Sq\.?\s*[Mm]eter|Sq\.?\s*[Ff]t|Acres?|Hectares?|Cents?|Grounds?))",
    "street": r"Street:\s*([\w\s]+)|([\w\s]+Street)"
}

# (New) Function based on user's specific regex for a full address line
def extract_full_address_line(text):
    """Extracts a complete address line based on the 'Property situated at' pattern."""
    pattern = r"Property situated at (.*?)(?=\n|$)"
    match = re.search(pattern, text, re.IGNORECASE | re.DOTALL)
    return match.group(1).strip() if match else None


# 2. Owner Name Extraction (Updated with user's keywords)
def extract_owner_names(text):
    """Extracts owner names using a more comprehensive set of patterns."""
    patterns = [
        # User-suggested keywords
        r"Owner[\s\.,]*Name\(?s?\)?[:\s]*([\w\s.,]+)",
        r"Title[\s\.,]*Owner\(?s?\)?[:\s]*([\w\s.,]+)",
        # Original patterns
        r"Owner\(s\):\s*([\w\s.,]+)",
        r"Name\(s\)[\s\.,]*of[\s\.,]*Owner\(s\)[:\s]*([\w\s.,]+)"
    ]
    for pattern in patterns:
        match = re.search(pattern, text, re.IGNORECASE)
        if match:
            return match.group(1).strip()
    return None

# 3. Property Extent Distinction (Updated with user's keywords)
def extract_property_extent(text):
    """Extracts land and built-up area using more varied keywords."""
    # Merged patterns from original and user's suggestion
    land_pattern = r"(?:Land[\s\.,]*Area|Plot[\s\.,]*Area|admeasuring)[\s:]*([\d.,]+\s*(?:Sq\.?\s*Yard|Sq\.?\s*[Mm]eter|Sq\.?\s*[Ff]t|Acres?|Hectares?|Cents?|Grounds?|Ankanams?|Guntas?))"
    built_up_pattern = r"(?:Built[\s-]*up[\s\.,]*Area|Constructed[\s\.,]*Area)[\s:]*([\d.,]+\s*(?:Sq\.?\s*Yard|Sq\.?\s*[Mm]eter|Sq\.?\s*[Ff]t|sft))"

    land_match = re.search(land_pattern, text, re.IGNORECASE)
    built_up_match = re.search(built_up_pattern, text, re.IGNORECASE)

    return {
        "land_area": land_match.group(1).strip() if land_match else "Not Found",
        "built_up_area": built_up_match.group(1).strip() if built_up_match else "Not Found"
    }

# 4. Title History Processing (Updated: Removed skipping of patta documents to include everything)
def process_title_history(title_history_list):
    processed = []
    if not isinstance(title_history_list, list):
        return [] # Return empty list if input is not a list

    for entry in title_history_list:
        if not isinstance(entry, dict):
            continue

        # No skipping anymore - include all entries

        # Ensure proper field names
        executed_by = entry.get("executed_by", entry.get("from", "Unknown"))
        in_favour_of = entry.get("in_favour_of", entry.get("to", "Unknown"))

        # Correct transfer nature
        nature_raw = entry.get("nature_of_transfer", "Transfer")
        nature = str(nature_raw).strip() # Ensure it's a string
        nature_lower = nature.lower()

        # Standardize transfer nature names
        if "deed/patta" in nature_lower or "registered deed/patta" in nature_lower:
            nature = "Deed/Patta"
        elif "will" in nature_lower:
            nature = "Will"
        elif "court decree" in nature_lower:
            nature = "Court Decree"
        elif "sale" in nature_lower:
            nature = "Sale"
        elif "gift" in nature_lower:
            nature = "Gift"
        elif "partition" in nature_lower:
            nature = "Partition"
        # Keep original if no match
        
        processed.append({
            "document_date": entry.get("document_date", ""),
            "document_name": entry.get("document_name", ""),
            "executed_by": executed_by,
            "in_favour_of": in_favour_of,
            "nature_of_transfer": nature,
            "extent_owned": entry.get("extent_owned", "Full Property")
        })
    return processed
 

# 5. Context Length Optimization (New)
def optimize_context(document_text, max_length=30000):
    """Reduce context length while preserving key sections"""
    if len(document_text) <= max_length:
        return document_text

    # Preserve key sections using headers
    preserved_sections = []
    headers = [
        "PROPERTY ADDRESS", "BOUNDARIES", "OWNER NAME", "PROPERTY EXTENT",
        "TITLE HISTORY", "DOCUMENTS EXAMINED", "MORTGAGE DOCUMENTS",
        "LEGAL OPINION", "CONCLUSION"
    ]

    # Create a regex to find content between headers
    header_pattern_str = '|'.join(map(re.escape, headers))
    # This pattern finds a header and captures everything until the next header or end of string
    pattern = re.compile(fr"(\b(?:{header_pattern_str})\b.*?(?=\n\b(?:{header_pattern_str})\b|\Z))",
                       re.IGNORECASE | re.DOTALL)

    matches = pattern.findall(document_text)
    if matches:
        preserved_sections = [match.strip() for match in matches]

    optimized_text = "\n\n".join(preserved_sections)

    # If key sections are still too long or not found, truncate the original
    if not optimized_text or len(optimized_text) > max_length:
        return document_text[:max_length]

    return optimized_text

# 6. Legal Opinion Extraction (Updated with user's pattern)
def extract_legal_opinion(text):
    """Extracts the legal opinion using a list of common section headers."""
    patterns = [
        # User-suggested pattern (modified for robustness)
        r"Conclusion/ observation, if any\.(.*?)(?=\n\n[A-Z\s]{3,}:|\Z)",
        # Original patterns
        r"LEGAL OPINION(.*?)(?=\n\n[A-Z\s]{3,}:|\Z)",
        r"OPINION(.*?)(?=\n\n[A-Z\s]{3,}:|\Z)",
        r"REMARKS(.*?)(?=\n\n[A-Z\s]{3,}:|\Z)"
    ]

    for pattern in patterns:
        match = re.search(pattern, text, re.IGNORECASE | re.DOTALL)
        if match:
            # Clean up the extracted text
            opinion = match.group(1).strip()
            # Remove leading list-like characters (e.g., "1.", "a)", "-")
            opinion = re.sub(r'^\s*[\d\w][\.\)]\s*', '', opinion)
            return opinion

    return "Not Found"


try:
    index = faiss.read_index(r"chola_faiss_index.faiss")
    with open(r'chola_metadata.json', 'r', encoding='utf-8') as f:
        metadata = json.load(f)
    with open(r'chola_answers.json', 'r', encoding='utf-8') as f:
        answers = json.load(f)
    print(f"Loaded {len(metadata)} metadata entries and {len(answers)} answers")
    RAG_AVAILABLE = True
except Exception as e:
    print(f"Error loading FAISS index or related files: {e}")
    RAG_AVAILABLE = False

def ensure_dir(directory):
    path = Path(directory)
    if not path.exists():
        try:
            path.mkdir(parents=True, exist_ok=True)
        except OSError as e:
            st.error(f"Failed to create directory {directory}: {e}")
            st.stop()

def check_poppler(poppler_path_config):
    if not poppler_path_config and platform.system() != "Windows" and not IS_MAC:
        if shutil.which("pdfinfo"): return True
        else:
            print("Poppler Check Failed: pdfinfo not found in PATH.")
            return False
    elif not poppler_path_config and platform.system() == "Windows":
        print("Poppler Check Failed: POPPLER_PATH not configured for Windows.")
        return False

    # For Mac, poppler_path_config is set, so this will run. For others, it checks the configured path.
    try:
        # On Mac, the config path is a directory, not a file path.
        if IS_MAC:
            pdfinfo_command = str(Path(poppler_path_config) / 'pdfinfo')
        else:
            pdfinfo_command = 'pdfinfo' if not poppler_path_config else str(Path(poppler_path_config) / 'pdfinfo')

        process = subprocess.Popen([pdfinfo_command, '-h'], stdout=subprocess.PIPE, stderr=subprocess.PIPE, shell=platform.system()=="Windows")
        _, err = process.communicate(timeout=5)
        if process.returncode in [0, 1]: return True
        else:
            print(f"Poppler Check Failed from '{poppler_path_config}'. RC: {process.returncode}. Err: {err.decode(errors='ignore')}")
            return False
    except FileNotFoundError:
        print(f"Poppler Check Failed: pdfinfo not found at '{poppler_path_config}'.")
        return False
    except subprocess.TimeoutExpired:
        print(f"Poppler Check Timed Out from '{poppler_path_config}'.")
        return False
    except Exception as e:
        print(f"Poppler Check Failed: Unexpected error: {e}. Path: '{poppler_path_config}'.")
        return False

def pdf_bytes_to_images(pdf_bytes, pdf_filename, poppler_path_config, dpi=450):
    images = []
    try:
        actual_poppler_path = poppler_path_config if poppler_path_config and Path(poppler_path_config).is_dir() else None
        images = convert_from_bytes(pdf_bytes, dpi=dpi, poppler_path=actual_poppler_path, thread_count=os.cpu_count() or 1, fmt='jpeg', timeout=600)
        if not images:
            print(f"Poppler converted '{pdf_filename}' but returned no images.")
            return []
    except Exception as e:
        print(f"Error converting PDF '{pdf_filename}' to images: {e}")
        if "poppler" in str(e).lower() or "pdfinfo" in str(e).lower() or "pdftoppm" in str(e).lower():
            print("Poppler issue. Check Poppler installation/PATH/POPPLER_PATH. PDF might be corrupted/protected.")
        return None
    return images

def ocr_image(image_pil: Image.Image, page_num: int, original_filename: str):
    """
    Performs OCR on a single image using the Doctr model.
    Tesseract logic has been removed as requested.
    """
    page_all_texts_untranslated = []
    debug_log_messages = []

    if not DOCTR_AVAILABLE or not DOCTR_PREDICTOR:
        msg = f"Doctr OCR Error (p{page_num}, {original_filename}): Doctr model is not available or not loaded."
        debug_log_messages.append(msg)
        print(f"ERROR: ocr_image (Doctr): {msg}")
        return f"[Doctr OCR Error on page {page_num}: Model not available]", debug_log_messages

    debug_log_messages.append(f"Page {page_num}: Using Doctr OCR engine.")
    print(f"DEBUG: ocr_image (Page {page_num}, File {original_filename}): Using Doctr engine.")
    try:
        # Convert PIL Image to numpy array as required by doctr
        doc_page_np = [np.array(image_pil.convert('RGB'))]
        result = DOCTR_PREDICTOR(doc_page_np)
        json_output = result.export()

        if json_output and json_output.get('pages'):
            page_data = json_output['pages'][0]

            # Reconstruct full text from all detected blocks for a natural reading order
            page_content_blocks = []
            for block in page_data.get('blocks', []):
                block_text = ""
                for line in block.get('lines', []):
                    # Join words in a line with a space
                    line_text = ' '.join([word['value'] for word in line.get('words', [])])
                    block_text += line_text + '\n' # Add newline after each line
                page_content_blocks.append(block_text.strip())

            full_page_text = "\n\n".join(page_content_blocks)
            page_all_texts_untranslated.append(f"\n--- Full Page OCR Text (Doctr) (p{page_num}) ---\n{full_page_text}\n--- End Full Page OCR Text ---\n")
            debug_log_messages.append(f"  Page {page_num}: Doctr full page OCR successful. Length: {len(full_page_text)}")

            # Extract tables detected by Doctr's layout analysis
            if page_data.get('tables'):
                debug_log_messages.append(f"  Page {page_num}: Doctr detected {len(page_data['tables'])} table(s).")
                for table_idx, table_structure in enumerate(page_data['tables']):
                    table_as_string = ""
                    # Reconstruct the table from its cell structure
                    for row in table_structure.get('body', []):
                        row_cells = [cell.get('value', '') for cell in row.get('cells', [])]
                        table_as_string += " | ".join(row_cells) + "\n"

                    demarcated_table_text = f"\n--- Start of AI Detected Table (Doctr p{page_num}-t{table_idx+1}) ---\n{table_as_string.strip()}\n--- End of AI Detected Table (Doctr) ---\n"
                    page_all_texts_untranslated.append(demarcated_table_text)
                    debug_log_messages.append(f"    Page {page_num}: Extracted Doctr table {table_idx+1}.")
            else:
                debug_log_messages.append(f"  Page {page_num}: Doctr detected no tables on this page.")
        else:
            debug_log_messages.append(f"  Page {page_num}: Doctr OCR returned no pages in its output.")
            page_all_texts_untranslated.append(f"[Doctr returned no content for page {page_num}]")

    except Exception as e:
        msg = f"Doctr OCR Error (p{page_num}, {original_filename}): {e}"
        debug_log_messages.append(msg)
        print(f"ERROR: ocr_image (Doctr): {msg}\n{traceback.format_exc()}")
        page_all_texts_untranslated.append(f"[Doctr OCR Error on page {page_num}]")

    # --- Final Text Combination ---
    final_combined_page_text = "\n\n".join(pt for pt in page_all_texts_untranslated if pt.strip() and not pt.startswith("[No text found")).strip()
    if not final_combined_page_text:
        final_combined_page_text = f"[No text extracted from page {page_num} via Doctr engine]"
        debug_log_messages.append(f"Page {page_num}: Final combined text is empty.")

    return final_combined_page_text, debug_log_messages

def extract_text_from_pdf(file_path_obj: Path):
    try:
        from PyPDF2 import PdfReader
        reader = PdfReader(str(file_path_obj))
        if reader.is_encrypted:
            try:
                if reader.decrypt("") == 0:
                    print(f"PDF '{file_path_obj.name}' encrypted, decryption failed.")
                    return "[PDF Encrypted - Decryption Failed]"
            except Exception as decrypt_e:
                print(f"Error decrypting PDF '{file_path_obj.name}': {decrypt_e}")
                return "[PDF Encrypted - Decryption Error]"
        text_content = []
        num_pages_with_text = 0
        for page_num, page in enumerate(reader.pages):
            try:
                page_text = page.extract_text()
                if page_text and page_text.strip():
                    text_content.append(f"--- Page {page_num + 1} ---\n{page_text.strip()}")
                    num_pages_with_text += 1
            except Exception as page_err:
                print(f"PyPDF2: Error extracting text from page {page_num + 1} of '{file_path_obj.name}': {page_err}")
                text_content.append(f"--- Page {page_num + 1} ---\n[Text Extraction Error on this page]")
        if num_pages_with_text > 0 :
                        print(f"PyPDF2: Found parsable text on {num_pages_with_text}/{len(reader.pages)} pages of '{file_path_obj.name}'.")
                        return "\n\n".join(text_content)
        else:
            print(f"PyPDF2: No parsable text in '{file_path_obj.name}'. OCR needed.")
            return None
    except ImportError:
                print("PyPDF2 not installed. pip install PyPDF2")
                return "[PyPDF2 Import Error]"
    except Exception as e:
        print(f"PyPDF2: Error extracting text from PDF '{file_path_obj.name}': {e}. OCR needed.")
        return None

def extract_tables_with_tabula(pdf_path_obj: Path, filename_str: str):
    all_tables_dfs_with_pages = []
    plain_text_tables_for_llm = []
    embedded_html_tables_str = "<!-- No tables attempted or extracted -->"
    print(f"  -> Starting Tabula table extraction for '{filename_str}'...")
    java_home = os.environ.get("JAVA_HOME")
    java_exe_found = False
    if java_home:
        if platform.system() == "Windows" and (Path(java_home) / "bin" / "java.exe").exists(): java_exe_found = True
        elif platform.system() != "Windows" and (Path(java_home) / "bin" / "java").exists(): java_exe_found = True
    if not java_exe_found and not shutil.which("java"):
        print(f"Tabula: Java runtime not found. Table extraction will fail.")
        return "<!-- Error: Tabula - Java Not Found -->\n", [], []
    try:
        dfs_from_pdf = tabula.read_pdf(str(pdf_path_obj), pages='all', multiple_tables=True,
                                       stream=True, lattice=True, silent=True, encoding='utf-8',
                                       java_options=["-Dfile.encoding=UTF8"])
        if not dfs_from_pdf:
            print(f"  Tabula: No tables detected in '{filename_str}'.")
            return "<!-- No tables detected -->\n", [], []
        temp_dfs_with_pages = []
        for i, df in enumerate(dfs_from_pdf):
            if isinstance(df, pd.DataFrame) and not df.empty:
                df.dropna(axis=0, how='all', inplace=True)
                df.dropna(axis=1, how='all', inplace=True)
                if not df.empty: temp_dfs_with_pages.append((df.fillna(''), 'Unknown'))
        if not temp_dfs_with_pages:
            print(f"  Tabula: No tables detected after cleaning in '{filename_str}'.")
            return "<!-- No valid tables after cleaning -->\n", [], []
        print(f"  Tabula: Found {len(temp_dfs_with_pages)} potential table(s) in '{filename_str}'. Formatting...")
        all_tables_dfs_with_pages = temp_dfs_with_pages
        embedded_html_tables_str = "\n\n<!-- Tabula Extracted Tables Start -->\n"
        for i, (df, page_num_str) in enumerate(all_tables_dfs_with_pages):
            if df.empty: continue
            table_id_html = f"Table {i+1} (Page: {page_num_str})"
            table_id_plain = f"--- Table {i+1} (Page: {page_num_str}) ---"
            try: embedded_html_tables_str += f"<!-- HTML Table Start: {table_id_html} -->\n{df.to_html(index=False, border=1, classes='tabula-table streamlit-table', na_rep='')}\n<!-- HTML Table End: {table_id_html} -->\n\n"
            except Exception as html_err: print(f"Could not convert table {i+1} from '{filename_str}' to HTML: {html_err}")
            try:
                with pd.option_context('display.max_rows', None, 'display.max_columns', None, 'display.width', 2000, 'display.max_colwidth', None):
                    plain_text_table = df.to_string(index=False, na_rep='')
                plain_text_tables_for_llm.append(f"{table_id_plain}\n{plain_text_table}")
            except Exception as plain_err: print(f"Could not convert table {i+1} from '{filename_str}' to plain text: {plain_err}")
        embedded_html_tables_str += "<!-- Tabula Extracted Tables End -->\n\n"
        return embedded_html_tables_str, all_tables_dfs_with_pages, plain_text_tables_for_llm
    except FileNotFoundError as e:
        if "java" in str(e).lower(): print(f"Tabula: Java runtime not found. Error: {e}. Extraction failed.")
        else: print(f"Tabula: File not found error for '{filename_str}': {e}")
        return "<!-- Error: Tabula - Java/File Not Found -->\n", [], []
    except Exception as e:
        print(f"Tabula: Unexpected error for '{filename_str}': {e}.")
        print(traceback.format_exc())
        return f"<!-- Error: Tabula failed for {filename_str} -->\n", [], []

def clean_processing_artifacts(text, filename_for_debug=""):
    if not text: return ""
    cleaned_text = text
    patterns_to_remove_at_start = [
        re.compile(r"^\s*Okay, I'm ready to process.*?\.\s*", re.IGNORECASE | re.DOTALL),
        re.compile(r"^\s*Okay, I understand.*?Here's the Markdown table with the extracted data\.\s*", re.IGNORECASE | re.DOTALL),
        re.compile(r"^\s*Here's the Markdown table.*?\.\s*", re.IGNORECASE | re.DOTALL),
        re.compile(r"^\s*Here is the English translation.*?\n", re.IGNORECASE | re.DOTALL),
        re.compile(r"^\s*Here's the English translation.*?\n", re.IGNORECASE | re.DOTALL),
        re.compile(r"^\s*Here's a translation of the provided text.*?\n", re.IGNORECASE | re.DOTALL),
        re.compile(r"^\s*This document appears to be.*?translation.*?\n", re.IGNORECASE | re.DOTALL),
        re.compile(r"^\s*This appears to be nonsensical text.*?\n", re.IGNORECASE | re.DOTALL),
        re.compile(r"^\s*This appears to be a collection of.*original text to translate\..*?\n", re.IGNORECASE | re.DOTALL),
        re.compile(r"^\s*I apologize, but.*?fluent English translation due to the significant damage.*\n", re.IGNORECASE | re.MULTILINE),
        re.compile(r"^\s*```json\s*$", re.IGNORECASE | re.MULTILINE),
        re.compile(r"^\s*Here is the JSON output:?\s*```json\s*", re.IGNORECASE | re.DOTALL),
        re.compile(r"^\s*```\s*$", re.IGNORECASE | re.MULTILINE),
    ]
    for pattern in patterns_to_remove_at_start:
        cleaned_text = pattern.sub("", cleaned_text, count=1).lstrip()
    cleaned_text = re.sub(r"\[Information redacted.*?\]", "", cleaned_text, flags=re.IGNORECASE)
    cleaned_text = re.sub(r"\[.*?fragmented text.*?\]", "", cleaned_text, flags=re.IGNORECASE | re.DOTALL)
    cleaned_text = re.sub(r"\[\s*Content of image.*?could not be reliably translated\s*\]", "", cleaned_text, flags=re.IGNORECASE)
    cleaned_text = re.sub(r"\[.*?No translation available due to .*?\]", "", cleaned_text, flags=re.IGNORECASE | re.DOTALL)
    cleaned_text = re.sub(r'\r\n', '\n', cleaned_text)
    cleaned_text = re.sub(r'\r', '\n', cleaned_text)
    cleaned_text = re.sub(r'\n{3,}', '\n\n', cleaned_text).strip()
    return cleaned_text

def count_non_english_chars(text):
    if not text: return 0
    return sum(1 for char in text if ord(char) > 127)

def extract_address_components(text):
    """Extracts individual address components from the document text using regex."""
    address_components = {}
    for component, pattern in COMPONENT_PATTERNS.items():
        match = re.search(pattern, text, re.IGNORECASE)
        if match:
            # For patterns with multiple capture groups (like street, village, etc)
            captured_value = next((group for group in match.groups() if group), None)
            if captured_value:
                address_components[component] = captured_value.strip()

    return address_components

def construct_address(address_components):
    """Combines extracted address components into a standardized address string."""
    address_parts = []
    door_survey = []
    if "door_no" in address_components:
        door_survey.append(f"Door No. {address_components['door_no']}")
    if "survey_no" in address_components:
        door_survey.append(f"Survey No. {address_components['survey_no']}")
    if door_survey:
        address_parts.append("/".join(door_survey))
    if "street" in address_components:
        address_parts.append(address_components['street'])
    if "district" in address_components:
        address_parts.append(address_components['district'])
    if "state" in address_components:
        address_parts.append(address_components['state'])
    return ", ".join(address_parts) if address_parts else "Not Found"


# --- Document Processing Functions ---
def process_pdf_file(file_bytes, filename, poppler_path_config, ocr_langs):
    # Set OCR engine to Doctr permanently
    ocr_engine = "Doctr"
    print(f"--- Starting PDF Pre-processing for: {filename} (OCR Engine: {ocr_engine}) ---")
    final_english_text = ""
    original_untranslated_text = ""
    processing_notes = []
    detected_language_info = {"code": "unknown", "name": "Unknown", "source": "unknown"}
    embedded_html_tables = "<!-- No tables attempted or extracted yet -->"
    all_tables_dfs_with_pages = []
    plain_text_tables_for_llm = []
    processing_error = None
    processing_warning = None
    ensure_dir(OUTPUT_DIR)
    temp_pdf_path = Path(OUTPUT_DIR) / f"temp_{Path(filename).stem}_{datetime.now().strftime('%Y%m%d%H%M%S%f')}.pdf"
    try:
        with open(temp_pdf_path, "wb") as temp_file: temp_file.write(file_bytes)
        print(f"  Saved temporary PDF to '{temp_pdf_path.name}'")
    except Exception as e:
        print(f"Failed to save temporary PDF for {filename}: {e}")
        return {"error": f"Failed to save temp PDF: {e}", "translation": "[Error Saving Temp PDF]", "original_text": "", "notes": [f"Failed to save temp PDF: {e}"], "detected_language": {"code":"error", "name":"Error", "source":"system"}, "tables_dfs": [], "embedded_html_tables": "<!-- File Save Error -->", "plain_text_tables": []}
    try:
        extracted_text_pypdf = extract_text_from_pdf(temp_pdf_path)
        if extracted_text_pypdf and not extracted_text_pypdf.startswith("["):
            original_untranslated_text = extracted_text_pypdf
            processing_notes.append("Used parsable text from PDF (PyPDF2).")
            detected_language_info["source"] = "direct_pdf_text"
            if original_untranslated_text:
                safe_stem_direct = "".join(c if c.isalnum() or c in ('_', '-') else '_' for c in Path(filename).stem)
                direct_untranslated_save_filename = Path(OUTPUT_DIR) / f"untranslated_direct_pdf_{safe_stem_direct}.txt"
                try:
                    with open(direct_untranslated_save_filename, "w", encoding="utf-8") as f_direct:
                        f_direct.write(f"# Untranslated Direct Text (PyPDF2) from: {filename} @ {datetime.now().isoformat()}\n# ---\n\n{original_untranslated_text}")
                    note_msg = f"Saved untranslated direct PDF text to '{direct_untranslated_save_filename.name}'."
                    processing_notes.append(note_msg); print(f"  {note_msg}")
                except Exception as direct_save_e:
                    err_msg = f"Untranslated direct PDF text save error for '{filename}': {direct_save_e}"
                    processing_notes.append(err_msg); print(f"  ERROR: {err_msg}")
            final_english_text = original_untranslated_text
            non_english_char_count = count_non_english_chars(original_untranslated_text)
            if non_english_char_count > NON_ENGLISH_CHAR_THRESHOLD:
                try:
                    lang_info = translator.identify_language(original_untranslated_text)
                    if not isinstance(lang_info, dict) or 'code' not in lang_info: lang_info = {"code": "en", "name": "English (Defaulted on ID error)"}
                    detected_language_info.update(lang_info)
                    if lang_info.get("code", "en") != "en":
                        print(f"  Translating direct PDF text from {lang_info.get('name','Unknown Language')}...")
                        translated_text = translator.translate_to_english(original_untranslated_text, lang_info)
                        final_english_text = translated_text if translated_text and translated_text.strip() else original_untranslated_text
                        processing_notes.append(f"Translated direct PDF text from {lang_info.get('name','Unknown Language')}.")
                    else: processing_notes.append("Direct PDF text identified as English.")
                except Exception as lang_e:
                    print(f"  Language ID/Translation error for direct PDF text: {lang_e}. Using untranslated text.")
                    processing_notes.append(f"Language ID/Translation error (direct PDF text): {lang_e}")
                    detected_language_info.update({"code":"error", "name": f"Error ({lang_e})"})
            else:
                detected_language_info.update({"code": "en", "name": "English (assumed from direct PDF text)"})
                processing_notes.append(f"Assumed English for direct PDF text (low non-ASCII: {non_english_char_count}).")
        elif extracted_text_pypdf and (extracted_text_pypdf.startswith("[PDF Encrypted") or extracted_text_pypdf.startswith("[PyPDF2 Import Error]")):
            processing_error = extracted_text_pypdf[1:-1]
            original_untranslated_text = final_english_text = extracted_text_pypdf
            processing_notes.append(processing_error)
            detected_language_info["source"] = "direct_pdf_error"
        else: # Path 2: PyPDF2 failed or no text, proceed to OCR
            processing_notes.append(f"No parsable text from PyPDF2 or error, proceeding to OCR with '{ocr_engine}'.")
            original_untranslated_text = f"[OCR Attempted with {ocr_engine} as PyPDF2 Failed]"
            images = pdf_bytes_to_images(file_bytes, filename, poppler_path_config)
            if images is None:
                processing_error = "PDF to Image conversion failed."
                final_english_text = original_untranslated_text = "[PDF Conversion Error]"
                detected_language_info["source"] = "ocr_conversion_error"
            elif not images:
                processing_warning = "PDF converted to 0 images (empty PDF?)."
                final_english_text = original_untranslated_text = "[Empty PDF - No Pages for OCR]"
                detected_language_info["source"] = "ocr_empty_pdf"
            else: # OCR proceeds
                if not DOCTR_AVAILABLE or not DOCTR_PREDICTOR:
                     processing_error = "Doctr OCR Engine is not available or failed to load. Cannot process scanned PDF."
                     final_english_text = original_untranslated_text = "[OCR Engine Not Available]"
                     detected_language_info["source"] = "ocr_engine_error"
                else:
                    processing_notes.append(f"Attempting OCR on {len(images)} page(s) using '{ocr_engine}' engine.")
                    full_original_ocr_text_list = []
                    full_final_english_ocr_text_list = []
                    page_level_lang_info = {}
                    detected_language_info["source"] = "ocr"

                    for i, img in enumerate(images):
                        page_num = i + 1
                        page_ocr_untranslated_text, page_debug_logs = ocr_image(
                            img, page_num, filename
                        )
                        if page_debug_logs:
                            processing_notes.extend(page_debug_logs)
                        full_original_ocr_text_list.append(f"--- Page {page_num} ---\n{page_ocr_untranslated_text}")
                        page_final_english_segment = page_ocr_untranslated_text
                        if page_ocr_untranslated_text and not page_ocr_untranslated_text.startswith("["):
                            non_english_char_count_ocr = count_non_english_chars(page_ocr_untranslated_text)
                            current_page_lang_info = {"code": "en", "name": "English (assumed from OCR)"}
                            if non_english_char_count_ocr > NON_ENGLISH_CHAR_THRESHOLD:
                                try:
                                    lang_info_ocr_page = translator.identify_language(page_ocr_untranslated_text)
                                    if not isinstance(lang_info_ocr_page, dict) or 'code' not in lang_info_ocr_page: lang_info_ocr_page = {"code": "en", "name": "English (Defaulted on page ID error)"}
                                    current_page_lang_info = lang_info_ocr_page
                                    if lang_info_ocr_page.get("code", "en") != "en":
                                        print(f"  Translating OCR text from page {page_num} ({lang_info_ocr_page.get('name','Unknown')})...")
                                        page_translated = translator.translate_to_english(page_ocr_untranslated_text, lang_info_ocr_page)
                                        page_final_english_segment = page_translated if page_translated and page_translated.strip() else page_ocr_untranslated_text
                                        processing_notes.append(f"Translated OCR text from page {page_num} ({lang_info_ocr_page.get('name','Unknown')}).")
                                    else: processing_notes.append(f"OCR text on page {page_num} identified as English.")
                                except Exception as page_lang_e:
                                    print(f"  Lang ID/Trans error for OCR page {page_num}: {page_lang_e}. Using untranslated OCR.")
                                    processing_notes.append(f"Lang ID/Translation error (OCR page {page_num}): {page_lang_e}")
                            else: processing_notes.append(f"Assumed English OCR text on page {page_num} (low non-ASCII: {non_english_char_count_ocr}).")
                            page_level_lang_info[page_num] = current_page_lang_info
                        full_final_english_ocr_text_list.append(f"--- Page {page_num} ---\n{page_final_english_segment}")
                        try: img.close()
                        except: pass
                    original_untranslated_text = "\n\n".join(full_original_ocr_text_list)
                    final_english_text = "\n\n".join(full_final_english_ocr_text_list)
                    if page_level_lang_info:
                        first_page_with_lang = next((lang for p, lang in sorted(page_level_lang_info.items()) if lang.get("code") != "en" or lang.get("name") != "English (assumed from OCR)"), None)
                        if first_page_with_lang: detected_language_info.update(first_page_with_lang)
                        elif page_level_lang_info.get(1): detected_language_info.update(page_level_lang_info[1])
                    if original_untranslated_text and not original_untranslated_text.strip().startswith("["):
                        safe_stem_ocr = "".join(c if c.isalnum() or c in ('_', '-') else '_' for c in Path(filename).stem)
                        ocr_untranslated_save_filename = Path(OUTPUT_DIR) / f"untranslated_full_doc_ocr_{safe_stem_ocr}.txt"
                        try:
                            with open(ocr_untranslated_save_filename, "w", encoding="utf-8") as f_ocr:
                                f_ocr.write(f"# Untranslated Full Doc OCR from: {filename} @ {datetime.now().isoformat()}\n# OCR Langs: {ocr_langs}\n# OCR Engine: {ocr_engine}\n# ---\n\n{original_untranslated_text}")
                            note_msg = f"Saved untranslated full-doc raw OCR to '{ocr_untranslated_save_filename.name}'."
                            processing_notes.append(note_msg); print(f"  {note_msg}")
                        except Exception as ocr_save_e:
                            err_msg = f"Untranslated full-doc raw OCR save error for '{filename}': {ocr_save_e}"
                            processing_notes.append(err_msg); print(f"  ERROR: {err_msg}")
        # Note: Tabula runs independently of the OCR engine choice for now.
        html_tables_str, dfs_list, plain_tables_list = extract_tables_with_tabula(temp_pdf_path, filename)
        embedded_html_tables = html_tables_str
        all_tables_dfs_with_pages = dfs_list
        plain_text_tables_for_llm = plain_tables_list
        if plain_text_tables_for_llm: processing_notes.append(f"Tabula formatted {len(plain_text_tables_for_llm)} table(s).")
        else: processing_notes.append("No tables by Tabula or tables were empty.")
        if not processing_error and final_english_text and not final_english_text.startswith("["):
            cleaned_text = clean_processing_artifacts(final_english_text, filename_for_debug=filename)
            if cleaned_text != final_english_text: processing_notes.append("Applied artifact cleaning to final English text.")
            final_english_text = cleaned_text
        if not processing_error and final_english_text and not final_english_text.startswith("["):
            safe_stem = "".join(c if c.isalnum() or c in ('_', '-') else '_' for c in Path(filename).stem)
            processed_english_save_filename = Path(OUTPUT_DIR) / f"processed_english_for_llm_{safe_stem}.txt"
            try:
                with open(processed_english_save_filename, "w", encoding="utf-8") as f:
                    f.write(f"# Final Processed English Text (for LLM) from: {filename} @ {datetime.now().isoformat()}\n")
                    f.write(f"# Orig Text Source: {detected_language_info.get('source', 'unknown')}\n")
                    f.write(f"# Detected Lang: {detected_language_info.get('name','N/A')} ({detected_language_info.get('code','N/A')})\n")
                    f.write(f"# Tables (Plain Text): {len(plain_text_tables_for_llm)}\n# ---\n\n{final_english_text}\n\n")
                    if plain_text_tables_for_llm:
                        f.write("## Extracted Tables (Plain Text for LLM):\n")
                        for pt_table in plain_text_tables_for_llm: f.write(f"```text\n{pt_table}\n```\n\n")
                processing_notes.append(f"Saved final processed English text to {processed_english_save_filename.name}")
            except Exception as save_e: processing_notes.append(f"Save error (final English text): {save_e}")

        final_status = {"original_text": original_untranslated_text, "detected_language": detected_language_info, "translation": final_english_text, "tables_dfs": all_tables_dfs_with_pages, "embedded_html_tables": embedded_html_tables, "plain_text_tables": plain_text_tables_for_llm, "notes": processing_notes}
        if processing_error: final_status["error"] = processing_error
        if processing_warning: final_status["warning"] = processing_warning
        print(f"--- Finished PDF Pre-processing for: {filename} ---")
        return final_status
    except Exception as e:
        print(f"Critical PDF processing error '{filename}': {e}")
        detailed_tb = traceback.format_exc()
        print(f"Traceback: {detailed_tb}")
        return {"error": f"Critical PDF error: {e}", "translation": "[Critical PDF Error]", "original_text": f"[Critical PDF Error processing {filename}]", "notes": [f"Critical error: {e}", detailed_tb], "detected_language": {"code":"error", "name":"Error", "source":"system"}, "tables_dfs": [], "embedded_html_tables": "<!-- Critical Error -->", "plain_text_tables": []}
    finally:
        try:
            if temp_pdf_path.exists(): temp_pdf_path.unlink()
        except Exception as cleanup_error: print(f"Could not delete temp file '{temp_pdf_path.name}': {cleanup_error}")

def process_txt_file(file_bytes, filename):
    print(f"--- Starting TXT Pre-processing for: {filename} ---")
    original_text, final_english_text = "", ""
    notes, detected_language_info = [], {"code": "unknown", "name": "Unknown", "source": "unknown"}
    processing_error, processing_warning = None, None
    try:
        try: original_text = file_bytes.decode('utf-8'); notes.append("Decoded utf-8.")
        except UnicodeDecodeError:
            try: original_text = file_bytes.decode('latin-1'); notes.append("Decoded latin-1.")
            except UnicodeDecodeError as de:
                processing_error = f"Decode error: {de}"; print(f"Decode error for {filename}: {de}")
                return {"error": processing_error, "translation": "[Decode Error]", "original_text": "[Decode Error]", "notes": [processing_error], "detected_language": {"code":"error"}, "tables_dfs": [], "embedded_html_tables": "<!-- Decode Error -->", "plain_text_tables": []}
    except Exception as e:
        processing_error = f"Unexpected decode error: {e}"; print(f"Unexpected decode error for {filename}: {e}")
        return {"error": processing_error, "translation": "[Decode Error]", "original_text": "[Decode Error]", "notes": [processing_error], "detected_language": {"code":"error"}, "tables_dfs": [], "embedded_html_tables": "<!-- Decode Error -->", "plain_text_tables": []}
    if not original_text.strip():
        processing_warning = "TXT file empty/whitespace."; notes.append(processing_warning)
        final_english_text = "[File empty]"; print(f"TXT file '{filename}' is empty.")
    else:
        detected_language_info["source"] = "text_file"
        non_english_char_count = count_non_english_chars(original_text)
        if non_english_char_count > NON_ENGLISH_CHAR_THRESHOLD:
            try:
                lang_info = translator.identify_language(original_text)
                if not isinstance(lang_info, dict) or 'code' not in lang_info: lang_info = {"code": "en", "name": "English (Defaulted)"}
                detected_language_info.update(lang_info)
                if lang_info.get("code", "en") != "en":
                    print(f"  Translating TXT from {lang_info.get('name','Unknown')}...")
                    translated_text = translator.translate_to_english(original_text, lang_info)
                    final_english_text = translated_text if translated_text and translated_text.strip() else original_text
                    notes.append(f"Translated from {lang_info.get('name','Unknown')}.")
                else: final_english_text = original_text; notes.append("TXT identified as English.")
            except Exception as lang_e:
                print(f"  Lang ID/Trans error for TXT: {lang_e}. Using raw text.")
                final_english_text = original_text; notes.append(f"Lang ID/Trans error: {lang_e}")
        else:
                        final_english_text = original_text
                        detected_language_info.update({"code": "en", "name": "English (assumed)"})
                        notes.append(f"Assumed English (low non-ASCII: {non_english_char_count}).")
        if final_english_text and not final_english_text.startswith("["):
            cleaned_text = clean_processing_artifacts(final_english_text, filename_for_debug=filename)
        if cleaned_text != final_english_text: notes.append("Applied artifact cleaning.")
        final_english_text = cleaned_text
    if final_english_text and not final_english_text.startswith("["):
        ensure_dir(OUTPUT_DIR)
        safe_stem = "".join(c if c.isalnum() or c in ('_', '-') else '_' for c in Path(filename).stem)
        save_filename = Path(OUTPUT_DIR) / f"processed_english_for_llm_{safe_stem}.txt"
        try:
            with open(save_filename, "w", encoding="utf-8") as f:
                f.write(f"# Processed from: {filename} @ {datetime.now().isoformat()}\n# Lang ({detected_language_info.get('source', '?')}): {detected_language_info.get('name','N/A')} ({detected_language_info.get('code','N/A')})\n# ---\n\n{final_english_text}")
            notes.append(f"Saved to {save_filename.name}")
        except Exception as save_e: notes.append(f"Save error: {save_e}")

    final_result = {"original_text": original_text, "translation": final_english_text, "detected_language": detected_language_info, "notes": notes, "tables_dfs": [], "embedded_html_tables": "<!-- No tables for TXT -->", "plain_text_tables": []}
    if processing_error: final_result["error"] = processing_error
    if processing_warning: final_result["warning"] = processing_warning
    print(f"--- Finished TXT Pre-processing for: {filename} ---")
    return final_result

def call_ollama_api(prompt, ollama_url, model_name, step_name="Analysis"):
    """
    Calls the Ollama API. UI calls have been removed for a cleaner frontend.
    All feedback is now logged to the console.
    """
    api_url_to_call = ollama_url or os.getenv("OLLAMA_URL_ANALYSIS", DEFAULT_OLLAMA_URL_ANALYSIS)
    model_name_to_call = model_name or os.getenv("ANALYSIS_MODEL_NAME", DEFAULT_ANALYSIS_MODEL_NAME)

    # Increased context window to 25000 as requested
    payload = {"model": model_name_to_call, "prompt": prompt, "stream": False, "options": {"num_ctx": 25000, "temperature": 0.0, "repeat_penalty": 1.15, "top_k":30, "top_p":0.85, "seed": 42}}
    full_response_text = ""
    print(f"  Calling Ollama ({step_name}) API ({api_url_to_call}) model '{model_name_to_call}'...")
    try:
        response = requests.post(api_url_to_call, json=payload, timeout=700)
        response.raise_for_status()
        data = response.json()
        print(f"  Ollama ({step_name}) API call successful.")

        if 'response' in data and isinstance(data['response'], str):
            full_response_text = data['response']
        elif 'message' in data and isinstance(data.get('message'), dict) and 'content' in data['message']:
            full_response_text = data['message']['content']
        elif 'content' in data and isinstance(data['content'], str) and 'choices' not in data:
            full_response_text = data['content']
        elif isinstance(data, dict) and 'error' in data:
            print(f"Ollama API error ({step_name}): {data['error']}"); return None
        else:
            print(f"Unexpected Ollama response ({step_name}). Raw: {str(data)[:300]}...")
            possible_responses = [v for k, v in data.items() if isinstance(v, str) and len(v) > 20]
            if possible_responses:
                full_response_text = possible_responses[0]
                print("Used fallback response extraction.")
            else:
                print("Couldn't extract content.")
                return None
        return clean_processing_artifacts(full_response_text)
    except requests.exceptions.Timeout:
        print(f"Ollama request ({step_name}) timed out.")
        return None
    except requests.exceptions.RequestException as e:
        print(f"Ollama request error ({step_name}): {e}. Check server.")
        return None
    except json.JSONDecodeError as e:
        print(f"Ollama response JSON decode error ({step_name}): {e}. Raw: {response.text[:500]}...")
        return None
    except Exception as e:
        print(f"Unexpected error in Ollama call ({step_name}): {e}.")
        return None

# Update the convert_to_sqft function with more units and better parsing
def convert_to_sqft(measurement_str):
    """
    Parses a measurement string, converts it to square feet, and returns the
    converted value as a formatted string and the original string.
    """
    if not isinstance(measurement_str, str) or measurement_str.lower() in ["not found", "n/a", "", None]:
        return None, measurement_str

    # Conversion factors to square feet (expanded list)
    CONVERSION_FACTORS = {
        # Square meters
        'sq.m': 10.7639, 'sq m': 10.7639, 'sqm': 10.7639,
        'square meter': 10.7639, 'square meters': 10.7639, 'm²': 10.7639,

        # Acres
        'acre': 43560, 'acres': 43560,

        # Hectares
        'hectare': 107639, 'hectares': 107639, 'ha': 107639,

        # Cents
        'cent': 435.6, 'cents': 435.6,

        # Grounds
        'ground': 2400, 'grounds': 2400,

        # Ankanams
        'ankanam': 72, 'ankanams': 72,

        # Guntas
        'gunta': 1089, 'guntas': 1089,

        # Square feet (no conversion needed)
        'sq.ft': 1, 'sq ft': 1, 'sqft': 1,
        'square foot': 1, 'square feet': 1, 'sft': 1, 'ft²': 1,

        # Square yards
        'sq.yd': 9, 'sq yd': 9, 'sqyd': 9,
        'square yard': 9, 'square yards': 9
    }

    # Regex to find number and unit (handles commas, spaces, and various formats)
    measurement_str_cleaned = measurement_str.lower().replace(',', '')

    # Match numbers with decimals and fractions
    number_pattern = r'(\d+\.\d+|\d+\s+\d+/\d+|\d+/\d+|\d+)'

    # Match units (with optional 's' at the end)
    unit_pattern = '|'.join(re.escape(key) for key in CONVERSION_FACTORS.keys())

    # More robust regex pattern
    pattern = rf'{number_pattern}\s*({unit_pattern})s?\b'
    match = re.search(pattern, measurement_str_cleaned)

    if match:
        try:
            # Handle fractions in numbers
            number_str = match.group(1)
            if '/' in number_str:
                if ' ' in number_str:  # Mixed number (e.g., 1 1/2)
                    whole, fraction = number_str.split(' ')
                    num1, num2 = fraction.split('/')
                    value = float(whole) + (float(num1) / float(num2))
                else:  # Simple fraction (e.g., 1/2)
                    num1, num2 = number_str.split('/')
                    value = float(num1) / float(num2)
            else:
                value = float(number_str)

            unit = match.group(2).strip()
            if unit in CONVERSION_FACTORS:
                sqft_value = value * CONVERSION_FACTORS[unit]
                return f"{sqft_value:,.2f} sq. ft.", measurement_str
        except (ValueError, IndexError, ZeroDivisionError):
            # Failed to parse number or unit
            return None, measurement_str

    # If no unit is matched, return original
    return None, measurement_str


def create_single_doc_extraction_prompt_v2(filename, doc_text_content, doc_plain_tables):
    """Creates a heavily revised prompt for the LLM with detailed instructions for the 7 high-priority fields."""
    prompt = f"""You are an expert Legal Document Analyst AI. Your task is to perform a high-precision extraction from the document '{filename}'. You must generate a single JSON object as output.

**PRIMARY OBJECTIVE**: Focus all your analytical power on flawlessly extracting the following 7 high-priority sections. Accuracy and completeness for these sections are paramount.
1.  **PROPERTY ADDRESS**: The full, complete address of the primary property.
2.  **PROPERTY EXTENT DETAILS**: All measurements of the property.
3.  **BOUNDARIES**: The four cardinal boundaries (North, South, East, West).
4.  **TITLE HISTORY**: The chronological flow of ownership.
5.  **DOCUMENTS EXAMINED**: The list of all legal documents reviewed.
6.  **MORTGAGE DOCUMENTS**: Specific documents for pre and post-disbursal.
7.  **LEGAL OPINION**: A detailed, initial legal assessment based *only* on the text.



**DETAILED EXTRACTION INSTRUCTIONS**:

Extract the information into a single JSON object with the following 11 keys. Adhere strictly to the formats provided.

1.  **PROPERTY ADDRESS** (High Priority):
   - A Property address is a identifier to a property which usually contains Plot numbers, Survey Numbers and more. Look for property address with most Identification numbers. Do not take address which is in R/O that is usually associated with Owner name or their details. Look for actual Property details with expanded details.
   - First, check 'Extracted Tables (Plain Text)' for address information.
   - Then, check 'General Text Content (English)'. Look for "DESCRIPTION OF THE PROPERTY", "Property Information", "Address", "SCHEDULE","SCHEDULE: OF PROPERTY", "DETAILS OF PROPERTY".
   - Extract the most complete physical address with property identification numbers, a complete address will usually be in the format consider this as an Example only and the order might change PLOT Number, Some vernacular property identification Number, a street or nagar name ,a village or town name,a district name, a state name and pin code in 6 digits.
   - House number/Door number/Unit number look for terms: "House No.", "H.No.", "Door No.", "Unit No.", "Building No." .
   - Survey number/Plot number/Patta number look for terms: "Survey No.", "S.No.", "Plot No.", "Patta No.", "Khata No." .
   - Indian vernacular terms like "khasra number", "Khewat", "Khatauni", "Patta", "Jamabandi", "Fard" numbers look for the address context and extract any of the vernacular property identification numbers
   - Street, locality, city, and PIN code
   - Combine all these elements into a single address string.
   - If multiple addresses exist, choose the one that appears in the main property description section
   - Format: A single string with all combined elements separated by commas.

2.  **BOUNDARIES** (High Priority):
    *   Extract the literal description for North, South, East, and West boundaries.
    *   Capture exactly what is written, e.g., "North: Property of [name]", "South: [measurement] Public Road".
    *   If a boundary is missing, use "Not Found" for that specific direction.
    *   If more than one set of boundaries are present in the document extract them all as separate objects in the list.
    *   Format: [{{"north": "[description]", "south": "[description]", "east": "[description]", "west": "[description]"}}, ...]
3.  **OWNER NAME(S)**:
    *   Identify the name(s) of the current legal owner(s) of the property.
    *   List multiple owners separated by commas.
    *   Format: "[Name1], [Name2]"

4.  **PROPERTY EXTENT DETAILS** (High Priority):
    *   Extract all available measurements of the property.
    *   Capture the original text exactly, including units (e.g., "[number] acres", "[number] sq. ft.", "[number] cents", "[number] grounds").
    *   Format: {{"land_area": "e.g., [number] acres", "built_up_area": "e.g., [number] sq. ft.", "floor_wise_areas": "e.g., Ground Floor: [number] sq. ft., First Floor: [number] sq. ft."}}
    *   If a measurement is not mentioned, use "Not Found".
5.  **DOCUMENTS EXAMINED** (High Priority):
    *   Extract the entire information from the "Documents Examined", "Documents Scrutinized", "Documents Verified", "Documents Reviewed", or any similar section exactly as it appears in the document. This is critical: capture EVERY SINGLE DOCUMENT listed, including all details provided for each one, without summarizing, filtering, omitting duplicates, or altering any content—even if they are photocopies, originals, or repeated entries. Do not decide to omit or filter any documents; include absolutely everything listed in the section.
    *   Preserve the full description, including any dates, parties involved, document numbers, and other details verbatim.
    *   If a document has a number (e.g., Doc No [number]/[year]), include it in the document_name field as "Sale Deed Doc No [number]/[year]".
    *   If no specific dates or parties are mentioned for a document, use "Not Specified" for those fields.
    *   Format: [{{"document_name": "Sale Deed Doc No [number]/[year]", "creation_date": "DD-MM-YYYY", "execution_date": "DD-MM-YYYY", "executed_by": "[Issued by]", "in_favour_of": "[Receiver ]", "full_description": "[Verbatim full text/description of the document entry from the document]"}}]

6.  **TITLE HISTORY (CHRONOLOGICAL ORDER)** (High Priority):
    *   Trace the flow of ownership from the earliest document to the latest.
    *   Each entry in the list represents a transfer event (e.g., a sale, gift, partition).
    *   Critical : Extract all the transactions in title flow even if it points to multiple properites in the document but differentiate it.
    *   Every transcation revolving arround the property should be conisdered even if a small extent is sold it counts so extract all the transcation in title flow. Even draft settlement deed counts as a transcation.
    *   Ensure no transactions are skipped; include every mentioned transfer in the chain, with no data left out. Include EVERY single transaction mentioned in the title history section, without omission, even if repetitive or minor.
    *   Maintain the exact chronological order found in the document.
    *   Include the document number if mentioned (e.g., Doc No [number]/[year]) in the document_name field as "Sale Deed Doc No [number]/[year]".
    *   Format: [{{"document_name": "Sale Deed Doc No [number]/[year]", "document_date": "DD-MM-YYYY", "executed_by": "[Seller Name]", "in_favour_of": "[Buyer Name]", "nature_of_transfer": "Sale", "extent_owned": "[Area]"}}]


7.  **MORTGAGE DOCUMENTS** (High Priority):
    *   Identify and extract ALL documents required or suggested for mortgage creation, exactly as listed in the document. These are documents that the lawyer recommends the company to obtain from the customer before or after the loan process. Differentiate strictly between pre-disbursal and post-disbursal based on the document's context or section headings. Do not decide to omit or filter any documents; include absolutely everything listed in the sections.
    *   Pre-disbursal: Extract EVERY SINGLE DOCUMENT listed under sections like "Pre-Disbursal", "Prior to Disbursal", "Before Disbursal", "Documents Required Before Loan Disbursement", or similar phrases. Capture them exactly as mentioned, without skipping, summarizing, or altering any—include all details, descriptions, or notes provided.
    *   Post-disbursal: Extract EVERY SINGLE DOCUMENT listed under sections like "Post-Disbursal", "After Disbursal", "At the Time of Handling", "Documents to be Executed Post-Disbursement", or similar phrases. Capture them exactly as mentioned, without skipping, summarizing, or altering any—include all details, descriptions, or notes provided.
    *   If a document appears in both categories or is ambiguous, include it in both arrays with a note in the description. Ensure clear differentiation: do not merge or confuse the two lists.
    *   Format: {{"pre_disbursal": [{{"document_name": "[Exact Name as Listed]", "document_description": "[Full Verbatim Description/Notes from Document]" }}], "post_disbursal": [{{"document_name": "[Exact Name as Listed]", "document_description": "[Full Verbatim Description/Notes from Document]" }}]}}

8.  **LEGAL OPINION (INITIAL & DETAILED)** (High Priority):
    *   **This section must be VERY DESCRIPTIVE and based ONLY on the document's text.**
    *   Provide a thorough, multi-sentence justification for each verdict. Do not use single-word answers.
    *   Format: {{
        "title_clear": {{
            "verdict": "Clear / Not Clear / Clear with Conditions",
            "detailed_justification": "Provide a comprehensive explanation for the title verdict. Reference specific documents, ownership links, and any breaks or concerns in the chain of title. Explain WHY the title is considered clear or not."
        }},
        "encumbrances": {{
            "verdict": "None Found / Encumbrances Found",
            "detailed_justification": "If encumbrances are found, list each one with its nature (e.g., mortgage, lien), amount, and the document reference. If none, state that the Encumbrance Certificate was checked for a specific period and found to be nil."
        }},
        "mortgage_viability": {{
            "verdict": "Suitable / Not Suitable / Suitable with Conditions",
            "detailed_justification": "Elaborate in detail on why the property is or is not suitable for creating a mortgage. Consider title clarity, property type, legal status, and any other relevant factors mentioned in the text."
        }},
        "risk_level": "Low / Medium / High",
        "recommendations": "Extract the specific, actionable recommendations made by the lawyer in the document.",
        "conclusion": "Provide a detailed summary of the lawyer's overall conclusion on the property's legal standing.",
        "lawyers_opinion_as_per_document": "Extract the verbatim legal opinion paragraph(s) from the document as a multi-line string.",
        "summarized_opinion": "Provide a neat, to-the-point summary of the lawyer's opinion in 2-4 sentences, incorporating key keywords from the document (e.g., 'clear title', 'encumbrance free', 'suitable for mortgage'), and ensuring it aligns directly with the verbatim opinion."
    }}

9.  **PROPERTY NATURE**:
    *   Format: {{"type": "Residential", "ownership_type": "Single", "access_type": "Direct Public Access", "special_conditions": "None", "classification": "Building"}}

10. **POTENTIAL ISSUES (FLAGS)**:
    *   List any risks, discrepancies, or missing information you found.
    *   For each issue, include the related_section (one of: "property_address", "boundaries", "owner_names", "property_extent", "title_history", "documents_examined", "mortgage_documents", "legal_opinion", "property_nature").
    *   Format: [{{"issue_type": "Title Gap", "description": "Missing link document between [year1] and [year2].", "severity": "High", "evidence": "Quote from document or page number", "related_section": "title_history"}}]

11. **GUIDANCE QUESTIONS**:
    *   Based on your analysis, generate specific questions for each section to check against internal company guides, focusing on verification, compliance, and best practices.
    *   Format: {{"boundaries_questions": ["question1", "question2"], "title_history_questions": ["question1"], "property_nature_questions": ["question1"], "potential_issues_questions": ["question1"], "legal_opinion_questions": ["question1"]}}
    *   Generate at least one question per section, more if specific discrepancies or flags are found.

**SPECIFIC FORMATTING INSTRUCTIONS:**
1.  For **TITLE HISTORY**:
    *   Use "executed_by" instead of "from".
    *   Use "in_favour_of" instead of "to".
    *   Omit the "owner" field in the final JSON structure for this key.
    *   For transfer nature: Use exact terms like "Sale", "Will", "Court Decree", "Gift", "Partition".

2.  For **PROPERTY EXTENT DETAILS**:
    *   Clearly separate "land_area" and "built_up_area".
    *   Include original units in the extracted string.

3.  For **OWNER NAME(S)**:
    *   Extract ONLY from sections explicitly labeled "Owner", "Title Owner".
    *   If a dedicated owner section is not found, return "Not Found".

Output only the single, complete JSON object. Do not add any text before or after the JSON.
---
**DOCUMENT CONTENT for '{filename}'**:
{doc_text_content}
---
"""
    return prompt


def parse_single_doc_extraction_response_v2(response_text, filename, doc_text_content=""):
    """Parses LLM response and enhances address extraction with regex fallback."""
    try:
        # Extract JSON from response
        json_match = re.search(r'```json\s*({.*?})\s*```', response_text, re.DOTALL)
        if not json_match:
            # Fallback if no code block is found, assume the entire response is JSON
            if response_text.strip().startswith('{') and response_text.strip().endswith('}'):
                json_str = response_text.strip()
            else:
                return {"error": "No JSON object found in LLM response"}
        else:
            json_str = json_match.group(1)

        data = json.loads(json_str)

        key_mapping = {
            "PROPERTY ADDRESS": "property_address",
            "BOUNDARIES": "boundaries",
            "OWNER NAME(S)": "owner_names",
            "PROPERTY EXTENT DETAILS": "property_extent",
            "TITLE HISTORY": "title_history",
            "DOCUMENTS EXAMINED": "documents_examined",
            "MORTGAGE DOCUMENTS": "mortgage_documents",
            "LEGAL OPINION": "legal_opinion",
            "PROPERTY NATURE": "property_nature",
            "POTENTIAL ISSUES (FLAGS)": "potential_issues",
            "GUIDANCE QUESTIONS": "guidance_questions" # Updated key
        }

        standardized_data = {}

        # Lowercase all keys in the parsed data for case-insensitive matching
        data_lower = {k.lower().replace(" ", "_").replace("(", "").replace(")", "").replace("-", "_"): v for k, v in data.items()}

        for aiprompt_key, python_key in key_mapping.items():
            # Handle the old and new key for questions for backward compatibility
            lookup_key = aiprompt_key.lower().replace(" ", "_").replace("(", "").replace(")", "").replace("-", "_")
            if lookup_key in data_lower:
                standardized_data[python_key] = data_lower[lookup_key]
            # Specific fallback check for the old "regulatory_questions" key name
            elif python_key == "guidance_questions" and "regulatory_questions" in data_lower:
                 standardized_data[python_key] = data_lower["regulatory_questions"]
            else:
                standardized_data[python_key] = "Not Found"


        # Enhanced address extraction fallback
        if not standardized_data.get("property_address") or str(standardized_data.get("property_address", "")).lower() in ["not found", "n/a"]:
            # First, try the user's specific pattern for explicit full address
            full_address = extract_full_address_line(doc_text_content)
            if full_address:
                 standardized_data["property_address"] = full_address  # Extract as is
            else:
                # If that fails, try the component-based approach and format as specified
                address_components = extract_address_components(doc_text_content)
                if address_components:
                    constructed_address = construct_address(address_components)
                    standardized_data["property_address"] = constructed_address
                else:
                    # Final fallback to paragraph with most keywords
                    paragraphs = doc_text_content.split('\n\n')
                    best_paragraph = None
                    max_keywords = 0
                    for para in paragraphs:
                        para_lower = para.lower()
                        keyword_count = sum(1 for keyword in ADDRESS_KEYWORDS if keyword in para_lower)
                        if keyword_count > max_keywords:
                            max_keywords = keyword_count
                            best_paragraph = para
                    if best_paragraph:
                        standardized_data["property_address"] = best_paragraph.strip()
                    else:
                        standardized_data["property_address"] = "Not Found"


        # (New) Add owner name extraction fallback from document text
        if not standardized_data.get("owner_names") or standardized_data.get("owner_names") == "Not Found":
            owner_names = extract_owner_names(doc_text_content)
            if owner_names:
                standardized_data["owner_names"] = owner_names

        # (New) Process property extent with fallback
        if not standardized_data.get("property_extent") or standardized_data.get("property_extent") == "Not Found":
            standardized_data["property_extent"] = extract_property_extent(doc_text_content)


        # Validate mortgage document structure
        mortgage_docs = standardized_data.get("mortgage_documents", {})
        if isinstance(mortgage_docs, dict):
            if not isinstance(mortgage_docs.get("pre_disbursal"), list):
                mortgage_docs["pre_disbursal"] = []
            if not isinstance(mortgage_docs.get("post_disbursal"), list):
                mortgage_docs["post_disbursal"] = []
            standardized_data["mortgage_documents"] = mortgage_docs
        else:
            # If the format is wrong, reset it to the expected structure
            standardized_data["mortgage_documents"] = {"pre_disbursal": [], "post_disbursal": []}

        # Standardize and convert property extent details
        if "property_extent" in standardized_data and isinstance(standardized_data["property_extent"], dict):
            extent = standardized_data["property_extent"]

            # Land Area
            land_area_orig = extent.get("land_area", "Not Found")
            converted_land_area, _ = convert_to_sqft(land_area_orig)
            if converted_land_area:
                extent["land_area_sqft"] = converted_land_area

            # Built-up Area
            built_up_area_orig = extent.get("built_up_area", "Not Found")
            converted_built_up_area, _ = convert_to_sqft(built_up_area_orig)
            if converted_built_up_area:
                extent["built_up_area_sqft"] = converted_built_up_area

        # Enhanced property classification
        if "property_nature" in standardized_data and isinstance(standardized_data["property_nature"], dict):
            nature = standardized_data["property_nature"]
            boundaries_list = standardized_data.get("boundaries", [])

            # Ensure potential_issues is a list
            if "potential_issues" not in standardized_data or not isinstance(standardized_data["potential_issues"], list):
                standardized_data["potential_issues"] = []

            # Per-set landlocked check
            landlocked_sets = []
            has_access = False
            for idx, boundary in enumerate(boundaries_list):
                boundary_text = ' '.join([str(v).lower() for v in boundary.values()])
                access_present = any(kw in boundary_text for kw in ACCESS_KEYWORDS)
                if access_present:
                    has_access = True
                if not access_present:
                    landlocked_sets.append(idx + 1)
                    landlocked_issue = {
                        "issue_type": "Landlocked Property",
                        "description": f"S.No {idx + 1} is a landlocked property based on boundary descriptions.",
                        "severity": "High",
                        "evidence": "No access-related keywords found in this set of boundaries.",
                        "related_section": "boundaries"
                    }
                    # Check if already flagged to avoid duplicates
                    if not any(issue.get("description") == landlocked_issue["description"] for issue in standardized_data["potential_issues"]):
                        standardized_data["potential_issues"].append(landlocked_issue)

            # Determine overall access type
            if len(boundaries_list) > 0:
                if landlocked_sets == []:
                    nature["access_type"] = "Direct Public Access"
                elif len(landlocked_sets) == len(boundaries_list):
                    nature["access_type"] = "Landlocked"
                else:
                    nature["access_type"] = "Mixed Access"

            # Classify property type based on indicators in the full document text
            building_indicators = ["floor", "storey", "building", "constructed", "sq.ft", "sq ft", "built-up"]
            building_mentioned = any(indicator in doc_text_content.lower() for indicator in building_indicators)
            nature["classification"] = "Building" if building_mentioned else "Plot"

        # (New) Enhanced title history parsing
        title_history_raw = standardized_data.get("title_history")
        history_as_list = []

        if isinstance(title_history_raw, list):
            history_as_list = title_history_raw
        elif isinstance(title_history_raw, dict):
            # Convert dictionary of histories into a list
            history_as_list = [v for k, v in title_history_raw.items() if isinstance(v, dict)]

        if history_as_list:
            # Use the new processing function
            processed_history = process_title_history(history_as_list)
            # Add sequence numbers back in
            final_history = []
            for i, entry in enumerate(processed_history):
                final_history.append({"sequence": i + 1, **entry})
            standardized_data["title_history"] = final_history
        else:
            # If no history, ensure it's an empty list for consistency
            standardized_data["title_history"] = []

        # Enhanced legal opinion handling
        if "legal_opinion" in standardized_data and isinstance(standardized_data["legal_opinion"], dict):
            # Get verbatim opinion using the improved extractor
            opinion_from_doc = extract_legal_opinion(doc_text_content)
            
            # Preserve exact formatting and content
            standardized_data["legal_opinion"]["lawyers_opinion_as_per_document"] = opinion_from_doc

        # Ensure boundaries are parsed into a list of dicts for multiple sets
        boundaries_raw = standardized_data.get("boundaries", "Not Found")
        boundaries_list = []
        if isinstance(boundaries_raw, list):
            for b in boundaries_raw:
                if isinstance(b, dict):
                    boundary_dict = {k.lower(): v for k, v in b.items()}
                    boundaries_list.append(boundary_dict)
        elif isinstance(boundaries_raw, dict):
            boundary_dict = {k.lower(): v for k, v in boundaries_raw.items()}
            boundaries_list = [boundary_dict]
        elif isinstance(boundaries_raw, str) and boundaries_raw != "Not Found":
            # Parse string fallback for single set
            boundary_dict = {}
            parts = [p.strip() for p in boundaries_raw.split(',') if p.strip()]
            for part in parts:
                if ':' in part:
                    key, val = part.split(':', 1)
                    key_clean = key.strip().lower()
                    if key_clean in ['north', 'south', 'east', 'west']:
                        boundary_dict[key_clean] = val.strip()
            if boundary_dict:
                boundaries_list = [boundary_dict]
        if not boundaries_list:
            boundaries_list = [{"north": "Not Found", "south": "Not Found", "east": "Not Found", "west": "Not Found"}]
        standardized_data["boundaries"] = boundaries_list

        return standardized_data

    except Exception as e:
        return {"error": f"Parsing failed: {str(e)}", "raw_response": response_text}


def perform_rag_lookups_integrated(extracted_data):
    """
    Perform RAG lookups using the questions directly generated by the LLM
    and integrate the results back into the original data structure
    """
    if not RAG_AVAILABLE:
        print("RAG not available - FAISS index or related files not loaded")
        return extracted_data, []

    # Get questions from the extracted data (using the new key)
    guidance_questions = extracted_data.get("guidance_questions", {})
    if not guidance_questions or not isinstance(guidance_questions, dict):
        print("No guidance questions found in the extracted data")
        return extracted_data, []

    # Process each section's questions
    all_lookups = []
    enhanced_data = copy.deepcopy(extracted_data)

    # Create section mapping with fallbacks
    section_mapping = {
        "boundaries_questions": ["boundaries", "property_address", "potential_issues"],
        "title_history_questions": ["title_history", "documents_examined", "legal_opinion"],
        "property_nature_questions": ["property_nature", "property_extent", "legal_opinion"],
        "potential_issues_questions": ["potential_issues", "legal_opinion", "mortgage_documents"],
        "legal_opinion_questions": ["legal_opinion", "mortgage_documents", "potential_issues"]
    }

    for question_section, data_sections in section_mapping.items():
        questions = guidance_questions.get(question_section, [])
        if not questions or not isinstance(questions, list):
            continue

        # Perform FAISS lookups for each question
        section_lookups = []
        for question in questions:
            try:
                # Use the retrieve_top_k function from the imported module
                results = retrieve_top_k(question, top_k=3)  # Increased top_k for consistency

                if results:
                    # Select the best result with highest similarity
                    top_result = max(results, key=lambda x: x["similarity"])
                    if top_result["similarity"] < 0.5:  # Add threshold for relevance
                        continue
                    lookup = {
                        "query": question,
                        "answer": top_result["answer"],
                        "regulation_area": top_result["regulation_area"],
                        "issued_on": top_result["issued_on"],
                        "similarity": top_result["similarity"]
                    }
                    section_lookups.append(lookup)
                    all_lookups.append({
                        "section": data_sections[0], # Attribute to the primary section
                        "lookup": lookup
                    })
            except Exception as e:
                print(f"RAG lookup error: {str(e)}")

        # Add lookups to the enhanced data for all relevant sections
        if section_lookups:
            # Attach to ALL relevant sections (fallback through list)
            for data_section in data_sections:
                if data_section not in enhanced_data:
                    continue

                # Create consistent storage format
                if isinstance(enhanced_data[data_section], dict):
                    if "rag_lookups" not in enhanced_data[data_section]:
                        enhanced_data[data_section]["rag_lookups"] = []
                    enhanced_data[data_section]["rag_lookups"].extend(section_lookups)

    return enhanced_data, all_lookups

def create_final_grounded_prompt_integrated(enhanced_data, all_lookups):
    """
    Create a prompt for the final LLM pass that requests a concise, grounded opinion.
    """
    # Create a clean version of the data for the prompt
    prompt_data = copy.deepcopy(enhanced_data)

    # Remove the guidance_questions section as it's no longer needed
    if "guidance_questions" in prompt_data:
        del prompt_data["guidance_questions"]

    # Remove RAG lookups from the prompt data to avoid duplication, as they are provided separately.
    for section in prompt_data.values():
        if isinstance(section, dict) and "rag_lookups" in section:
            del section["rag_lookups"]

    # Format the data as JSON for the prompt
    formatted_data = json.dumps(prompt_data, indent=2)

    # Format the lookups in a readable way for the prompt
    formatted_lookups = "\n\n".join(
        f"Section: {lookup['section']}\n"
        f"Query: {lookup['lookup']['query']}\n"
        f"Guidance (from {lookup['lookup']['regulation_area']} issued on {lookup['lookup']['issued_on']}): {lookup['lookup']['answer']}"
        for lookup in all_lookups
    )

    # Create the prompt
    prompt = f"""You are a Chief Legal Officer with expertise in Loan Against Property company guides.
Review the following property document analysis and the associated guidance retrieved from the CIFCL knowledge base.

**DOCUMENT ANALYSIS SUMMARY:**
```json
{formatted_data}
```

**RELEVANT GUIDANCE:**
---
{formatted_lookups}
---

**YOUR TASK:**
Synthesize all the above information into a **brief, executive-level summary**. Your opinion must be grounded in the provided company guidance.

- **Final Recommendation:** State a clear, one-word recommendation (e.g., "Approve", "Approve with Conditions", "Reject").
- **Critical Issues:** Briefly mention the 1-2 most critical issues identified and explicitly link them to the guidance provided.
- **Brevity is Key:** Keep the entire opinion to **one or two short paragraphs**. Be direct and avoid lengthy explanations. The goal is a quick, actionable summary for a busy executive.

**FINAL GUIDANCE-GROUNDED OPINION (BRIEF SUMMARY):**
"""
    return prompt

def render_single_doc_view():
    st.header("Legal Document Analysis")

    if "single_doc_result_v2" not in st.session_state:
        st.session_state.single_doc_result_v2 = None

    if "enhanced_doc_result" not in st.session_state:
        st.session_state.enhanced_doc_result = None

    if "final_grounded_opinion" not in st.session_state:
        st.session_state.final_grounded_opinion = None

    if "rag_lookups" not in st.session_state:
        st.session_state.rag_lookups = []

    uploaded_file = st.file_uploader(
        "Upload legal document (PDF/TXT)",
        type=["pdf", "txt"],
        key="single_doc_uploader_v2"
    )

    if uploaded_file and st.button("🔎 Analyze Document", use_container_width=True):
        # Clear previous results
        st.session_state.single_doc_result_v2 = None
        st.session_state.enhanced_doc_result = None
        st.session_state.final_grounded_opinion = None
        st.session_state.rag_lookups = []

        with st.status("Processing...", expanded=True) as status:
            # Initial document processing
            status.update(label="Pre-processing document...")
            file_bytes = uploaded_file.getvalue()
            # OCR Engine is now hardcoded to Doctr
            processed_data = process_pdf_file(file_bytes, uploaded_file.name, POPPLER_PATH, LANGUAGES_FOR_OCR) if uploaded_file.type == "application/pdf" else process_txt_file(file_bytes, uploaded_file.name)

            if processed_data.get("error"):
                status.update(label="Processing failed", state="error")
                st.error(processed_data["error"])
                return

            # (New) Optimize context length before sending to LLM
            doc_text_content = optimize_context(processed_data["translation"])


            # First Ollama call: Extract data AND generate guidance questions
            status.update(label="Performing comprehensive analysis with AI...")
            # Use default values for Ollama URL and model name, removing dependency on session state from sidebar
            prompt = create_single_doc_extraction_prompt_v2(uploaded_file.name, doc_text_content, processed_data.get("plain_text_tables", []))
            llm_response = call_ollama_api(prompt, DEFAULT_OLLAMA_URL_ANALYSIS, DEFAULT_ANALYSIS_MODEL_NAME, "LegalAnalysis")
            with open("legal_analysis_output.txt", "w", encoding="utf-8") as file: file.write(llm_response)
 

            if not llm_response:
                status.update(label="AI analysis failed", state="error")
                st.session_state.single_doc_result_v2 = {"error": "The AI model failed to return a response."}
                return

            # Parse the extraction with guidance questions
            extracted_data = parse_single_doc_extraction_response_v2(llm_response, uploaded_file.name, processed_data["translation"]) # Pass full text for parsing fallbacks
            st.session_state.single_doc_result_v2 = extracted_data

            if "error" in extracted_data:
                status.update(label="Analysis parsing failed", state="error")
                return

            # Perform RAG lookups using the generated questions
            if RAG_AVAILABLE and st.session_state.get("enable_rag", True):
                status.update(label="Retrieving legislative clarifications from CIFCL guides and circulars...")
                try:
                    enhanced_data, all_lookups = perform_rag_lookups_integrated(extracted_data)
                    st.session_state.enhanced_doc_result = enhanced_data
                    st.session_state.rag_lookups = all_lookups

                    if all_lookups:
                        # Second Ollama call: Generate final grounded opinion
                        status.update(label="Generating final guidance-grounded legal opinion...")
                        final_prompt = create_final_grounded_prompt_integrated(enhanced_data, all_lookups)
                        final_opinion = call_ollama_api(final_prompt, DEFAULT_OLLAMA_URL_ANALYSIS,
                                                      DEFAULT_ANALYSIS_MODEL_NAME, "GroundedOpinion")

                        if final_opinion:
                            st.session_state.final_grounded_opinion = final_opinion
                except Exception as e:
                    print(f"Error during RAG enhancement: {e}")
                    traceback.print_exc()

            status.update(label="Analysis complete!", state="complete")
        st.rerun()

    # Display results
    if st.session_state.single_doc_result_v2:
        result = st.session_state.single_doc_result_v2
        enhanced_result = st.session_state.enhanced_doc_result or result
        rag_lookups = st.session_state.rag_lookups or []

        if "error" in result:
            st.error(f"Analysis error: {result['error']}")
            if "raw_response" in result:
                with st.expander("Show Raw AI Response"):
                    st.code(result["raw_response"])
            return

        st.subheader(f"Legal Analysis Report for: {uploaded_file.name}")

        # --- RESTRUCTURED UI SECTIONS ---

        # 1. Property Summary
        with st.expander("Property Summary", expanded=True):
            cols = st.columns(3)
            cols[0].write("**Address:**")
            cols[0].info(result.get("property_address", "Not Found"))
            cols[1].write("**Owner(s):**")
            cols[1].info(result.get("owner_names", "Not Found"))

            cols[2].write("**Property Extent:**")
            extent_data = result.get("property_extent", {})
            if isinstance(extent_data, dict):
                land_area_orig = extent_data.get("land_area", "Not Found")
                land_area_sqft = extent_data.get("land_area_sqft")
                built_up_area_orig = extent_data.get("built_up_area", "Not Found")
                built_up_area_sqft = extent_data.get("built_up_area_sqft")
                floor_wise_area = extent_data.get("floor_wise_areas", "Not Found")

                display_text_list = []
                if land_area_sqft:
                    display_text_list.append(f"**Land:** {land_area_orig} ({land_area_sqft})")
                else:
                    display_text_list.append(f"**Land:** {land_area_orig}")

                if built_up_area_sqft:
                    display_text_list.append(f"**Built-up:** {built_up_area_orig} ({built_up_area_sqft})")
                else:
                    if built_up_area_orig.lower() not in ["not found", ""]:
                        display_text_list.append(f"**Built-up:** {built_up_area_orig}")

                if floor_wise_area.lower() not in ["not found", ""]:
                    display_text_list.append(f"**Floor-wise:** {floor_wise_area}")

                cols[2].info("\n\n".join(display_text_list))
            else:
                cols[2].info(str(extent_data))

            st.write("**Boundaries:**")
            boundaries_data = result.get("boundaries", [])
            if isinstance(boundaries_data, list) and boundaries_data:
                # Display as table with multiple rows if multiple sets
                boundary_rows = []
                for i, b in enumerate(boundaries_data):
                    boundary_rows.append({
                        "S.No": i + 1,
                        "North": b.get("north", "Not Found"),
                        "South": b.get("south", "Not Found"),
                        "East": b.get("east", "Not Found"),
                        "West": b.get("west", "Not Found")
                    })
                boundary_df = pd.DataFrame(boundary_rows)
                st.dataframe(boundary_df, use_container_width=True, hide_index=True)
            else:
                st.info("Not Found")

            rag_content_boundaries = [l for l in rag_lookups if l.get("section") == "boundaries"]
            if rag_content_boundaries:
                st.markdown("### Guidance for Boundaries")
                for lookup in rag_content_boundaries:
                    st.markdown(f"**Question:** `{lookup['lookup']['query']}`")
                    st.markdown(f"**Guide:** {lookup['lookup']['regulation_area']} (Issued: {lookup['lookup']['issued_on']})")
                    st.info(f"**Guidance:** {lookup['lookup']['answer']}")
                    st.markdown("---")

            # Discrepancies for this section
            section_issues = [issue for issue in result.get("potential_issues", []) if isinstance(issue, dict) and issue.get("related_section") == "boundaries"]
            if section_issues:
                st.subheader("Discrepancies")
                df_discrepancies = pd.DataFrame(section_issues)
                st.dataframe(df_discrepancies, use_container_width=True, hide_index=True)
            else:
                st.success("No discrepancies found.")

        # 4. Documents Examined (Moved before Title History)
        with st.expander("Documents Examined", expanded=False):
            docs_examined = result.get("documents_examined", [])

            # No filtering, no deduplication — show all documents exactly as extracted
            if isinstance(docs_examined, list) and docs_examined:
                st.dataframe(pd.DataFrame(docs_examined), use_container_width=True, hide_index=True)
            else:
                st.warning("No examined document details found or format is incorrect.")
            # Discrepancies for this section
            section_issues = [issue for issue in result.get("potential_issues", []) if isinstance(issue, dict) and issue.get("related_section") == "documents_examined"]
            if section_issues:
                st.subheader("Discrepancies")
                df_discrepancies = pd.DataFrame(section_issues)
                st.dataframe(df_discrepancies, use_container_width=True, hide_index=True)
            else:
                st.success("No discrepancies found.")

        # 2. Title History (Updated)
        with st.expander("📜 Title History", expanded=True):
            title_history = result.get("title_history", [])
            if isinstance(title_history, list) and title_history:
                df_history = pd.DataFrame(title_history)
                # Updated column order as per request
                column_order = ['sequence', 'document_date', 'document_name','executed_by', 'in_favour_of', 'nature_of_transfer', 'extent_owned']
                df_history_ordered = df_history.reindex(columns=[col for col in column_order if col in df_history.columns])
                st.dataframe(df_history_ordered, use_container_width=True, hide_index=True)

                st.subheader("TITLE FLOW")
                for entry in title_history:
                    with st.container(border=True):
                        cols = st.columns([1, 4])
                        # Changed from st.metric to st.write for normal size
                        section_issues = [issue for issue in result.get("potential_issues", []) if isinstance(issue, dict) and issue.get("related_section") == "property_nature"]
                        cols[0].write(f"**#{entry.get('sequence', '')}**")
                        cols[1].write(f"**Date:** {entry.get('document_date', '')}")
                        cols[1].write(f"**Document:** {entry.get('document_name', '')}")
                        cols[1].write(f"**Executed By:** {entry.get('executed_by', '')}")
                        cols[1].write(f"**In Favour of:** {entry.get('in_favour_of', '')}")

                        # Owner field removed as requested

                rag_content_title = [l for l in rag_lookups if l["section"] == "title_history"]
                if rag_content_title:
                    st.markdown("### Guidance for Title History")
                    for lookup in rag_content_title:
                        st.markdown(f"**Question:** `{lookup['lookup']['query']}`")
                        st.markdown(f"**Guide:** {lookup['lookup']['regulation_area']} (Issued: {lookup['lookup']['issued_on']})")
                        st.info(f"**Guidance:** {lookup['lookup']['answer']}")
                        st.markdown("---")
            else:
                st.warning("No title history found or format is incorrect.")

            # Discrepancies for this section
            section_issues = [issue for issue in result.get("potential_issues", []) if isinstance(issue, dict) and issue.get("related_section") == "title_history"]
            if section_issues:
                st.subheader("Discrepancies")
                df_discrepancies = pd.DataFrame(section_issues)
                st.dataframe(df_discrepancies, use_container_width=True, hide_index=True)
            else:
                st.success("No discrepancies found.")

        # 3. Property Nature
        with st.expander("Property Nature", expanded=False):
            nature = result.get("property_nature", {})
            if isinstance(nature, dict):
                # Changed from st.metric to st.write/st.info for smaller, consistent font
                st.write("**Property Classification**")
                st.info(nature.get("classification", "N/A"))

                cols = st.columns(4)
                with cols[0]:
                    st.write("**Type**")
                    st.info(nature.get("type", "N/A"))
                with cols[1]:
                    st.write("**Ownership**")
                    st.info(nature.get("ownership_type", "N/A"))
                with cols[2]:
                    st.write("**Access**")
                    st.info(nature.get("access_type", "N/A"))
                with cols[3]:
                    st.write("**Conditions**")
                    st.info(nature.get("special_conditions", "N/A"))

                rag_content_nature = [l for l in rag_lookups if l["section"] == "property_nature"]
                if rag_content_nature:
                    st.markdown("### Guidance for Property Nature")
                    for lookup in rag_content_nature:
                        st.markdown(f"**Question:** `{lookup['lookup']['query']}`")
                        st.markdown(f"**Guide:** {lookup['lookup']['regulation_area']} (Issued: {lookup['lookup']['issued_on']})")
                        st.info(f"**Guidance:** {lookup['lookup']['answer']}")
                        st.markdown("---")
            else:
                st.warning("Property nature details not found or in an incorrect format.")

            # Discrepancies for this section
            section_issues = [issue for issue in result.get("potential_issues", []) if isinstance(issue, dict) and issue.get("related_section") == "property_nature"]
            if section_issues:
                st.subheader("Discrepancies")
                df_discrepancies = pd.DataFrame(section_issues)
                st.dataframe(df_discrepancies, use_container_width=True, hide_index=True)
            else:
                st.success("No discrepancies found.")

        # 5. Mortgage Documents
        with st.expander("Mortgage Documents", expanded=False):
            mortgage_docs = result.get("mortgage_documents", {})
            st.write("**Pre-Disbursal Documents**")
            pre_disbursal = mortgage_docs.get("pre_disbursal", [])
            if isinstance(pre_disbursal, list) and pre_disbursal:
                section_issues = [issue for issue in result.get("potential_issues", []) if isinstance(issue, dict) and issue.get("related_section") == "mortgage_documents"]
                df_pre = pd.DataFrame(pre_disbursal)
                # Ensure new column is present
                if "document_name" in df_pre.columns:
                    df_pre = df_pre.rename(columns={"document_name": "Document Name", "document_description": "Document Description"})
                    # Reorder columns
                    if "Document Description" in df_pre.columns:
                         df_pre = df_pre[["Document Name", "Document Description"]]
                st.dataframe(df_pre, use_container_width=True, hide_index=True)
            else: st.warning("No pre-disbursal documents found.")

            st.write("**Post-Disbursal Documents**")
            post_disbursal = mortgage_docs.get("post_disbursal", [])
            if isinstance(post_disbursal, list) and post_disbursal:
                df_post = pd.DataFrame(post_disbursal)
                if "document_name" in df_post.columns:
                    df_post = df_post.rename(columns={"document_name": "Document Name", "document_description": "Document Description"})
                    if "Document Description" in df_post.columns:
                        df_post = df_post[["Document Name", "Document Description"]]
                st.dataframe(df_post, use_container_width=True, hide_index=True)
            else: st.warning("No post-disbursal documents found.")

            # Discrepancies for this section
            section_issues = [issue for issue in result.get("potential_issues", []) if isinstance(issue, dict) and issue.get("related_section") == "mortgage_documents"]
            if section_issues:
                st.subheader("Discrepancies")
                df_discrepancies = pd.DataFrame(section_issues)
                st.dataframe(df_discrepancies, use_container_width=True, hide_index=True)
            else:
                st.success("No discrepancies found.")

        # 6. Initial Legal Opinion
        with st.expander("Initial Legal Opinion (Detailed View)", expanded=True):
            opinion = result.get("legal_opinion", {})
            if isinstance(opinion, dict):
                # The key "detailed_justification" is used as per the new prompt
                metrics = {
                    "title_clear": ("Title Status", "detailed_justification"),
                    "encumbrances": ("Encumbrances", "detailed_justification"),
                    "mortgage_viability": ("Mortgage Viability", "detailed_justification")
                }
                for key, (title, just_key) in metrics.items():
                    st.markdown(f"**{title}**")
                    metric_data = opinion.get(key, {})
                    if isinstance(metric_data, dict):
                        verdict = metric_data.get("verdict", "N/A")
                        justification = metric_data.get(just_key, "No detailed justification provided.")
                        col1, col2 = st.columns([1, 3])
                        with col1:
                            st.write("**Verdict**")
                            st.info(f"{verdict}")
                        with col2:
                            st.write("**Detailed Justification**")
                            st.info(justification)
                    else: st.warning(f"Data for '{title}' is not in the expected format.")
                    st.markdown("---")

                st.write("**Recommendations**")
                st.success(f'**Expert Recommendation:** {opinion.get("recommendations", "No recommendations provided")}')
                st.write("**Expert Conclusion**")
                st.info(f'**Expert Conclusion:** {opinion.get("conclusion", "No conclusion provided")}')

                if "summarized_opinion" in opinion:
                    st.write("**Summarized Lawyer's Opinion**")
                    st.info(f'{opinion.get("summarized_opinion", "No summarized opinion found")}')

                rag_content_opinion = [l for l in rag_lookups if l["section"] == "legal_opinion"]
                if rag_content_opinion:
                    st.markdown("### Guidance for Legal Opinion")
                    for lookup in rag_content_opinion:
                        st.markdown(f"**Question:** `{lookup['lookup']['query']}`")
                        st.markdown(f"**Guide:** {lookup['lookup']['regulation_area']} (Issued: {lookup['lookup']['issued_on']})")
                        st.info(f"**Guidance:** {lookup['lookup']['answer']}")
                        st.markdown("---")
            else: st.error("Legal opinion section is missing or in an incorrect format.")

            # Discrepancies for this section
            section_issues = [issue for issue in result.get("potential_issues", []) if isinstance(issue, dict) and issue.get("related_section") == "potential_issues"]
            if section_issues:
                st.subheader("Discrepancies")
                df_discrepancies = pd.DataFrame(section_issues)
                st.dataframe(df_discrepancies, use_container_width=True, hide_index=True)
            else:
                st.success("No discrepancies found.")

        # 7. Potential Issues (Flags)
        with st.expander("🚨 Potential Issues (Flags)", expanded=True):
            issues = result.get("potential_issues")
            if isinstance(issues, list) and issues:
                df_issues = pd.DataFrame(issues)
                def highlight_severity(s):
                    color = 'gray'
                    s_lower = str(s).lower()
                    if 'high' in s_lower: color = '#FF4B4B'
                    elif 'medium' in s_lower: color = '#FFC400'
                    elif 'low' in s_lower: color = '#22A762'
                    return f'color: {color}; font-weight: bold;'
                st.dataframe(df_issues.style.applymap(highlight_severity, subset=['severity']), use_container_width=True, hide_index=True)

                rag_content_issues = [l for l in rag_lookups if l["section"] == "potential_issues"]
                if rag_content_issues:
                    st.markdown("### Guidance for Potential Issues")
                    for lookup in rag_content_issues:
                        st.markdown(f"**Question:** `{lookup['lookup']['query']}`")
                        st.markdown(f"**Guide:** {lookup['lookup']['regulation_area']} (Issued: {lookup['lookup']['issued_on']})")
                        st.info(f"**Guidance:** {lookup['lookup']['answer']}")
                        st.markdown("---")
            else: st.success("✅ No potential issues flagged by the AI.")

        # 8. Final Grounded Opinion
        if st.session_state.final_grounded_opinion:
            with st.expander("🔍 Final Guidance-Grounded Opinion", expanded=True):
                st.markdown("### Final Legal Opinion (Grounded in CIFCL Guides)")
                st.markdown(st.session_state.final_grounded_opinion)

        # Download report
        st.markdown("---")
        report_data = enhanced_result if st.session_state.enhanced_doc_result else result
        if st.session_state.final_grounded_opinion:
            if isinstance(report_data, dict):
                report_data["final_grounded_opinion"] = st.session_state.final_grounded_opinion

        pdf_data = generate_legal_report_pdf(uploaded_file.name, report_data, rag_lookups)
        st.download_button("📥 Download Full Legal Report", pdf_data,
                          file_name=f"legal_analysis_{Path(uploaded_file.name).stem}.pdf",
                          mime="application/pdf", use_container_width=True)


def generate_legal_report_pdf(filename, analysis_data, rag_lookups=None):
    buffer = io.BytesIO()
    doc_template = SimpleDocTemplate(buffer, pagesize=letter, title=f"Legal Analysis: {filename}",
                            leftMargin=0.5*inch, rightMargin=0.5*inch,
                            topMargin=0.5*inch, bottomMargin=0.5*inch)

    styles = getSampleStyleSheet()
    styles.add(ParagraphStyle(name='Justify', alignment=4, fontSize=8, leading=9)) # Reduced font and leading for conciseness
    styles.add(ParagraphStyle(name='SmallNormal', parent=styles['Normal'], fontSize=7, leading=8)) # Smaller font
    styles.add(ParagraphStyle(name='Tiny', parent=styles['Normal'], fontSize=6, leading=7)) # Even smaller for long texts
    styles['Heading2'].fontSize = 10
    styles['Heading3'].fontSize = 9
    styles['Title'].fontSize = 12

    elements = []

    # Title
    elements.append(Paragraph(f"LEGAL ANALYSIS REPORT", styles['Title']))
    elements.append(Paragraph(f"<b>Document:</b> {filename}", styles['SmallNormal']))
    elements.append(Paragraph(f"<b>Generated:</b> {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}", styles['SmallNormal']))
    elements.append(Spacer(1, 6))  # Reduced spacer

    # Final Grounded Opinion (if available)
    if "final_grounded_opinion" in analysis_data and analysis_data["final_grounded_opinion"]:
        elements.append(Paragraph("Final Guidance-Grounded Opinion", styles['Heading2']))
        opinion_text = analysis_data["final_grounded_opinion"].replace('\n', '<br/>')
        elements.append(Paragraph(opinion_text, styles['Justify']))
        elements.append(Spacer(1, 6))

    # Summary
    elements.append(Paragraph("Property Summary", styles['Heading2']))

    extent_data = analysis_data.get("property_extent", {})
    extent_str = ""
    if isinstance(extent_data, dict):
        land_area_orig = extent_data.get("land_area", "Not Found")
        land_area_sqft = extent_data.get("land_area_sqft")
        built_up_area_orig = extent_data.get("built_up_area", "Not Found")
        built_up_area_sqft = extent_data.get("built_up_area_sqft")

        if land_area_sqft:
            extent_str += f"Land: {land_area_orig} ({land_area_sqft})<br/>"
        else:
            extent_str += f"Land: {land_area_orig}<br/>"

        if built_up_area_sqft:
            extent_str += f"Built-up: {built_up_area_orig} ({built_up_area_sqft})<br/>"
        elif built_up_area_orig.lower() not in ["not found", ""]:
            extent_str += f"Built-up: {built_up_area_orig}<br/>"
    else:
        extent_str = str(analysis_data.get("property_extent", "Not Found"))

    summary_data = [
        [Paragraph("<b>Address:</b>", styles['SmallNormal']), Paragraph(str(analysis_data.get("property_address", "Not Found")), styles['SmallNormal'])],
        [Paragraph("<b>Owners:</b>", styles['SmallNormal']), Paragraph(str(analysis_data.get("owner_names", "Not Found")), styles['SmallNormal'])],
        [Paragraph("<b>Property Extent:</b>", styles['SmallNormal']), Paragraph(extent_str, styles['SmallNormal'])],
    ]
    summary_table = Table(summary_data, colWidths=[1.5*inch, 5.5*inch])
    summary_table.setStyle(TableStyle([('VALIGN', (0,0), (-1,-1), 'TOP'), ('GRID', (0,0), (-1,-1), 0.5, colors.grey)]))
    elements.append(summary_table)

    # Boundaries as table
    boundaries_data = analysis_data.get("boundaries", [])
    if isinstance(boundaries_data, list) and boundaries_data:
        elements.append(Paragraph("Boundaries", styles['Heading3']))
        boundary_row = [
            Paragraph("<b>S.No</b>", styles['SmallNormal']),
            Paragraph("<b>North</b>", styles['SmallNormal']),
            Paragraph("<b>South</b>", styles['SmallNormal']),
            Paragraph("<b>East</b>", styles['SmallNormal']),
            Paragraph("<b>West</b>", styles['SmallNormal'])
        ]
        boundaries_table_data = [boundary_row]
        for i, b in enumerate(boundaries_data):
            boundary_descriptions = [
                Paragraph(str(i + 1), styles['SmallNormal']),
                Paragraph(str(b.get("north", "Not Found")), styles['SmallNormal']),
                Paragraph(str(b.get("south", "Not Found")), styles['SmallNormal']),
                Paragraph(str(b.get("east", "Not Found")), styles['SmallNormal']),
                Paragraph(str(b.get("west", "Not Found")), styles['SmallNormal'])
            ]
            boundaries_table_data.append(boundary_descriptions)
        boundaries_table = Table(boundaries_table_data, colWidths=[0.5*inch] + [1.75*inch] * 4, repeatRows=1)  # Reduced colWidths
        boundaries_table.setStyle(TableStyle([
            ('BACKGROUND', (0,0), (-1,0), colors.lightgrey),
            ('GRID', (0,0), (-1,-1), 0.5, colors.black),
            ('VALIGN', (0,0), (-1,-1), 'TOP')
        ]))
        elements.append(boundaries_table)

    # Add RAG lookups for boundaries
    boundary_lookups = [l for l in (rag_lookups or []) if l["section"] == "boundaries"]
    if boundary_lookups:
        elements.append(Spacer(1, 6))
        elements.append(Paragraph("Guidance Clarifications for Boundaries", styles['Heading3']))
        for lookup in boundary_lookups:
            elements.append(Paragraph(f"<b>Query:</b> {lookup['lookup']['query']}", styles['Tiny']))
            elements.append(Paragraph(f"<b>CIFCL Guide:</b> {lookup['lookup']['regulation_area']} (Issued: {lookup['lookup']['issued_on']})", styles['Tiny']))
            elements.append(Paragraph(f"<b>Answer:</b> {lookup['lookup']['answer']}", styles['Tiny']))
            elements.append(Spacer(1, 3))

    # Property Nature section
    if "property_nature" in analysis_data and isinstance(analysis_data.get("property_nature"), dict):
        elements.append(Spacer(1, 6))
        elements.append(Paragraph("Property Nature", styles['Heading2']))
        prop_nature = analysis_data["property_nature"]
        nature_data = [
            [Paragraph("<b>Classification:</b>", styles['SmallNormal']), Paragraph(str(prop_nature.get("classification", "Not Found")), styles['SmallNormal'])],
            [Paragraph("<b>Type:</b>", styles['SmallNormal']), Paragraph(str(prop_nature.get("type", "Not Found")), styles['SmallNormal'])],
            [Paragraph("<b>Ownership Type:</b>", styles['SmallNormal']), Paragraph(str(prop_nature.get("ownership_type", "Not Found")), styles['SmallNormal'])],
            [Paragraph("<b>Access Type:</b>", styles['SmallNormal']), Paragraph(str(prop_nature.get("access_type", "Not Specified")), styles['SmallNormal'])],
            [Paragraph("<b>Special Conditions:</b>", styles['SmallNormal']), Paragraph(str(prop_nature.get("special_conditions", "None")), styles['SmallNormal'])]
        ]
        nature_table = Table(nature_data, colWidths=[1.5*inch, 5.5*inch])
        nature_table.setStyle(TableStyle([('VALIGN', (0,0), (-1,-1), 'TOP'), ('GRID', (0,0), (-1,-1), 0.5, colors.grey)]))
        elements.append(nature_table)

        nature_lookups = [l for l in (rag_lookups or []) if l["section"] == "property_nature"]
        if nature_lookups:
            elements.append(Spacer(1, 6))
            elements.append(Paragraph("Guidance Clarifications for Property Nature", styles['Heading3']))
            for lookup in nature_lookups:
                elements.append(Paragraph(f"<b>Query:</b> {lookup['lookup']['query']}", styles['Tiny']))
                elements.append(Paragraph(f"<b>CIFCL Guide:</b> {lookup['lookup']['regulation_area']} (Issued: {lookup['lookup']['issued_on']})", styles['Tiny']))
                elements.append(Paragraph(f"<b>Answer:</b> {lookup['lookup']['answer']}", styles['Tiny']))
                elements.append(Spacer(1, 3))

    # Potential Issues section
    if "potential_issues" in analysis_data and isinstance(analysis_data.get("potential_issues"), list) and analysis_data["potential_issues"]:
        elements.append(Spacer(1, 6))
        elements.append(Paragraph("Potential Issues (Flags)", styles['Heading2']))
        issues_data = [[Paragraph("<b>Type</b>", styles['SmallNormal']), Paragraph("<b>Description</b>", styles['SmallNormal']), Paragraph("<b>Severity</b>", styles['SmallNormal']), Paragraph("<b>Related Section</b>", styles['SmallNormal'])]]
        for issue in analysis_data["potential_issues"]:
            row = [Paragraph(str(issue.get("issue_type", "")), styles['SmallNormal']), Paragraph(str(issue.get("description", "")), styles['SmallNormal']), Paragraph(str(issue.get("severity", "")), styles['SmallNormal']), Paragraph(str(issue.get("related_section", "")), styles['SmallNormal'])]
            issues_data.append(row)

        issues_table = Table(issues_data, colWidths=[1.0*inch, 4.0*inch, 0.75*inch, 1.25*inch], repeatRows=1)  # Reduced colWidths
        issues_table.setStyle(TableStyle([
            ('BACKGROUND', (0,0), (-1,0), colors.lightgrey), ('TEXTCOLOR', (0,0), (-1,0), colors.black),
            ('ALIGN', (0,0), (-1,-1), 'LEFT'), ('VALIGN', (0,0), (-1,-1), 'TOP'),
            ('FONTNAME', (0,0), (-1,0), 'Helvetica-Bold'), ('GRID', (0,0), (-1,-1), 0.5, colors.black),
            ('WORDWRAP', (0,0), (-1,-1), 'CJK')
        ]))
        elements.append(issues_table)

        issue_lookups = [l for l in (rag_lookups or []) if l["section"] == "potential_issues"]
        if issue_lookups:
            elements.append(Spacer(1, 6))
            elements.append(Paragraph("Guidance Clarifications for Potential Issues", styles['Heading3']))
            for lookup in issue_lookups:
                elements.append(Paragraph(f"<b>Query:</b> {lookup['lookup']['query']}", styles['Tiny']))
                elements.append(Paragraph(f"<b>CIFCL Guide:</b> {lookup['lookup']['regulation_area']} (Issued: {lookup['lookup']['issued_on']})", styles['Tiny']))
                elements.append(Paragraph(f"<b>Answer:</b> {lookup['lookup']['answer']}", styles['Tiny']))
                elements.append(Spacer(1, 3))

    # Documents Examined Section (Added)
    docs_examined = analysis_data.get("documents_examined", [])
    if isinstance(docs_examined, list) and docs_examined:
        elements.append(Spacer(1, 6))
        elements.append(Paragraph("Documents Examined", styles['Heading2']))
        examined_data = [[Paragraph("<b>Name</b>", styles['SmallNormal']), Paragraph("<b>Creation Date</b>", styles['SmallNormal']), Paragraph("<b>Execution Date</b>", styles['SmallNormal']), Paragraph("<b>Executed By</b>", styles['SmallNormal']), Paragraph("<b>In Favour Of</b>", styles['SmallNormal']), Paragraph("<b>Full Description</b>", styles['SmallNormal'])]]
        for doc in docs_examined:
            row = [
                Paragraph(str(doc.get("document_name", "N/A")), styles['SmallNormal']),
                Paragraph(str(doc.get("creation_date", "N/A")), styles['SmallNormal']),
                Paragraph(str(doc.get("execution_date", "N/A")), styles['SmallNormal']),
                Paragraph(str(doc.get("executed_by", "N/A")), styles['SmallNormal']),
                Paragraph(str(doc.get("in_favour_of", "N/A")), styles['SmallNormal']),
                Paragraph(str(doc.get("full_description", "N/A")), styles['Tiny'])  # Use Tiny for long descriptions
            ]
            examined_data.append(row)
        examined_table = Table(examined_data, colWidths=[1.5*inch, 1*inch, 1*inch, 1*inch, 1*inch, 2*inch], repeatRows=1)  # Adjusted widths for conciseness
        examined_table.setStyle(TableStyle([
            ('BACKGROUND', (0,0), (-1,0), colors.lightgrey), ('GRID', (0,0), (-1,-1), 0.5, colors.black),
            ('VALIGN', (0,0), (-1,-1), 'TOP'), ('FONTNAME', (0,0), (-1,0), 'Helvetica-Bold')
        ]))
        elements.append(examined_table)

    # Enhanced Title History section
    if isinstance(analysis_data.get("title_history"), list) and analysis_data["title_history"]:
        elements.append(Spacer(1, 6))
        elements.append(Paragraph("Title History", styles['Heading2']))
        history_data = [[
            Paragraph("<b>Seq</b>", styles['SmallNormal']), Paragraph("<b>Document</b>", styles['SmallNormal']),
            Paragraph("<b>Date</b>", styles['SmallNormal']), Paragraph("<b>Executed By</b>", styles['SmallNormal']), Paragraph("<b>In Favor Of</b>", styles['SmallNormal']),
            Paragraph("<b>Transfer Type</b>", styles['SmallNormal']), Paragraph("<b>Extent</b>", styles['SmallNormal'])
        ]]
        for entry in analysis_data["title_history"]:
            row = [
                Paragraph(str(entry.get("sequence", "")), styles['SmallNormal']),
                Paragraph(str(entry.get("document_name", "")), styles['SmallNormal']),
                Paragraph(str(entry.get("document_date", "")), styles['SmallNormal']),
                Paragraph(str(entry.get("executed_by", "")), styles['SmallNormal']), Paragraph(str(entry.get("in_favour_of", "")), styles['SmallNormal']),
                Paragraph(str(entry.get("nature_of_transfer", "")), styles['SmallNormal']), Paragraph(str(entry.get("extent_owned", "")), styles['SmallNormal'])
            ]
            history_data.append(row)
        history_table = Table(history_data, colWidths=[0.5*inch, 1.5*inch, 1*inch, 1.5*inch, 1.5*inch, 1*inch, 1*inch], repeatRows=1)  # Reduced colWidths
        history_table.setStyle(TableStyle([
            ('BACKGROUND', (0,0), (-1,0), colors.lightgrey), ('TEXTCOLOR', (0,0), (-1,0), colors.black),
            ('ALIGN', (0,0), (-1,-1), 'CENTER'), ('VALIGN', (0,0), (-1,-1), 'TOP'),
            ('FONTNAME', (0,0), (-1,0), 'Helvetica-Bold'), ('GRID', (0,0), (-1,-1), 0.5, colors.black),
            ('FONTSIZE', (0,0), (-1,-1), 7), ('WORDWRAP', (0,0), (-1,-1), 'CJK')
        ]))
        elements.append(history_table)

        title_lookups = [l for l in (rag_lookups or []) if l["section"] == "title_history"]
        if title_lookups:
            elements.append(Spacer(1, 6))
            elements.append(Paragraph("Guidance Clarifications for Title History", styles['Heading3']))
            for lookup in title_lookups:
                elements.append(Paragraph(f"<b>Query:</b> {lookup['lookup']['query']}", styles['Tiny']))
                elements.append(Paragraph(f"<b>CIFCL Guide:</b> {lookup['lookup']['regulation_area']} (Issued: {lookup['lookup']['issued_on']})", styles['Tiny']))
                elements.append(Paragraph(f"<b>Answer:</b> {lookup['lookup']['answer']}", styles['Tiny']))
                elements.append(Spacer(1, 3))

    # Enhanced Legal Opinion
    elements.append(Spacer(1, 6))
    opinion = analysis_data.get("legal_opinion", {})
    elements.append(Paragraph("Detailed Initial Legal Opinion", styles['Heading2']))
    metrics = {
        "title_clear": ("Title Status", "detailed_justification"),
        "encumbrances": ("Encumbrances", "detailed_justification"),
        "mortgage_viability": ("Mortgage Viability", "detailed_justification")
    }
    for key, (title, just_key) in metrics.items():
        metric_content = opinion.get(key, {})
        if isinstance(metric_content, dict):
            elements.append(Spacer(1, 6))
            elements.append(Paragraph(f"<b>{title}</b>", styles['Heading3']))
            verdict = metric_content.get("verdict", "N/A")
            justification = metric_content.get(just_key, "No detailed justification provided.")
            just_para = Paragraph(str(justification), styles['Tiny'])  # Use Tiny for justifications
            metric_data = [ [Paragraph("<b>Verdict:</b>", styles['SmallNormal']), Paragraph(f"{verdict}", styles['SmallNormal'])], [Paragraph("<b>Justification:</b>", styles['SmallNormal']), just_para] ]
            metric_table = Table(metric_data, colWidths=[1.5*inch, 5.5*inch])
            metric_table.setStyle(TableStyle([
                ('VALIGN', (0,0), (-1,-1), 'TOP'), ('GRID', (0,0), (-1,-1), 0.5, colors.lightgrey),
                ('BACKGROUND', (0,0), (0,-1), colors.whitesmoke)
            ]))
            elements.append(metric_table)

    elements.append(Spacer(1, 6))
    elements.append(Paragraph("<b>Recommendations</b>", styles['Heading3']))
    elements.append(Paragraph(str(opinion.get("recommendations", "No recommendations provided")), styles['SmallNormal']))
    elements.append(Spacer(1, 6))
    elements.append(Paragraph("<b>Expert Conclusion</b>", styles['Heading3']))
    elements.append(Paragraph(str(opinion.get("conclusion", "No conclusion provided")).replace('\n', '<br/>'), styles['Justify']))

    elements.append(Spacer(1, 6))
    elements.append(Paragraph("<b>Summarized Lawyer's Opinion</b>", styles['Heading3']))
    elements.append(Paragraph(str(opinion.get("summarized_opinion", "No summarized opinion found")).replace('\n', '<br/>'), styles['Justify']))

    # Enhanced Mortgage Documents Section for PDF
    mortgage_docs = analysis_data.get("mortgage_documents", {})
    pre_disbursal = mortgage_docs.get("pre_disbursal", [])
    post_disbursal = mortgage_docs.get("post_disbursal", [])
    if pre_disbursal or post_disbursal:
        elements.append(Spacer(1, 6))
        elements.append(Paragraph("Mortgage Documents", styles['Heading2']))
        if pre_disbursal:
            elements.append(Paragraph("Pre-Disbursal Documents", styles['Heading3']))
            pre_data = [[Paragraph("<b>Document Name</b>", styles['SmallNormal']), Paragraph("<b>Document Description</b>", styles['SmallNormal'])]]
            for mortgage_doc in pre_disbursal:
                pre_data.append([Paragraph(mortgage_doc.get("document_name", "N/A"), styles['SmallNormal']), Paragraph(mortgage_doc.get("document_description", "N/A"), styles['Tiny'])])  # Tiny for descriptions
            pre_table = Table(pre_data, colWidths=[2.0*inch, 5.0*inch], repeatRows=1)  # Reduced widths
            pre_table.setStyle(TableStyle([
                ('BACKGROUND', (0,0), (-1,0), colors.lightgrey), ('GRID', (0,0), (-1,-1), 0.5, colors.black),
                ('VALIGN', (0,0), (-1,-1), 'TOP'), ('FONTNAME', (0,0), (-1,0), 'Helvetica-Bold')
            ]))
            elements.append(pre_table)
            elements.append(Spacer(1, 3))
        if post_disbursal:
            elements.append(Paragraph("Post-Disbursal Documents", styles['Heading3']))
            post_data = [[Paragraph("<b>Document Name</b>", styles['SmallNormal']), Paragraph("<b>Document Description</b>", styles['SmallNormal'])]]
            for mortgage_doc in post_disbursal:
                post_data.append([Paragraph(mortgage_doc.get("document_name", "N/A"), styles['SmallNormal']), Paragraph(mortgage_doc.get("document_description", "N/A"), styles['Tiny'])])
            post_table = Table(post_data, colWidths=[2.0*inch, 5.0*inch], repeatRows=1)
            post_table.setStyle(TableStyle([
                ('BACKGROUND', (0,0), (-1,0), colors.lightgrey), ('GRID', (0,0), (-1,-1), 0.5, colors.black),
                ('VALIGN', (0,0), (-1,-1), 'TOP'), ('FONTNAME', (0,0), (-1,0), 'Helvetica-Bold')
            ]))
            elements.append(post_table)
    
    # Removed "Lawyer's Opinion (As Per Document)" from the PDF as requested

    opinion_lookups = [l for l in (rag_lookups or []) if l["section"] == "legal_opinion"]
    if opinion_lookups:
        elements.append(Spacer(1, 6))
        elements.append(Paragraph("Guidance Clarifications for Legal Opinion", styles['Heading3']))
        for lookup in opinion_lookups:
            elements.append(Paragraph(f"<b>Query:</b> {lookup['lookup']['query']}", styles['Tiny']))
            elements.append(Paragraph(f"<b>CIFCL Guide:</b> {lookup['lookup']['regulation_area']} (Issued: {lookup['lookup']['issued_on']})", styles['Tiny']))
            elements.append(Paragraph(f"<b>Answer:</b> {lookup['lookup']['answer']}", styles['Tiny']))
            elements.append(Spacer(1, 3))

    doc_template.build(elements)
    return buffer.getvalue()

def main():
    st.markdown("""
        <style>
            [data-testid="stHeader"] {visibility: hidden; height: 0%; position: fixed;}
            .st-emotion-cache-1avcm0n {height: 100%; display: flex; align-items: center; justify-content: center;}
        </style>""", unsafe_allow_html=True)

    # Sidebar has been removed. Configurations are now hardcoded or use defaults.

    logo_col, title_col = st.columns([1, 4])
    with logo_col:
        try: st.image('chola.png', width=200)
        except Exception: st.write(" ")

    # Directly render the single document view as it's the only mode
    render_single_doc_view()

    st.markdown("---")
    if st.button("🔄 Clear All State & Refresh Page"):
        keys_to_clear = list(st.session_state.keys())
        for key in keys_to_clear: del st.session_state[key]
        output_dir_path = Path(OUTPUT_DIR)
        if output_dir_path.exists():
            try:
                shutil.rmtree(output_dir_path)
                ensure_dir(OUTPUT_DIR)
                st.toast(f"Cleared output directory '{OUTPUT_DIR}'.")
            except Exception as e:
                st.warning(f"Could not clear '{OUTPUT_DIR}': {e}.")
        st.rerun()

if __name__ == "__main__":
    main()
```

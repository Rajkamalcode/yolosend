# -*- coding: utf-8 -*-
import json
import re
import requests
from typing import Dict, Any, Tuple, Optional

# --- Configuration Constants ---
# Ollama API endpoint (ensure Ollama is running and accessible)
OLLAMA_API = "http://localhost:11434/api/generate" 

# Models to use (MUST BE PULLED in Ollama: e.g., `ollama pull gemma3:4b`)
IDENTIFICATION_MODEL = "gemma3:4b" # Model specialized for quick identification tasks
TRANSLATION_MODEL = "gemma3:4b"    # Model for translation (gemma3:4b is often good enough and faster) 
                                   # Or use "gemma3:12b" if higher quality is needed and resources allow

# Temperature for LLM responses (lower = more deterministic)
TEMPERATURE_IDENTIFY = 0.0
TEMPERATURE_TRANSLATE = 0.2

# List of major Indian languages (ISO 639-1/639-2/639-3 codes)
# Keep this comprehensive for better identification prompts/parsing
INDIAN_LANGUAGES = {
    "en": "English", # Add English for cases where it might be input
    "hi": "Hindi",
    "bn": "Bengali",
    "te": "Telugu",
    "mr": "Marathi",
    "ta": "Tamil",
    "ur": "Urdu",
    "gu": "Gujarati",
    "kn": "Kannada",
    "ml": "Malayalam",
    "pa": "Punjabi",
    "or": "Odia",
    "as": "Assamese",
    "sa": "Sanskrit",
    "sd": "Sindhi",
    "ks": "Kashmiri",
    "ne": "Nepali",
    "doi": "Dogri",
    "kok": "Konkani",
    "mai": "Maithili",
    "bho": "Bhojpuri",
    "mni": "Manipuri",
    "sat": "Santali",
    "unknown": "Unknown" # Added for consistency
}

# --- Custom Exception ---
class OllamaError(Exception):
    """Custom exception for Ollama API errors."""
    pass

# --- Core Functions ---

def _query_ollama(
    prompt: str,
    model: str,
    system_prompt: str = "",
    temperature: float = 0.1
) -> str:
    """
    Internal function to send a query to Ollama API and handle errors.

    Args:
        prompt: The user prompt.
        model: The model to use.
        system_prompt: Optional system prompt.
        temperature: Control randomness.

    Returns:
        The response text from Ollama.

    Raises:
        OllamaError: If there's an error communicating with Ollama or parsing the response.
        requests.exceptions.RequestException: For network-level errors.
    """
    payload = {
        "model": model,
        "prompt": prompt,
        "stream": False,
        "temperature": temperature,
         "options": {"num_ctx": 58192} # Increased context window to reduce truncation
    }

    if system_prompt:
        payload["system"] = system_prompt

    try:
        response = requests.post(OLLAMA_API, json=payload, timeout=300) # 5 min timeout per query
        response.raise_for_status() # Raise HTTPError for bad responses (4xx or 5xx)

        result = response.json()

        # Handle different possible response structures from Ollama
        if 'response' in result and isinstance(result['response'], str):
            return result['response'].strip()
        elif 'message' in result and isinstance(result.get('message'), dict) and 'content' in result['message'] and isinstance(result['message']['content'], str):
            # Handle chat completion response format if used by the endpoint
             return result['message']['content'].strip()
        elif isinstance(result, dict) and 'error' in result:
             raise OllamaError(f"Ollama API returned an error: {result['error']}")
        else:
             # Fallback if response structure is unexpected
             raise OllamaError(f"Unexpected response structure from Ollama: {str(result)[:200]}...")

    except requests.exceptions.Timeout as e:
         raise OllamaError(f"Timeout connecting to Ollama API at {OLLAMA_API}: {e}")
    except requests.exceptions.ConnectionError as e:
         raise OllamaError(f"Connection refused connecting to Ollama API at {OLLAMA_API}. Is 'ollama serve' running? Error: {e}")
    except requests.exceptions.RequestException as e:
        # Re-raise other request exceptions directly or wrap them
        raise OllamaError(f"Ollama API request failed: {e}") # Wrap general request errors
    except json.JSONDecodeError as e:
        raise OllamaError(f"Failed to decode JSON response from Ollama: {e}. Response text: {response.text[:200]}...")
    except Exception as e:
        # Catch any other unexpected error during the API call
        raise OllamaError(f"An unexpected error occurred during Ollama query: {e}")


def identify_language(text: str) -> Dict[str, str]:
    """
    Identify the language of the input text using Ollama.

    Args:
        text: The input text (a snippet is usually sufficient).

    Returns:
        Dictionary {"code": language_code, "name": language_name}
        Returns {"code": "unknown", "name": "Unknown"} on failure or if language is not identifiable as Indian/English.

    Raises:
        OllamaError: If communication with Ollama fails.
    """
    # Use only a portion of the text for efficiency
    text_snippet = text[:1000] # Analyze first 1000 characters for better accuracy
    if not text_snippet.strip():
        return {"code": "none", "name": "No Text Provided"} # Handle empty input

    # Generate the list of language codes for the prompt dynamically
    lang_codes_list = ", ".join(INDIAN_LANGUAGES.keys())

    system_prompt = f"""You are a language identification expert specialized in Indian languages and English.
Analyze the provided text snippet and identify its primary language.
Respond ONLY with the ISO 639-1 (or appropriate ISO 639-x) code for the detected language.
Choose from this list: {lang_codes_list}.
If the language is not in the list or you are uncertain, respond with 'unknown'.
Do not add any explanation or surrounding text. Just the code."""

    prompt = f"Identify the language of this text: \"{text_snippet}\""

    try:
        response = _query_ollama(prompt, model=IDENTIFICATION_MODEL, system_prompt=system_prompt, temperature=TEMPERATURE_IDENTIFY)

        # Clean the response: extract potential code, handle variations
        # Look for a 2 or 3 letter code, possibly surrounded by quotes or whitespace
        match = re.search(r'\b([a-z]{2,3})\b', response.lower().strip())
        lang_code = match.group(1) if match else "unknown"

        # Validate against our list
        if lang_code not in INDIAN_LANGUAGES:
            lang_code = "unknown"

        lang_name = INDIAN_LANGUAGES.get(lang_code, "Unknown") # Default to "Unknown"

        return {"code": lang_code, "name": lang_name}

    except Exception as e:
        # If Ollama query itself fails, re-raise the error
        # The calling function (`process_input`) will catch this.
        raise OllamaError(f"Language identification query failed: {e}")


def count_non_english_words(text: str) -> int:
    """
    Count the number of likely non-English words in the text.
    
    Args:
        text: The input text to analyze.
        
    Returns:
        The count of likely non-English words.
    """
    if not text or not isinstance(text, str):
        return 0
        
    # Simple regex to split text into words
    words = re.findall(r'\b[^\s\d\W]+\b', text.lower())
    
    # Common English words and patterns to check against
    english_patterns = [
        # Common English word endings
        r'ing$', r'ed$', r'ly$', r'ment$', r'tion$', r'ness$', r'ity$',
        # Common English letter patterns
        r'^[a-z]+$',  # Only ASCII letters
        r'^th', r'the', r'and', r'that', r'have', r'for', r'not', r'with', r'this', r'but',
        r'from', r'they', r'say', r'will', r'one', r'all', r'would', r'there', r'their',
        r'what', r'out', r'about', r'who', r'get', r'which', r'when', r'make', r'can', r'like',
        r'time', r'just', r'know', r'take', r'people', r'year', r'your', r'good', r'some', r'could'
    ]
    
    non_english_count = 0
    
    for word in words:
        # Skip very short words (1-2 letters) as they're hard to classify
        if len(word) <= 2:
            continue
            
        # Check if the word matches any English pattern
        is_likely_english = any(re.search(pattern, word) for pattern in english_patterns)
        
        # Check for non-ASCII characters (strong indicator of non-English)
        has_non_ascii = any(ord(c) > 127 for c in word)
        
        # Count as non-English if it has non-ASCII chars or doesn't match English patterns
        if has_non_ascii or not is_likely_english:
            non_english_count += 1
            
    return non_english_count



def translate_to_english(text: str, source_lang_info: Dict[str, str]) -> str:
    """
    Translate the input text to English using Ollama.

    Args:
        text: The input text to translate.
        source_lang_info: Dictionary {"code": code, "name": name} of the source language.

    Returns:
        The translated text in English.

    Raises:
        OllamaError: If communication with Ollama fails.
    """
    if not text.strip():
        return "" # Return empty if input is empty

    source_lang_code = source_lang_info.get("code", "unknown")
    source_lang_name = source_lang_info.get("name", "the detected language")

    # Handle English input - no translation needed
    if source_lang_code == "en":
        return text

    # Handle unknown source language in the prompt
    if source_lang_code == "unknown":
        source_description = "the source language (which was not confidently identified)"
    else:
        source_description = source_lang_name

    system_prompt = f"""You are a professional translator. Translate the following text from {source_description} into natural, fluent English.
Focus on accuracy and preserving the original meaning.
If the input text is already in English, return it as is.
Respond ONLY with the English translation. Do not add any extra commentary, greetings, or explanations."""

    prompt = f"Translate this to English:\n\n{text}"

    try:
        # Using the configured TRANSLATION_MODEL
        translated_text = _query_ollama(prompt, model=TRANSLATION_MODEL, system_prompt=system_prompt, temperature=TEMPERATURE_TRANSLATE)
        return translated_text
    except Exception as e:
        # Re-raise errors for the calling function
        raise OllamaError(f"Translation query failed: {e}")


# --- Main Processing Function for Streamlit App ---

def process_input(text: str, force_translation: bool = False) -> Dict[str, Any]:
    """
    Processes input text: identifies language and translates to English.
    This is the main function intended to be called by the Streamlit app.

    Args:
        text: The input text (can be a full document or a page chunk).
        force_translation: If True, will translate even if language is detected as English

    Returns:
        A dictionary containing:
        - "original_text": The input text.
        - "detected_language": {"code": str, "name": str}
        - "translation": The translated English text, or the original if English, or an error message.
        - "error": None if successful, or an error message string if failed.
        - "note": Optional notes about the process (e.g., if language was unknown).
    """
    result = {
        "original_text": text,
        "detected_language": {"code": "unknown", "name": "Unknown"},
        "translation": None,
        "error": None,
        "note": None
    }

    if not text or not text.strip():
        result["error"] = "Input text is empty."
        result["translation"] = "[No text provided for processing]"
        result["detected_language"] = {"code": "none", "name": "No Text Provided"}
        return result

    try:
        # 1. Identify Language
        lang_info = identify_language(text)
        result["detected_language"] = lang_info

        # Add note if language identification was uncertain
        if lang_info["code"] == "unknown":
             result["note"] = "Source language could not be confidently identified. Translation attempted assuming source is non-English."

        # 2. Translate to English (skip if already English or identification failed badly)
        if lang_info["code"] == "en" and not force_translation:
            # Check if there are significant non-English words despite being detected as English
            non_english_count = count_non_english_words(text)
            if non_english_count > 50:  # Threshold for mixed language content
                result["note"] = f"Detected as English but contains {non_english_count} non-English words. Translating anyway."
                translated_text = translate_to_english(text, {"code": "unknown", "name": "Mixed Language"})
                result["translation"] = translated_text
            else:
                result["translation"] = text  # Return original text if English
                result["note"] = "Input text detected as English. No translation performed."
        elif lang_info["code"] == "none":  # Should be caught earlier, but defensive check
             result["translation"] = "[No text provided for translation]"
             result["error"] = "Cannot translate empty text."
        else:
            # Proceed with translation
            translated_text = translate_to_english(text, lang_info)
            result["translation"] = translated_text

    except OllamaError as e:
        # Catch errors specifically from Ollama interactions
        error_message = f"Ollama processing failed: {e}"
        result["error"] = error_message
        # Provide a placeholder in translation field indicating error
        result["translation"] = f"[Processing Error: {e}]"
        # Mark language as error state if it happened during identification
        if "Language identification" in str(e):
            result["detected_language"] = {"code": "error", "name": "Identification Failed"}
        else: # Assume translation failed if language ID succeeded
             # Keep the detected language, but mark translation as failed
             pass

    except Exception as e:
        # Catch any other unexpected errors during processing
        error_message = f"Unexpected error during processing: {e}"
        result["error"] = error_message
        result["translation"] = f"[Unexpected Error: {e}]"
        result["detected_language"] = {"code": "error", "name": "Processing Error"}

    # Final check for empty translation result
    if result["translation"] is None and result["error"] is None:
         result["translation"] = "[Translation process completed but yielded no result]"
         if result["note"] is None:
              result["note"] = "Translation yielded empty result."
         else:
              result["note"] += " Translation yielded empty result."

    return result


# --- Optional: Code to run only when script is executed directly (for testing) ---
if __name__ == "__main__":
    print("--- Running translation_with_ollama.py in standalone test mode ---")

    # Example usage for testing:
    test_text_hindi = "यह हिंदी में एक परीक्षण वाक्य है।"
    test_text_telugu = "ఇది తెలుగులో ఒక పరీక్ష వాక్యం."
    test_text_english = "This is a test sentence in English."
    test_text_empty = ""

    print("\nTesting Hindi:")
    result_hi = process_input(test_text_hindi)
    print(json.dumps(result_hi, indent=2, ensure_ascii=False))

    print("\nTesting Telugu:")
    result_te = process_input(test_text_telugu)
    print(json.dumps(result_te, indent=2, ensure_ascii=False))

    print("\nTesting English:")
    result_en = process_input(test_text_english)
    print(json.dumps(result_en, indent=2, ensure_ascii=False))

    print("\nTesting Empty:")
    result_empty = process_input(test_text_empty)
    print(json.dumps(result_empty, indent=2, ensure_ascii=False))

    print("\n--- Standalone test mode finished ---")

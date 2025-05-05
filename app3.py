import os
import numpy as np
from flask import Flask, request, jsonify, Response
from pdf2image import convert_from_path
import cv2
import tempfile
import multiprocessing
from multiprocessing import Pool
import json
from flask_cors import CORS
import pywt  # Make sure to install PyWavelets: pip install PyWavelets

app = Flask(__name__)
CORS(app)

# Configuration
UPLOAD_FOLDER = tempfile.gettempdir()
ALLOWED_EXTENSIONS = {'pdf'}
THRESHOLDS = {
    'noise_mad': 1.36,           # Updated from 4.0
    'noise_wavelet': 0.36,       # Updated from 2.0
    'noise_laplacian': 200.0,    # Updated from 50.0
    'edge_density': 0.022,       # Updated from 0.1
    'edge_strength': 36.0,       # Updated from 20.0
    'print_pattern_energy': 203.0, # Updated
    'print_pattern_variance': 614.0 # Updated from 100.0
}

app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER

def allowed_file(filename):
    return '.' in filename and filename.rsplit('.', 1)[1].lower() in ALLOWED_EXTENSIONS

def calculate_noise_features(gray):
    """Calculate multiple noise metrics from grayscale image."""
    try:
        # Median Absolute Deviation
        med = np.median(gray)
        noise_mad = np.median(np.abs(gray - med))
        
        # Wavelet-based noise estimation
        coeffs = pywt.wavedec(gray.ravel(), 'db1', level=3)
        noise_wavelet = np.median(np.abs(coeffs[-1])) / 0.6745
        
        # Laplacian-based noise estimation
        laplacian = cv2.Laplacian(gray, cv2.CV_64F)
        noise_laplacian = np.var(laplacian)
        
        return {
            'noise_mad': float(noise_mad) if not np.isnan(noise_mad) else 0.0,
            'noise_wavelet': float(noise_wavelet) if not np.isnan(noise_wavelet) else 0.0,
            'noise_laplacian': float(noise_laplacian) if not np.isnan(noise_laplacian) else 0.0
        }
    except Exception as e:
        print(f"Error calculating noise features: {e}")
        return {
            'noise_mad': 0.0,
            'noise_wavelet': 0.0,
            'noise_laplacian': 0.0
        }

def calculate_edge_features(gray):
    """Calculate edge-related features from grayscale image."""
    try:
        # Apply Canny edge detection
        edges = cv2.Canny(gray, 100, 200)
        
        # Calculate edge density
        edge_density = np.sum(edges > 0) / (edges.shape[0] * edges.shape[1])
        
        # Calculate edge strength
        gradient_x = cv2.Sobel(gray, cv2.CV_64F, 1, 0, ksize=3)
        gradient_y = cv2.Sobel(gray, cv2.CV_64F, 0, 1, ksize=3)
        gradient_magnitude = np.sqrt(gradient_x**2 + gradient_y**2)
        edge_strength = np.mean(gradient_magnitude)
        
        return {
            'edge_density': float(edge_density) if not np.isnan(edge_density) else 0.0,
            'edge_strength': float(edge_strength) if not np.isnan(edge_strength) else 0.0
        }
    except Exception as e:
        print(f"Error calculating edge features: {e}")
        return {
            'edge_density': 0.0,
            'edge_strength': 0.0
        }

def calculate_print_pattern_features(gray):
    """Calculate print pattern features using Fourier transform."""
    try:
        # Fourier transform to detect periodic patterns
        f_transform = np.fft.fft2(gray)
        f_shift = np.fft.fftshift(f_transform)
        magnitude_spectrum = 20 * np.log(np.abs(f_shift) + 1)
        
        # Analyze high-frequency components (printer patterns)
        h, w = gray.shape
        center_h, center_w = h//2, w//2
        high_freq_region = magnitude_spectrum[center_h-h//4:center_h+h//4, center_w-w//4:center_w+w//4]
        high_freq_energy = np.mean(high_freq_region)
        high_freq_variance = np.var(high_freq_region)
        
        return {
            'print_pattern_energy': float(high_freq_energy) if not np.isnan(high_freq_energy) else 0.0,
            'print_pattern_variance': float(high_freq_variance) if not np.isnan(high_freq_variance) else 0.0
        }
    except Exception as e:
        print(f"Error calculating print pattern features: {e}")
        return {
            'print_pattern_energy': 0.0,
            'print_pattern_variance': 0.0
        }

def calculate_metrics(image):
    """Calculate all metrics for an image."""
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    
    # Collect all features
    metrics = {}
    metrics.update(calculate_noise_features(gray))
    metrics.update(calculate_edge_features(gray))
    metrics.update(calculate_print_pattern_features(gray))
    
    return metrics

def classify_page(metrics):
    """Classify a page as original or photocopy based on weighted metrics."""
    # Define weights for each feature
    weights = {
        'noise_mad': 0.15,            # Reduced from 0.25
        'noise_wavelet': 0.25,        # Increased from 0.15
        'noise_laplacian': 0.05,      # Reduced from 0.15
        'edge_density': 0.05,         # Reduced from 0.15
        'edge_strength': 0.05,        # Reduced from 0.15
        'print_pattern_energy': 0.05, # Unchanged
        'print_pattern_variance': 0.40 # Increased from 0.10
    }

    
    # Define direction (True if original has higher values)
    directions = {
        'noise_mad': True,             # Unchanged (higher in originals)
        'noise_wavelet': True,         # Unchanged (higher in originals)
        'noise_laplacian': False,      # Changed to False (higher in photocopies)
        'edge_density': False,         # Changed to False (higher in photocopies)
        'edge_strength': True,         # Unchanged
        'print_pattern_energy': False, # Unchanged
        'print_pattern_variance': True # Unchanged (higher in originals)
    }

    
    # Calculate weighted score
    score = 0
    for feature, weight in weights.items():
        if feature in metrics and feature in THRESHOLDS:
            # Check if the feature value indicates original or photocopy
            is_above_threshold = metrics[feature] >= THRESHOLDS[feature]
            # If direction is True, above threshold means original
            # If direction is False, below threshold means original
            is_original = (is_above_threshold == directions[feature])
            # Add to score if it indicates original
            if is_original:
                score += weight
    
    # Classify based on total score
    return "original" if score >= 0.5 else "photocopy"

def process_page(args):
    """Process a single page - used by multiprocessing pool"""
    idx, image_array = args
    img = cv2.cvtColor(image_array, cv2.COLOR_RGB2BGR)
    metrics = calculate_metrics(img)
    classification = classify_page(metrics)
    return {
        'page': idx + 1,
        'classification': classification,
        'metrics': metrics
    }

@app.route('/analyze', methods=['POST'])
def analyze_pdf():
    if 'file' not in request.files:
        return jsonify({'error': 'No file uploaded'}), 400

    file = request.files['file']
    if file and allowed_file(file.filename):
        temp_path = os.path.join(app.config['UPLOAD_FOLDER'], file.filename)
        file.save(temp_path)

        try:
            # Convert PDF to images
            images = convert_from_path(temp_path, dpi=300)
            
            # Prepare data for multiprocessing
            image_data = [(idx, np.array(image)) for idx, image in enumerate(images)]
            
            # Use multiprocessing to analyze pages in parallel
            num_processes = min(multiprocessing.cpu_count(), len(images))
            with Pool(processes=num_processes) as pool:
                results = pool.map(process_page, image_data)
            
            os.remove(temp_path)
            return jsonify({'results': results})
        except Exception as e:
            return jsonify({'error': str(e)}), 500

    return jsonify({'error': 'Invalid file'}), 400

@app.route('/analyze-stream', methods=['POST'])
def analyze_pdf_stream():
    if 'file' not in request.files:
        return jsonify({'error': 'No file uploaded'}), 400

    file = request.files['file']
    if file and allowed_file(file.filename):
        temp_path = os.path.join(app.config['UPLOAD_FOLDER'], file.filename)
        file.save(temp_path)

        def generate():
            try:
                # Convert PDF to images
                images = convert_from_path(temp_path, dpi=300)
                total_pages = len(images)
                
                # Send total pages info
                yield f"data: {json.dumps({'status': 'start', 'total_pages': total_pages})}\n\n"
                
                results = []
                for idx, image in enumerate(images):
                    img = cv2.cvtColor(np.array(image), cv2.COLOR_RGB2BGR)
                    metrics = calculate_metrics(img)
                    classification = classify_page(metrics)
                    page_result = {
                        'page': idx + 1,
                        'classification': classification,
                        'metrics': metrics
                    }
                    results.append(page_result)
                    
                    # Send progress update
                    yield f"data: {json.dumps({'status': 'progress', 'current': idx + 1, 'total': total_pages, 'page_result': page_result})}\n\n"
                
                # Send final results
                yield f"data: {json.dumps({'status': 'complete', 'results': results})}\n\n"
                
                os.remove(temp_path)
            except Exception as e:
                yield f"data: {json.dumps({'status': 'error', 'message': str(e)})}\n\n"
        
        return Response(generate(), mimetype='text/event-stream')

    return jsonify({'error': 'Invalid file'}), 400

if __name__ == '__main__':
    app.run(host='0.0.0.0', port=5022, debug=True)

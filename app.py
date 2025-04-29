from flask import Flask, render_template, request, jsonify
from flask_cors import CORS
import torch
from transformers import pipeline, CLIPProcessor, CLIPModel
import base64
import io
import re
import json
import numpy as np
from PIL import Image
import fitz  # PyMuPDF
import os
from datetime import datetime
import uuid
import requests
from geopy.geocoders import Nominatim
from sentence_transformers import SentenceTransformer, util
import spacy
import pytesseract
from langdetect import detect
from deep_translator import GoogleTranslator
import logging
from functools import lru_cache
import time
import math
from pyngrok import ngrok
import threading
import asyncio
import concurrent.futures

app = Flask(__name__)
CORS(app)  # Enable CORS for frontend

# Configure logging
def setup_logging():
    log_dir = os.environ.get('LOG_DIR', '/app/logs')
    try:
        os.makedirs(log_dir, exist_ok=True)
        log_file = os.path.join(log_dir, 'app.log')
        
        # Ensure log file exists and is writable
        if not os.path.exists(log_file):
            open(log_file, 'a').close()
        os.chmod(log_file, 0o666)
        
        logging.basicConfig(
            level=logging.INFO,
            format='%(asctime)s - %(levelname)s - %(message)s',
            handlers=[
                logging.FileHandler(log_file),
                logging.StreamHandler()
            ]
        )
        return logging.getLogger(__name__)
    except Exception as e:
        # Fallback to console-only logging if file logging fails
        logging.basicConfig(
            level=logging.INFO,
            format='%(asctime)s - %(levelname)s - %(message)s',
            handlers=[logging.StreamHandler()]
        )
        return logging.getLogger(__name__)

logger = setup_logging()

# Initialize geocoder
geocoder = Nominatim(user_agent="indian_property_verifier", timeout=10)

# Cache models
@lru_cache(maxsize=10)
def load_model(task, model_name):
    try:
        logger.info(f"Loading model: {model_name} for task: {task}")
        return pipeline(task, model=model_name, device=-1)
    except Exception as e:
        logger.error(f"Error loading model {model_name}: {str(e)}")
        raise

# Initialize CLIP model
try:
    clip_processor = CLIPProcessor.from_pretrained("openai/clip-vit-base-patch32")
    clip_model = CLIPModel.from_pretrained("openai/clip-vit-base-patch32")
    has_clip_model = True
    logger.info("CLIP model loaded successfully")
except Exception as e:
    logger.error(f"Error loading CLIP model: {str(e)}")
    has_clip_model = False

# Initialize sentence transformer
try:
    sentence_model = SentenceTransformer('paraphrase-MiniLM-L6-v2')
    logger.info("Sentence transformer loaded successfully")
except Exception as e:
    logger.error(f"Error loading sentence transformer: {str(e)}")
    sentence_model = None

# Initialize spaCy
try:
    nlp = spacy.load('en_core_web_md')
    logger.info("spaCy model loaded successfully")
except Exception as e:
    logger.error(f"Error loading spaCy model: {str(e)}")
    nlp = None

def make_json_serializable(obj):
    try:
        if isinstance(obj, (bool, int, float, str, type(None))):
            return obj
        elif isinstance(obj, (list, tuple)):
            return [make_json_serializable(item) for item in obj]
        elif isinstance(obj, dict):
            return {str(key): make_json_serializable(value) for key, value in obj.items()}
        elif torch.is_tensor(obj):
            return obj.item() if obj.numel() == 1 else obj.tolist()
        elif np.isscalar(obj):
            return obj.item() if hasattr(obj, 'item') else float(obj)
        elif isinstance(obj, np.ndarray):
            return obj.tolist()
        else:
            return str(obj)
    except Exception as e:
        logger.error(f"Error serializing object: {str(e)}")
        return str(obj)

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/get-location', methods=['POST'])
def get_location():
    try:
        data = request.json or {}
        latitude = data.get('latitude')
        longitude = data.get('longitude')

        if not latitude or not longitude:
            logger.warning("Missing latitude or longitude")
            return jsonify({
                'status': 'error',
                'message': 'Latitude and longitude are required'
            }), 400

        # Validate coordinates are within India
        try:
            lat, lng = float(latitude), float(longitude)
            if not (6.5 <= lat <= 37.5 and 68.0 <= lng <= 97.5):
                return jsonify({
                    'status': 'error',
                    'message': 'Coordinates are outside India'
                }), 400
        except ValueError:
            return jsonify({
                'status': 'error',
                'message': 'Invalid coordinates format'
            }), 400

        # Retry geocoding up to 3 times
        for attempt in range(3):
            try:
                location = geocoder.reverse((latitude, longitude), exactly_one=True)
                if location:
                    address_components = location.raw.get('address', {})
                    
                    # Extract Indian-specific address components
                    city = address_components.get('city', '')
                    if not city:
                        city = address_components.get('town', '')
                    if not city:
                        city = address_components.get('village', '')
                    if not city:
                        city = address_components.get('suburb', '')
                    
                    state = address_components.get('state', '')
                    if not state:
                        state = address_components.get('state_district', '')
                    
                    # Get postal code and validate Indian format
                    postal_code = address_components.get('postcode', '')
                    if postal_code and not re.match(r'^\d{6}$', postal_code):
                        postal_code = ''
                    
                    # Get road/street name
                    road = address_components.get('road', '')
                    if not road:
                        road = address_components.get('street', '')
                    
                    # Get area/locality
                    area = address_components.get('suburb', '')
                    if not area:
                        area = address_components.get('neighbourhood', '')
                    
                    return jsonify({
                        'status': 'success',
                        'address': location.address,
                        'street': road,
                        'area': area,
                        'city': city,
                        'state': state,
                        'country': 'India',
                        'postal_code': postal_code,
                        'latitude': latitude,
                        'longitude': longitude,
                        'formatted_address': f"{road}, {area}, {city}, {state}, India - {postal_code}"
                    })
                logger.warning(f"Geocoding failed on attempt {attempt + 1}")
                time.sleep(1)  # Wait before retry
            except Exception as e:
                logger.error(f"Geocoding error on attempt {attempt + 1}: {str(e)}")
                time.sleep(1)

        return jsonify({
            'status': 'error',
            'message': 'Could not determine location after retries'
        }), 500

    except Exception as e:
        logger.error(f"Error in get_location: {str(e)}")
        return jsonify({
            'status': 'error',
            'message': str(e)
        }), 500

def calculate_final_verdict(results):
    """
    Calculate a comprehensive final verdict based on all analysis results.
    This function combines all verification scores, fraud indicators, and quality assessments
    to determine if a property listing is legitimate, suspicious, or fraudulent.
    """
    try:
        # Initialize verdict components
        verdict = {
            'status': 'unknown',
            'confidence': 0.0,
            'score': 0.0,
            'reasons': [],
            'critical_issues': [],
            'warnings': [],
            'recommendations': []
        }
        
        # Extract key components from results
        trust_score = results.get('trust_score', {}).get('score', 0)
        fraud_classification = results.get('fraud_classification', {})
        quality_assessment = results.get('quality_assessment', {})
        specs_verification = results.get('specs_verification', {})
        cross_validation = results.get('cross_validation', [])
        location_analysis = results.get('location_analysis', {})
        price_analysis = results.get('price_analysis', {})
        legal_analysis = results.get('legal_analysis', {})
        document_analysis = results.get('document_analysis', {})
        image_analysis = results.get('image_analysis', {})
        
        # Calculate component scores (0-100)
        component_scores = {
            'trust': trust_score,
            'fraud': 100 - (fraud_classification.get('alert_score', 0) * 100),
            'quality': quality_assessment.get('score', 0),
            'specs': specs_verification.get('verification_score', 0),
            'location': location_analysis.get('completeness_score', 0),
            'price': price_analysis.get('confidence', 0) * 100 if price_analysis.get('has_price') else 0,
            'legal': legal_analysis.get('completeness_score', 0),
            'documents': min(100, (document_analysis.get('pdf_count', 0) / 3) * 100) if document_analysis.get('pdf_count') else 0,
            'images': min(100, (image_analysis.get('image_count', 0) / 5) * 100) if image_analysis.get('image_count') else 0
        }
        
        # Calculate weighted final score with adjusted weights
        weights = {
            'trust': 0.20,
            'fraud': 0.25,  # Increased weight for fraud detection
            'quality': 0.15,
            'specs': 0.10,
            'location': 0.10,
            'price': 0.05,
            'legal': 0.05,
            'documents': 0.05,
            'images': 0.05
        }
        
        final_score = sum(score * weights.get(component, 0) for component, score in component_scores.items())
        verdict['score'] = final_score
        
        # Determine verdict status based on multiple factors
        fraud_level = fraud_classification.get('alert_level', 'minimal')
        high_risk_indicators = len(fraud_classification.get('high_risk', []))
        critical_issues = []
        warnings = []
        
        # Check for critical issues
        if fraud_level in ['critical', 'high']:
            critical_issues.append(f"High fraud risk detected: {fraud_level} alert level")
        
        if trust_score < 40:
            critical_issues.append(f"Very low trust score: {trust_score}%")
        
        if quality_assessment.get('score', 0) < 30:
            critical_issues.append(f"Very low content quality: {quality_assessment.get('score', 0)}%")
        
        if specs_verification.get('verification_score', 0) < 40:
            critical_issues.append(f"Property specifications verification failed: {specs_verification.get('verification_score', 0)}%")
        
        # Check for warnings
        if fraud_level == 'medium':
            warnings.append(f"Medium fraud risk detected: {fraud_level} alert level")
        
        if trust_score < 60:
            warnings.append(f"Low trust score: {trust_score}%")
        
        if quality_assessment.get('score', 0) < 60:
            warnings.append(f"Low content quality: {quality_assessment.get('score', 0)}%")
        
        if specs_verification.get('verification_score', 0) < 70:
            warnings.append(f"Property specifications have issues: {specs_verification.get('verification_score', 0)}%")
        
        # Check cross-validation results
        for check in cross_validation:
            if check.get('status') in ['inconsistent', 'invalid', 'suspicious', 'no_match']:
                warnings.append(f"Cross-validation issue: {check.get('message', 'Unknown issue')}")
        
        # Check for missing critical information
        missing_critical = []
        if not location_analysis.get('completeness_score', 0) > 70:
            missing_critical.append("Location information is incomplete")
        
        if not price_analysis.get('has_price', False):
            missing_critical.append("Price information is missing")
        
        if not legal_analysis.get('completeness_score', 0) > 70:
            missing_critical.append("Legal information is incomplete")
        
        if document_analysis.get('pdf_count', 0) == 0:
            missing_critical.append("No supporting documents provided")
        
        if image_analysis.get('image_count', 0) == 0:
            missing_critical.append("No property images provided")
        
        if missing_critical:
            warnings.append(f"Missing critical information: {', '.join(missing_critical)}")
        
        # Enhanced verdict determination with more strict criteria
        if critical_issues or (fraud_level in ['critical', 'high'] and trust_score < 50) or high_risk_indicators > 0:
            verdict['status'] = 'fraudulent'
            verdict['confidence'] = min(100, max(70, 100 - (trust_score * 0.5)))
        elif warnings or (fraud_level == 'medium' and trust_score < 70) or specs_verification.get('verification_score', 0) < 60:
            verdict['status'] = 'suspicious'
            verdict['confidence'] = min(100, max(50, trust_score * 0.8))
        else:
            verdict['status'] = 'legitimate'
            verdict['confidence'] = min(100, max(70, trust_score * 0.9))
        
        # Add reasons to verdict
        verdict['critical_issues'] = critical_issues
        verdict['warnings'] = warnings
        
        # Add recommendations based on issues
        if critical_issues:
            verdict['recommendations'].append("Do not proceed with this property listing")
            verdict['recommendations'].append("Report this listing to the platform")
        elif warnings:
            verdict['recommendations'].append("Proceed with extreme caution")
            verdict['recommendations'].append("Request additional verification documents")
            verdict['recommendations'].append("Verify all information with independent sources")
        else:
            verdict['recommendations'].append("Proceed with standard due diligence")
            verdict['recommendations'].append("Verify final details before transaction")
        
        # Add specific recommendations based on missing information
        for missing in missing_critical:
            verdict['recommendations'].append(f"Request {missing.lower()}")
        
        return verdict
    except Exception as e:
        logger.error(f"Error calculating final verdict: {str(e)}")
        return {
            'status': 'error',
            'confidence': 0.0,
            'score': 0.0,
            'reasons': [f"Error calculating verdict: {str(e)}"],
            'critical_issues': [],
            'warnings': [],
            'recommendations': ["Unable to determine property status due to an error"]
        }

@app.route('/verify', methods=['POST'])
def verify_property():
    try:
        if not request.form and not request.files:
            logger.warning("No form data or files provided")
            return jsonify({
                'error': 'No data provided',
                'status': 'error'
            }), 400

        # Extract form data
        data = {
            'property_name': request.form.get('property_name', '').strip(),
            'property_type': request.form.get('property_type', '').strip(),
            'status': request.form.get('status', '').strip(),
            'description': request.form.get('description', '').strip(),
            'address': request.form.get('address', '').strip(),
            'city': request.form.get('city', '').strip(),
            'state': request.form.get('state', '').strip(),
            'country': request.form.get('country', 'India').strip(),
            'zip': request.form.get('zip', '').strip(),
            'latitude': request.form.get('latitude', '').strip(),
            'longitude': request.form.get('longitude', '').strip(),
            'bedrooms': request.form.get('bedrooms', '').strip(),
            'bathrooms': request.form.get('bathrooms', '').strip(),
            'total_rooms': request.form.get('total_rooms', '').strip(),
            'year_built': request.form.get('year_built', '').strip(),
            'parking': request.form.get('parking', '').strip(),
            'sq_ft': request.form.get('sq_ft', '').strip(),
            'market_value': request.form.get('market_value', '').strip(),
            'amenities': request.form.get('amenities', '').strip(),
            'nearby_landmarks': request.form.get('nearby_landmarks', '').strip(),
            'legal_details': request.form.get('legal_details', '').strip()
        }

        # Validate required fields
        required_fields = ['property_name', 'property_type', 'address', 'city', 'state']
        missing_fields = [field for field in required_fields if not data[field]]
        if missing_fields:
            logger.warning(f"Missing required fields: {', '.join(missing_fields)}")
            return jsonify({
                'error': f"Missing required fields: {', '.join(missing_fields)}",
                'status': 'error'
            }), 400

        # Process images
        images = []
        image_analysis = []
        if 'images' in request.files:
            # Get unique image files by filename to prevent duplicates
            image_files = {}
            for img_file in request.files.getlist('images'):
                if img_file.filename and img_file.filename.lower().endswith(('.jpg', '.jpeg', '.png')):
                    image_files[img_file.filename] = img_file

            # Process unique images
            for img_file in image_files.values():
                    try:
                        img = Image.open(img_file)
                        buffered = io.BytesIO()
                        img.save(buffered, format="JPEG")
                        img_str = base64.b64encode(buffered.getvalue()).decode('utf-8')
                        images.append(img_str)
                        image_analysis.append(analyze_image(img))
                    except Exception as e:
                        logger.error(f"Error processing image {img_file.filename}: {str(e)}")
                        image_analysis.append({'error': str(e), 'is_property_related': False})

        # Process PDFs
        pdf_texts = []
        pdf_analysis = []
        if 'documents' in request.files:
            # Get unique PDF files by filename to prevent duplicates
            pdf_files = {}
            for pdf_file in request.files.getlist('documents'):
                if pdf_file.filename and pdf_file.filename.lower().endswith('.pdf'):
                    pdf_files[pdf_file.filename] = pdf_file

            # Process unique PDFs
            for pdf_file in pdf_files.values():
                    try:
                        pdf_text = extract_pdf_text(pdf_file)
                        pdf_texts.append({
                            'filename': pdf_file.filename,
                            'text': pdf_text
                        })
                        pdf_analysis.append(analyze_pdf_content(pdf_text, data))
                    except Exception as e:
                        logger.error(f"Error processing PDF {pdf_file.filename}: {str(e)}")
                        pdf_analysis.append({'error': str(e)})

        # Create consolidated text for analysis
        consolidated_text = f"""
        Property Name: {data['property_name']}
        Property Type: {data['property_type']}
        Status: {data['status']}
        Description: {data['description']}
        Location: {data['address']}, {data['city']}, {data['state']}, {data['country']}, {data['zip']}
        Coordinates: Lat {data['latitude']}, Long {data['longitude']}
        Specifications: {data['bedrooms']} bedrooms, {data['bathrooms']} bathrooms, {data['total_rooms']} total rooms
        Year Built: {data['year_built']}
        Parking: {data['parking']}
        Size: {data['sq_ft']} sq. ft.
        Market Value: ₹{data['market_value']}
        Amenities: {data['amenities']}
        Nearby Landmarks: {data['nearby_landmarks']}
        Legal Details: {data['legal_details']}
        """

        # Process description translation if needed
        try:
            description = data['description']
            if description and len(description) > 10:
                text_language = detect(description)
                if text_language != 'en':
                    translated_description = GoogleTranslator(source=text_language, target='en').translate(description)
                    data['description_translated'] = translated_description
                else:
                    data['description_translated'] = description
            else:
                data['description_translated'] = description
        except Exception as e:
            logger.error(f"Error in language detection/translation: {str(e)}")
            data['description_translated'] = data['description']

        # Run all analyses in parallel using asyncio
        async def run_analyses():
            with concurrent.futures.ThreadPoolExecutor() as executor:
                loop = asyncio.get_event_loop()
                tasks = [
                    loop.run_in_executor(executor, generate_property_summary, data),
                    loop.run_in_executor(executor, classify_fraud, consolidated_text, data),
                    loop.run_in_executor(executor, generate_trust_score, consolidated_text, image_analysis, pdf_analysis),
                    loop.run_in_executor(executor, generate_suggestions, consolidated_text, data),
                    loop.run_in_executor(executor, assess_text_quality, data['description_translated']),
                    loop.run_in_executor(executor, verify_address, data),
                    loop.run_in_executor(executor, perform_cross_validation, data),
                    loop.run_in_executor(executor, analyze_location, data),
                    loop.run_in_executor(executor, analyze_price, data),
                    loop.run_in_executor(executor, analyze_legal_details, data['legal_details']),
                    loop.run_in_executor(executor, verify_property_specs, data),
                    loop.run_in_executor(executor, analyze_market_value, data)
                ]
                results = await asyncio.gather(*tasks)
                return results

        # Run analyses and get results
        loop = asyncio.new_event_loop()
        asyncio.set_event_loop(loop)
        analysis_results = loop.run_until_complete(run_analyses())
        loop.close()

        # Unpack results
        summary, fraud_classification, (trust_score, trust_reasoning), suggestions, quality_assessment, \
        address_verification, cross_validation, location_analysis, price_analysis, legal_analysis, \
        specs_verification, market_analysis = analysis_results

        # Prepare response
        document_analysis = {
            'pdf_count': len(pdf_texts),
            'pdf_texts': pdf_texts,
            'pdf_analysis': pdf_analysis
        }
        image_results = {
            'image_count': len(images),
            'image_analysis': image_analysis
        }

        report_id = str(uuid.uuid4())

        # Create results dictionary
        results = {
            'report_id': report_id,
            'timestamp': datetime.now().strftime('%Y-%m-%d %H:%M:%S'),
            'summary': summary,
            'fraud_classification': fraud_classification,
            'trust_score': {
                'score': trust_score,
                'reasoning': trust_reasoning
            },
            'suggestions': suggestions,
            'quality_assessment': quality_assessment,
            'address_verification': address_verification,
            'cross_validation': cross_validation,
            'location_analysis': location_analysis,
            'price_analysis': price_analysis,
            'legal_analysis': legal_analysis,
            'document_analysis': document_analysis,
            'image_analysis': image_results,
            'specs_verification': specs_verification,
            'market_analysis': market_analysis,
            'images': images
        }
        
        # Calculate final verdict
        final_verdict = calculate_final_verdict(results)
        results['final_verdict'] = final_verdict

        return jsonify(make_json_serializable(results))

    except Exception as e:
        logger.error(f"Error in verify_property: {str(e)}")
        return jsonify({
            'error': 'Server error occurred. Please try again later.',
            'status': 'error',
            'details': str(e)
        }), 500

def extract_pdf_text(pdf_file):
    try:
        pdf_document = fitz.Document(stream=pdf_file.read(), filetype="pdf")
        text = ""
        for page in pdf_document:
            text += page.get_text()
        pdf_document.close()
        return text
    except Exception as e:
        logger.error(f"Error extracting PDF text: {str(e)}")
        return ""

def analyze_image(image):
    try:
        if has_clip_model:
            img_rgb = image.convert('RGB')
            inputs = clip_processor(
                text=[
                    "real estate property interior",
                    "real estate property exterior",
                    "non-property-related image",
                    "office space",
                    "landscape"
                ],
                images=img_rgb,
                return_tensors="pt",
                padding=True
            )
            outputs = clip_model(**inputs)
            logits_per_image = outputs.logits_per_image
            probs = logits_per_image.softmax(dim=1).detach().numpy()[0]

            property_related_score = probs[0] + probs[1]
            is_property_related = property_related_score > 0.5

            quality = assess_image_quality(image)
            is_ai_generated = detect_ai_generated_image(image)

            return {
                'is_property_related': is_property_related,
                'property_confidence': float(property_related_score),
                'top_predictions': [
                    {'label': 'property interior', 'confidence': float(probs[0])},
                    {'label': 'property exterior', 'confidence': float(probs[1])},
                    {'label': 'non-property', 'confidence': float(probs[2])}
                ],
                'image_quality': quality,
                'is_ai_generated': is_ai_generated,
                'authenticity_score': 0.95 if not is_ai_generated else 0.60
            }
        else:
            logger.warning("CLIP model unavailable")
            return {
                'is_property_related': False,
                'property_confidence': 0.0,
                'top_predictions': [],
                'image_quality': assess_image_quality(image),
                'is_ai_generated': False,
                'authenticity_score': 0.5
            }
    except Exception as e:
        logger.error(f"Error analyzing image: {str(e)}")
        return {
            'is_property_related': False,
            'property_confidence': 0.0,
            'top_predictions': [],
            'image_quality': {'resolution': 'unknown', 'quality_score': 0},
            'is_ai_generated': False,
            'authenticity_score': 0.0,
            'error': str(e)
        }

def detect_ai_generated_image(image):
    try:
        img_array = np.array(image)
        if len(img_array.shape) == 3:
            gray = np.mean(img_array, axis=2)
        else:
            gray = img_array
        noise = gray - np.mean(gray)
        noise_std = np.std(noise)
        width, height = image.size
        perfect_dimensions = (width % 64 == 0 and height % 64 == 0)
        has_exif = hasattr(image, '_getexif') and image._getexif() is not None
        return noise_std < 0.05 or perfect_dimensions or not has_exif
    except Exception as e:
        logger.error(f"Error detecting AI-generated image: {str(e)}")
        return False

def analyze_pdf_content(document_text, property_data):
    try:
        if not document_text:
            return {
                'document_type': {'classification': 'unknown', 'confidence': 0.0},
                'authenticity': {'assessment': 'could not verify', 'confidence': 0.0},
                'key_info': {},
                'consistency_score': 0.0,
                'is_property_related': False,
                'summary': 'Empty document',
                'has_signatures': False,
                'has_dates': False,
                'verification_score': 0.0
            }

        # Use a more sophisticated model for document classification
        classifier = load_model("zero-shot-classification", "facebook/bart-large-mnli")
        
        # Enhanced document types with more specific categories
        doc_types = [
            "property deed", "sales agreement", "mortgage document",
            "property tax record", "title document", "khata certificate",
            "encumbrance certificate", "lease agreement", "rental agreement",
            "property registration document", "building permit", "other document"
        ]
        
        # Analyze document type with context
        doc_context = f"{document_text[:1000]} property_type:{property_data.get('property_type', '')} location:{property_data.get('city', '')}"
        doc_result = classifier(doc_context, doc_types)
        doc_type = doc_result['labels'][0]
        doc_confidence = doc_result['scores'][0]

        # Enhanced authenticity check with multiple aspects
        authenticity_aspects = [
            "authentic legal document",
            "questionable document",
            "forged document",
            "template document",
            "official document"
        ]
        authenticity_result = classifier(document_text[:1000], authenticity_aspects)
        authenticity = "likely authentic" if authenticity_result['labels'][0] == "authentic legal document" else "questionable"
        authenticity_confidence = authenticity_result['scores'][0]

        # Extract key information using NLP
        key_info = extract_document_key_info(document_text)
        
        # Enhanced consistency check
        consistency_score = check_document_consistency(document_text, property_data)
        
        # Property relation check with context
        property_context = f"{document_text[:1000]} property:{property_data.get('property_name', '')} type:{property_data.get('property_type', '')}"
        is_property_related = check_if_property_related(property_context)['is_related']
        
        # Generate summary using BART
        summary = summarize_text(document_text[:2000])

        # Enhanced signature and date detection
        has_signatures = bool(re.search(r'(?:sign|signature|signed|witness|notary|authorized).{0,50}(?:by|of|for)', document_text.lower()))
        has_dates = bool(re.search(r'\d{1,2}[/-]\d{1,2}[/-]\d{2,4}|\d{4}[/-]\d{1,2}[/-]\d{1,2}', document_text))

        # Calculate verification score with weighted components
        verification_weights = {
            'doc_type': 0.3,
            'authenticity': 0.3,
            'consistency': 0.2,
            'property_relation': 0.1,
            'signatures_dates': 0.1
        }
        
        verification_score = (
            doc_confidence * verification_weights['doc_type'] +
            authenticity_confidence * verification_weights['authenticity'] +
            consistency_score * verification_weights['consistency'] +
            float(is_property_related) * verification_weights['property_relation'] +
            float(has_signatures and has_dates) * verification_weights['signatures_dates']
        )

        return {
            'document_type': {'classification': doc_type, 'confidence': float(doc_confidence)},
            'authenticity': {'assessment': authenticity, 'confidence': float(authenticity_confidence)},
            'key_info': key_info,
            'consistency_score': float(consistency_score),
            'is_property_related': is_property_related,
            'summary': summary,
            'has_signatures': has_signatures,
            'has_dates': has_dates,
            'verification_score': float(verification_score)
        }
    except Exception as e:
        logger.error(f"Error analyzing PDF content: {str(e)}")
        return {
            'document_type': {'classification': 'unknown', 'confidence': 0.0},
            'authenticity': {'assessment': 'could not verify', 'confidence': 0.0},
            'key_info': {},
            'consistency_score': 0.0,
            'is_property_related': False,
            'summary': 'Could not analyze document',
            'has_signatures': False,
            'has_dates': False,
            'verification_score': 0.0,
            'error': str(e)
        }

def check_document_consistency(document_text, property_data):
    try:
        if not sentence_model:
            logger.warning("Sentence model unavailable")
            return 0.5
        property_text = ' '.join([
            property_data.get(key, '') for key in [
                'property_name', 'property_type', 'address', 'city',
                'state', 'market_value', 'sq_ft', 'bedrooms'
            ]
        ])
        property_embedding = sentence_model.encode(property_text)
        document_embedding = sentence_model.encode(document_text[:1000])
        similarity = util.cos_sim(property_embedding, document_embedding)[0][0].item()
        return max(0.0, min(1.0, float(similarity)))
    except Exception as e:
        logger.error(f"Error checking document consistency: {str(e)}")
        return 0.0

def extract_document_key_info(text):
    try:
        info = {}
        patterns = {
            'property_address': r'(?:property|premises|located at)[:\s]+([^\n.]+)',
            'price': r'(?:price|value|amount)[:\s]+(?:Rs\.?|₹)?[\s]*([0-9,.]+)',
            'date': r'(?:date|dated|executed on)[:\s]+([^\n.]+\d{4})',
            'seller': r'(?:seller|grantor|owner)[:\s]+([^\n.]+)',
            'buyer': r'(?:buyer|grantee|purchaser)[:\s]+([^\n.]+)',
            'size': r'(?:area|size|extent)[:\s]+([0-9,.]+)[\s]*(?:sq\.?[\s]*(?:ft|feet))',
            'registration_number': r'(?:registration|reg\.?|document)[\s]*(?:no\.?|number|#)[:\s]*([A-Za-z0-9\-/]+)'
        }
        for key, pattern in patterns.items():
            match = re.search(pattern, text, re.IGNORECASE)
            if match:
                info[key] = match.group(1).strip()
        return info
    except Exception as e:
        logger.error(f"Error extracting document key info: {str(e)}")
        return {}

def generate_property_summary(data):
    try:
        # Create a detailed context for summary generation
        property_context = f"""
        Property Name: {data.get('property_name', '')}
        Type: {data.get('property_type', '')}
        Status: {data.get('status', '')}
        Location: {data.get('address', '')}, {data.get('city', '')}, {data.get('state', '')}, {data.get('country', '')}
        Size: {data.get('sq_ft', '')} sq. ft.
        Price: ₹{data.get('market_value', '0')}
        Bedrooms: {data.get('bedrooms', '')}
        Bathrooms: {data.get('bathrooms', '')}
        Year Built: {data.get('year_built', '')}
        Description: {data.get('description', '')}
        """
        
        # Use BART for summary generation
        summarizer = load_model("summarization", "facebook/bart-large-cnn")
        
        # Generate initial summary
        summary_result = summarizer(property_context, max_length=150, min_length=50, do_sample=False)
        initial_summary = summary_result[0]['summary_text']
        
        # Enhance summary with key features
        key_features = []
        
        # Add property type and status
        if data.get('property_type') and data.get('status'):
            key_features.append(f"{data['property_type']} is {data['status'].lower()}")
        
        # Add location if available
        location_parts = []
        if data.get('city'):
            location_parts.append(data['city'])
        if data.get('state'):
            location_parts.append(data['state'])
        if location_parts:
            key_features.append(f"Located in {', '.join(location_parts)}")
        
        # Add size and price if available
        if data.get('sq_ft'):
            key_features.append(f"Spans {data['sq_ft']} sq. ft.")
        if data.get('market_value'):
            key_features.append(f"Valued at ₹{data['market_value']}")
        
        # Add rooms information
        rooms_info = []
        if data.get('bedrooms'):
            rooms_info.append(f"{data['bedrooms']} bedroom{'s' if data['bedrooms'] != '1' else ''}")
        if data.get('bathrooms'):
            rooms_info.append(f"{data['bathrooms']} bathroom{'s' if data['bathrooms'] != '1' else ''}")
        if rooms_info:
            key_features.append(f"Features {' and '.join(rooms_info)}")
        
        # Add amenities if available
        if data.get('amenities'):
            key_features.append(f"Amenities: {data['amenities']}")
        
        # Combine initial summary with key features
        enhanced_summary = initial_summary
        if key_features:
            enhanced_summary += " " + ". ".join(key_features) + "."
        
        # Clean up the summary
        enhanced_summary = enhanced_summary.replace("  ", " ").strip()
        
        return enhanced_summary
    except Exception as e:
        logger.error(f"Error generating property summary: {str(e)}")
        return "Could not generate summary."

def summarize_text(text):
    try:
        if not text or len(text.strip()) < 10:
            return "No text to summarize."
        summarizer = load_model("summarization", "facebook/bart-large-cnn")
        input_length = len(text.split())
        max_length = max(50, min(150, input_length // 2))
        min_length = max(20, input_length // 4)
        summary = summarizer(text[:2000], max_length=max_length, min_length=min_length, do_sample=False)
        return summary[0]['summary_text']
    except Exception as e:
        logger.error(f"Error summarizing text: {str(e)}")
        return text[:200] + "..." if len(text) > 200 else text

def classify_fraud(property_details, description):
    """
    Classify the risk of fraud in a property listing using zero-shot classification.
    This function analyzes property details and description to identify potential fraud indicators.
    """
    try:
        # Initialize fraud classification result
        fraud_classification = {
            'alert_level': 'minimal',
            'alert_score': 0.0,
            'high_risk': [],
            'medium_risk': [],
            'low_risk': [],
            'confidence_scores': {}
        }
        
        # Combine property details and description for analysis
        text_to_analyze = f"{property_details}\n{description}"
        
        # Define risk categories for zero-shot classification
        risk_categories = [
            "fraudulent listing",
            "misleading information",
            "fake property",
            "scam attempt",
            "legitimate listing"
        ]
        
        # Perform zero-shot classification
        classifier = pipeline("zero-shot-classification", model="facebook/bart-large-mnli")
        result = classifier(text_to_analyze, risk_categories, multi_label=True)
        
        # Process classification results
        fraud_score = 0.0
        for label, score in zip(result['labels'], result['scores']):
            if label != "legitimate listing":
                fraud_score += score
                fraud_classification['confidence_scores'][label] = score
        
        # Normalize fraud score to 0-1 range
        fraud_score = min(1.0, fraud_score / (len(risk_categories) - 1))
        fraud_classification['alert_score'] = fraud_score
        
        # Define fraud indicators to check
        fraud_indicators = {
            'high_risk': [
                r'urgent|immediate|hurry|limited time|special offer',
                r'bank|transfer|wire|payment|money',
                r'fake|scam|fraud|illegal|unauthorized',
                r'guaranteed|promised|assured|certain',
                r'contact.*whatsapp|whatsapp.*contact',
                r'price.*negotiable|negotiable.*price',
                r'no.*documents|documents.*not.*required',
                r'cash.*only|only.*cash',
                r'off.*market|market.*off',
                r'under.*table|table.*under'
            ],
            'medium_risk': [
                r'unverified|unconfirmed|unchecked',
                r'partial|incomplete|missing',
                r'different.*location|location.*different',
                r'price.*increased|increased.*price',
                r'no.*photos|photos.*not.*available',
                r'contact.*email|email.*contact',
                r'agent.*not.*available|not.*available.*agent',
                r'property.*not.*viewable|not.*viewable.*property',
                r'price.*changed|changed.*price',
                r'details.*updated|updated.*details'
            ],
            'low_risk': [
                r'new.*listing|listing.*new',
                r'recent.*update|update.*recent',
                r'price.*reduced|reduced.*price',
                r'contact.*phone|phone.*contact',
                r'agent.*available|available.*agent',
                r'property.*viewable|viewable.*property',
                r'photos.*available|available.*photos',
                r'documents.*available|available.*documents',
                r'price.*fixed|fixed.*price',
                r'details.*complete|complete.*details'
            ]
        }
        
        # Check for fraud indicators in text
        for risk_level, patterns in fraud_indicators.items():
            for pattern in patterns:
                matches = re.finditer(pattern, text_to_analyze, re.IGNORECASE)
                for match in matches:
                    indicator = match.group(0)
                    if indicator not in fraud_classification[risk_level]:
                        fraud_classification[risk_level].append(indicator)
        
        # Determine alert level based on fraud score and indicators
        if fraud_score > 0.7 or len(fraud_classification['high_risk']) > 0:
            fraud_classification['alert_level'] = 'critical'
        elif fraud_score > 0.5 or len(fraud_classification['medium_risk']) > 2:
            fraud_classification['alert_level'] = 'high'
        elif fraud_score > 0.3 or len(fraud_classification['medium_risk']) > 0:
            fraud_classification['alert_level'] = 'medium'
        elif fraud_score > 0.1 or len(fraud_classification['low_risk']) > 0:
            fraud_classification['alert_level'] = 'low'
        else:
            fraud_classification['alert_level'] = 'minimal'
        
        # Additional checks for common fraud patterns
        if re.search(r'price.*too.*good|too.*good.*price', text_to_analyze, re.IGNORECASE):
            fraud_classification['high_risk'].append("Unrealistically low price")
        
        if re.search(r'no.*inspection|inspection.*not.*allowed', text_to_analyze, re.IGNORECASE):
            fraud_classification['high_risk'].append("No property inspection allowed")
        
        if re.search(r'owner.*abroad|abroad.*owner', text_to_analyze, re.IGNORECASE):
            fraud_classification['medium_risk'].append("Owner claims to be abroad")
        
        if re.search(r'agent.*unavailable|unavailable.*agent', text_to_analyze, re.IGNORECASE):
            fraud_classification['medium_risk'].append("Agent unavailable for verification")
        
        # Check for inconsistencies in property details
        if 'price' in property_details and 'market_value' in property_details:
            try:
                price = float(re.search(r'\d+(?:,\d+)*(?:\.\d+)?', property_details['price']).group().replace(',', ''))
                market_value = float(re.search(r'\d+(?:,\d+)*(?:\.\d+)?', property_details['market_value']).group().replace(',', ''))
                if price < market_value * 0.5:
                    fraud_classification['high_risk'].append("Price significantly below market value")
            except (ValueError, AttributeError):
                pass
        
        return fraud_classification
    except Exception as e:
        logger.error(f"Error in fraud classification: {str(e)}")
        return {
            'alert_level': 'error',
            'alert_score': 1.0,
            'high_risk': [f"Error in fraud classification: {str(e)}"],
            'medium_risk': [],
            'low_risk': [],
            'confidence_scores': {}
        }

def generate_trust_score(text, image_analysis, pdf_analysis):
    try:
        classifier = load_model("zero-shot-classification", "facebook/bart-large-mnli")
        aspects = [
            "complete information provided",
            "verified location",
            "consistent data",
            "authentic documents",
            "authentic images",
            "reasonable pricing",
            "verified ownership",
            "proper documentation"
        ]
        result = classifier(text[:1000], aspects, multi_label=True)
        
        # Much stricter weights with higher emphasis on critical aspects
        weights = {
            "complete information provided": 0.25,
            "verified location": 0.20,
            "consistent data": 0.15,
            "authentic documents": 0.15,
            "authentic images": 0.10,
            "reasonable pricing": 0.05,
            "verified ownership": 0.05,
            "proper documentation": 0.05
        }
        
        score = 0
        reasoning_parts = []
        
        # Much stricter scoring for each aspect
        for label, confidence in zip(result['labels'], result['scores']):
            adjusted_confidence = confidence
            
            # Stricter document verification
            if label == "authentic documents":
                if not pdf_analysis or len(pdf_analysis) == 0:
                    adjusted_confidence = 0.0
                else:
                    doc_scores = [p.get('verification_score', 0) for p in pdf_analysis]
                    adjusted_confidence = sum(doc_scores) / max(1, len(doc_scores))
                    # Heavily penalize if any document has low verification score
                    if any(score < 0.7 for score in doc_scores):
                        adjusted_confidence *= 0.4
                    # Additional penalty for missing documents
                    if len(doc_scores) < 2:
                        adjusted_confidence *= 0.5
            
            # Stricter image verification
            elif label == "authentic images":
                if not image_analysis or len(image_analysis) == 0:
                    adjusted_confidence = 0.0
                else:
                    img_scores = [i.get('authenticity_score', 0) for i in image_analysis]
                    adjusted_confidence = sum(img_scores) / max(1, len(img_scores))
                    # Heavily penalize if any image has low authenticity score
                    if any(score < 0.8 for score in img_scores):
                        adjusted_confidence *= 0.4
                    # Additional penalty for AI-generated images
                    if any(i.get('is_ai_generated', False) for i in image_analysis):
                        adjusted_confidence *= 0.5
                    # Additional penalty for non-property related images
                    if any(not i.get('is_property_related', False) for i in image_analysis):
                        adjusted_confidence *= 0.6
            
            # Stricter consistency check
            elif label == "consistent data":
                # Check for inconsistencies in the data
                if "inconsistent" in text.lower() or "suspicious" in text.lower():
                    adjusted_confidence *= 0.3
                # Check for impossible values
                if "impossible" in text.lower() or "invalid" in text.lower():
                    adjusted_confidence *= 0.2
                # Check for missing critical information
                if "missing" in text.lower() or "not provided" in text.lower():
                    adjusted_confidence *= 0.4
            
            # Stricter completeness check
            elif label == "complete information provided":
                # Check for missing critical information
                if len(text) < 300 or "not provided" in text.lower() or "missing" in text.lower():
                    adjusted_confidence *= 0.4
                # Check for vague or generic descriptions
                if "generic" in text.lower() or "vague" in text.lower():
                    adjusted_confidence *= 0.5
                # Check for suspiciously short descriptions
                if len(text) < 150:
                    adjusted_confidence *= 0.3
            
            score += adjusted_confidence * weights.get(label, 0.1)
            reasoning_parts.append(f"{label} ({adjusted_confidence:.0%})")

        # Apply additional penalties for suspicious patterns
        if "suspicious" in text.lower() or "fraudulent" in text.lower():
            score *= 0.5
        
        # Apply penalties for suspiciously low values
        if "suspiciously low" in text.lower() or "unusually small" in text.lower():
            score *= 0.6
        
        # Apply penalties for inconsistencies
        if "inconsistent" in text.lower() or "mismatch" in text.lower():
            score *= 0.6
        
        # Apply penalties for missing critical information
        if "missing critical" in text.lower() or "incomplete" in text.lower():
            score *= 0.7
        
        # Ensure score is between 0 and 100
        score = min(100, max(0, int(score * 100)))
        reasoning = f"Based on: {', '.join(reasoning_parts)}"
        return score, reasoning
    except Exception as e:
        logger.error(f"Error generating trust score: {str(e)}")
        return 20, "Could not assess trust."

def generate_suggestions(text, data=None):
    try:
        classifier = load_model("zero-shot-classification", "facebook/bart-large-mnli")
        
        # Create comprehensive context for analysis
        suggestion_context = text
        if data:
            suggestion_context += f"""
            Additional Context:
            Property Type: {data.get('property_type', '')}
            Location: {data.get('city', '')}, {data.get('state', '')}
            Size: {data.get('sq_ft', '')} sq.ft.
            Year Built: {data.get('year_built', '')}
            """
        
        # Enhanced suggestion categories based on property context
        base_suggestions = {
            'documentation': {
                'label': "add more documentation",
                'categories': [
                    "complete documentation provided",
                    "missing essential documents",
                    "incomplete paperwork",
                    "documentation needs verification"
                ],
                'weight': 2.0,
                'improvements': {
                    'missing essential documents': [
                        "Add property deed or title documents",
                        "Include recent property tax records",
                        "Attach property registration documents"
                    ],
                    'incomplete paperwork': [
                        "Complete all required legal documents",
                        "Add missing ownership proof",
                        "Include property survey documents"
                    ]
                }
            },
            'details': {
                'label': "enhance property details",
                'categories': [
                    "detailed property information",
                    "basic information only",
                    "missing key details",
                    "comprehensive description"
                ],
                'weight': 1.8,
                'improvements': {
                    'basic information only': [
                        "Add more details about property features",
                        "Include information about recent renovations",
                        "Describe unique selling points"
                    ],
                    'missing key details': [
                        "Specify exact built-up area",
                        "Add floor plan details",
                        "Include maintenance costs"
                    ]
                }
            },
            'images': {
                'label': "improve visual content",
                'categories': [
                    "high quality images provided",
                    "poor image quality",
                    "insufficient images",
                    "missing key area photos"
                ],
                'weight': 1.5,
                'improvements': {
                    'poor image quality': [
                        "Add high-resolution property photos",
                        "Include better lighting in images",
                        "Provide professional photography"
                    ],
                    'insufficient images': [
                        "Add more interior photos",
                        "Include exterior and surrounding area images",
                        "Add photos of amenities"
                    ]
                }
            },
            'pricing': {
                'label': "pricing information",
                'categories': [
                    "detailed pricing breakdown",
                    "basic price only",
                    "missing price details",
                    "unclear pricing terms"
                ],
                'weight': 1.7,
                'improvements': {
                    'basic price only': [
                        "Add detailed price breakdown",
                        "Include maintenance charges",
                        "Specify additional costs"
                    ],
                    'missing price details': [
                        "Add price per square foot",
                        "Include tax implications",
                        "Specify payment terms"
                    ]
                }
            },
            'location': {
                'label': "location details",
                'categories': [
                    "comprehensive location info",
                    "basic location only",
                    "missing location details",
                    "unclear accessibility info"
                ],
                'weight': 1.6,
                'improvements': {
                    'basic location only': [
                        "Add nearby landmarks and distances",
                        "Include transportation options",
                        "Specify neighborhood facilities"
                    ],
                    'missing location details': [
                        "Add exact GPS coordinates",
                        "Include area development plans",
                        "Specify distance to key facilities"
                    ]
                }
            }
        }
        
        suggestions = []
        confidence_scores = []
        
        for aspect, config in base_suggestions.items():
            # Analyze each aspect with context
            result = classifier(suggestion_context[:1000], config['categories'])
            
            # Get the most relevant category
            top_category = result['labels'][0]
            confidence = float(result['scores'][0])
            
            # If the category indicates improvement needed (confidence < 0.6)
            if confidence < 0.6 and top_category in config['improvements']:
                weighted_confidence = confidence * config['weight']
                for improvement in config['improvements'][top_category]:
                    suggestions.append({
                        'aspect': aspect,
                        'category': top_category,
                        'suggestion': improvement,
                        'confidence': weighted_confidence
                    })
                confidence_scores.append(weighted_confidence)
        
        # Sort suggestions by confidence and priority
        suggestions.sort(key=lambda x: x['confidence'], reverse=True)
        
        # Property type specific suggestions
        if data and data.get('property_type'):
            property_type = data['property_type'].lower()
            type_specific_suggestions = {
                'residential': [
                    "Add information about school districts",
                    "Include details about neighborhood safety",
                    "Specify parking arrangements"
                ],
                'commercial': [
                    "Add foot traffic statistics",
                    "Include zoning information",
                    "Specify business licenses required"
                ],
                'industrial': [
                    "Add power supply specifications",
                    "Include environmental clearances",
                    "Specify loading/unloading facilities"
                ],
                'land': [
                    "Add soil testing reports",
                    "Include development potential analysis",
                    "Specify available utilities"
                ]
            }
            
            for type_key, type_suggestions in type_specific_suggestions.items():
                if type_key in property_type:
                    for suggestion in type_suggestions:
                        suggestions.append({
                            'aspect': 'property_type_specific',
                            'category': 'type_specific_requirements',
                            'suggestion': suggestion,
                            'confidence': 0.8  # High confidence for type-specific suggestions
                        })
        
        # Add market-based suggestions
        if data and data.get('market_value'):
            try:
                market_value = float(data['market_value'].replace('₹', '').replace(',', ''))
                if market_value > 10000000:  # High-value property
                    premium_suggestions = [
                        "Add virtual tour of the property",
                        "Include detailed investment analysis",
                        "Provide historical price trends"
                    ]
                    for suggestion in premium_suggestions:
                        suggestions.append({
                            'aspect': 'premium_property',
                            'category': 'high_value_requirements',
                            'suggestion': suggestion,
                            'confidence': 0.9
                        })
            except ValueError:
                pass
        
        # Calculate overall completeness score
        completeness_score = sum(confidence_scores) / len(confidence_scores) if confidence_scores else 0
        completeness_score = min(100, max(0, completeness_score * 100))
        
        return {
            'suggestions': suggestions[:10],  # Return top 10 suggestions
            'completeness_score': completeness_score,
            'priority_aspects': [s['aspect'] for s in suggestions[:3]],
            'improvement_summary': f"Focus on improving {', '.join([s['aspect'] for s in suggestions[:3]])}",
            'total_suggestions': len(suggestions)
        }
    except Exception as e:
        logger.error(f"Error generating suggestions: {str(e)}")
        return {
            'suggestions': [
                {
                    'aspect': 'general',
                    'category': 'basic_requirements',
                    'suggestion': 'Please provide more property details',
                    'confidence': 0.5
                }
            ],
            'completeness_score': 0,
            'priority_aspects': ['general'],
            'improvement_summary': "Add basic property information",
            'total_suggestions': 1
        }

def assess_text_quality(text):
    try:
        if not text or len(text.strip()) < 20:
            return {
                'assessment': 'insufficient',
                'score': 0,
                'reasoning': 'Text too short.',
                'is_ai_generated': False,
                'quality_metrics': {}
            }

        classifier = load_model("zero-shot-classification", "facebook/bart-large-mnli")
        
        # Enhanced quality categories with more specific indicators
        quality_categories = [
            "detailed and informative",
            "adequately detailed",
            "basic information",
            "vague description",
            "misleading content",
            "professional listing",
            "amateur listing",
            "spam-like content",
            "template-based content",
            "authentic description"
        ]
        
        # Analyze text with multiple aspects
        quality_result = classifier(text[:1000], quality_categories, multi_label=True)
        
        # Get top classifications with confidence scores
        top_classifications = []
        for label, score in zip(quality_result['labels'][:3], quality_result['scores'][:3]):
            if score > 0.3:  # Only include if confidence is above 30%
                top_classifications.append({
                    'classification': label,
                    'confidence': float(score)
                })
        
        # AI generation detection with multiple models
        ai_check = classifier(text[:1000], ["human-written", "AI-generated", "template-based", "authentic"])
        is_ai_generated = (
            (ai_check['labels'][0] == "AI-generated" and ai_check['scores'][0] > 0.6) or
            (ai_check['labels'][0] == "template-based" and ai_check['scores'][0] > 0.7)
        )
        
        # Calculate quality metrics
        quality_metrics = {
            'detail_level': sum(score for label, score in zip(quality_result['labels'], quality_result['scores']) 
                              if label in ['detailed and informative', 'adequately detailed']),
            'professionalism': sum(score for label, score in zip(quality_result['labels'], quality_result['scores']) 
                                 if label in ['professional listing', 'authentic description']),
            'clarity': sum(score for label, score in zip(quality_result['labels'], quality_result['scores']) 
                         if label not in ['vague description', 'misleading content', 'spam-like content']),
            'authenticity': 1.0 - sum(score for label, score in zip(quality_result['labels'], quality_result['scores']) 
                                    if label in ['template-based content', 'spam-like content'])
        }
        
        # Calculate overall score with weighted metrics
        weights = {
            'detail_level': 0.3,
            'professionalism': 0.25,
            'clarity': 0.25,
            'authenticity': 0.2
        }
        
        score = sum(metric * weights[metric_name] for metric_name, metric in quality_metrics.items())
        score = score * 100  # Convert to percentage
        
        # Adjust score for AI-generated content
        if is_ai_generated:
            score = score * 0.7  # Reduce score by 30% for AI-generated content
        
        # Generate detailed reasoning
        reasoning_parts = []
        if top_classifications:
            primary_class = top_classifications[0]['classification']
            reasoning_parts.append(f"Primary assessment: {primary_class}")
        
        if quality_metrics['detail_level'] > 0.7:
            reasoning_parts.append("Contains comprehensive details")
        elif quality_metrics['detail_level'] > 0.4:
            reasoning_parts.append("Contains adequate details")
        else:
            reasoning_parts.append("Lacks important details")
        
        if quality_metrics['professionalism'] > 0.7:
            reasoning_parts.append("Professional listing style")
        elif quality_metrics['professionalism'] < 0.4:
            reasoning_parts.append("Amateur listing style")
        
        if quality_metrics['clarity'] < 0.5:
            reasoning_parts.append("Content clarity issues detected")
        
        if is_ai_generated:
            reasoning_parts.append("Content appears to be AI-generated")

        return {
            'assessment': top_classifications[0]['classification'] if top_classifications else 'could not assess',
            'score': int(score),
            'reasoning': '. '.join(reasoning_parts),
            'is_ai_generated': is_ai_generated,
            'quality_metrics': quality_metrics,
            'top_classifications': top_classifications
        }
    except Exception as e:
        logger.error(f"Error assessing text quality: {str(e)}")
        return {
            'assessment': 'could not assess',
            'score': 50,
            'reasoning': 'Technical error.',
            'is_ai_generated': False,
            'quality_metrics': {},
            'top_classifications': []
        }

def verify_address(data):
    try:
        address_results = {
            'address_exists': False,
            'pincode_valid': False,
            'city_state_match': False,
            'coordinates_match': False,
            'confidence': 0.0,
            'issues': [],
            'verification_score': 0.0
        }

        if data['zip']:
            try:
                response = requests.get(f"https://api.postalpincode.in/pincode/{data['zip']}", timeout=5)
                if response.status_code == 200:
                    pin_data = response.json()
                    if pin_data[0]['Status'] == 'Success':
                        address_results['pincode_valid'] = True
                        post_offices = pin_data[0]['PostOffice']
                        cities = {po['Name'].lower() for po in post_offices}
                        states = {po['State'].lower() for po in post_offices}
                        if data['city'].lower() in cities or data['state'].lower() in states:
                            address_results['city_state_match'] = True
                        else:
                            address_results['issues'].append("City/state may not match pincode")
                    else:
                        address_results['issues'].append(f"Invalid pincode: {data['zip']}")
                else:
                    address_results['issues'].append("Pincode API error")
            except Exception as e:
                logger.error(f"Pincode API error: {str(e)}")
                address_results['issues'].append("Pincode validation failed")

        full_address = ', '.join(filter(None, [data['address'], data['city'], data['state'], data['country'], data['zip']]))
        for attempt in range(3):
            try:
                location = geocoder.geocode(full_address)
                if location:
                    address_results['address_exists'] = True
                    address_results['confidence'] = 0.9
                    if data['latitude'] and data['longitude']:
                        try:
                            provided_coords = (float(data['latitude']), float(data['longitude']))
                            geocoded_coords = (location.latitude, location.longitude)
                            from geopy.distance import distance
                            dist = distance(provided_coords, geocoded_coords).km
                            address_results['coordinates_match'] = dist < 1.0
                            if not address_results['coordinates_match']:
                                address_results['issues'].append(f"Coordinates {dist:.2f}km off")
                        except:
                            address_results['issues'].append("Invalid coordinates")
                    break
                time.sleep(1)
            except Exception as e:
                logger.error(f"Geocoding error on attempt {attempt + 1}: {str(e)}")
                time.sleep(1)
        else:
            address_results['issues'].append("Address geocoding failed")

        verification_points = (
            address_results['address_exists'] * 0.4 +
            address_results['pincode_valid'] * 0.3 +
            address_results['city_state_match'] * 0.2 +
            address_results['coordinates_match'] * 0.1
        )
        address_results['verification_score'] = verification_points

        return address_results
    except Exception as e:
        logger.error(f"Error verifying address: {str(e)}")
        address_results['issues'].append(str(e))
        return address_results

def perform_cross_validation(data):
    try:
        cross_checks = []

        # Check bedroom count consistency
        try:
            bedrooms = int(data['bedrooms']) if data['bedrooms'] else 0
            desc_bhk = re.findall(r'(\d+)\s*bhk', data['description'].lower())
            if desc_bhk and int(desc_bhk[0]) != bedrooms:
                cross_checks.append({
                    'check': 'bedroom_count',
                    'status': 'inconsistent',
                    'message': f"Description mentions {desc_bhk[0]} BHK, form says {bedrooms}"
                })
            else:
                cross_checks.append({
                    'check': 'bedroom_count',
                    'status': 'consistent',
                    'message': f"Bedrooms: {bedrooms}"
                })
        except:
            cross_checks.append({
                'check': 'bedroom_count',
                'status': 'invalid',
                'message': 'Invalid bedroom data'
            })

        # Check room count consistency
        try:
            bedrooms = int(data['bedrooms']) if data['bedrooms'] else 0
            bathrooms = float(data['bathrooms']) if data['bathrooms'] else 0
            total_rooms = int(data['total_rooms']) if data['total_rooms'] else 0
            
            # More thorough room count validation
            if total_rooms > 0:
                if total_rooms < bedrooms + bathrooms:
                    cross_checks.append({
                        'check': 'room_count',
                        'status': 'inconsistent',
                        'message': f"Total rooms ({total_rooms}) less than bedrooms ({bedrooms}) + bathrooms ({bathrooms})"
                    })
                elif total_rooms > bedrooms + bathrooms + 5:  # Allow for some extra rooms
                    cross_checks.append({
                        'check': 'room_count',
                        'status': 'suspicious',
                        'message': f"Total rooms ({total_rooms}) seems unusually high compared to bedrooms ({bedrooms}) + bathrooms ({bathrooms})"
                    })
                else:
                    cross_checks.append({
                        'check': 'room_count',
                        'status': 'consistent',
                        'message': f"Rooms consistent: {total_rooms} total, {bedrooms} bedrooms, {bathrooms} bathrooms"
                    })
            else:
                cross_checks.append({
                    'check': 'room_count',
                    'status': 'missing',
                    'message': 'Total room count not provided'
                })
        except:
            cross_checks.append({
                'check': 'room_count',
                'status': 'invalid',
                'message': 'Invalid room count data'
            })

        # Check year built consistency
        try:
            year_built = int(data['year_built']) if data['year_built'] else 0
            current_year = datetime.now().year
            
            if year_built > 0:
                if year_built > current_year:
                    cross_checks.append({
                        'check': 'year_built',
                        'status': 'invalid',
                        'message': f"Year built ({year_built}) is in the future"
                    })
                elif year_built < 1800:
                    cross_checks.append({
                        'check': 'year_built',
                        'status': 'suspicious',
                        'message': f"Year built ({year_built}) seems unusually old"
                    })
                elif current_year - year_built > 200:
                    cross_checks.append({
                        'check': 'year_built',
                        'status': 'suspicious',
                        'message': f"Property age ({current_year - year_built} years) seems unusually old"
                    })
                else:
                    cross_checks.append({
                        'check': 'year_built',
                        'status': 'reasonable',
                        'message': f"Year built reasonable: {year_built}"
                    })
            else:
                cross_checks.append({
                    'check': 'year_built',
                    'status': 'missing',
                    'message': 'Year built not provided'
                })
        except:
            cross_checks.append({
                'check': 'year_built',
                'status': 'invalid',
                'message': 'Invalid year built data'
            })

        # Check square footage consistency
        try:
            sq_ft = float(re.sub(r'[^\d.]', '', data['sq_ft'])) if data['sq_ft'] else 0
            bedrooms = int(data['bedrooms']) if data['bedrooms'] else 0
            
            if sq_ft > 0 and bedrooms > 0:
                sq_ft_per_bedroom = sq_ft / bedrooms
                
                if sq_ft_per_bedroom < 50:  # Unusually small per bedroom
                    cross_checks.append({
                        'check': 'sq_ft_per_bedroom',
                        'status': 'suspicious',
                        'message': f"Square footage per bedroom ({sq_ft_per_bedroom:.1f} sq.ft.) seems unusually small"
                    })
                elif sq_ft_per_bedroom > 1000:  # Unusually large per bedroom
                    cross_checks.append({
                        'check': 'sq_ft_per_bedroom',
                        'status': 'suspicious',
                        'message': f"Square footage per bedroom ({sq_ft_per_bedroom:.1f} sq.ft.) seems unusually large"
                    })
                else:
                    cross_checks.append({
                        'check': 'sq_ft_per_bedroom',
                        'status': 'reasonable',
                        'message': f"Square footage per bedroom ({sq_ft_per_bedroom:.1f} sq.ft.) is reasonable"
                    })
            elif sq_ft > 0:
                cross_checks.append({
                    'check': 'sq_ft',
                    'status': 'incomplete',
                    'message': f"Square footage provided: {sq_ft} sq.ft., but bedroom count missing"
                })
            elif bedrooms > 0:
                cross_checks.append({
                    'check': 'sq_ft',
                    'status': 'missing',
                    'message': f"Square footage not provided, but {bedrooms} bedrooms listed"
                })
            else:
                cross_checks.append({
                    'check': 'sq_ft',
                    'status': 'missing',
                    'message': 'Square footage not provided'
                })
        except:
            cross_checks.append({
                'check': 'sq_ft',
                'status': 'invalid',
                'message': 'Invalid square footage data'
            })

        # Check price per square foot
        try:
            market_value = float(data['market_value'].replace('₹', '').replace(',', '')) if data['market_value'] else 0
            sq_ft = float(re.sub(r'[^\d.]', '', data['sq_ft'])) if data['sq_ft'] else 0
            
            if market_value > 0 and sq_ft > 0:
                price_per_sqft = market_value / sq_ft
                
                # Check for suspiciously low price per sq ft
                if price_per_sqft < 100:
                    cross_checks.append({
                        'check': 'price_per_sqft',
                        'status': 'suspiciously low',
                        'message': f"Price per sq.ft.: ₹{price_per_sqft:.2f} is suspiciously low"
                    })
                # Check for suspiciously high price per sq ft
                elif price_per_sqft > 50000:
                    cross_checks.append({
                        'check': 'price_per_sqft',
                        'status': 'suspiciously high',
                        'message': f"Price per sq.ft.: ₹{price_per_sqft:.2f} is suspiciously high"
                    })
                else:
                    cross_checks.append({
                        'check': 'price_per_sqft',
                        'status': 'reasonable',
                        'message': f"Price per sq.ft.: ₹{price_per_sqft:.2f} is reasonable"
                    })
            elif market_value > 0:
                cross_checks.append({
                    'check': 'price_per_sqft',
                    'status': 'incomplete',
                    'message': f"Market value provided: ₹{market_value:,.2f}, but square footage missing"
                })
            elif sq_ft > 0:
                cross_checks.append({
                    'check': 'price_per_sqft',
                    'status': 'incomplete',
                    'message': f"Square footage provided: {sq_ft} sq.ft., but market value missing"
                })
            else:
                cross_checks.append({
                    'check': 'price_per_sqft',
                    'status': 'missing',
                    'message': 'Price per sq.ft. cannot be calculated (missing data)'
                })
        except:
            cross_checks.append({
                'check': 'price_per_sqft',
                'status': 'invalid',
                'message': 'Invalid price per sq.ft. data'
            })

        # Check location consistency
        try:
            latitude = float(data['latitude']) if data['latitude'] else 0
            longitude = float(data['longitude']) if data['longitude'] else 0
            address = data['address'].lower() if data['address'] else ''
            city = data['city'].lower() if data['city'] else ''
            state = data['state'].lower() if data['state'] else ''
            country = data['country'].lower() if data['country'] else 'india'
            
            # Check if coordinates are within India
            if latitude != 0 and longitude != 0:
                if 6.5 <= latitude <= 35.5 and 68.1 <= longitude <= 97.4:
                    cross_checks.append({
                        'check': 'coordinates',
                        'status': 'valid',
                        'message': 'Coordinates within India'
                    })
                else:
                    cross_checks.append({
                        'check': 'coordinates',
                        'status': 'invalid',
                        'message': 'Coordinates outside India'
                    })
            else:
                cross_checks.append({
                    'check': 'coordinates',
                    'status': 'missing',
                    'message': 'Coordinates not provided'
                })
            
            # Check if address contains city and state
            if address and city and state:
                if city in address and state in address:
                    cross_checks.append({
                        'check': 'address_consistency',
                        'status': 'consistent',
                        'message': 'Address contains city and state'
                    })
                else:
                    cross_checks.append({
                        'check': 'address_consistency',
                        'status': 'inconsistent',
                        'message': 'Address does not contain city or state'
                    })
            else:
                cross_checks.append({
                    'check': 'address_consistency',
                    'status': 'incomplete',
                    'message': 'Address consistency check incomplete (missing data)'
                })
        except:
            cross_checks.append({
                'check': 'location',
                'status': 'invalid',
                'message': 'Invalid location data'
            })

        # Check property type consistency
        try:
            property_type = data['property_type'].lower() if data['property_type'] else ''
            description = data['description'].lower() if data['description'] else ''
            
            if property_type and description:
                property_types = ['apartment', 'house', 'condo', 'townhouse', 'villa', 'land', 'commercial']
                found_types = [pt for pt in property_types if pt in description]
                
                if found_types and property_type not in found_types:
                    cross_checks.append({
                        'check': 'property_type',
                        'status': 'inconsistent',
                        'message': f"Description mentions {', '.join(found_types)}, but property type is {property_type}"
                    })
                else:
                    cross_checks.append({
                        'check': 'property_type',
                        'status': 'consistent',
                        'message': f"Property type consistent: {property_type}"
                    })
            else:
                cross_checks.append({
                    'check': 'property_type',
                    'status': 'incomplete',
                    'message': 'Property type consistency check incomplete (missing data)'
                })
        except:
            cross_checks.append({
                'check': 'property_type',
                'status': 'invalid',
                'message': 'Invalid property type data'
            })

        # Check for suspiciously low market value
        try:
            market_value = float(data['market_value'].replace('₹', '').replace(',', '')) if data['market_value'] else 0
            property_type = data['property_type'].lower() if data['property_type'] else ''
            
            if market_value > 0 and property_type:
                # Define minimum reasonable values for different property types
                min_values = {
                    'apartment': 500000,
                    'house': 1000000,
                    'condo': 800000,
                    'townhouse': 900000,
                    'villa': 2000000,
                    'land': 300000,
                    'commercial': 2000000
                }
                
                min_value = min_values.get(property_type, 500000)
                
                if market_value < min_value:
                    cross_checks.append({
                        'check': 'market_value',
                        'status': 'suspiciously low',
                        'message': f"Market value (₹{market_value:,.2f}) seems suspiciously low for a {property_type}"
                    })
                else:
                    cross_checks.append({
                        'check': 'market_value',
                        'status': 'reasonable',
                        'message': f"Market value (₹{market_value:,.2f}) is reasonable for a {property_type}"
                    })
            elif market_value > 0:
                cross_checks.append({
                    'check': 'market_value',
                    'status': 'incomplete',
                    'message': f"Market value provided: ₹{market_value:,.2f}, but property type missing"
                })
            else:
                cross_checks.append({
                    'check': 'market_value',
                    'status': 'missing',
                    'message': 'Market value not provided'
                })
        except:
            cross_checks.append({
                'check': 'market_value',
                'status': 'invalid',
                'message': 'Invalid market value data'
            })

        return cross_checks
    except Exception as e:
        logger.error(f"Error performing cross validation: {str(e)}")
        return [{
            'check': 'cross_validation',
            'status': 'error',
            'message': f'Error performing cross validation: {str(e)}'
        }]

def analyze_location(data):
    try:
        classifier = load_model("zero-shot-classification", "facebook/bart-large-mnli")
        
        # Create a detailed location text for analysis
        location_text = ' '.join(filter(None, [
            data['address'], data['city'], data['state'], data['country'],
            data['zip'], f"Lat: {data['latitude']}", f"Long: {data['longitude']}",
            data['nearby_landmarks']
        ]))
        
        # Classify location completeness
        categories = ["complete", "partial", "minimal", "missing"]
        result = classifier(location_text, categories)

        # Verify location quality
        location_quality = "unknown"
        if data['city'] and data['state']:
            for attempt in range(3):
                try:
                    location = geocoder.geocode(f"{data['city']}, {data['state']}, India")
                    if location:
                        location_quality = "verified"
                        break
                    time.sleep(1)
                except:
                    time.sleep(1)
            else:
                location_quality = "unverified"

        # Check coordinates
        coord_check = "missing"
        if data['latitude'] and data['longitude']:
            try:
                lat, lng = float(data['latitude']), float(data['longitude'])
                if 6.5 <= lat <= 37.5 and 68.0 <= lng <= 97.5:
                    coord_check = "in_india"
                    # Further validate coordinates against known Indian cities
                    if any(city in data['city'].lower() for city in ["mumbai", "delhi", "bangalore", "hyderabad", "chennai", "kolkata", "pune"]):
                        coord_check = "in_metro_city"
                else:
                    coord_check = "outside_india"
            except:
                coord_check = "invalid"

        # Calculate location completeness with weighted scoring
        completeness = calculate_location_completeness(data)
        
        # Analyze landmarks
        landmarks_analysis = {
            'provided': bool(data['nearby_landmarks']),
            'count': len(data['nearby_landmarks'].split(',')) if data['nearby_landmarks'] else 0,
            'types': []
        }
        
        if data['nearby_landmarks']:
            landmark_types = {
                'transport': ['station', 'metro', 'bus', 'railway', 'airport'],
                'education': ['school', 'college', 'university', 'institute'],
                'healthcare': ['hospital', 'clinic', 'medical'],
                'shopping': ['mall', 'market', 'shop', 'store'],
                'entertainment': ['park', 'garden', 'theater', 'cinema'],
                'business': ['office', 'business', 'corporate']
            }
            
            landmarks = data['nearby_landmarks'].lower().split(',')
            for landmark in landmarks:
                for type_name, keywords in landmark_types.items():
                    if any(keyword in landmark for keyword in keywords):
                        if type_name not in landmarks_analysis['types']:
                            landmarks_analysis['types'].append(type_name)

        # Determine location assessment
        assessment = "complete" if completeness >= 80 else "partial" if completeness >= 50 else "minimal"
        
        # Add city tier information
        city_tier = "unknown"
        if data['city']:
            city_lower = data['city'].lower()
            if any(city in city_lower for city in ["mumbai", "delhi", "bangalore", "hyderabad", "chennai", "kolkata", "pune"]):
                city_tier = "metro"
            elif any(city in city_lower for city in ["ahmedabad", "jaipur", "surat", "lucknow", "kanpur", "nagpur", "indore", "thane", "bhopal", "visakhapatnam"]):
                city_tier = "tier2"
            else:
                city_tier = "tier3"

        return {
            'assessment': assessment,
            'confidence': float(result['scores'][0]),
            'coordinates_check': coord_check,
            'landmarks_analysis': landmarks_analysis,
            'completeness_score': completeness,
            'location_quality': location_quality,
            'city_tier': city_tier,
            'formatted_address': f"{data['address']}, {data['city']}, {data['state']}, India - {data['zip']}",
            'verification_status': "verified" if location_quality == "verified" and coord_check in ["in_india", "in_metro_city"] else "unverified"
        }
    except Exception as e:
        logger.error(f"Error analyzing location: {str(e)}")
        return {
            'assessment': 'error',
            'confidence': 0.0,
            'coordinates_check': 'error',
            'landmarks_analysis': {'provided': False, 'count': 0, 'types': []},
            'completeness_score': 0,
            'location_quality': 'error',
            'city_tier': 'unknown',
            'formatted_address': '',
            'verification_status': 'error'
        }

def calculate_location_completeness(data):
    # Define weights for different fields
    weights = {
        'address': 0.25,
        'city': 0.20,
        'state': 0.15,
        'country': 0.05,
        'zip': 0.10,
        'latitude': 0.10,
        'longitude': 0.10,
        'nearby_landmarks': 0.05
    }
    
    # Calculate weighted score
    score = 0
    for field, weight in weights.items():
        if data[field]:
            score += weight
    
    return int(score * 100)

def analyze_price(data):
    try:
        price_str = data['market_value'].replace('$', '').replace(',', '').strip()
        price = float(price_str) if price_str else 0
        sq_ft = float(re.sub(r'[^\d.]', '', data['sq_ft'])) if data['sq_ft'] else 0
        price_per_sqft = price / sq_ft if sq_ft else 0

        if not price:
            return {
                'assessment': 'no price',
                'confidence': 0.0,
                'price': 0,
                'formatted_price': '₹0',
                'price_per_sqft': 0,
                'formatted_price_per_sqft': '₹0',
                'price_range': 'unknown',
                'location_price_assessment': 'cannot assess',
                'has_price': False,
                'market_trends': {},
                'price_factors': {},
                'risk_indicators': []
            }

        # Use a more sophisticated model for price analysis
        classifier = load_model("zero-shot-classification", "facebook/bart-large-mnli")
        
        # Create a detailed context for price analysis
        price_context = f"""
        Property Type: {data.get('property_type', '')}
        Location: {data.get('city', '')}, {data.get('state', '')}
        Size: {sq_ft} sq.ft.
        Price: ₹{price:,.2f}
        Price per sq.ft.: ₹{price_per_sqft:,.2f}
        Property Status: {data.get('status', '')}
        Year Built: {data.get('year_built', '')}
        Bedrooms: {data.get('bedrooms', '')}
        Bathrooms: {data.get('bathrooms', '')}
        Amenities: {data.get('amenities', '')}
        """
        
        # Enhanced price categories with more specific indicators
        price_categories = [
            "reasonable market price",
            "suspiciously low price",
            "suspiciously high price",
            "average market price",
            "luxury property price",
            "budget property price",
            "premium property price",
            "mid-range property price",
            "overpriced for location",
            "underpriced for location",
            "price matches amenities",
            "price matches property age",
            "price matches location value",
            "price matches property condition",
            "price matches market trends"
        ]
        
        # Analyze price with multiple aspects
        price_result = classifier(price_context, price_categories, multi_label=True)
        
        # Get top classifications with enhanced confidence calculation
        top_classifications = []
        for label, score in zip(price_result['labels'][:5], price_result['scores'][:5]):
            if score > 0.25:  # Lower threshold for better sensitivity
                top_classifications.append({
                    'classification': label,
                    'confidence': float(score)
                })
        
        # Determine price range based on AI classification and market data
        price_range = 'unknown'
        if top_classifications:
            primary_class = top_classifications[0]['classification']
            if 'luxury' in primary_class:
                price_range = 'luxury'
            elif 'premium' in primary_class:
                price_range = 'premium'
            elif 'mid-range' in primary_class:
                price_range = 'mid_range'
            elif 'budget' in primary_class:
                price_range = 'budget'
        
        # Enhanced location-specific price assessment
        location_assessment = "unknown"
        market_trends = {}
        if data.get('city') and price_per_sqft:
            city_lower = data['city'].lower()
        metro_cities = ["mumbai", "delhi", "bangalore", "hyderabad", "chennai", "kolkata", "pune"]
            
            # Define price ranges for different city tiers
        if any(city in city_lower for city in metro_cities):
                market_trends = {
                    'city_tier': 'metro',
                    'avg_price_range': {
                        'min': 5000,
                        'max': 30000,
                        'trend': 'stable'
                    },
                    'price_per_sqft': {
                        'current': price_per_sqft,
                        'market_avg': 15000,
                        'deviation': abs(price_per_sqft - 15000) / 15000 * 100
                    }
                }
                location_assessment = (
                    "reasonable" if 5000 <= price_per_sqft <= 30000 else
                    "suspiciously low" if price_per_sqft < 5000 else
                    "suspiciously high"
                )
        else:
                market_trends = {
                    'city_tier': 'non-metro',
                    'avg_price_range': {
                        'min': 1500,
                        'max': 15000,
                        'trend': 'stable'
                    },
                    'price_per_sqft': {
                        'current': price_per_sqft,
                        'market_avg': 7500,
                        'deviation': abs(price_per_sqft - 7500) / 7500 * 100
                    }
                }
                location_assessment = (
                    "reasonable" if 1500 <= price_per_sqft <= 15000 else
                    "suspiciously low" if price_per_sqft < 1500 else
                    "suspiciously high"
                )
        
        # Enhanced price analysis factors
        price_factors = {}
        risk_indicators = []
        
        # Property age factor
        try:
            year_built = int(data.get('year_built', 0))
            current_year = datetime.now().year
            property_age = current_year - year_built
            
            if property_age > 0:
                depreciation_factor = max(0.5, 1 - (property_age * 0.01))  # 1% depreciation per year, min 50%
                price_factors['age_factor'] = {
                    'property_age': property_age,
                    'depreciation_factor': depreciation_factor,
                    'impact': 'high' if property_age > 30 else 'medium' if property_age > 15 else 'low'
                }
        except:
            price_factors['age_factor'] = {'error': 'Invalid year built'}
        
        # Size factor
        if sq_ft > 0:
            size_factor = {
                'size': sq_ft,
                'price_per_sqft': price_per_sqft,
                'efficiency': 'high' if 800 <= sq_ft <= 2000 else 'medium' if 500 <= sq_ft <= 3000 else 'low'
            }
            price_factors['size_factor'] = size_factor
            
            # Add risk indicators based on size
            if sq_ft < 300:
                risk_indicators.append('Unusually small property size')
            elif sq_ft > 10000:
                risk_indicators.append('Unusually large property size')
        
        # Amenities factor
        if data.get('amenities'):
            amenities_list = [a.strip() for a in data['amenities'].split(',')]
            amenities_score = min(1.0, len(amenities_list) * 0.1)  # 10% per amenity, max 100%
            price_factors['amenities_factor'] = {
                'count': len(amenities_list),
                'score': amenities_score,
                'impact': 'high' if amenities_score > 0.7 else 'medium' if amenities_score > 0.4 else 'low'
            }
        
        # Calculate overall confidence with weighted factors
        confidence_weights = {
            'primary_classification': 0.3,
            'location_assessment': 0.25,
            'age_factor': 0.2,
            'size_factor': 0.15,
            'amenities_factor': 0.1
        }
        
        confidence_scores = []
        
        # Primary classification confidence
        if top_classifications:
            confidence_scores.append(price_result['scores'][0] * confidence_weights['primary_classification'])
        
        # Location assessment confidence
        location_confidence = 0.8 if location_assessment == "reasonable" else 0.4
        confidence_scores.append(location_confidence * confidence_weights['location_assessment'])
        
        # Age factor confidence
        if 'age_factor' in price_factors and 'depreciation_factor' in price_factors['age_factor']:
            age_confidence = price_factors['age_factor']['depreciation_factor']
            confidence_scores.append(age_confidence * confidence_weights['age_factor'])
        
        # Size factor confidence
        if 'size_factor' in price_factors:
            size_confidence = 0.8 if price_factors['size_factor']['efficiency'] == 'high' else 0.6
            confidence_scores.append(size_confidence * confidence_weights['size_factor'])
        
        # Amenities factor confidence
        if 'amenities_factor' in price_factors:
            amenities_confidence = price_factors['amenities_factor']['score']
            confidence_scores.append(amenities_confidence * confidence_weights['amenities_factor'])
        
        overall_confidence = sum(confidence_scores) / sum(confidence_weights.values())

        return {
            'assessment': top_classifications[0]['classification'] if top_classifications else 'could not classify',
            'confidence': float(overall_confidence),
            'price': price,
            'formatted_price': f"₹{price:,.0f}",
            'price_per_sqft': price_per_sqft,
            'formatted_price_per_sqft': f"₹{price_per_sqft:,.2f}",
            'price_range': price_range,
            'location_price_assessment': location_assessment,
            'has_price': True,
            'market_trends': market_trends,
            'price_factors': price_factors,
            'risk_indicators': risk_indicators,
            'top_classifications': top_classifications
        }
    except Exception as e:
        logger.error(f"Error analyzing price: {str(e)}")
        return {
            'assessment': 'error',
            'confidence': 0.0,
            'price': 0,
            'formatted_price': '₹0',
            'price_per_sqft': 0,
            'formatted_price_per_sqft': '₹0',
            'price_range': 'unknown',
            'location_price_assessment': 'error',
            'has_price': False,
            'market_trends': {},
            'price_factors': {},
            'risk_indicators': [],
            'top_classifications': []
        }

def analyze_legal_details(legal_text):
    try:
        if not legal_text or len(legal_text.strip()) < 5:
            return {
                'assessment': 'insufficient',
                'confidence': 0.0,
                'summary': 'No legal details provided',
                'completeness_score': 0,
                'potential_issues': False,
                'legal_metrics': {},
                'reasoning': 'No legal details provided for analysis',
                'top_classifications': []
            }

        classifier = load_model("zero-shot-classification", "facebook/bart-large-mnli")
        
        # Enhanced legal categories with more specific indicators
        categories = [
            "comprehensive legal documentation",
            "basic legal documentation",
            "missing critical legal details",
            "potential legal issues",
            "standard property documentation",
            "title verification documents",
            "encumbrance certificates",
            "property tax records",
            "building permits",
            "land use certificates",
            "clear title documentation",
            "property registration documents",
            "ownership transfer documents",
            "legal compliance certificates",
            "property dispute records"
        ]
        
        # Create a more detailed context for analysis
        legal_context = f"""
        Legal Documentation Analysis:
        {legal_text[:1000]}
        
        Key aspects to verify:
        - Title and ownership documentation
        - Property registration status
        - Tax compliance
        - Building permits and approvals
        - Land use compliance
        - Encumbrance status
        - Dispute history
        """
        
        # Analyze legal text with multiple aspects
        legal_result = classifier(legal_context, categories, multi_label=True)
        
        # Get top classifications with confidence scores
        top_classifications = []
        for label, score in zip(legal_result['labels'][:3], legal_result['scores'][:3]):
            if score > 0.3:  # Only include if confidence is above 30%
                top_classifications.append({
                    'classification': label,
                    'confidence': float(score)
                })
        
        # Generate summary using BART
        summary = summarize_text(legal_text[:1000])
        
        # Calculate legal metrics with weighted scoring
        legal_metrics = {
            'completeness': sum(score for label, score in zip(legal_result['labels'], legal_result['scores']) 
                              if label in ['comprehensive legal documentation', 'standard property documentation']),
            'documentation_quality': sum(score for label, score in zip(legal_result['labels'], legal_result['scores']) 
                                      if label in ['title verification documents', 'encumbrance certificates', 'clear title documentation']),
            'compliance': sum(score for label, score in zip(legal_result['labels'], legal_result['scores']) 
                            if label in ['building permits', 'land use certificates', 'legal compliance certificates']),
            'risk_level': sum(score for label, score in zip(legal_result['labels'], legal_result['scores']) 
                            if label in ['missing critical legal details', 'potential legal issues', 'property dispute records'])
        }
        
        # Calculate completeness score with weighted components
        completeness_score = (
            legal_metrics['completeness'] * 0.4 +
            legal_metrics['documentation_quality'] * 0.4 +
            legal_metrics['compliance'] * 0.2
        ) * 100
        
        # Determine if there are potential issues with threshold
        potential_issues = legal_metrics['risk_level'] > 0.3
        
        # Generate detailed reasoning with specific points
        reasoning_parts = []
        
        # Primary assessment
        if top_classifications:
            primary_class = top_classifications[0]['classification']
            confidence = top_classifications[0]['confidence']
            reasoning_parts.append(f"Primary assessment: {primary_class} (confidence: {confidence:.0%})")
        
        # Documentation completeness
        if legal_metrics['completeness'] > 0.7:
            reasoning_parts.append("Comprehensive legal documentation present")
        elif legal_metrics['completeness'] > 0.4:
            reasoning_parts.append("Basic legal documentation present")
        else:
            reasoning_parts.append("Insufficient legal documentation")
        
        # Documentation quality
        if legal_metrics['documentation_quality'] > 0.6:
            reasoning_parts.append("Quality documentation verified (title, encumbrance)")
        elif legal_metrics['documentation_quality'] > 0.3:
            reasoning_parts.append("Basic documentation quality verified")
        
        # Compliance status
        if legal_metrics['compliance'] > 0.6:
            reasoning_parts.append("Full compliance documentation present")
        elif legal_metrics['compliance'] > 0.3:
            reasoning_parts.append("Partial compliance documentation present")
        
        # Risk assessment
        if potential_issues:
            if legal_metrics['risk_level'] > 0.6:
                reasoning_parts.append("High risk: Multiple potential legal issues detected")
            else:
                reasoning_parts.append("Moderate risk: Some potential legal issues detected")
        else:
            reasoning_parts.append("No significant legal issues detected")
        
        # Calculate overall confidence
        overall_confidence = min(1.0, (
            legal_metrics['completeness'] * 0.4 +
            legal_metrics['documentation_quality'] * 0.4 +
            (1 - legal_metrics['risk_level']) * 0.2
        ))

        return {
            'assessment': top_classifications[0]['classification'] if top_classifications else 'could not assess',
            'confidence': float(overall_confidence),
            'summary': summary,
            'completeness_score': int(completeness_score),
            'potential_issues': potential_issues,
            'legal_metrics': legal_metrics,
            'reasoning': '. '.join(reasoning_parts),
            'top_classifications': top_classifications
        }
    except Exception as e:
        logger.error(f"Error analyzing legal details: {str(e)}")
        return {
            'assessment': 'could not assess',
            'confidence': 0.0,
            'summary': 'Error analyzing legal details',
            'completeness_score': 0,
            'potential_issues': False,
            'legal_metrics': {},
            'reasoning': 'Technical error occurred during analysis',
            'top_classifications': []
        }

def verify_property_specs(data):
    """
    Verify property specifications for reasonableness and consistency.
    This function checks if the provided property details are within reasonable ranges
    for the Indian real estate market.
    """
    specs_verification = {
        'is_valid': True,
        'bedrooms_reasonable': True,
        'bathrooms_reasonable': True,
        'total_rooms_reasonable': True,
        'year_built_reasonable': True,
        'parking_reasonable': True,
        'sq_ft_reasonable': True,
        'market_value_reasonable': True,
        'issues': []
    }
    
    try:
        # Validate property type
        valid_property_types = [
            'Apartment', 'House', 'Villa', 'Independent House', 'Independent Villa',
            'Studio', 'Commercial', 'Office', 'Shop', 'Warehouse', 'Industrial'
        ]
        
        if 'property_type' not in data or data['property_type'] not in valid_property_types:
            specs_verification['is_valid'] = False
            specs_verification['issues'].append(f"Invalid property type: {data.get('property_type', 'Not specified')}")

        # Validate bedrooms
        if 'bedrooms' in data:
            try:
                bedrooms = int(data['bedrooms'])
                if data['property_type'] in ['Apartment', 'Studio']:
                    if bedrooms > 5 or bedrooms < 0:
                        specs_verification['bedrooms_reasonable'] = False
                        specs_verification['issues'].append(f"Invalid number of bedrooms for {data['property_type']}: {bedrooms}. Should be between 0 and 5.")
                elif data['property_type'] in ['House', 'Villa', 'Independent House', 'Independent Villa']:
                    if bedrooms > 8 or bedrooms < 0:
                        specs_verification['bedrooms_reasonable'] = False
                        specs_verification['issues'].append(f"Invalid number of bedrooms for {data['property_type']}: {bedrooms}. Should be between 0 and 8.")
                elif data['property_type'] in ['Commercial', 'Office', 'Shop', 'Warehouse', 'Industrial']:
                    if bedrooms > 0:
                        specs_verification['bedrooms_reasonable'] = False
                        specs_verification['issues'].append(f"Commercial properties typically don't have bedrooms: {bedrooms}")
            except ValueError:
                specs_verification['bedrooms_reasonable'] = False
                specs_verification['issues'].append("Invalid bedrooms data: must be a number")

        # Validate bathrooms
        if 'bathrooms' in data:
            try:
                bathrooms = float(data['bathrooms'])
                if data['property_type'] in ['Apartment', 'Studio']:
                    if bathrooms > 4 or bathrooms < 0:
                        specs_verification['bathrooms_reasonable'] = False
                        specs_verification['issues'].append(f"Invalid number of bathrooms for {data['property_type']}: {bathrooms}. Should be between 0 and 4.")
                elif data['property_type'] in ['House', 'Villa', 'Independent House', 'Independent Villa']:
                    if bathrooms > 6 or bathrooms < 0:
                        specs_verification['bathrooms_reasonable'] = False
                        specs_verification['issues'].append(f"Invalid number of bathrooms for {data['property_type']}: {bathrooms}. Should be between 0 and 6.")
                elif data['property_type'] in ['Commercial', 'Office', 'Shop', 'Warehouse', 'Industrial']:
                    if bathrooms > 0:
                        specs_verification['bathrooms_reasonable'] = False
                        specs_verification['issues'].append(f"Commercial properties typically don't have bathrooms: {bathrooms}")
            except ValueError:
                specs_verification['bathrooms_reasonable'] = False
                specs_verification['issues'].append("Invalid bathrooms data: must be a number")

        # Validate total rooms
        if 'total_rooms' in data:
            try:
                total_rooms = int(data['total_rooms'])
                if total_rooms < 0:
                    specs_verification['total_rooms_reasonable'] = False
                    specs_verification['issues'].append(f"Invalid total rooms: {total_rooms}. Cannot be negative.")
                elif 'bedrooms' in data and 'bathrooms' in data:
                    try:
                        bedrooms = int(data['bedrooms'])
                        bathrooms = int(float(data['bathrooms']))
                        if total_rooms < (bedrooms + bathrooms):
                            specs_verification['total_rooms_reasonable'] = False
                            specs_verification['issues'].append(f"Total rooms ({total_rooms}) is less than bedrooms + bathrooms ({bedrooms + bathrooms})")
                    except ValueError:
                        pass
            except ValueError:
                specs_verification['total_rooms_reasonable'] = False
                specs_verification['issues'].append("Invalid total rooms data: must be a number")

        # Validate year built
        if 'year_built' in data:
            try:
                year_built = int(data['year_built'])
                current_year = datetime.now().year
                if year_built > current_year:
                    specs_verification['year_built_reasonable'] = False
                    specs_verification['issues'].append(f"Year built ({year_built}) is in the future")
                elif year_built < 1800:
                    specs_verification['year_built_reasonable'] = False
                    specs_verification['issues'].append(f"Year built ({year_built}) seems unreasonably old")
            except ValueError:
                specs_verification['year_built_reasonable'] = False
                specs_verification['issues'].append("Invalid year built data: must be a number")

        # Validate parking
        if 'parking' in data:
            try:
                parking = int(data['parking'])
                if data['property_type'] in ['Apartment', 'Studio']:
                    if parking > 2 or parking < 0:
                        specs_verification['parking_reasonable'] = False
                        specs_verification['issues'].append(f"Invalid parking spaces for {data['property_type']}: {parking}. Should be between 0 and 2.")
                elif data['property_type'] in ['House', 'Villa', 'Independent House', 'Independent Villa']:
                    if parking > 4 or parking < 0:
                        specs_verification['parking_reasonable'] = False
                        specs_verification['issues'].append(f"Invalid parking spaces for {data['property_type']}: {parking}. Should be between 0 and 4.")
                elif data['property_type'] in ['Commercial', 'Office', 'Shop', 'Warehouse', 'Industrial']:
                    if parking < 0:
                        specs_verification['parking_reasonable'] = False
                        specs_verification['issues'].append(f"Invalid parking spaces: {parking}. Cannot be negative.")
            except ValueError:
                specs_verification['parking_reasonable'] = False
                specs_verification['issues'].append("Invalid parking data: must be a number")

        # Validate square footage
        if 'sq_ft' in data:
            try:
                sq_ft = float(data['sq_ft'].replace(',', ''))
                if sq_ft <= 0:
                    specs_verification['sq_ft_reasonable'] = False
                    specs_verification['issues'].append(f"Invalid square footage: {sq_ft}. Must be greater than 0.")
                else:
                    if data['property_type'] in ['Apartment', 'Studio']:
                        if sq_ft > 5000:
                            specs_verification['sq_ft_reasonable'] = False
                            specs_verification['issues'].append(f"Square footage ({sq_ft}) seems unreasonably high for {data['property_type']}")
                        elif sq_ft < 200:
                            specs_verification['sq_ft_reasonable'] = False
                            specs_verification['issues'].append(f"Square footage ({sq_ft}) seems unreasonably low for {data['property_type']}")
                    elif data['property_type'] in ['House', 'Villa', 'Independent House', 'Independent Villa']:
                        if sq_ft > 10000:
                            specs_verification['sq_ft_reasonable'] = False
                            specs_verification['issues'].append(f"Square footage ({sq_ft}) seems unreasonably high for {data['property_type']}")
                        elif sq_ft < 500:
                            specs_verification['sq_ft_reasonable'] = False
                            specs_verification['issues'].append(f"Square footage ({sq_ft}) seems unreasonably low for {data['property_type']}")
            except ValueError:
                specs_verification['sq_ft_reasonable'] = False
                specs_verification['issues'].append("Invalid square footage data: must be a number")

        # Validate market value
        if 'market_value' in data:
            try:
                market_value = float(data['market_value'].replace(',', '').replace('₹', '').strip())
                if market_value <= 0:
                    specs_verification['market_value_reasonable'] = False
                    specs_verification['issues'].append(f"Invalid market value: {market_value}. Must be greater than 0.")
                else:
                    if data['property_type'] in ['Apartment', 'Studio']:
                        if market_value > 500000000:  # 5 crore limit for apartments
                            specs_verification['market_value_reasonable'] = False
                            specs_verification['issues'].append(f"Market value (₹{market_value:,.2f}) seems unreasonably high for {data['property_type']}")
                        elif market_value < 500000:  # 5 lakh minimum
                            specs_verification['market_value_reasonable'] = False
                            specs_verification['issues'].append(f"Market value (₹{market_value:,.2f}) seems unreasonably low for {data['property_type']}")
                    elif data['property_type'] in ['House', 'Villa', 'Independent House', 'Independent Villa']:
                        if market_value > 2000000000:  # 20 crore limit for houses
                            specs_verification['market_value_reasonable'] = False
                            specs_verification['issues'].append(f"Market value (₹{market_value:,.2f}) seems unreasonably high for {data['property_type']}")
                        elif market_value < 1000000:  # 10 lakh minimum
                            specs_verification['market_value_reasonable'] = False
                            specs_verification['issues'].append(f"Market value (₹{market_value:,.2f}) seems unreasonably low for {data['property_type']}")
                    elif data['property_type'] in ['Commercial', 'Office', 'Shop']:
                        if market_value < 2000000:  # 20 lakh minimum
                            specs_verification['market_value_reasonable'] = False
                            specs_verification['issues'].append(f"Market value (₹{market_value:,.2f}) seems unreasonably low for {data['property_type']}")
                    elif data['property_type'] in ['Warehouse', 'Industrial']:
                        if market_value < 5000000:  # 50 lakh minimum
                            specs_verification['market_value_reasonable'] = False
                            specs_verification['issues'].append(f"Market value (₹{market_value:,.2f}) seems unreasonably low for {data['property_type']}")
                    
                    # Check price per square foot
                    if 'sq_ft' in data and float(data['sq_ft'].replace(',', '')) > 0:
                        try:
                            sq_ft = float(data['sq_ft'].replace(',', ''))
                            price_per_sqft = market_value / sq_ft
                            
                            if data['property_type'] in ['Apartment', 'Studio']:
                                if price_per_sqft < 1000:  # Less than ₹1000 per sq ft
                                    specs_verification['market_value_reasonable'] = False
                                    specs_verification['issues'].append(f"Price per sq ft (₹{price_per_sqft:,.2f}) seems unreasonably low for {data['property_type']}")
                                elif price_per_sqft > 50000:  # More than ₹50k per sq ft
                                    specs_verification['market_value_reasonable'] = False
                                    specs_verification['issues'].append(f"Price per sq ft (₹{price_per_sqft:,.2f}) seems unreasonably high for {data['property_type']}")
                            elif data['property_type'] in ['House', 'Villa', 'Independent House', 'Independent Villa']:
                                if price_per_sqft < 500:  # Less than ₹500 per sq ft
                                    specs_verification['market_value_reasonable'] = False
                                    specs_verification['issues'].append(f"Price per sq ft (₹{price_per_sqft:,.2f}) seems unreasonably low for {data['property_type']}")
                                elif price_per_sqft > 100000:  # More than ₹1 lakh per sq ft
                                    specs_verification['market_value_reasonable'] = False
                                    specs_verification['issues'].append(f"Price per sq ft (₹{price_per_sqft:,.2f}) seems unreasonably high for {data['property_type']}")
                        except ValueError:
                            pass
            except ValueError:
                specs_verification['market_value_reasonable'] = False
                specs_verification['issues'].append("Invalid market value data: must be a number")

        # Calculate verification score
        valid_checks = sum([
            specs_verification['bedrooms_reasonable'],
            specs_verification['bathrooms_reasonable'],
            specs_verification['total_rooms_reasonable'],
            specs_verification['year_built_reasonable'],
            specs_verification['parking_reasonable'],
            specs_verification['sq_ft_reasonable'],
            specs_verification['market_value_reasonable']
        ])
        
        total_checks = 7
        specs_verification['verification_score'] = (valid_checks / total_checks) * 100

        # Overall validity
        specs_verification['is_valid'] = all([
            specs_verification['bedrooms_reasonable'],
            specs_verification['bathrooms_reasonable'],
            specs_verification['total_rooms_reasonable'],
            specs_verification['year_built_reasonable'],
            specs_verification['parking_reasonable'],
            specs_verification['sq_ft_reasonable'],
            specs_verification['market_value_reasonable']
        ])
        
    except Exception as e:
        logger.error(f"Error in property specs verification: {str(e)}")
        specs_verification['is_valid'] = False
        specs_verification['issues'].append(f"Error in verification: {str(e)}")
    
    return specs_verification

def analyze_market_value(data):
    """
    Analyzes the market value of a property based on its specifications and location
    for the Indian real estate market.
    """
    specs_verification = {
        'is_valid': True,
        'bedrooms_reasonable': True,
        'bathrooms_reasonable': True,
        'total_rooms_reasonable': True,
        'parking_reasonable': True,
        'sq_ft_reasonable': True,
        'market_value_reasonable': True,
        'year_built_reasonable': True,  # Added missing field
        'issues': []
    }
    
    try:
        # Validate property type
        valid_property_types = [
            'Apartment', 'House', 'Villa', 'Independent House', 'Independent Villa',
            'Studio', 'Commercial', 'Office', 'Shop', 'Warehouse', 'Industrial'
        ]
        
        if 'property_type' not in data or data['property_type'] not in valid_property_types:
            specs_verification['is_valid'] = False
            specs_verification['issues'].append(f"Invalid property type: {data.get('property_type', 'Not specified')}")

        # Validate bedrooms
        if 'bedrooms' in data:
            try:
                bedrooms = int(data['bedrooms'])
                if data['property_type'] in ['Apartment', 'Studio']:
                    if bedrooms > 5 or bedrooms < 0:
                        specs_verification['bedrooms_reasonable'] = False
                        specs_verification['issues'].append(f"Invalid number of bedrooms for {data['property_type']}: {bedrooms}. Should be between 0 and 5.")
                elif data['property_type'] in ['House', 'Villa', 'Independent House', 'Independent Villa']:
                    if bedrooms > 8 or bedrooms < 0:
                        specs_verification['bedrooms_reasonable'] = False
                        specs_verification['issues'].append(f"Invalid number of bedrooms for {data['property_type']}: {bedrooms}. Should be between 0 and 8.")
                elif data['property_type'] in ['Commercial', 'Office', 'Shop', 'Warehouse', 'Industrial']:
                    if bedrooms > 0:
                        specs_verification['bedrooms_reasonable'] = False
                        specs_verification['issues'].append(f"Commercial properties typically don't have bedrooms: {bedrooms}")
            except ValueError:
                specs_verification['bedrooms_reasonable'] = False
                specs_verification['issues'].append("Invalid bedrooms data: must be a number")

        # Validate bathrooms
        if 'bathrooms' in data:
            try:
                bathrooms = float(data['bathrooms'])
                if data['property_type'] in ['Apartment', 'Studio']:
                    if bathrooms > 4 or bathrooms < 0:
                        specs_verification['bathrooms_reasonable'] = False
                        specs_verification['issues'].append(f"Invalid number of bathrooms for {data['property_type']}: {bathrooms}. Should be between 0 and 4.")
                elif data['property_type'] in ['House', 'Villa', 'Independent House', 'Independent Villa']:
                    if bathrooms > 6 or bathrooms < 0:
                        specs_verification['bathrooms_reasonable'] = False
                        specs_verification['issues'].append(f"Invalid number of bathrooms for {data['property_type']}: {bathrooms}. Should be between 0 and 6.")
                elif data['property_type'] in ['Commercial', 'Office', 'Shop', 'Warehouse', 'Industrial']:
                    if bathrooms > 0:
                        specs_verification['bathrooms_reasonable'] = False
                        specs_verification['issues'].append(f"Commercial properties typically don't have bathrooms: {bathrooms}")
            except ValueError:
                specs_verification['bathrooms_reasonable'] = False
                specs_verification['issues'].append("Invalid bathrooms data: must be a number")

        # Validate total rooms
        if 'total_rooms' in data:
            try:
                total_rooms = int(data['total_rooms'])
                if total_rooms < 0:
                    specs_verification['total_rooms_reasonable'] = False
                    specs_verification['issues'].append(f"Invalid total rooms: {total_rooms}. Cannot be negative.")
                elif 'bedrooms' in data and 'bathrooms' in data:
                    try:
                        bedrooms = int(data['bedrooms'])
                        bathrooms = int(float(data['bathrooms']))
                        if total_rooms < (bedrooms + bathrooms):
                            specs_verification['total_rooms_reasonable'] = False
                            specs_verification['issues'].append(f"Total rooms ({total_rooms}) is less than bedrooms + bathrooms ({bedrooms + bathrooms})")
                    except ValueError:
                        pass
            except ValueError:
                specs_verification['total_rooms_reasonable'] = False
                specs_verification['issues'].append("Invalid total rooms data: must be a number")

        # Validate parking
        if 'parking' in data:
            try:
                parking = int(data['parking'])
                if data['property_type'] in ['Apartment', 'Studio']:
                    if parking > 2 or parking < 0:
                        specs_verification['parking_reasonable'] = False
                        specs_verification['issues'].append(f"Invalid parking spaces for {data['property_type']}: {parking}. Should be between 0 and 2.")
                elif data['property_type'] in ['House', 'Villa', 'Independent House', 'Independent Villa']:
                    if parking > 4 or parking < 0:
                        specs_verification['parking_reasonable'] = False
                        specs_verification['issues'].append(f"Invalid parking spaces for {data['property_type']}: {parking}. Should be between 0 and 4.")
                elif data['property_type'] in ['Commercial', 'Office', 'Shop', 'Warehouse', 'Industrial']:
                    if parking < 0:
                        specs_verification['parking_reasonable'] = False
                        specs_verification['issues'].append(f"Invalid parking spaces: {parking}. Cannot be negative.")
            except ValueError:
                specs_verification['parking_reasonable'] = False
                specs_verification['issues'].append("Invalid parking data: must be a number")

        # Validate square footage
        if 'sq_ft' in data:
            try:
                sq_ft = float(data['sq_ft'].replace(',', ''))
                if sq_ft <= 0:
                    specs_verification['sq_ft_reasonable'] = False
                    specs_verification['issues'].append(f"Invalid square footage: {sq_ft}. Must be greater than 0.")
                else:
                    if data['property_type'] in ['Apartment', 'Studio']:
                        if sq_ft > 5000:
                            specs_verification['sq_ft_reasonable'] = False
                            specs_verification['issues'].append(f"Square footage ({sq_ft}) seems unreasonably high for {data['property_type']}")
                        elif sq_ft < 200:
                            specs_verification['sq_ft_reasonable'] = False
                            specs_verification['issues'].append(f"Square footage ({sq_ft}) seems unreasonably low for {data['property_type']}")
                    elif data['property_type'] in ['House', 'Villa', 'Independent House', 'Independent Villa']:
                        if sq_ft > 10000:
                            specs_verification['sq_ft_reasonable'] = False
                            specs_verification['issues'].append(f"Square footage ({sq_ft}) seems unreasonably high for {data['property_type']}")
                        elif sq_ft < 500:
                            specs_verification['sq_ft_reasonable'] = False
                            specs_verification['issues'].append(f"Square footage ({sq_ft}) seems unreasonably low for {data['property_type']}")
            except ValueError:
                specs_verification['sq_ft_reasonable'] = False
                specs_verification['issues'].append("Invalid square footage data: must be a number")

        # Validate market value
        if 'market_value' in data:
            try:
                market_value = float(data['market_value'].replace(',', '').replace('₹', '').strip())
                if market_value <= 0:
                    specs_verification['market_value_reasonable'] = False
                    specs_verification['issues'].append(f"Invalid market value: {market_value}. Must be greater than 0.")
                else:
                    if data['property_type'] in ['Apartment', 'Studio']:
                        if market_value > 500000000:  # 5 crore limit for apartments
                            specs_verification['market_value_reasonable'] = False
                            specs_verification['issues'].append(f"Market value (₹{market_value:,.2f}) seems unreasonably high for {data['property_type']}")
                        elif market_value < 500000:  # 5 lakh minimum
                            specs_verification['market_value_reasonable'] = False
                            specs_verification['issues'].append(f"Market value (₹{market_value:,.2f}) seems unreasonably low for {data['property_type']}")
                    elif data['property_type'] in ['House', 'Villa', 'Independent House', 'Independent Villa']:
                        if market_value > 2000000000:  # 20 crore limit for houses
                            specs_verification['market_value_reasonable'] = False
                            specs_verification['issues'].append(f"Market value (₹{market_value:,.2f}) seems unreasonably high for {data['property_type']}")
                        elif market_value < 1000000:  # 10 lakh minimum
                            specs_verification['market_value_reasonable'] = False
                            specs_verification['issues'].append(f"Market value (₹{market_value:,.2f}) seems unreasonably low for {data['property_type']}")
                    elif data['property_type'] in ['Commercial', 'Office', 'Shop']:
                        if market_value < 2000000:  # 20 lakh minimum
                            specs_verification['market_value_reasonable'] = False
                            specs_verification['issues'].append(f"Market value (₹{market_value:,.2f}) seems unreasonably low for {data['property_type']}")
                    elif data['property_type'] in ['Warehouse', 'Industrial']:
                        if market_value < 5000000:  # 50 lakh minimum
                            specs_verification['market_value_reasonable'] = False
                            specs_verification['issues'].append(f"Market value (₹{market_value:,.2f}) seems unreasonably low for {data['property_type']}")
                    
                    # Check price per square foot
                    if 'sq_ft' in data and float(data['sq_ft'].replace(',', '')) > 0:
                        try:
                            sq_ft = float(data['sq_ft'].replace(',', ''))
                            price_per_sqft = market_value / sq_ft
                            
                            if data['property_type'] in ['Apartment', 'Studio']:
                                if price_per_sqft < 1000:  # Less than ₹1000 per sq ft
                                    specs_verification['market_value_reasonable'] = False
                                    specs_verification['issues'].append(f"Price per sq ft (₹{price_per_sqft:,.2f}) seems unreasonably low for {data['property_type']}")
                                elif price_per_sqft > 50000:  # More than ₹50k per sq ft
                                    specs_verification['market_value_reasonable'] = False
                                    specs_verification['issues'].append(f"Price per sq ft (₹{price_per_sqft:,.2f}) seems unreasonably high for {data['property_type']}")
                            elif data['property_type'] in ['House', 'Villa', 'Independent House', 'Independent Villa']:
                                if price_per_sqft < 500:  # Less than ₹500 per sq ft
                                    specs_verification['market_value_reasonable'] = False
                                    specs_verification['issues'].append(f"Price per sq ft (₹{price_per_sqft:,.2f}) seems unreasonably low for {data['property_type']}")
                                elif price_per_sqft > 100000:  # More than ₹1 lakh per sq ft
                                    specs_verification['market_value_reasonable'] = False
                                    specs_verification['issues'].append(f"Price per sq ft (₹{price_per_sqft:,.2f}) seems unreasonably high for {data['property_type']}")
                        except ValueError:
                            pass
            except ValueError:
                specs_verification['market_value_reasonable'] = False
                specs_verification['issues'].append("Invalid market value data: must be a number")

        # Calculate verification score
        valid_checks = sum([
            specs_verification['bedrooms_reasonable'],
            specs_verification['bathrooms_reasonable'],
            specs_verification['total_rooms_reasonable'],
            specs_verification['year_built_reasonable'],
            specs_verification['parking_reasonable'],
            specs_verification['sq_ft_reasonable'],
            specs_verification['market_value_reasonable']
        ])
        
        total_checks = 7
        specs_verification['verification_score'] = (valid_checks / total_checks) * 100

        # Overall validity
        specs_verification['is_valid'] = all([
            specs_verification['bedrooms_reasonable'],
            specs_verification['bathrooms_reasonable'],
            specs_verification['total_rooms_reasonable'],
            specs_verification['year_built_reasonable'],
            specs_verification['parking_reasonable'],
            specs_verification['sq_ft_reasonable'],
            specs_verification['market_value_reasonable']
        ])
        
    except Exception as e:
        logger.error(f"Error in property specs verification: {str(e)}")
        specs_verification['is_valid'] = False
        specs_verification['issues'].append(f"Error in verification: {str(e)}")
    
    return specs_verification

def assess_image_quality(img):
    try:
        width, height = img.size
        resolution = width * height
        quality_score = min(100, resolution // 20000)
        return {
            'resolution': f"{width}x{height}",
            'quality_score': quality_score
        }
    except Exception as e:
        logger.error(f"Error assessing image quality: {str(e)}")
        return {
            'resolution': 'unknown',
            'quality_score': 0
        }

def check_if_property_related(text):
    try:
        classifier = load_model("zero-shot-classification", "facebook/bart-large-mnli")
        result = classifier(text[:1000], ["property-related", "non-property-related"])
        is_related = result['labels'][0] == "property-related"
        return {
            'is_related': is_related,
            'confidence': float(result['scores'][0])
        }
    except Exception as e:
        logger.error(f"Error checking property relation: {str(e)}")
        return {
            'is_related': False,
            'confidence': 0.0
        }

if __name__ == '__main__':
    # Run Flask app
    app.run(host='0.0.0.0', port=8000, debug=True, use_reloader=False)
import os
import re
import joblib
import logging
from flask import Flask, render_template, request, jsonify
from langdetect import detect as detect_language, LangDetectException

# --- অ্যাপ এবং লগিং কনফিগারেশন ---
app = Flask(__name__)
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

# --- গ্লোবাল ভ্যারিয়েবল ---
MODELS = {}
VECTORIZERS = {}

# --- মডেল লোড করার ফাংশন ---
def load_all_models():
    """
    অ্যাপ্লিকেশন শুরু হওয়ার সময় models ফোল্ডার থেকে সব মডেল লোড করে।
    """
    global MODELS, VECTORIZERS
    languages = {'bn': 'বাংলা', 'en': 'ইংরেজি'}
    logger.info("মডেল লোড করা শুরু হচ্ছে...")
    
    for lang_code, lang_name in languages.items():
        model_path = os.path.join('models', f'model_{lang_code}.joblib')
        vectorizer_path = os.path.join('models', f'vectorizer_{lang_code}.joblib')

        if os.path.exists(model_path) and os.path.exists(vectorizer_path):
            try:
                MODELS[lang_code] = joblib.load(model_path)
                VECTORIZERS[lang_code] = joblib.load(vectorizer_path)
                logger.info(f"✅ {lang_name} ({lang_code}) মডেল সফলভাবে লোড হয়েছে।")
            except Exception as e:
                logger.error(f"❌ {lang_name} ({lang_code}) মডেল লোড করার সময় ত্রুটি: {e}")
        else:
            logger.warning(f"⚠️ {lang_name} ({lang_code}) মডেল ফাইল খুঁজে পাওয়া যায়নি: {model_path}")

# --- মডেল লোড করা (Render-এ ইম্পোর্টের সময়েই) ---
load_all_models()

# --- টেক্সট প্রি-প্রসেসিং ---
def preprocess_text(text):
    if not isinstance(text, str): return ""
    text = re.sub(r'<.*?>', '', text)
    text = re.sub(r'[^\w\s\u0980-\u09FFa-zA-Z]', '', text)
    text = re.sub(r'\s+', ' ', text).strip().lower()
    return text

# --- Flask Routes ---
@app.route('/')
def home():
    return render_template('index.html')

@app.route('/detect', methods=['POST'])
def detect():
    """ভাষা শনাক্ত করে এবং লোড করা মডেল ব্যবহার করে প্রেডিকশন দেয়।"""
    if not MODELS or not VECTORIZERS:
        return jsonify({'error': 'মডেল সার্ভারে সঠিকভাবে লোড হয়নি। অনুগ্রহ করে পরে আবার চেষ্টা করুন।'}), 503

    try:
        text = request.get_json().get('text', '').strip()
        if not text:
            return jsonify({'error': 'Please enter some text to analyze.'}), 400

        # ভাষা শনাক্তকরণ
        try:
            lang_code = detect_language(text)
        except LangDetectException:
            lang_code = 'en' if re.search(r'[a-zA-Z]', text) else 'bn'
        
        logger.info(f"শনাক্ত করা ভাষা: {lang_code}")
        
        # সঠিক মডেল বেছে নেওয়া
        model = MODELS.get(lang_code)
        vectorizer = VECTORIZERS.get(lang_code)
        
        if not model or not vectorizer:
            logger.warning(f"'{lang_code}' ভাষার জন্য মডেল পাওয়া যায়নি। ফলব্যাক করা হচ্ছে...")
            # ফলব্যাক হিসেবে ইংরেজি বা বাংলা মডেল ব্যবহার করা
            model = MODELS.get('en') or MODELS.get('bn')
            vectorizer = VECTORIZERS.get('en') or VECTORIZERS.get('bn')

        if not model or not vectorizer:
            return jsonify({'error': 'No suitable model found on the server.'}), 501

        processed_text = preprocess_text(text)
        vectorized_text = vectorizer.transform([processed_text] if processed_text else [""])
        
        prediction = int(model.predict(vectorized_text)[0])
        probability = model.predict_proba(vectorized_text)[0]
        
        hate_prob = round(float(probability[1]) * 100, 2)
        normal_prob = round(float(probability[0]) * 100, 2)
        result = "Hate speech" if prediction == 1 else "Normal speech"
        return jsonify({
            'text': text,
            'result': result,
            'hate_prob': hate_prob,
            'normal_prob': normal_prob
        })

    except Exception as e:
        logger.error(f"'/detect' রুটে ত্রুটি: {e}", exc_info=True)
        return jsonify({'error': 'An unexpected server error occurred.'}), 500

# --- লোকাল টেস্টিং-এর জন্য ---
if __name__ == '__main__':
    app.run(host='0.0.0.0', port=5000)
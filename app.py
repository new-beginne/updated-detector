import os
import re
import joblib
import logging
import time
from flask import Flask, render_template, request, jsonify
from langdetect import detect as detect_language, LangDetectException

# --- অ্যাপ এবং লগিং কনফিগারেশন ---
app = Flask(__name__)
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

# --- ক্যাশ বাস্টিং এর জন্য কনটেক্সট প্রসেসর ---
@app.context_processor
def inject_version():
    """CSS ফাইলের জন্য ক্যাশ বাস্টিং ভার্সন তৈরি করে।"""
    return {'version': int(time.time())}

# --- গ্লোবাল ভ্যারিয়েবল ---
# অ্যাপ চালু হওয়ার সময় মডেল এবং ভেক্টরাইজার লোড করে এখানে রাখা হবে
MODELS = {}
VECTORIZERS = {}

# --- Helper Functions ---
def load_all_models():
    """
    অ্যাপ্লিকেশন শুরু হওয়ার সময় বাংলা এবং ইংরেজি উভয় মডেল লোড করে।
    """
    global MODELS, VECTORIZERS
    languages = {'bn': 'বাংলা', 'en': 'ইংরেজি'}
    logger.info("সকল মডেল লোড করা শুরু হচ্ছে...")
    for lang_code, lang_name in languages.items():
        model_path = os.path.join('models', f'model_{lang_code}.joblib')
        vectorizer_path = os.path.join('models', f'vectorizer_{lang_code}.joblib')

        if os.path.exists(model_path) and os.path.exists(vectorizer_path):
            try:
                MODELS[lang_code] = joblib.load(model_path)
                VECTORIZERS[lang_code] = joblib.load(vectorizer_path)
                logger.info(f"✅ {lang_name} ({lang_code}) মডেল সফলভাবে লোড হয়েছে।")
            except Exception as e:
                logger.error(f"❌ {lang_name} ({lang_code}) মডেল লোড করার সময় ত্রুটি: {e}", exc_info=True)
        else:
            # এই সতর্কবার্তাটি গুরুত্বপূর্ণ, কারণ এটি ডিপ্লয়মেন্টের সময় কোনো সমস্যা হলে ধরিয়ে দেবে
            logger.warning(f"⚠️ {lang_name} ({lang_code}) মডেল ফাইল খুঁজে পাওয়া যায়নি।")

def preprocess_text(text):
    """শুধুমাত্র প্রেডিকশনের জন্য টেক্সট পরিষ্কার করে।"""
    if not isinstance(text, str): return ""
    text = re.sub(r'<.*?>', '', text)
    text = re.sub(r'[^\w\s\u0980-\u09FFa-zA-Z]', '', text)
    text = re.sub(r'\s+', ' ', text).strip().lower()
    return text

# --- Flask Routes ---
@app.route('/')
def home():
    """হোমপেজ রেন্ডার করে।"""
    return render_template('index.html')

@app.route('/detect', methods=['POST'])
def detect():
    """ভাষা শনাক্ত করে এবং সঠিক মডেল ব্যবহার করে হ্যাট স্পিচ ডিটেক্ট করে।"""
    if not MODELS or not VECTORIZERS:
        return jsonify({'error': 'মডেল সার্ভারে সঠিকভাবে লোড হয়নি। অনুগ্রহ করে পরে আবার চেষ্টা করুন।'}), 503

    try:
        text = request.get_json().get('text', '').strip()
        if not text:
            return jsonify({'error': 'দয়া করে টেক্সট লিখুন'}), 400

        # ভাষা শনাক্ত করা
        try:
            lang_code = detect_language(text)
        except LangDetectException:
            # যদি ভাষা শনাক্ত না করা যায়, ইংরেজি অক্ষর আছে কিনা তার উপর ভিত্তি করে সিদ্ধান্ত নেওয়া হবে
            lang_code = 'en' if re.search(r'[a-zA-Z]', text) else 'bn'
        
        logger.info(f"শনাক্ত করা ভাষা: '{lang_code}'")
        
        # সঠিক মডেল বেছে নেওয়া
        model, vectorizer = MODELS.get(lang_code), VECTORIZERS.get(lang_code)
        
        # যদি নির্দিষ্ট ভাষার মডেল না থাকে, একটি উপযুক্ত ফলব্যাক বেছে নেওয়া
        if not model:
            logger.warning(f"'{lang_code}' ভাষার জন্য নির্দিষ্ট মডেল নেই। একটি ফলব্যাক মডেল ব্যবহার করা হচ্ছে।")
            if 'en' in MODELS and re.search(r'[a-zA-Z]', text):
                model, vectorizer = MODELS['en'], VECTORIZERS['en']
            elif 'bn' in MODELS:
                model, vectorizer = MODELS['bn'], VECTORIZERS['bn']

        if not model:
            return jsonify({'error': 'কোনো উপযুক্ত মডেল সার্ভারে পাওয়া যায়নি।'}), 501

        processed_text = preprocess_text(text)
        vectorized_text = vectorizer.transform([processed_text] if processed_text else [""])
        
        prediction = int(model.predict(vectorized_text)[0])
        probability = model.predict_proba(vectorized_text)[0]
        
        hate_prob = round(float(probability[1]) * 100, 2)
        normal_prob = round(float(probability[0]) * 100, 2)
        
        result = "হ্যাট স্পিচ" if prediction == 1 else "নরমাল স্পিচ"
        
        return jsonify({
            'text': text,
            'result': result,
            'hate_prob': hate_prob,
            'normal_prob': normal_prob
        })

    except Exception as e:
        logger.error(f"'/detect' রুটে ত্রুটি: {e}", exc_info=True)
        return jsonify({'error': 'সার্ভারে একটি অপ্রত্যাশিত সমস্যা হয়েছে'}), 500

# --- অ্যাপ চালনার প্রধান অংশ ---
if __name__ == '__main__':
    load_all_models()
    # Gunicorn ব্যবহারের জন্য host, port, debug এখানে উল্লেখ করার প্রয়োজন নেই
    app.run()
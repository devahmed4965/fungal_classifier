from flask import Flask, request, render_template, flash, send_from_directory, redirect, url_for
import tensorflow as tf
from tensorflow.keras.preprocessing import image
from werkzeug.utils import secure_filename
import numpy as np
import os
import uuid
import logging
import requests # <-- تأكد من تثبيت هذه المكتبة (pip install requests)

# إعداد التسجيل الأساسي
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

app = Flask(__name__)
# !!! هام: غير هذا المفتاح في بيئة الإنتاج !!! استخدم مفتاحًا عشوائيًا وقويًا
app.secret_key = os.environ.get("FLASK_SECRET_KEY", "change_this_to_a_strong_random_secret_key_too") # Change this key
app.config['UPLOAD_FOLDER'] = 'uploads'
# app.config['MAX_CONTENT_LENGTH'] = 16 * 1024 * 1024 # مثال: 16 ميغابايت
os.makedirs(app.config['UPLOAD_FOLDER'], exist_ok=True)

# --- قاموس التصنيف العلمي الموسع ---
# !!! ################################################################# !!!
# !!!   تحذير: لا يزال هذا القاموس بحاجة لتحديث ليعكس الـ 13 فئة الصحيحة  !!!
# !!! ################################################################# !!!
#
# 1. ابحث عن أسماء المجلدات الـ 13 في مجلد التدريب (`data/resized/train`).
# 2. تأكد من أن هذا القاموس يحتوي على مفتاح لكل اسم مجلد من هذه المجلدات الـ 13.
# 3. استبدل الأسماء العامة (مثل Pezizales) والمفاتيح المؤقتة بالأسماء الصحيحة.
# 4. أضف معلومات التصنيف الصحيحة لهذه الفئات.
# 5. العدد الإجمالي للمفاتيح في هذا القاموس يجب أن يكون 13.
#
taxonomy_map = {
    # --- تأكد من أن هذه المفاتيح تطابق مجلدات التدريب ---
    'Amanita': {
        'Domain': 'Eukaryota', 'Kingdom': 'Fungi', 'Phylum': 'Basidiomycota', 'Class': 'Agaricomycetes',
        'Order': 'Agaricales', 'Family': 'Amanitaceae', 'Genus': 'Amanita', 'Note': 'تضم أنواع سامة و صالحة للأكل'
    },
    'Agaricus': {
        'Domain': 'Eukaryota', 'Kingdom': 'Fungi', 'Phylum': 'Basidiomycota', 'Class': 'Agaricomycetes',
        'Order': 'Agaricales', 'Family': 'Agaricaceae', 'Genus': 'Agaricus', 'Note': 'مثل فطر الشامبنيون الشائع'
    },
    'Boletus': {
        'Domain': 'Eukaryota', 'Kingdom': 'Fungi', 'Phylum': 'Basidiomycota', 'Class': 'Agaricomycetes',
        'Order': 'Boletales', 'Family': 'Boletaceae', 'Genus': 'Boletus (sensu lato)', 'Note': 'فطر ذو مسام بدلاً من خياشيم'
    },
    'Cantharellus': {
        'Domain': 'Eukaryota', 'Kingdom': 'Fungi', 'Phylum': 'Basidiomycota', 'Class': 'Agaricomycetes',
        'Order': 'Cantharellales', 'Family': 'Cantharellaceae', 'Genus': 'Cantharellus', 'Note': 'معروف باسم فطر الشانتريل'
    },
    'Lactarius': {
        'Domain': 'Eukaryota', 'Kingdom': 'Fungi', 'Phylum': 'Basidiomycota', 'Class': 'Agaricomycetes',
        'Order': 'Russulales', 'Family': 'Russulaceae', 'Genus': 'Lactarius', 'Note': 'يفرز سائلاً لبنياً عند الكسر'
    },
    'Ganoderma': {
        'Domain': 'Eukaryota', 'Kingdom': 'Fungi', 'Phylum': 'Basidiomycota', 'Class': 'Agaricomycetes',
        'Order': 'Polyporales', 'Family': 'Ganodermataceae', 'Genus': 'Ganoderma', 'Note': 'مثل فطر الريشي الطبي'
    },
    'Trametes': {
        'Domain': 'Eukaryota', 'Kingdom': 'Fungi', 'Phylum': 'Basidiomycota', 'Class': 'Agaricomycetes',
        'Order': 'Polyporales', 'Family': 'Polyporaceae', 'Genus': 'Trametes', 'Note': 'مثل فطر ذيل الديك الرومي'
    },
    'Russula': {
        'Domain': 'Eukaryota', 'Kingdom': 'Fungi', 'Phylum': 'Basidiomycota', 'Class': 'Agaricomycetes',
        'Order': 'Russulales', 'Family': 'Russulaceae', 'Genus': 'Russula', 'Note': 'يتميز بقبعات ملونة وساق قابلة للكسر'
    },
    # --- هذه المفاتيح التالية قد تحتاج لتغييرها لأسماء مجلدات التدريب الفعلية ---
    'Pezizales_Placeholder': { # <-- استبدل بالاسم الصحيح للفئة 9
        'Domain': 'Eukaryota', 'Kingdom': 'Fungi', 'Phylum': 'Ascomycota',
        'Class': 'Pezizomycetes', 'Order': 'Pezizales', 'Family': 'متعددة',
        'Genus': 'غير محدد (مستوى الرتبة)', 'Species': 'غير محدد'
    },
    'Erysiphales_Placeholder': { # <-- استبدل بالاسم الصحيح للفئة 10
        'Domain': 'Eukaryota', 'Kingdom': 'Fungi', 'Phylum': 'Ascomycota',
        'Class': 'Leotiomycetes', 'Order': 'Erysiphales', 'Family': 'Erysiphaceae',
        'Genus': 'متعددة', 'Species': 'غير محدد (مستوى الرتبة)'
    },
    'Saccharomycetales_Placeholder': { # <-- استبدل بالاسم الصحيح للفئة 11
         'Domain': 'Eukaryota', 'Kingdom': 'Fungi', 'Phylum': 'Ascomycota',
         'Class': 'Saccharomycetes', 'Order': 'Saccharomycetales', 'Family': 'متعددة',
         'Genus': 'متعددة', 'Species': 'غير محدد (مستوى الرتبة)'
    },
    'Glomeromycota_Placeholder': { # <-- استبدل بالاسم الصحيح للفئة 12
         'Domain': 'Eukaryota', 'Kingdom': 'Fungi', 'Phylum': 'Glomeromycota',
         'Class': 'متعددة', 'Order': 'متعددة', 'Family': 'متعددة',
         'Genus': 'متعددة', 'Species': 'غير محدد (مستوى الشعبة)'
    },
    'Others': { # <-- تأكد من أن 'Others' كان أحد مجلدات التدريب الـ 13
        'Domain': 'غير محدد', 'Kingdom': 'غير محدد', 'Phylum': 'غير محدد',
        'Class': 'غير محدد', 'Order': 'غير محدد', 'Family': 'غير محدد',
        'Genus': 'غير محدد', 'Note': 'فئة أخرى / غير مصنف'
    }
    # تأكد من أن العدد الإجمالي للمفاتيح هنا هو 13 بالضبط وأنها تطابق مجلدات التدريب
}
# --- نهاية قاموس التصنيف ---


# --- تحميل النموذج ---
# الرابط المباشر الذي قدمته
MODEL_URL = "https://drive.google.com/uc?export=download&id=1f2YooYzN3fBVLwkemU_jRS82BfqoIHcn"
MODEL_LOCAL_PATH = "downloaded_model.h5" # اسم الملف الذي سيتم حفظه محليًا

model = None
model_output_units = -1 # قيمة مبدئية

# التحميل فقط إذا لم يكن الملف موجودًا محليًا
if not os.path.exists(MODEL_LOCAL_PATH):
    try:
        logging.info(f"Downloading model from Google Drive...")
        response = requests.get(MODEL_URL, stream=True)
        response.raise_for_status() # للتحقق من أخطاء التحميل (مثل 404 Not Found أو 403 Forbidden)
        with open(MODEL_LOCAL_PATH, 'wb') as f:
            for chunk in response.iter_content(chunk_size=8192):
                f.write(chunk)
        logging.info(f"Model downloaded successfully to {MODEL_LOCAL_PATH}")
    except requests.exceptions.RequestException as download_e:
        # التعامل مع أخطاء الشبكة أو الرابط بشكل محدد
        logging.exception(f"Failed to download model due to network/URL error: {download_e}")
        # model يبقى None
    except Exception as download_e:
        # التعامل مع أخطاء أخرى محتملة أثناء التنزيل
        logging.exception(f"An unexpected error occurred during model download: {download_e}")
        # model يبقى None

# محاولة تحميل النموذج من الملف المحلي
if os.path.exists(MODEL_LOCAL_PATH):
    try:
        model = tf.keras.models.load_model(MODEL_LOCAL_PATH)
        logging.info(f"Model loaded successfully from {MODEL_LOCAL_PATH}")
        try:
            output_shape = model.output_shape
            model_output_units = output_shape[-1]
            logging.info(f"Loaded model expects {model_output_units} output units (classes).")
        except Exception as shape_e:
            logging.warning(f"Could not determine model output shape automatically: {shape_e}")
    except OSError as load_e:
        # قد يشير خطأ OSError أحيانًا إلى ملف تالف أو غير مكتمل
        logging.exception(f"Critical Error loading model (potential file corruption or I/O issue): {load_e}")
        model = None
    except Exception as load_e:
        # التعامل مع أخطاء التحميل الأخرى (مثل نفاد الذاكرة - قد لا يتم التقاطه هنا دائمًا)
        logging.exception(f"Critical Error loading model: {load_e}")
        model = None
else:
     logging.error(f"Model file could not be downloaded or found at {MODEL_LOCAL_PATH}. Cannot load model.")
     # model يبقى None

# --- استخراج أسماء الفئات والتحقق ---
class_labels = sorted(list(taxonomy_map.keys()))
logging.info(f"Class labels based on taxonomy map (sorted): {class_labels}")
num_expected_classes = len(class_labels)
logging.info(f"Number of classes expected by app based on taxonomy map: {num_expected_classes}")

# التحقق الفوري من التطابق بعد محاولة التحميل
if model is not None:
    if model_output_units != -1 and model_output_units != num_expected_classes:
        logging.error("###########################################################################")
        logging.error(f"CRITICAL MISMATCH DETECTED ON LOAD:")
        logging.error(f"  Model output units: {model_output_units}")
        logging.error(f"  Classes in taxonomy_map: {num_expected_classes}")
        logging.error("  Please CORRECT the taxonomy_map in app.py to match the model!")
        logging.error("###########################################################################")
        # قد ترغب في منع التطبيق من الاستمرار إذا كان النموذج موجودًا ولكن الفئات غير متطابقة
        model = None # اجعل النموذج None لمنع التشغيل بمعلومات خاطئة
        logging.error("Model set to None due to class mismatch.")
elif model is None:
     logging.warning("Model is None after download/load attempts. Check previous errors.")


# --- حجم الصورة المتوقع من قبل النموذج ---
EXPECTED_IMG_SIZE = (224, 224)


@app.route('/', methods=['GET', 'POST'])
def upload_file():
    image_url_display = None
    predictions_display = []
    full_taxonomy_display = None
    taxonomy_message = None

    if request.method == 'POST':
        # التحقق من تحميل النموذج
        if model is None:
             # هذه الرسالة ستظهر إذا فشل تحميل النموذج لأي سبب (تنزيل، تحميل، عدم تطابق)
             flash('خطأ حرج: نموذج التصنيف غير مُحمّل أو غير متوافق. لا يمكن المتابعة. يرجى مراجعة سجلات الخادم.', 'error')
             return render_template('upload.html', class_labels=class_labels)

        # --- باقي كود معالجة POST ---
        if 'file' not in request.files:
            flash('لم يتم إرفاق أي ملف.', 'warning'); return redirect(request.url)
        file = request.files['file']
        if file.filename == '':
            flash('لم يتم اختيار ملف.', 'warning'); return redirect(request.url)

        allowed_extensions = {'png', 'jpg', 'jpeg'}
        file_ext = file.filename.rsplit('.', 1)[1].lower() if '.' in file.filename else ''
        if file_ext in allowed_extensions:
            try:
                filename = f"{uuid.uuid4().hex}_{secure_filename(file.filename)}"
                file_path = os.path.join(app.config['UPLOAD_FOLDER'], filename)
                file.save(file_path)
                logging.info(f"File saved to: {file_path}")
                image_url_display = filename

                img = image.load_img(file_path, target_size=EXPECTED_IMG_SIZE)
                img_array = image.img_to_array(img)
                img_array_scaled = img_array / 255.0
                img_batch = np.expand_dims(img_array_scaled, axis=0)

                # --- التنبؤ باستخدام النموذج المحمل ---
                # يجب أن يكون model ليس None هنا إذا نجح التحميل والتوافق
                logging.info(f"Predicting using loaded model...")
                preds_raw = model.predict(img_batch)[0]
                logging.info(f"Model produced {len(preds_raw)} probability outputs.")

                # التحقق من التوافق (للتأكيد مرة أخرى، رغم التحقق عند التحميل)
                if len(preds_raw) != num_expected_classes:
                    # هذا لا يجب أن يحدث إذا نجح التحقق عند التحميل، لكنه احتياطي
                    error_msg = (f'خطأ داخلي: عدد مخرجات النموذج ({len(preds_raw)}) لا يطابق عدد الفئات المتوقع ({num_expected_classes}).')
                    flash(error_msg, 'error'); logging.error(error_msg)
                    if os.path.exists(file_path): os.remove(file_path)
                    return redirect(request.url)

                # ربط الاحتمالات مع أسماء الفئات (من taxonomy_map)
                predictions_data = [
                    {'class': label, 'confidence': float(prob) * 100}
                    for label, prob in zip(class_labels, preds_raw)
                ]
                predictions_data.sort(key=lambda x: x['confidence'], reverse=True)

                top_prediction_label = None
                if predictions_data:
                     top_prediction_label = predictions_data[0]['class']
                     logging.info(f"Top prediction: {top_prediction_label} ({predictions_data[0]['confidence']:.2f}%)")
                     predictions_data[0]['is_top'] = True
                     for i in range(1, len(predictions_data)): predictions_data[i]['is_top'] = False

                predictions_display = [
                    {
                        'class': pred_data['class'],
                        'confidence': f"{pred_data['confidence']:.2f}%",
                        'is_top': pred_data.get('is_top', False)
                    }
                    for pred_data in predictions_data[:5] # عرض أفضل 5
                ]

                if top_prediction_label:
                    full_taxonomy_display = taxonomy_map.get(top_prediction_label)
                    if not full_taxonomy_display:
                         warning_msg = f"لم يتم العثور على معلومات تصنيف للفئة المتوقعة '{top_prediction_label}' في القاموس."
                         taxonomy_message = warning_msg; logging.warning(warning_msg)
                    elif not isinstance(full_taxonomy_display, dict):
                              error_msg = f"خطأ في البيانات: قيمة التصنيف لـ '{top_prediction_label}' ليست قاموسًا."
                              taxonomy_message = error_msg; logging.error(error_msg); full_taxonomy_display = None
                else:
                    taxonomy_message = "لم يتم تحديد فئة ذات ثقة كافية للحصول على التصنيف."

            except FileNotFoundError:
                 flash('خطأ: الملف الذي تم تحميله لم يتم العثور عليه للمعالجة.', 'error'); logging.error(f"File not found after saving: {file_path}"); return redirect(request.url)
            except Exception as e:
                logging.exception(f'حدث خطأ غير متوقع أثناء معالجة الملف {file.filename}: {e}')
                flash(f'حدث خطأ أثناء معالجة الصورة. يرجى المحاولة مرة أخرى أو مراجعة السجلات.', 'error')
                if 'file_path' in locals() and os.path.exists(file_path):
                    try: os.remove(file_path); logging.info(f"Removed corrupted/problematic file: {file_path}")
                    except OSError as remove_error: logging.error(f"Error removing file {file_path}: {remove_error}")
                # لا تقم بإعادة التوجيه، اعرض نفس الصفحة مع رسالة خطأ
                return render_template('upload.html', image_url=image_url_display, predictions=[], full_taxonomy=None, taxonomy_message="فشلت معالجة الصورة.", class_labels=class_labels)
        else:
            flash('نوع الملف غير مسموح به. يرجى رفع صورة بامتداد png, jpg, أو jpeg.', 'error')
            return redirect(request.url)

    # عرض القالب في حالة GET أو بعد معالجة POST
    return render_template(
        'upload.html', image_url=image_url_display, predictions=predictions_display,
        full_taxonomy=full_taxonomy_display, taxonomy_message=taxonomy_message, class_labels=class_labels
    )

@app.route('/uploads/<filename>')
def uploaded_file(filename):
    try: return send_from_directory(app.config['UPLOAD_FOLDER'], filename)
    except FileNotFoundError: logging.warning(f"Attempted to access non-existent file: {filename}"); return "File not found", 404

if __name__ == '__main__':
    # تعيين debug=False للإنتاج
    # استخدم host='0.0.0.0' لجعله متاحًا على شبكتك (استخدم بحذر)
    # Gunicorn سيتجاهل هذا البلوك عند تشغيل app:app
    app.run(debug=False, host='127.0.0.1', port=5001)

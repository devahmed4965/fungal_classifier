from flask import Flask, request, render_template, flash, send_from_directory, redirect, url_for
import tensorflow as tf # استخدام tensorflow لتحميل النموذج
from tensorflow.keras.preprocessing import image
from werkzeug.utils import secure_filename
import numpy as np
import os
import uuid
import logging # لإضافة تسجيل أفضل للأخطاء

# إعداد التسجيل الأساسي
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

app = Flask(__name__)
# !!! هام: غير هذا المفتاح في بيئة الإنتاج !!! استخدم مفتاحًا عشوائيًا وقويًا
app.secret_key = os.environ.get("FLASK_SECRET_KEY", "change_this_to_a_strong_random_secret_key") # Change this key
app.config['UPLOAD_FOLDER'] = 'uploads'
# app.config['MAX_CONTENT_LENGTH'] = 16 * 1024 * 1024 # مثال: 16 ميغابايت
os.makedirs(app.config['UPLOAD_FOLDER'], exist_ok=True)

# --- قاموس التصنيف العلمي الموسع ---
# !!! ################################################################# !!!
# !!!   الخطوة الأهم: عدّل هذا القاموس ليحتوي على الـ 13 فئة بالضبط   !!!
# !!! ################################################################# !!!
#
# 1. ابحث عن أسماء المجلدات الـ 13 في مجلد التدريب (`data/resized/train`).
# 2. استبدل 'Your_Missing_Class_11_Name', 'Your_Missing_Class_12_Name',
#    و 'Your_Missing_Class_13_Name' بأسماء المجلدات الثلاثة الفعلية.
# 3. أضف معلومات التصنيف الصحيحة لهذه الفئات الثلاث.
# 4. تأكد من أن المفاتيح الـ 10 الأخرى تطابق أيضًا مجلدات التدريب.
# 5. العدد الإجمالي للمفاتيح في هذا القاموس يجب أن يكون 13.
#
taxonomy_map = {
    # --- Basidiomycota / Agaricales ---
    'Amanita': {
        'Domain': 'Eukaryota (حقيقيات النوى)', 'Kingdom': 'Fungi (فطريات)', 'Phylum': 'Basidiomycota (فطريات بازيدية)',
        'Class': 'Agaricomycetes (أغاريقونيات)', 'Order': 'Agaricales (غاريقونيات)', 'Family': 'Amanitaceae (أمانيتية)',
        'Genus': 'Amanita (أمانيت)', 'Species': 'غير محدد (مستوى الجنس)'
    },
    'Agaricus': {
        'Domain': 'Eukaryota', 'Kingdom': 'Fungi', 'Phylum': 'Basidiomycota', 'Class': 'Agaricomycetes',
        'Order': 'Agaricales', 'Family': 'Agaricaceae (غاريقية)', 'Genus': 'Agaricus (غاريقون)', 'Species': 'غير محدد'
    },
    'Boletus': {
        'Domain': 'Eukaryota', 'Kingdom': 'Fungi', 'Phylum': 'Basidiomycota', 'Class': 'Agaricomycetes',
        'Order': 'Boletales (بوليطيات)', 'Family': 'Boletaceae (بوليطية)', 'Genus': 'Boletus (بوليط)', 'Species': 'غير محدد'
        # ملاحظة: جنس Boletus تم تقسيمه، قد تكون الأنواع الآن في أجناس أخرى. تحقق من صحة البيانات المستخدمة للتدريب.
    },
    'Cantharellus': {
        'Domain': 'Eukaryota', 'Kingdom': 'Fungi', 'Phylum': 'Basidiomycota', 'Class': 'Agaricomycetes',
        'Order': 'Cantharellales (كويزيات)', 'Family': 'Cantharellaceae (كويزية)', 'Genus': 'Cantharellus (كويزي)', 'Species': 'غير محدد'
    },
    'Lactarius': {
        'Domain': 'Eukaryota', 'Kingdom': 'Fungi', 'Phylum': 'Basidiomycota', 'Class': 'Agaricomycetes',
        'Order': 'Russulales (روسوليات)', 'Family': 'Russulaceae (روسولية)', 'Genus': 'Lactarius (خريق)', 'Species': 'غير محدد'
    },
     # --- Basidiomycota / Polyporales ---
    'Ganoderma': {
        'Domain': 'Eukaryota', 'Kingdom': 'Fungi', 'Phylum': 'Basidiomycota', 'Class': 'Agaricomycetes',
        'Order': 'Polyporales (متعددات المسام)', 'Family': 'Ganodermataceae (غصنية)', 'Genus': 'Ganoderma (غانوديرما)', 'Species': 'غير محدد'
    },
    'Trametes': {
        'Domain': 'Eukaryota', 'Kingdom': 'Fungi', 'Phylum': 'Basidiomycota', 'Class': 'Agaricomycetes',
        'Order': 'Polyporales', 'Family': 'Polyporaceae (بوليبوراسية)', 'Genus': 'Trametes (طرْمِيد)', 'Species': 'غير محدد'
    },
    # --- Basidiomycota / Russulales ---
    'Russula': {
        'Domain': 'Eukaryota', 'Kingdom': 'Fungi', 'Phylum': 'Basidiomycota', 'Class': 'Agaricomycetes',
        'Order': 'Russulales', 'Family': 'Russulaceae', 'Genus': 'Russula (روسولا)', 'Species': 'غير محدد'
    },
    # --- Ascomycota ---
    'Pezizales': { # فئة على مستوى الرتبة
        'Domain': 'Eukaryota', 'Kingdom': 'Fungi', 'Phylum': 'Ascomycota (فطريات زقية)',
        'Class': 'Pezizomycetes (فنجانيانية)', 'Order': 'Pezizales (فنجانيات)', 'Family': 'متعددة',
        'Genus': 'غير محدد (مستوى الرتبة)', 'Species': 'غير محدد'
    },
    'Erysiphales': { # فئة على مستوى الرتبة (البياض الدقيقي)
        'Domain': 'Eukaryota', 'Kingdom': 'Fungi', 'Phylum': 'Ascomycota',
        'Class': 'Leotiomycetes (ليوتانيانية)', 'Order': 'Erysiphales (إريسيبيات)', 'Family': 'Erysiphaceae (إريسيبية)',
        'Genus': 'متعددة (مثل Erysiphe, Podosphaera)', 'Species': 'غير محدد (مستوى الرتبة)'
    },
     'Saccharomycetales': { # فئة على مستوى الرتبة (الخمائر)
        'Domain': 'Eukaryota', 'Kingdom': 'Fungi', 'Phylum': 'Ascomycota',
        'Class': 'Saccharomycetes (سكريانية)', 'Order': 'Saccharomycetales (سكريات)', 'Family': 'متعددة (مثل Saccharomycetaceae)',
        'Genus': 'متعددة (مثل Saccharomyces)', 'Species': 'غير محدد (مستوى الرتبة)'
    },
    # --- Glomeromycota ---
     'Glomeromycota': { # فئة على مستوى الشعبة (الفطريات الجذرية التكافلية)
        'Domain': 'Eukaryota', 'Kingdom': 'Fungi', 'Phylum': 'Glomeromycota (كبيبيائية)',
        'Class': 'متعددة', 'Order': 'متعددة', 'Family': 'متعددة',
        'Genus': 'متعددة (مثل Glomus)', 'Species': 'غير محدد (مستوى الشعبة)'
    },
    # --- Others ---
    'Others': {
        'Domain': 'غير محدد', 'Kingdom': 'غير محدد', 'Phylum': 'غير محدد',
        'Class': 'غير محدد', 'Order': 'غير محدد', 'Family': 'غير محدد',
        'Genus': 'غير محدد', 'Species': 'فئة أخرى / غير مصنف'
    }
}
# --- نهاية قاموس التصنيف ---

# تحميل النموذج المدرب
# تأكد من أن هذا يطابق اسم النموذج الأساسي الذي استخدمته للتدريب
MODEL_BASENAME = "VGG16" # أو غيره
MODEL_PATH = f'fungal_classifier_{MODEL_BASENAME}_best.h5'
model = None
model_output_units = -1 # قيمة مبدئية

try:
    if os.path.exists(MODEL_PATH):
        model = tf.keras.models.load_model(MODEL_PATH)
        logging.info(f"Model loaded successfully from {MODEL_PATH}")
        try:
            output_shape = model.output_shape
            model_output_units = output_shape[-1]
            logging.info(f"Loaded model expects {model_output_units} output units (classes).")
        except Exception as shape_e:
            logging.warning(f"Could not determine model output shape automatically: {shape_e}")
    else:
        logging.error(f"Error: Model file not found at {MODEL_PATH}")
except Exception as e:
    logging.exception(f"Critical Error loading model from {MODEL_PATH}: {e}")
    model = None

# --- استخراج أسماء الفئات ---
class_labels = sorted(list(taxonomy_map.keys()))
logging.info(f"Class labels based on taxonomy map (sorted): {class_labels}")
num_expected_classes = len(class_labels)
logging.info(f"Number of classes expected by app based on taxonomy map: {num_expected_classes}")

# التحقق الفوري من التطابق إذا تم تحميل النموذج بنجاح
if model is not None and model_output_units != -1 and model_output_units != num_expected_classes:
    logging.error("###########################################################################")
    logging.error(f"CRITICAL MISMATCH DETECTED ON LOAD:")
    logging.error(f"  Model output units: {model_output_units}")
    logging.error(f"  Classes in taxonomy_map: {num_expected_classes}")
    logging.error("  Please CORRECT the taxonomy_map in app.py to match the model!")
    logging.error("###########################################################################")
    # يمكنك اختيار إيقاف التطبيق هنا إذا أردت
    # raise ValueError("Model output size does not match taxonomy_map size!")


# --- حجم الصورة المتوقع من قبل النموذج ---
EXPECTED_IMG_SIZE = (224, 224)


@app.route('/', methods=['GET', 'POST'])
def upload_file():
    image_url_display = None
    predictions_display = []
    full_taxonomy_display = None
    taxonomy_message = None

    if request.method == 'POST':
        if model is None:
             flash('خطأ حرج: نموذج التصنيف غير مُحمّل. لا يمكن المتابعة. يرجى مراجعة سجلات الخادم.', 'error')
             return render_template('upload.html', class_labels=class_labels)

        if 'file' not in request.files:
            flash('لم يتم إرفاق أي ملف.', 'warning')
            return redirect(request.url)

        file = request.files['file']
        if file.filename == '':
            flash('لم يتم اختيار ملف.', 'warning')
            return redirect(request.url)

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

                logging.info(f"Predicting using model: {MODEL_PATH}")
                preds_raw = model.predict(img_batch)[0]
                logging.info(f"Model produced {len(preds_raw)} probability outputs.")

                # التحقق من التوافق (يجب أن ينجح إذا تم تعديل taxonomy_map)
                if len(preds_raw) != num_expected_classes:
                    error_msg = (f'خطأ في التوافق: النموذج أخرج {len(preds_raw)} احتمالات، '
                                 f'بينما القائمة المتوقعة تحتوي على {num_expected_classes} فئات ({class_labels}). '
                                 f'هل تم تحديث taxonomy_map في app.py بشكل صحيح ليشمل جميع الفئات الـ {len(preds_raw)} التي يعرفها النموذج؟')
                    flash(error_msg, 'error')
                    logging.error(error_msg)
                    if os.path.exists(file_path): os.remove(file_path)
                    return redirect(request.url)

                # ربط الاحتمالات مع أسماء الفئات (يجب أن يكون الطول متطابقًا الآن)
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
                         taxonomy_message = warning_msg
                         logging.warning(warning_msg)
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
                return render_template('upload.html', image_url=None, predictions=[], full_taxonomy=None, taxonomy_message="فشلت معالجة الصورة.", class_labels=class_labels)
        else:
            flash('نوع الملف غير مسموح به. يرجى رفع صورة بامتداد png, jpg, أو jpeg.', 'error')
            return redirect(request.url)

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
    app.run(debug=False, host='127.0.0.1', port=5001) # استخدام منفذ 5001 كمثال
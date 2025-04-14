import os
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import classification_report, confusion_matrix

# --- اختيار النموذج الأساسي ---
# قم بتغيير هذا لاستخدام نموذج مختلف إذا أردت
# الخيارات الممكنة: VGG16, ResNet50V2, EfficientNetB0, EfficientNetB4, InceptionV3, MobileNetV2
BASE_MODEL_NAME = "VGG16" # أو "ResNet50V2", "EfficientNetB0" etc.

# استيراد النموذج المحدد ودالة المعالجة المسبقة الخاصة به (إذا لزم الأمر)
if BASE_MODEL_NAME == "VGG16":
    from tensorflow.keras.applications import VGG16 as BaseModel
    from tensorflow.keras.applications.vgg16 import preprocess_input # VGG يتطلب فقط تحجيم [0,1] أو [-1,1] ، هذا قد لا يكون ضروريا إذا استخدمنا rescale=1./255
elif BASE_MODEL_NAME == "ResNet50V2":
    from tensorflow.keras.applications import ResNet50V2 as BaseModel
    from tensorflow.keras.applications.resnet_v2 import preprocess_input
elif BASE_MODEL_NAME == "EfficientNetB0":
    from tensorflow.keras.applications import EfficientNetB0 as BaseModel
    from tensorflow.keras.applications.efficientnet import preprocess_input
# أضف المزيد من الشروط للنماذج الأخرى إذا أردت تجربتها...
else:
    raise ValueError(f"Unsupported BASE_MODEL_NAME: {BASE_MODEL_NAME}")

from tensorflow.keras.layers import Dense, Flatten, Dropout, Input, GlobalAveragePooling2D
from tensorflow.keras.models import Model
from tensorflow.keras.optimizers import Adam # أو AdamW
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.callbacks import EarlyStopping, ModelCheckpoint, ReduceLROnPlateau
import tensorflow as tf # للتأكد من النسخة وإعدادات الذاكرة

# --- Configuration ---
TRAIN_DIR = 'data/resized/train'
VAL_DIR = 'data/resized/validation' # تأكد من تطابق اسم المجلد
IMG_SIZE = (224, 224) # حجم الإدخال القياسي للعديد من النماذج، تأكد من توافقه مع النموذج الأساسي
BATCH_SIZE = 32 # اضبطه بناءً على ذاكرة GPU (16, 32, 64)
INITIAL_EPOCHS = 50 # عدد الفترات للمرحلة الأولى (تدريب الرأس) - زيادة القيمة
FINE_TUNE_EPOCHS = 25 # عدد الفترات لمرحلة الضبط الدقيق (Fine-tuning)
TOTAL_EPOCHS = INITIAL_EPOCHS + FINE_TUNE_EPOCHS
INITIAL_LEARNING_RATE = 1e-4 # معدل التعلم الأولي
FINE_TUNE_LEARNING_RATE = 1e-5 # معدل تعلم أقل بكثير للضبط الدقيق
MODEL_SAVE_PATH = f'fungal_classifier_{BASE_MODEL_NAME}_best.h5' # اسم ملف النموذج يتضمن اسم النموذج الأساسي
HISTORY_PLOT_PATH = f'training_history_{BASE_MODEL_NAME}.png'
CONFUSION_MATRIX_PATH = f'confusion_matrix_{BASE_MODEL_NAME}.png'

# --- إعدادات GPU (اختياري ولكن مفيد) ---
print(f"TensorFlow Version: {tf.__version__}")
gpus = tf.config.list_physical_devices('GPU')
if gpus:
    try:
        # حاول تمكين نمو الذاكرة لتجنب حجز كل ذاكرة GPU مرة واحدة
        for gpu in gpus:
            tf.config.experimental.set_memory_growth(gpu, True)
        print(f"Detected {len(gpus)} GPUs, Memory Growth enabled.")
    except RuntimeError as e:
        # قد يحدث خطأ إذا تم تعيين نمو الذاكرة بالفعل
        print(e)
else:
    print("No GPU detected, running on CPU.")


def build_model(input_shape, num_classes, freeze_base=True):
    """يبني النموذج باستخدام التعلم بالنقل."""
    print(f"\nBuilding model with {BASE_MODEL_NAME} base for {num_classes} classes...")
    print(f"Input shape: {input_shape}")

    inputs = Input(shape=input_shape)

    # --- معالجة الإدخال (إذا لزم الأمر) ---
    # بعض النماذج تتطلب معالجة مسبقة محددة (تحجيم مختلف أو تطبيع)
    # إذا كان النموذج يتطلب preprocess_input:
    # x = preprocess_input(inputs)
    # base_model_input = x
    # إذا كان النموذج يعمل جيدًا مع التحجيم [0,1] من ImageDataGenerator (rescale=1./255):
    base_model_input = inputs

    # تحميل النموذج الأساسي المُدرّب مسبقًا على ImageNet
    # include_top=False لإزالة طبقة التصنيف الأصلية
    base_model = BaseModel(weights='imagenet', include_top=False, input_shape=input_shape)

    # تجميد النموذج الأساسي (أو لا، حسب المرحلة)
    base_model.trainable = not freeze_base
    if freeze_base:
        print(f"Base model '{base_model.name}' loaded. Layers are FROZEN.")
    else:
        print(f"Base model '{base_model.name}' loaded. Layers are UNFROZEN (for fine-tuning).")


    # إضافة طبقات التصنيف المخصصة
    x = base_model(base_model_input, training=not freeze_base) # training=False when base is frozen

    # --- اختيار طبقة التجميع ---
    # Flatten جيد لـ VGG، لكن GlobalAveragePooling2D غالبًا أفضل للنماذج الأحدث (ResNet, EfficientNet)
    # لأنه يقلل عدد المعلمات بشكل كبير ويساعد على منع التجهيز الزائد (Overfitting)
    if BASE_MODEL_NAME in ["ResNet50V2", "EfficientNetB0", "InceptionV3", "MobileNetV2"]:
         x = GlobalAveragePooling2D(name='global_avg_pool')(x)
         print("Using GlobalAveragePooling2D head.")
    else: # For VGG16 or others where Flatten might be standard
         x = Flatten(name='flatten')(x)
         print("Using Flatten head.")

    x = Dense(256, activation='relu', name='dense_head_1')(x)
    x = Dropout(0.5, name='dropout_head')(x) # Dropout لتقليل التجهيز الزائد
    outputs = Dense(num_classes, activation='softmax', name='predictions')(x) # طبقة الإخراج النهائية

    model = Model(inputs=inputs, outputs=outputs)

    # طباعة ملخص للتحقق (خاصة عند إلغاء التجميد)
    # model.summary() # يمكن إلغاء التعليق لعرض الملخص التفصيلي هنا

    return model, base_model # إرجاع النموذج الأساسي للتحكم في التجميد لاحقًا


def plot_history(history_initial, history_fine=None, filename='training_history.png'):
    """يرسم سجل دقة وخسارة التدريب والتحقق."""
    acc = history_initial.history['accuracy']
    val_acc = history_initial.history['val_accuracy']
    loss = history_initial.history['loss']
    val_loss = history_initial.history['val_loss']

    initial_epochs = len(acc)
    epochs_range_initial = range(initial_epochs)

    plt.figure(figsize=(16, 8)) # شكل أعرض

    # رسم المرحلة الأولى
    plt.subplot(1, 2, 1)
    plt.plot(epochs_range_initial, acc, label='Training Accuracy (Initial)')
    plt.plot(epochs_range_initial, val_acc, label='Validation Accuracy (Initial)')

    plt.subplot(1, 2, 2)
    plt.plot(epochs_range_initial, loss, label='Training Loss (Initial)')
    plt.plot(epochs_range_initial, val_loss, label='Validation Loss (Initial)')

    # إضافة بيانات مرحلة الضبط الدقيق إذا كانت موجودة
    if history_fine:
        acc_fine = history_fine.history['accuracy']
        val_acc_fine = history_fine.history['val_accuracy']
        loss_fine = history_fine.history['loss']
        val_loss_fine = history_fine.history['val_loss']

        epochs_range_fine = range(initial_epochs, initial_epochs + len(acc_fine))

        plt.subplot(1, 2, 1)
        plt.plot(epochs_range_fine, acc_fine, label='Training Accuracy (Fine-tune)')
        plt.plot(epochs_range_fine, val_acc_fine, label='Validation Accuracy (Fine-tune)')
        # خط عمودي يوضح بداية الضبط الدقيق
        plt.axvline(initial_epochs -1 , linestyle='--', color='gray', label='Start Fine-tuning')


        plt.subplot(1, 2, 2)
        plt.plot(epochs_range_fine, loss_fine, label='Training Loss (Fine-tune)')
        plt.plot(epochs_range_fine, val_loss_fine, label='Validation Loss (Fine-tune)')
        plt.axvline(initial_epochs -1, linestyle='--', color='gray', label='Start Fine-tuning')

    # إعدادات الرسم البياني للدقة
    plt.subplot(1, 2, 1)
    plt.legend(loc='lower right')
    plt.title('Training and Validation Accuracy')
    plt.xlabel('Epoch')
    plt.ylabel('Accuracy')
    plt.grid(True, linestyle='--', alpha=0.6)
    plt.ylim([min(plt.ylim())-0.05, 1.05]) # ضبط حدود المحور Y

    # إعدادات الرسم البياني للخسارة
    plt.subplot(1, 2, 2)
    plt.legend(loc='upper right')
    plt.title('Training and Validation Loss')
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.grid(True, linestyle='--', alpha=0.6)
    plt.ylim([0, max(plt.ylim())+0.1]) # ضبط حدود المحور Y

    plt.suptitle(f"Training History ({BASE_MODEL_NAME})", fontsize=16)
    plt.tight_layout(rect=[0, 0.03, 1, 0.95]) # تعديل التخطيط مع العنوان الرئيسي

    try:
        plt.savefig(filename)
        print(f"\nTraining history plot saved to '{filename}'")
    except Exception as e:
        print(f"\nError saving training plot: {e}")
    # plt.show() # إلغاء التعليق لعرض الرسم مباشرة


def plot_confusion_matrix(y_true, y_pred_classes, class_labels, filename='confusion_matrix.png'):
     """يرسم مصفوفة الارتباك."""
     conf_matrix = confusion_matrix(y_true, y_pred_classes)
     plt.figure(figsize=(14, 12)) # حجم أكبر لاستيعاب أسماء الفئات
     sns.heatmap(conf_matrix, annot=True, fmt='d', cmap='Blues',
                 xticklabels=class_labels, yticklabels=class_labels)
     plt.yticks(rotation=0)
     plt.xticks(rotation=75, ha='right') # تدوير أكبر إذا كانت الأسماء طويلة
     plt.ylabel('True Label')
     plt.xlabel('Predicted Label')
     plt.title(f'Confusion Matrix ({BASE_MODEL_NAME})', fontsize=14)
     plt.tight_layout()
     try:
         plt.savefig(filename)
         print(f"Confusion matrix saved to '{filename}'")
     except Exception as e:
         print(f"Error saving confusion matrix plot: {e}")
     # plt.show()


def train_model():
    """يجهز مولدات البيانات، يبني، يجمع، ويدرب النموذج، ثم يقيمه."""

    # --- مولدات البيانات ---
    print("\nSetting up data generators...")
    # مولد بيانات التدريب مع زيادة البيانات (Augmentation)
    train_datagen = ImageDataGenerator(
        rescale=1./255,          # تحجيم قيم البكسل إلى [0, 1]
        rotation_range=30,       # دوران أكثر
        width_shift_range=0.2,
        height_shift_range=0.2,
        shear_range=0.2,
        zoom_range=0.2,
        horizontal_flip=True,
        brightness_range=[0.8, 1.2], # إضافة تغيير السطوع
        fill_mode='nearest'
    )

    # مولد بيانات التحقق (فقط التحجيم - بدون زيادة بيانات)
    val_datagen = ImageDataGenerator(rescale=1./255)

    # تحميل البيانات من المجلدات
    try:
        print(f"Looking for training data in: {TRAIN_DIR}")
        train_generator = train_datagen.flow_from_directory(
            TRAIN_DIR,
            target_size=IMG_SIZE,
            batch_size=BATCH_SIZE,
            class_mode='categorical', # لتصنيف متعدد الفئات
            shuffle=True
        )

        print(f"Looking for validation data in: {VAL_DIR}")
        val_generator = val_datagen.flow_from_directory(
            VAL_DIR,
            target_size=IMG_SIZE,
            batch_size=BATCH_SIZE,
            class_mode='categorical',
            shuffle=False # لا حاجة لخلط بيانات التحقق للتقييم
        )
    except FileNotFoundError as e:
         print(f"\nError: Data directory not found. {e}")
         print(f"Please ensure '{TRAIN_DIR}' and '{VAL_DIR}' exist and contain class subdirectories.")
         print("Did you run 'python data_preparation.py' successfully after organizing data in 'data/raw' (flattened structure)?")
         return # الخروج إذا لم يتم العثور على البيانات

    # الحصول على عدد الفئات وقائمة التسميات
    num_classes = train_generator.num_classes
    if num_classes == 0:
         print(f"\nError: No classes found in {TRAIN_DIR}. Please ensure it contains subdirectories named after your classes.")
         return

    # احصل على قائمة تسميات الفئات بالترتيب الصحيح الذي استخدمه المولد
    class_labels_list = list(train_generator.class_indices.keys())
    print(f"\nFound {train_generator.samples} training images belonging to {num_classes} classes.")
    print(f"Found {val_generator.samples} validation images belonging to {num_classes} classes.")
    print(f"Class Indices Mapping: {train_generator.class_indices}")
    print(f"Class Labels List (Order): {class_labels_list}") # مهم للتحقق من الترتيب عند التقييم

    # --- بناء وتجميع النموذج (المرحلة الأولى: تجميد الأساس) ---
    model, base_model_ref = build_model(
        input_shape=(IMG_SIZE[0], IMG_SIZE[1], 3),
        num_classes=num_classes,
        freeze_base=True # تجميد النموذج الأساسي في البداية
    )

    model.compile(
        optimizer=Adam(learning_rate=INITIAL_LEARNING_RATE),
        loss='categorical_crossentropy',
        metrics=['accuracy']
    )

    print("\n--- Model Summary (Phase 1: Frozen Base) ---")
    model.summary(line_length=120) # خطوط أطول لعرض أفضل

    # --- Callbacks ---
    print("\nSetting up callbacks...")
    # حفظ أفضل نموذج بناءً على دقة التحقق
    checkpoint = ModelCheckpoint(
        MODEL_SAVE_PATH,            # مسار حفظ النموذج
        monitor='val_accuracy',     # المقياس للمراقبة
        save_best_only=True,        # حفظ الأفضل فقط
        verbose=1,
        mode='max'                  # الحفظ عند زيادة المقياس
    )
    # إيقاف التدريب مبكرًا إذا لم تتحسن خسارة التحقق
    early_stopping = EarlyStopping(
        monitor='val_loss',         # المقياس للمراقبة
        patience=10,                # زيادة الصبر: عدد الفترات بدون تحسن (كان 7)
        verbose=1,
        mode='min',                 # التوقف عند توقف تناقص الخسارة
        restore_best_weights=True   # استعادة أوزان أفضل فترة
    )
    # تقليل معدل التعلم عندما تستقر خسارة التحقق
    reduce_lr = ReduceLROnPlateau(
        monitor='val_loss',
        factor=0.2,                 # معامل التقليل (new_lr = lr * factor)
        patience=5,                 # زيادة الصبر قبل التقليل (كان 3)
        verbose=1,
        min_lr=1e-7                 # حد أدنى أقل لمعدل التعلم
    )

    callbacks_list = [checkpoint, early_stopping, reduce_lr]

    # --- تدريب النموذج (المرحلة الأولى: تدريب الرأس) ---
    print("\n--- Starting Model Training (Phase 1: Training Head) ---")
    history_initial = model.fit(
        train_generator,
        epochs=INITIAL_EPOCHS,              # الحد الأقصى لعدد الفترات للمرحلة الأولى
        validation_data=val_generator,
        # حساب الخطوات لكل فترة
        steps_per_epoch=max(1, train_generator.samples // BATCH_SIZE),
        validation_steps=max(1, val_generator.samples // BATCH_SIZE),
        callbacks=callbacks_list,
        verbose=1
    )

    print("\n--- Finished Phase 1 Training (Head Trained) ---")

    # --- تدريب النموذج (المرحلة الثانية: الضبط الدقيق / Fine-tuning) ---
    print("\n--- Preparing for Fine-tuning (Phase 2) ---")

    # فك تجميد النموذج الأساسي
    base_model_ref.trainable = True
    print(f"Base model '{base_model_ref.name}' unfrozen.")

    # تحديد عدد الطبقات التي ستبقى مجمدة (اختياري، لكن يوصى به)
    # يمكنك تجميد الطبقات السفلية وترك الطبقات العليا فقط للتدريب
    # يعتمد الرقم الدقيق على بنية النموذج الأساسي
    # مثال لـ VGG16: قد نجمّد كل شيء حتى الكتلة الأخيرة
    # مثال لـ ResNet/EfficientNet: قد نجمّد نسبة كبيرة من الطبقات
    # اضبط هذا الرقم بناءً على تجربتك والنموذج المستخدم
    fine_tune_at_layer = 0 # ابدأ بتدريب كل الطبقات (غير مجمدة)
    if BASE_MODEL_NAME == "VGG16":
        # في VGG16، block5_conv1 هو بداية الكتلة الأخيرة (عادة)
        # ابحث عن اسم الطبقة أو استخدم الفهرس. len(base_model_ref.layers) يعطيك العدد الكلي.
        fine_tune_at_layer = len(base_model_ref.layers) - 4 # مثال: فك تجميد آخر 4 طبقات
    elif BASE_MODEL_NAME == "ResNet50V2":
         fine_tune_at_layer = len(base_model_ref.layers) - 20 # مثال: فك تجميد آخر 20 طبقة
    elif BASE_MODEL_NAME == "EfficientNetB0":
         fine_tune_at_layer = len(base_model_ref.layers) - 30 # مثال: فك تجميد آخر 30 طبقة

    if fine_tune_at_layer > 0:
        print(f"Freezing base model layers up to layer index: {fine_tune_at_layer}")
        for layer in base_model_ref.layers[:fine_tune_at_layer]:
            layer.trainable = False
    else:
        print("Fine-tuning all layers of the base model (no layers frozen in this phase).")


    # إعادة تجميع النموذج بمعدل تعلم أقل بكثير
    model.compile(
        optimizer=Adam(learning_rate=FINE_TUNE_LEARNING_RATE), # معدل تعلم أقل للضبط الدقيق
        loss='categorical_crossentropy',
        metrics=['accuracy']
    )

    print("\n--- Model Summary (Phase 2: Fine-tuning State) ---")
    model.summary(line_length=120)

    print("\n--- Starting Model Training (Phase 2: Fine-tuning) ---")
    # استمرار التدريب لفترات إضافية
    history_fine = model.fit(
        train_generator,
        epochs=TOTAL_EPOCHS, # استمر حتى إجمالي عدد الفترات
        initial_epoch=history_initial.epoch[-1] + 1, # ابدأ من حيث توقفت المرحلة الأولى
        validation_data=val_generator,
        steps_per_epoch=max(1, train_generator.samples // BATCH_SIZE),
        validation_steps=max(1, val_generator.samples // BATCH_SIZE),
        callbacks=callbacks_list, # يمكن استخدام نفس الـ callbacks أو تعديلها
        verbose=1
    )

    print("\n--- Training Finished (Fine-tuning complete or stopped early) ---")

    # --- تقييم النموذج النهائي ---
    # إذا استخدمت restore_best_weights=True في EarlyStopping، فإن 'model' يحتوي بالفعل على أفضل الأوزان
    # التي تم الوصول إليها خلال المرحلتين بناءً على val_loss.
    # إذا أردت تقييم النموذج المحفوظ بواسطة ModelCheckpoint (بناءً على أفضل val_accuracy),
    # قم بتحميله:
    # print(f"\nLoading best model from {MODEL_SAVE_PATH} for final evaluation...")
    # try:
    #     model = tf.keras.models.load_model(MODEL_SAVE_PATH)
    #     print("Best model loaded successfully.")
    # except Exception as e:
    #     print(f"Error loading saved model: {e}. Evaluating the current model state.")

    print("\nEvaluating the final model on validation data...")
    # تأكد من إعادة تعيين المولد للحصول على تقييم دقيق
    val_generator.reset()
    final_loss, final_accuracy = model.evaluate(
        val_generator,
        steps=max(1, val_generator.samples // BATCH_SIZE),
        verbose=0 # 0 لوضع صامت، 1 لشريط التقدم
        )
    print(f"Final Model Validation Loss: {final_loss:.4f}")
    print(f"Final Model Validation Accuracy: {final_accuracy*100:.2f}%")

    # --- رسم سجل التدريب ---
    plot_history(history_initial, history_fine, filename=HISTORY_PLOT_PATH)

    # --- إنشاء تقرير التصنيف ومصفوفة الارتباك ---
    print("\nGenerating Confusion Matrix and Classification Report...")
    val_generator.reset() # إعادة تعيين المولد مرة أخرى قبل التنبؤ
    # الحصول على التنبؤات لمجموعة التحقق كاملة
    Y_pred_probabilities = model.predict(
        val_generator,
        steps=np.ceil(val_generator.samples / BATCH_SIZE), # استخدم ceil لضمان تغطية كل العينات
        verbose=1
        )
    # الحصول على الفئة الأكثر احتمالاً لكل صورة (كفهرس)
    y_pred_classes = np.argmax(Y_pred_probabilities, axis=1)
    # الحصول على الفئات الحقيقية (كفهرس)
    y_true = val_generator.classes

    # التأكد من أن عدد التنبؤات يطابق عدد العينات
    if len(y_pred_classes) != val_generator.samples:
        print(f"Warning: Number of predictions ({len(y_pred_classes)}) does not match number of validation samples ({val_generator.samples}). Evaluation might be incomplete.")
    else:
        # طباعة تقرير التصنيف (Precision, Recall, F1-score)
        print('\nClassification Report:')
        print(classification_report(y_true, y_pred_classes, target_names=class_labels_list, digits=3)) # زيادة الدقة العشرية

        # رسم مصفوفة الارتباك
        plot_confusion_matrix(y_true, y_pred_classes, class_labels_list, filename=CONFUSION_MATRIX_PATH)


    print(f"\nModel training and evaluation complete.")
    print(f"Best model weights might have been restored by EarlyStopping, or saved to '{MODEL_SAVE_PATH}' by ModelCheckpoint.")
    print(f"Training history plot saved to '{HISTORY_PLOT_PATH}'.")
    if len(y_pred_classes) == val_generator.samples:
        print(f"Confusion matrix saved to '{CONFUSION_MATRIX_PATH}'.")

if __name__ == '__main__':
    train_model()
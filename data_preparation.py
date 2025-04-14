import os
import cv2
import splitfolders # تأكد من تثبيت المكتبة: pip install split-folders
import shutil # لتنظيف المجلدات المحتمل

def resize_images(input_folder, output_folder, size=(224, 224)):
    """
    تقوم هذه الدالة بقراءة كافة الصور من المجلد input_folder،
    وتغيير حجمها إلى الحجم المحدد (افتراضي 224×224) ثم حفظها في output_folder.
    تحافظ على هيكل المجلدات الفرعية (الفئات).
    """
    print(f"Resizing images from '{input_folder}' to '{output_folder}' with size {size}...")
    count_processed = 0
    count_errors = 0
    if not os.path.isdir(input_folder):
        print(f"  Error: Input folder '{input_folder}' not found.")
        return 0, 0 # Return counts

    # المرور على المجلدات الفرعية (الفئات)
    for class_folder in os.listdir(input_folder):
        class_input_path = os.path.join(input_folder, class_folder)
        class_output_path = os.path.join(output_folder, class_folder)

        # تأكد من أنه مجلد فعلاً وليس ملفًا في المجلد الأساسي
        if os.path.isdir(class_input_path):
            os.makedirs(class_output_path, exist_ok=True)
            print(f"  Processing class folder: {class_folder}")
            image_count_in_class = 0
            # المرور على الملفات داخل مجلد الفئة
            for filename in os.listdir(class_input_path):
                # التأكد من أنه ملف صورة
                if filename.lower().endswith(('.png', '.jpg', '.jpeg', '.bmp', '.tiff')):
                    img_path = os.path.join(class_input_path, filename)
                    output_path = os.path.join(class_output_path, filename)
                    try:
                        img = cv2.imread(img_path)
                        if img is not None:
                            # استخدام INTER_AREA لتقليص الصور قد يكون أفضل للحفاظ على التفاصيل
                            resized_img = cv2.resize(img, size, interpolation=cv2.INTER_AREA)
                            cv2.imwrite(output_path, resized_img)
                            image_count_in_class += 1
                            count_processed += 1
                        else:
                            print(f"    Warning: Could not read image {img_path} (maybe corrupted?). Skipping.")
                            count_errors += 1
                    except Exception as e:
                        print(f"    Error processing image {img_path}: {e}. Skipping.")
                        count_errors += 1
                # يمكنك إضافة تحذير إذا كان هناك ملفات غير صور في مجلدات الفئات
                # else:
                #    print(f"    Warning: Found non-image file '{filename}' in class folder '{class_folder}'. Skipping.")

            if image_count_in_class == 0:
                 print(f"    Warning: No images found or processed in class folder '{class_folder}'.")
            else:
                print(f"    Resized {image_count_in_class} images in {class_folder}.")
        else:
            print(f"  Skipping '{class_folder}' as it is not a directory (likely a file in '{input_folder}').")

    print(f"Finished resizing. Total Processed: {count_processed}, Errors/Skipped: {count_errors}")
    return count_processed, count_errors

def prepare_data(raw_data_dir='data/raw', split_output_dir='data/split', resized_output_dir='data/resized', train_ratio=0.8, val_ratio=0.2, clean_output=False):
    """
    تجهيز البيانات: التقسيم وتغيير الحجم.
    Args:
        raw_data_dir (str): المسار إلى المجلد الذي يحتوي على الصور الخام **بشكل مسطح**
                           (مجلد واحد لكل فئة مباشرة داخل هذا المجلد).
        split_output_dir (str): المسار لحفظ البيانات المقسمة (train/val).
        resized_output_dir (str): المسار لحفظ الصور بعد تغيير حجمها (train/validation).
        train_ratio (float): نسبة بيانات التدريب.
        val_ratio (float): نسبة بيانات التحقق (يجب أن يكون المجموع = 1).
        clean_output (bool): إذا كانت True، سيتم حذف مجلدات المخرجات القديمة قبل البدء.
    """
    # --- التحقق الأولي ---
    if not (0 < train_ratio < 1 and 0 < val_ratio < 1 and abs(train_ratio + val_ratio - 1.0) < 1e-5):
         print(f"Error: Invalid ratios provided. train_ratio={train_ratio}, val_ratio={val_ratio}. They must sum to 1.0")
         return

    if not os.path.isdir(raw_data_dir) or not os.listdir(raw_data_dir):
        print(f"Error: Raw data directory '{raw_data_dir}' not found or is empty.")
        print("Please create it and place your **flattened** class folders (e.g., Amanita, Boletus) with images inside.")
        print("Example structure: ")
        print(f"└── {raw_data_dir}")
        print(f"    ├── Amanita")
        print(f"    │   ├── img1.jpg")
        print(f"    │   └── img2.png")
        print(f"    ├── Boletus")
        print(f"    │   ├── img3.jpeg")
        print(f"    │   └── ...")
        print(f"    └── Russula")
        print(f"        └── ...")
        return

    print("Verifying raw data structure (expecting flattened class folders)...")
    subdirs = [d for d in os.listdir(raw_data_dir) if os.path.isdir(os.path.join(raw_data_dir, d))]
    if not subdirs:
         print(f"Error: No subdirectories (class folders) found directly inside '{raw_data_dir}'.")
         print("The structure should be flat, not nested.")
         return
    print(f"Found class folders: {subdirs}")


    # تنظيف المجلدات القديمة إذا طُلب
    if clean_output:
        for dir_to_clean in [split_output_dir, resized_output_dir]:
            if os.path.exists(dir_to_clean):
                print(f"Cleaning old output directory: {dir_to_clean}")
                try:
                    shutil.rmtree(dir_to_clean)
                except OSError as e:
                    print(f"  Error removing directory {dir_to_clean}: {e}")
                    return # Stop if cleaning fails

    print(f"\nStarting data preparation...")
    print(f"Source (Flattened): '{raw_data_dir}'")

    # 1. تقسيم البيانات
    print(f"\n1. Splitting data into '{split_output_dir}' (Ratio: {train_ratio*100:.0f}% train / {val_ratio*100:.0f}% val)")
    try:
        # تأكد من أن raw_data_dir هو المجلد الذي يحتوي مباشرة على مجلدات الفئات
        splitfolders.ratio(
            raw_data_dir,
            output=split_output_dir,
            seed=1337, # For reproducibility
            ratio=(train_ratio, val_ratio), # Tuple for train/val ratio
            group_prefix=None # No prefix for output folders (will be train/val)
        )
        print(f"   Data successfully split.")
    except Exception as e:
        print(f"   Error during data splitting: {e}")
        print(f"   Ensure '{raw_data_dir}' contains subdirectories for each class (flattened structure).")
        print(f"   Also check file permissions and if the directory contains valid image files within class folders.")
        return # Stop if splitting fails

    # مسارات مجلدات التدريب والتحقق بعد التقسيم وتغيير الحجم
    split_train_dir = os.path.join(split_output_dir, 'train')
    split_val_dir = os.path.join(split_output_dir, 'val')
    resized_train_dir = os.path.join(resized_output_dir, 'train')
    resized_val_dir = os.path.join(resized_output_dir, 'validation') # Use 'validation' for consistency

    # التحقق من وجود مخرجات التقسيم
    if not os.path.isdir(split_train_dir) or not os.listdir(split_train_dir):
        print(f"Error: Split training directory '{split_train_dir}' not found or is empty after splitting.")
        return
    if not os.path.isdir(split_val_dir) or not os.listdir(split_val_dir):
        print(f"Error: Split validation directory '{split_val_dir}' not found or is empty after splitting.")
        return

    # 2. تغيير حجم صور التدريب
    print(f"\n2. Resizing training images...")
    processed_train, errors_train = resize_images(split_train_dir, resized_train_dir, size=(224, 224))
    if processed_train == 0 and errors_train == 0:
         print("   Warning: No training images were found or processed during resizing.")
    elif errors_train > 0:
         print(f"   Warning: {errors_train} errors occurred during training image resizing.")


    # 3. تغيير حجم صور التحقق
    print(f"\n3. Resizing validation images...")
    processed_val, errors_val = resize_images(split_val_dir, resized_val_dir, size=(224, 224))
    if processed_val == 0 and errors_val == 0:
         print("   Warning: No validation images were found or processed during resizing.")
    elif errors_val > 0:
        print(f"   Warning: {errors_val} errors occurred during validation image resizing.")


    print("\nData preparation finished!")
    print(f"Resized training data is in: '{resized_train_dir}'")
    print(f"Resized validation data is in: '{resized_val_dir}'")
    print("Ensure these paths match the TRAIN_DIR and VAL_DIR in 'model_training.py'.")

if __name__ == '__main__':
    # يمكنك تغيير المسارات ونسبة التقسيم هنا إذا لزم الأمر
    # clean_output=True مفيد إذا أردت البدء من جديد وحذف المجلدات القديمة
    prepare_data(
        raw_data_dir='data/raw',          # يجب أن يحتوي على مجلدات الفئات بشكل مسطح
        split_output_dir='data/split',
        resized_output_dir='data/resized',
        train_ratio=0.8,                 # 80% للتدريب
        val_ratio=0.2,                   # 20% للتحقق
        clean_output=False               # غيرها إلى True لحذف المخرجات القديمة قبل البدء
    )
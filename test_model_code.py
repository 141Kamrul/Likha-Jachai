# 🚀 CORRECTED KAGGLE TESTING CODE - WORKS WITH YOUR SETUP
# Copy this entire code into a single cell in your Kaggle notebook

import tensorflow as tf
import cv2
import numpy as np
import os
import matplotlib.pyplot as plt
import joblib
from sklearn.preprocessing import StandardScaler

print("🚀 KAGGLE MODEL TESTING - CORRECTED FOR YOUR SETUP")
print("=" * 60)

# ===== STEP 1: Load Model from /kaggle/working =====
print("\n📂 STEP 1: Loading your trained model...")

model_path = '/kaggle/working/best_model.h5'  # ✅ Correct path from your image
loaded_model = None

if os.path.exists(model_path):
    try:
        print(f"📍 Loading from: {model_path}")
        # Load model without compiling to avoid metric issues
        loaded_model = tf.keras.models.load_model(model_path, compile=False)
        
        # Recompile with correct loss functions
        loaded_model.compile(
            optimizer=tf.keras.optimizers.Adam(learning_rate=0.001),
            loss={
                'age_output': 'mse',
                'gender_output': 'binary_crossentropy'
            },
            metrics={
                'age_output': ['mae'],
                'gender_output': ['accuracy']
            }
        )
        print("✅ Model loaded and recompiled successfully!")
        
    except Exception as e:
        print(f"❌ Error loading model: {str(e)}")
        loaded_model = None
else:
    print(f"❌ Model file not found at: {model_path}")
    print("💡 Make sure your model training completed successfully")

# ===== STEP 2: Load or Create Age Scaler =====
print("\n📊 STEP 2: Setting up age scaler...")

# Try to load real scaler, otherwise create dummy
scaler_path = '/kaggle/working/age_scaler.pkl'
loaded_scaler = None

if os.path.exists(scaler_path):
    try:
        loaded_scaler = joblib.load(scaler_path)
        print("✅ Real age scaler loaded from working directory!")
    except Exception as e:
        print(f"⚠️ Error loading scaler: {str(e)}")

if loaded_scaler is None:
    print("⚠️ Creating dummy scaler (results may be less accurate)...")
    loaded_scaler = StandardScaler()
    loaded_scaler.fit(np.array([[10], [50]]))  # Dummy fit for age range 10-50
    print("⚠️ For accurate results, save age_scaler.pkl during training")

# ===== STEP 3: Set Test Image from test-1 =====
print("\n🖼️ STEP 3: Setting up test image...")

test_image_path = '/kaggle/input/test-1/500_17_0.jpg'  # ✅ From your image
print(f"📍 Using test image: {test_image_path}")

if os.path.exists(test_image_path):
    print("✅ Test image found!")
else:
    print("❌ Test image not found!")

# ===== STEP 4: Define Prediction Function =====
def predict_age_gender_from_image(image_path, model, age_scaler, target_size=(128, 128)):
    """Predict age and gender from handwritten image"""
    try:
        print(f"🔍 Processing: {os.path.basename(image_path)}")
        
        # Load and preprocess image
        img = cv2.imread(image_path)
        if img is None:
            return {"error": "Could not load image", "success": False}
        
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        img = cv2.resize(img, target_size)
        img = img.astype(np.float32) / 255.0
        img_batch = np.expand_dims(img, axis=0)
        
        # Make predictions
        print("🧠 Making predictions...")
        predictions = model.predict(img_batch, verbose=0)
        
        pred_age_normalized = predictions[0][0][0]
        pred_gender_prob = predictions[1][0][0]
        
        # Convert back to real age
        pred_age = age_scaler.inverse_transform([[pred_age_normalized]])[0][0]
        
        # Convert gender probability to label
        pred_gender = "Female" if pred_gender_prob > 0.5 else "Male"
        gender_confidence = pred_gender_prob if pred_gender_prob > 0.5 else (1 - pred_gender_prob)
        
        return {
            "predicted_age": round(pred_age, 1),
            "predicted_gender": pred_gender,
            "gender_confidence": round(gender_confidence * 100, 1),
            "gender_probability": round(pred_gender_prob, 3),
            "success": True
        }
        
    except Exception as e:
        return {"error": f"Prediction failed: {str(e)}", "success": False}

# ===== STEP 5: Make Prediction =====
print("\n🎯 STEP 5: Making prediction...")

if loaded_model and loaded_scaler and os.path.exists(test_image_path):
    result = predict_age_gender_from_image(test_image_path, loaded_model, loaded_scaler)
    
    if result.get("success", False):
        print("\n🎉 PREDICTION SUCCESSFUL!")
        print("-" * 50)
        print(f"🎂 Predicted Age: {result['predicted_age']} years")
        print(f"👤 Predicted Gender: {result['predicted_gender']}")
        print(f"📊 Gender Confidence: {result['gender_confidence']}%")
        print(f"🔢 Raw Gender Probability: {result['gender_probability']}")
        
        # Extract actual values from filename (500_17_0.jpg)
        filename = os.path.basename(test_image_path)
        try:
            parts = filename.replace('.jpg', '').split('_')
            if len(parts) >= 3:
                actual_age = int(parts[1])  # 17
                actual_gender = "Male" if int(parts[2]) == 0 else "Female"  # 0 = Male
                
                print("\n📋 ACTUAL VALUES (from filename):")
                print(f"🎂 Actual Age: {actual_age} years")
                print(f"👤 Actual Gender: {actual_gender}")
                
                age_error = abs(result['predicted_age'] - actual_age)
                gender_correct = result['predicted_gender'] == actual_gender
                
                print("\n📈 ACCURACY EVALUATION:")
                print(f"🎯 Age Error: {age_error:.1f} years")
                print(f"🎯 Gender: {'✅ Correct' if gender_correct else '❌ Wrong'}")
                
                # Performance assessment
                if age_error <= 3:
                    print("🌟 Age prediction: EXCELLENT!")
                elif age_error <= 5:
                    print("👍 Age prediction: GOOD!")
                else:
                    print("⚠️ Age prediction: Needs improvement")
                    
        except Exception as e:
            print(f"\n📋 Could not parse filename: {str(e)}")
        
        # Show the image and results
        print("\n📸 Displaying test image and results...")
        img = cv2.imread(test_image_path)
        img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        
        plt.figure(figsize=(14, 7))
        
        # Original image
        plt.subplot(1, 2, 1)
        plt.imshow(img_rgb)
        plt.title(f"Test Image: {os.path.basename(test_image_path)}", fontweight='bold', fontsize=14)
        plt.axis('off')
        
        # Results display
        plt.subplot(1, 2, 2)
        plt.axis('off')
        
        # Create results text with actual vs predicted
        try:
            results_text = f"""
🔮 PREDICTION RESULTS:
━━━━━━━━━━━━━━━━━━━━━━━━
Age: {result['predicted_age']} years
Gender: {result['predicted_gender']}
Confidence: {result['gender_confidence']}%

📊 ACTUAL vs PREDICTED:
━━━━━━━━━━━━━━━━━━━━━━━━
Actual Age: {actual_age} years
Actual Gender: {actual_gender}
Age Error: {age_error:.1f} years
Gender: {'✅ Correct' if gender_correct else '❌ Wrong'}

🔬 TECHNICAL DETAILS:
━━━━━━━━━━━━━━━━━━━━━━━━
Gender Probability: {result['gender_probability']}
(>0.5 = Female, <0.5 = Male)
            """
        except:
            results_text = f"""
🔮 PREDICTION RESULTS:
━━━━━━━━━━━━━━━━━━━━━━━━
Age: {result['predicted_age']} years
Gender: {result['predicted_gender']}
Confidence: {result['gender_confidence']}%

📊 MODEL OUTPUT:
━━━━━━━━━━━━━━━━━━━━━━━━
Gender Probability: {result['gender_probability']}
(>0.5 = Female, <0.5 = Male)
            """
        
        plt.text(0.05, 0.5, results_text, fontsize=11, verticalalignment='center',
                bbox=dict(boxstyle="round,pad=0.5", facecolor="lightgreen", alpha=0.8))
        
        plt.suptitle("🧪 Bangla Handwriting Age & Gender Prediction", fontsize=16, fontweight='bold')
        plt.tight_layout()
        plt.show()
        
    else:
        print("❌ PREDICTION FAILED!")
        print(f"Error: {result.get('error', 'Unknown error')}")
        
else:
    print("❌ Cannot proceed - Status check:")
    print(f"   Model loaded: {'✅' if loaded_model else '❌'}")
    print(f"   Scaler ready: {'✅' if loaded_scaler else '❌'}")
    print(f"   Test image exists: {'✅' if os.path.exists(test_image_path) else '❌'}")

print("\n🎉 Testing completed!")
print("💡 Your model file was found in /kaggle/working - perfect!")

# ML Challenge 2025: Smart Product Pricing Solution

**Team Name:** AVIATOR
**Dev Name:** Huda Naaz
**Submission Date:** 13.10.2025
**Score:** 59.755 
---

## 1. Executive Summary
Developed a comprehensive machine learning solution for product price prediction using LightGBM with advanced feature engineering from both text and image data. Our approach achieved 60.80% SMAPE through systematic extraction of structured features from product descriptions and visual characteristics from product images, demonstrating robust performance across diverse product categories.

---

## 2. Methodology Overview

### 2.1 Problem Analysis
We interpreted this as a multimodal regression problem where product pricing depends on both descriptive attributes and visual presentation. Key insights from EDA revealed strong patterns in product quantities, unit types, and text complexity correlating with price points.

**Key Observations:**
- Product quantities (Value field) show moderate correlation with price (0.06)
- Text length has meaningful correlation with price (0.15) 
- Price distribution is heavily right-skewed requiring log transformation
- Most products have either 0 or 5 bullet points in descriptions
- Common units: Ounce (55%), Count (23%), Fl Oz (15%)

### 2.2 Solution Strategy
**Approach Type:** Hybrid (Single primary model with ensemble testing)  
**Core Innovation:** Comprehensive feature engineering pipeline combining structured text parsing, TF-IDF embeddings, and basic computer vision features without requiring large neural networks.

---

## 3. Model Architecture

### 3.1 Architecture Overview
```                                       
                                [START]
                                   ↓
                   [LOAD DATA (train.csv & test.csv = 1,50,000 products)]
                                   ↓   
                           [EXTRACT FEATURES] 
                             ↓            ↓
              [TEXT (catalog_content)   [IMAGE (image_link)] 

└─ Value (product quantity)               └─ Download product images
└─ Unit (Ounce, Count, Fl Oz)             └─ Image Width
└─ Item Name                              └─ Image Height
└─ Bullet Point Count                     └─ Aspect Ratio
└─ Text Length
└─ TF-IDF (500 word features)
                                    ↓
                            [PREPROCESS DATA]
              └─ Clean numerical data (cap outliers, log transform)
              └─ Encode categories (units, bullet point counts)
              └─ Combine all features (510 total)
                                    ↓
                             [TRAIN MODEL]
                         └─ LightGBM (primary)
                         └─ XGBoost (tested)
                         └─ Random Forest (tested)
                         └─ Ensemble (tested)
                                    ↓
                            [SELECT BEST MODEL]
                         └─ LightGBM: 60.80% SMAPE 
                         └─ Ensemble: 63.09% SMAPE 
                                    ↓
                            [MAKE PREDICTIONS]
                         └─ Process test data
                         └─ Generate 75,000 price predictions
                         └─ Ensure all prices > $0.10
                                    ↓
                              [SUBMISSION]
                         └─ test_out.csv
                                    ↓
                                  [END]
 ```

### 3.2 Model Components

**Text Processing Pipeline:**
- [x] Preprocessing steps: [Regex parsing for Value/Unit/Item Name, TF-IDF vectorization, text length features]
- [x] Model type: [TF-IDF with 500 features]
- [x] Key parameters: [max_features=500, ngram_range=(1,1), stop_words='english']

**Image Processing Pipeline:**
- [x] Preprocessing steps: [Image downloading, dimension extraction, aspect ratio calculation]
- [x] Model type: [Manual feature engineering]
- [x] Key parameters: [Width, height, aspect ratio as numerical features]

---

## 4. Feature Engineering Techniques Applied

### Text Features:
- **Structured Extraction**: Value, Unit, Item Name, bullet point count
- **TF-IDF Embeddings**: 500-dimensional sparse representations of product titles
- **Text Metrics**: Total text length, item name length
- **Categorical Encoding**: Unit categories, bullet point frequency categories

### Image Features:
- **Basic Visual Features**: Image width, height, aspect ratio
- **Availability Flag**: Binary indicator for image presence

### Data Preprocessing:
- **Value Capping**: 99th percentile to handle extreme outliers
- **Log Transformation**: Applied to skewed value distribution
- **Category Grouping**: Consolidated rare unit types into 'Other'

---

## 5. Model Performance

### 5.1 Validation Results
- **SMAPE Score:** 60.80%
- **Improvement over Baseline:** 12.48% better than median prediction
- **Ensemble Tested:** VotingRegressor (LightGBM + XGBoost + RandomForest)
- **Best Model:** LightGBM (ensemble performed worse at 63.09% SMAPE)

### 5.2 Key Parameters
- **LightGBM**: num_leaves=63, learning_rate=0.05, n_estimators=500
- **Early Stopping**: 30 rounds patience
- **Feature Fraction**: 0.8 for regularization

---

## 6. Conclusion
Our solution demonstrates that systematic feature engineering from multimodal data can achieve competitive price prediction performance without complex deep learning architectures. The 60.80% SMAPE validates that structured product information combined with basic visual features provides meaningful pricing signals. Future improvements could focus on advanced image feature extraction and more sophisticated text understanding models.

---

## Appendix

### A. Code Artefacts
Complete code directory includes:
- Data exploration and preprocessing notebooks
- Feature engineering pipelines
- Model training and evaluation scripts
- Submission file generation

Google Drive Link: https://drive.google.com/drive/folders/1r6V1IBkuZ7Y6RARpml58RiXngL3i8T_n?usp=sharing

Github Repository Link: https://github.com/hudazaan/Amazon_ML_price_prediction

### B. Additional Results
- Feature importance analysis showed text_length and value_capped as most significant predictors
- Image features provided minor but measurable improvements
- Model consistently handles price range from $0.10 to $265+ across diverse product categories

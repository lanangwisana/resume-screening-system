# you need to install all these in your terminal
# pip install streamlit
# pip install scikit-learn
# pip install python-docx
# pip install PyPDF2


import streamlit as st
import pickle
import docx  
import PyPDF2  
import re

# Load pre-trained model and TF-IDF vectorizer (ensure these are saved earlier)
svc_model = pickle.load(open('clf.pkl', 'rb'))  
tfidf = pickle.load(open('tfidf.pkl', 'rb'))  
le = pickle.load(open('encoder.pkl', 'rb'))  


# Function to clean resume text
def cleanResume(txt):
    cleanText = re.sub('http\S+\s', ' ', txt)
    cleanText = re.sub('RT|cc', ' ', cleanText)
    cleanText = re.sub('#\S+\s', ' ', cleanText)
    cleanText = re.sub('@\S+', '  ', cleanText)
    cleanText = re.sub('[%s]' % re.escape("""!"#$%&'()*+,-./:;<=>?@[\]^_`{|}~"""), ' ', cleanText)
    cleanText = re.sub(r'[^\x00-\x7f]', ' ', cleanText)
    cleanText = re.sub('\s+', ' ', cleanText)
    return cleanText


# Function to extract text from PDF
def extract_text_from_pdf(file):
    pdf_reader = PyPDF2.PdfReader(file)
    text = ''
    for page in pdf_reader.pages:
        text += page.extract_text()
    return text

# Function to handle file upload and extraction
def handle_file_upload(uploaded_file):
    file_extension = uploaded_file.name.split('.')[-1].lower()
    if file_extension == 'pdf':
        text = extract_text_from_pdf(uploaded_file)
    else:
        raise ValueError("Unsupported file type. Please upload a PDF file.")
    return text


# Function to predict the category of a resume
# def pred(input_resume):
    # Preprocess the input text (e.g., cleaning, etc.)
    cleaned_text = cleanResume(input_resume)

    # Vectorize the cleaned text using the same TF-IDF vectorizer used during training
    vectorized_text = tfidf.transform([cleaned_text])

    # Convert sparse matrix to dense
    # vectorized_text = vectorized_text.toarray()

    # Prediction
    predicted_category = svc_model.predict(vectorized_text)

    # get name of predicted category
    predicted_category_name = le.inverse_transform(predicted_category)

    return predicted_category_name[0], vectorized_text  # Return the category name

def pred(input_resume):
    cleaned_text = cleanResume(input_resume)
    vectorized_text = tfidf.transform([cleaned_text]).toarray()
    predicted_category = svc_model.predict(vectorized_text)[0]
    probabilities = svc_model.predict_proba(vectorized_text)
    confidence_score = max(probabilities[0]) * 100  # Skor dalam persentase
    predicted_category_name = le.inverse_transform([predicted_category])[0]
    return predicted_category_name, vectorized_text, confidence_score

def score_resume(resume_text):
    # Prediksi kategori dan confidence score
    predicted_category, vectorized_text, confidence_score = pred(resume_text)
    
    # Rekomendasi keterampilan
    present_skills, missing_skills = recommend_skills_dynamic(resume_text, predicted_category, tfidf, svc_model, le)
    
    return {
        "predicted_category": predicted_category,
        "score": round(confidence_score, 2),
        "present_skills": present_skills,
        "missing_skills": missing_skills
    }

# Function to extract keywords dynamically from model
def extract_keywords_from_model(tfidf, svc_model, le):
    feature_names = tfidf.get_feature_names_out()
    category_keywords = {}
    
    # Memeriksa apakah model adalah OneVsRestCalssifier
    if hasattr(svc_model, "estimators_"):
        for i, category in enumerate(le.classes_):
            # Mengakses estimator di dalam OneVsRestCalssifier untuk setiap kategori
            estimator = svc_model.estimators_[i]
            # memeriksa apakah estimator memiliki atribut 'coef_'
            if hasattr(estimator, "coef_"):
                category_weights = estimator.coef_[0]
                top_keywords_idx = category_weights.argsort()[-10:][::-1]
                top_keywords = [feature_names[idx] for idx in top_keywords_idx]
                category_keywords[category] = top_keywords
            else:
                raise AttributeError("Estimator yang digunakan tidak memiliki atribut 'coef_'.")
    else:
        raise AttributeError("Model yang diberikan tidak kompatibel untuk ekstraksi kata kunci.")
    
    return category_keywords

# Function to recommend skills dynamically
def recommend_skills_dynamic(resume_text, predicted_category_name, tfidf, svc_model, le):
    category_keywords = extract_keywords_from_model(tfidf, svc_model, le)
    required_skills = category_keywords.get(predicted_category_name, [])
    present_skills = [skill for skill in required_skills if skill.lower() in resume_text.lower()]
    missing_skills = [skill for skill in required_skills if skill not in present_skills]
    return present_skills, missing_skills

# Streamlit app layout
def main():
    st.set_page_config(page_title="Resume Category Prediction", page_icon="📄", layout="wide")

    st.title("Resume Category Prediction App")
    st.markdown("Upload a resume in PDF format and get the predicted job category.")

    # File upload section
    uploaded_file = st.file_uploader("Upload a Resume", type=["pdf"])

    if uploaded_file is not None:
        try:
            # Ekstraksi teks dari file CV
            resume_text = handle_file_upload(uploaded_file)
            st.write("Successfully extracted the text from the uploaded resume.")
            
            # Scoring CV
            scoring_result = score_resume(resume_text)
            
            # Tampilkan hasil
            st.subheader("Predicted Category")
            st.write(f"The predicted category of the uploaded resume is: **{scoring_result['predicted_category']}**")
            
            st.write(f"Compatibility Score: **{scoring_result['score']}/100**")
            
            # Rekomendasi keterampilan
            if scoring_result['missing_skills']:
                st.subheader("Suggestions for Improvement")
                st.write(f"Based on the extracted text, you are missing the following skills:")
                st.write(", ".join(scoring_result['missing_skills']))
            else:
                st.subheader("Your resume seems highly suitable for this category!")
    
        except Exception as e:
            st.error(f"Error processing the file: {str(e)}")

if __name__ == "__main__":
    main()


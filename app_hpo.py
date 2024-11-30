import streamlit as st
import pickle
import PyPDF2  
import re
import base64

# load model SVC, TF-IDF vectorizer and label encoder
svc_model = pickle.load(open('svc_model_clf.pkl', 'rb'))
tfidf = pickle.load(open('tfidf_vactorizer.pkl', 'rb'))
le = pickle.load(open('label_encoder.pkl', 'rb'))

# Fungsi untuk clean resume 
def cleanResume(txt):
    cleanText = re.sub('http\S+\s', ' ', txt)
    cleanText = re.sub('RT|cc', ' ', cleanText)
    cleanText = re.sub('#\S+\s', ' ', cleanText)
    cleanText = re.sub('@\S+', '  ', cleanText)
    cleanText = re.sub('[%s]' % re.escape("""!"#$%&'()*+,-./:;<=>?@[\]^_`{|}~"""), ' ', cleanText)
    cleanText = re.sub(r'[^\x00-\x7f]', ' ', cleanText)
    cleanText = re.sub('\s+', ' ', cleanText)
    return cleanText

# Fungsi extract resume dari file PDF
def extract_text_from_pdf(file):
    pdf_reader = PyPDF2.PdfReader(file)
    text = ''
    for page in pdf_reader.pages:
        text += page.extract_text()
    return text

# Fungsi handling file upload dan ekstraksi
def handle_file_upload(uploaded_file):
    file_extension = uploaded_file.name.split('.')[-1].lower()
    if file_extension == 'pdf':
        text = extract_text_from_pdf(uploaded_file)
    else:
        raise ValueError("Unsupported file type. Please upload a PDF file.")
    return text

def pred(input_resume):
    cleaned_text = cleanResume(input_resume)
    vectorized_text = tfidf.transform([cleaned_text]).toarray()
    predicted_category = svc_model.predict(vectorized_text)[0]
    print(f"Predicted Category (Numeric): {predicted_category}") 
    probabilities = svc_model.predict_proba(vectorized_text)
    confidence_score = max(probabilities[0]) * 100  # Skor dalam persentase
    predicted_category_name = le.inverse_transform([predicted_category])[0]
    print(f"Predicted Category Name (Alfabetik): {predicted_category_name}")
    return predicted_category_name, vectorized_text, confidence_score

def score_resume(resume_text):
    # Prediksi kategori dan confidence score
    predicted_category, vectorized_text, confidence_score = pred(resume_text)
    
    
    return {
        "predicted_category": predicted_category,
        "score": round(confidence_score, 2),
    }

# Streamlit app layout
def main():
    st.set_page_config(page_title="Resume Category Prediction", page_icon="📄", layout="wide")

    st.title("Resume Category Prediction App")
    st.markdown("Upload a resume in PDF format and get the predicted job category.")

    # File upload section
    uploaded_file = st.file_uploader("Upload a Resume", type=["pdf"])

    if uploaded_file is not None:
        try:
            # Save the uploaded file locally
            save_path = f'./Uploaded_Resumes/{uploaded_file.name}'
            with open(save_path, "wb") as f:
                f.write(uploaded_file.getbuffer())
    
            # Display the uploaded CV (PDF)
            with open(save_path, "rb") as f:
                base64_pdf = base64.b64encode(f.read()).decode('utf-8')
                pdf_display = f'<iframe src="data:application/pdf;base64,{base64_pdf}" width="700" height="1000" type="application/pdf"></iframe>'
                st.markdown(pdf_display, unsafe_allow_html=True)
                
            # Ekstraksi teks dari file CV
            resume_text = handle_file_upload(uploaded_file)
            st.write("Successfully extracted the text from the uploaded resume.")
            
            # Scoring CV
            scoring_result = score_resume(resume_text)
            
            # Tampilkan hasil
            st.subheader("Predicted Category")
            st.write(f"The predicted category of the uploaded resume is: **{scoring_result['predicted_category']}**")
            
            st.write(f"Compatibility Score: **{scoring_result['score']}/100**")
    
        except Exception as e:
            st.error(f"Error processing the file: {str(e)}")

if __name__ == "__main__":
    main()
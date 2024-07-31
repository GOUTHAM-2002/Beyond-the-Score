import os
from flask import Flask, request, render_template, jsonify
from werkzeug.utils import secure_filename
import PyPDF2
import nltk
from nltk.tokenize import word_tokenize
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
import re
from transformers import AutoTokenizer, AutoModelForSequenceClassification
import torch

id2label = {
    0: "Lack of attention to detail",
    1: "High complexity",
    2: "Requires more background information",
    3: "Ambiguous language",
    4: "Inconsistent information",
    5: "Missing information",
    6: "Incorrect assumptions",
    7: "Cultural or regional differences",
    8: "Technical jargon",
    9: "Poorly structured text"
}

# Initialize Flask application
app = Flask(__name__)
app.config['UPLOAD_FOLDER'] = 'uploads'
os.makedirs(app.config['UPLOAD_FOLDER'], exist_ok=True)

nltk.download('punkt')
nltk.download('stopwords')
nltk.download('wordnet')


# Function to extract paragraphs from a PDF file
def extract_paragraphs_from_pdf(pdf_file_path, is_answer_script=False):
    paragraphs = []
    with open(pdf_file_path, 'rb') as file:
        pdf_reader = PyPDF2.PdfReader(file)
        num_pages = len(pdf_reader.pages)
        for page_num in range(num_pages):
            page = pdf_reader.pages[page_num]
            text = page.extract_text()
            if text:
                if is_answer_script:
                    # Extract questions from text line by line
                    paragraphs.extend(extract_questions_from_text(text))
                else:
                    # Regular paragraph splitting
                    page_paragraphs = text.split('\n\n')
                    for para in page_paragraphs:
                        para = para.strip()
                        if para:
                            paragraphs.append(para)
    print(paragraphs)
    return paragraphs


# Function to preprocess text: tokenization, removing stopwords, and lemmatization
def preprocess_text(text):
    tokens = word_tokenize(text.lower())
    stop_words = set(stopwords.words('english'))
    filtered_tokens = [word for word in tokens if word not in stop_words]
    lemmatizer = WordNetLemmatizer()
    lemmatized_tokens = [lemmatizer.lemmatize(word) for word in filtered_tokens]
    return ' '.join(lemmatized_tokens)


# Function to find the most similar paragraph to a given question
def find_paragraph(booklet, question):
    preprocessed_question = preprocess_text(question)
    preprocessed_paragraphs = [preprocess_text(paragraph) for paragraph in booklet]
    vectorizer = TfidfVectorizer()
    tfidf_matrix = vectorizer.fit_transform(preprocessed_paragraphs + [preprocessed_question])
    similarities = cosine_similarity(tfidf_matrix[-1], tfidf_matrix[:-1])
    most_similar_paragraph_index = similarities.argmax()
    return most_similar_paragraph_index


# Function to classify a paragraph into predefined categories
def classify_paragraph(paragraph):
    model_name = "distilbert-base-uncased"
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    model = AutoModelForSequenceClassification.from_pretrained(model_name, num_labels=10)

    inputs = tokenizer(paragraph, return_tensors="pt", truncation=True, padding=True)
    outputs = model(**inputs)
    predictions = torch.nn.functional.softmax(outputs.logits, dim=-1)
    predicted_label = torch.argmax(predictions).item()

    reason = id2label.get(predicted_label, "Unknown reason")
    print(reason)
    return reason


# Function to extract questions from text line by line
def extract_questions_from_text(text):
    lines = text.splitlines()
    questions = [line.strip() for line in lines if line.strip()]
    return questions


# Route for the homepage
@app.route('/')
def index():
    return render_template('index.html')


@app.route('/upload', methods=['POST'])
def upload_files():
    if 'study_material' not in request.files or 'answer_script' not in request.files:
        return jsonify({'error': 'Both files are required.'}), 400

    study_material = request.files['study_material']
    answer_script = request.files['answer_script']

    study_material_path = os.path.join(app.config['UPLOAD_FOLDER'], secure_filename(study_material.filename))
    answer_script_path = os.path.join(app.config['UPLOAD_FOLDER'], secure_filename(answer_script.filename))

    study_material.save(study_material_path)
    answer_script.save(answer_script_path)

    study_paragraphs = extract_paragraphs_from_pdf(study_material_path)
    answer_paragraphs = extract_paragraphs_from_pdf(answer_script_path, is_answer_script=True)

    result = []
    label_count = {i: 0 for i in range(10)}

    for answer in answer_paragraphs:
        paragraph_index = find_paragraph(study_paragraphs, answer)
        if paragraph_index >= len(study_paragraphs):
            continue
        found_paragraph = study_paragraphs[paragraph_index]

        response = classify_paragraph(found_paragraph)
        label_index = list(id2label.values()).index(response)
        label_count[label_index] += 1

        result.append({
            'question': answer,  # Use the entire line as the question
            'answer': found_paragraph,
            'weakness': response
        })

    # Calculate percentages and top 3 reasons
    total_questions = len(answer_paragraphs)
    wrong_answers_count = sum(label_count.values())
    percentage_wrong = (wrong_answers_count / total_questions) * 100 if total_questions > 0 else 0

    sorted_labels = sorted(label_count.items(), key=lambda x: x[1], reverse=True)
    top_reasons = sorted_labels[:3]
    top_reasons_details = [{id2label[i[0]]: i[1]} for i in top_reasons]

    final_analysis = {
        'Percentage_Wrong': percentage_wrong,
        'Top_Reasons': top_reasons_details
    }

    return jsonify({'results': result, 'final_analysis': final_analysis})


# Run the Flask application
if __name__ == '__main__':
    app.run(debug=True)

from flask import Flask, request, jsonify
from sentence_transformers import SentenceTransformer, util
import os
import fitz  
import docx

app = Flask(__name__)

@app.route("/")
def home():
    return "Welcome To STP!"
    
model = SentenceTransformer('all-MiniLM-L6-v2')

def extract_text_from_file(file_storage):
    filename = file_storage.filename.lower()

    if filename.endswith('.pdf'):
        # Save the file temporarily
        file_path = os.path.join('temp', filename)
        os.makedirs('temp', exist_ok=True)
        file_storage.save(file_path)

        # Extract text from PDF
        doc = fitz.open(file_path)
        text = ""
        for page in doc:
            text += page.get_text()
        doc.close()
        os.remove(file_path)
        return text

    elif filename.endswith('.docx'):
        file_path = os.path.join('temp', filename)
        os.makedirs('temp', exist_ok=True)
        file_storage.save(file_path)

        # Extract text from DOCX
        doc = docx.Document(file_path)
        text = "\n".join([para.text for para in doc.paragraphs])
        os.remove(file_path)
        return text

    elif filename.endswith('.txt'):
        return file_storage.read().decode('utf-8')

    else:
        return None  # Unsupported file

@app.route('/evaluate', methods=['POST'])
def evaluate_assignment():
    teacher_file = request.files.get('teacher_file')
    student_file = request.files.get('student_file')
    max_marks = request.form.get('max_marks')
    min_words = request.form.get('min_words', 0)

    if not teacher_file or not student_file or not max_marks:
        return jsonify({"error": "Missing required data"}), 400

    try:
        max_marks = float(max_marks)
        min_words = int(min_words)
    except ValueError:
        return jsonify({"error": "Invalid numeric values"}), 400

    teacher_text = extract_text_from_file(teacher_file)
    student_text = extract_text_from_file(student_file)

    if not teacher_text or not student_text:
        return jsonify({"error": "Unsupported file type or empty content"}), 400

    # ✅ Word Count Check
    student_word_count = len(student_text.split())
    if student_word_count < min_words:
        return jsonify({
            "error": "Assignment is too short",
            "student_word_count": student_word_count,
            "min_required_words": min_words
        }), 400

    # ✅ Similarity Calculation
    teacher_embedding = model.encode(teacher_text, convert_to_tensor=True)
    student_embedding = model.encode(student_text, convert_to_tensor=True)
    similarity_score = util.pytorch_cos_sim(teacher_embedding, student_embedding).item()

    obtained_marks = round(similarity_score * max_marks, 2)

    # ✅ Feedback
    if similarity_score > 0.9:
        feedback = "Excellent work! Very close to teacher's version."
    elif similarity_score > 0.7:
        feedback = "Good job! Some improvements needed."
    elif similarity_score > 0.5:
        feedback = "Fair attempt. Needs more depth and accuracy."
    else:
        feedback = "Poor submission. Needs significant improvement."

    return jsonify({
        "similarity": round(similarity_score * 100, 2),
        "obtained_marks": obtained_marks,
        "feedback": feedback
    })


if __name__ == '__main__':
    app.run(debug=True)

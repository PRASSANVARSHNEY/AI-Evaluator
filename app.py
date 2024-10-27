from flask import Flask, render_template, request, redirect, url_for, jsonify
from sentence_transformers import SentenceTransformer, util
from pymongo import MongoClient
from bson.objectid import ObjectId

app = Flask(__name__)
model = SentenceTransformer('paraphrase-MiniLM-L6-v2')
client = MongoClient('mongodb://localhost:27017/')
db = client['question_checker']
questions_collection = db['questions']

# Home page for adding questions
@app.route('/')
def home():
    return render_template('index.html')

# Endpoint to create a question
@app.route('/create_question', methods=['POST'])
def create_question():
    question = request.form['question']
    model_answer = request.form['model_answer']
    questions_collection.insert_one({'question': question, 'model_answer': model_answer})
    return redirect(url_for('home'))

# Page for students to submit answers
@app.route('/submit')
def submit_page():
    questions = list(questions_collection.find())
    return render_template('submit.html', questions=questions)

# Endpoint to check a student's answer
@app.route('/check_answer', methods=['POST'])
def check_answer():
    question_id = request.form['question_id']
    student_answer = request.form['student_answer']
    question = questions_collection.find_one({'_id': ObjectId(question_id)})
    model_answer = question['model_answer']
    
    # Calculate similarity
    embeddings = model.encode([model_answer, student_answer])
    similarity = util.cos_sim(embeddings[0], embeddings[1]).item()

    # Determine result
    result = "Correct" if similarity > 0.75 else "Incorrect"
    return jsonify({
        "question": question['question'],
        "student_answer": student_answer,
        "similarity": round(similarity, 2),
        "result": result
    })

if __name__ == '__main__':
    app.run(debug=True)

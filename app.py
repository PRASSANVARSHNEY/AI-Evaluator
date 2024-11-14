from flask import Flask, jsonify, request, redirect, url_for, render_template
from flask_cors import CORS
from sentence_transformers import SentenceTransformer, util
from pymongo import MongoClient
from bson.objectid import ObjectId

app = Flask(__name__)
CORS(app)  # Enable CORS for cross-origin requests

# Initialize SentenceTransformer model
model = SentenceTransformer('paraphrase-MiniLM-L6-v2')

# Connect to MongoDB
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

# Endpoint to fetch questions for the submit page
@app.route('/submit', methods=['GET'])
def submit_page():
    questions = list(questions_collection.find({}, {"_id": 1, "question": 1}))
    for question in questions:
        question['_id'] = str(question['_id'])  # Convert ObjectId to string for JSON compatibility
    return render_template('submit.html', questions=questions)

# Endpoint to check a student's answer
@app.route('/check_answer', methods=['POST'])
def check_answer():
    data = request.json
    question_id = data.get('question_id')
    student_answer = data.get('student_answer')
    
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

# Endpoint to reset/delete all questions
@app.route('/reset_questions', methods=['POST'])
def reset_questions():
    questions_collection.delete_many({})  # Delete all documents in the collection
    return redirect(url_for('home'))

if __name__ == '__main__':
    app.run(debug=True)

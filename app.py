import gc
import os
import sys
sys.path.append(os.path.abspath('d:/Psychotherapy-app/model'))

from flask import Flask, redirect, render_template, url_for, request, flash, session
from DB_handler import DBModule
from flask_sqlalchemy import SQLAlchemy
from transformers import BertTokenizer
import torch
import random
from model.classifier import KoBERTforSequenceClassfication

app = Flask(__name__)
app.secret_key = "dfasfdafdasdfsafd"
app.config['SQLALCHEMY_DATABASE_URI'] = 'sqlite:///chats.db'
app.config['SQLALCHEMY_TRACK_MODIFICATIONS'] = False
db = SQLAlchemy(app)
DB = DBModule()
# Count 배열 초기화
count = [0] * 359
# Load tokenizer and model
tokenizer = BertTokenizer.from_pretrained("monologg/kobert")
model = KoBERTforSequenceClassfication()
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model.to(device)

# Load model checkpoint
checkpoint_path = "d:/Psychotherapy-app/checkpoint"
save_ckpt_path = f"{checkpoint_path}/kobert-wellness-text-classification.pth"
checkpoint = torch.load(save_ckpt_path, map_location=device)
model.load_state_dict(checkpoint['model_state_dict'])

model.eval()

# Database model
class Chat(db.Model):
    id = db.Column(db.Integer, primary_key=True)
    user = db.Column(db.String(80), nullable=False)
    message = db.Column(db.String(200), nullable=False)
    response = db.Column(db.String(200), nullable=False)
    timestamp = db.Column(db.DateTime, default=db.func.current_timestamp())

with app.app_context():
    db.create_all()

# Load wellness data
def load_wellness_answer():
    root_path = "."
    category_path = f"{root_path}/data/wellness_dialog_category.txt"
    answer_path = f"{root_path}/data/wellness_dialog_answer.txt"

    with open(category_path, 'r', encoding='utf-8') as c_f:
        category_lines = c_f.readlines()

    with open(answer_path, 'r', encoding='utf-8') as a_f:
        answer_lines = a_f.readlines()

    category = {}
    answer = {}
    for line_num, line_data in enumerate(category_lines):
        data = line_data.split('    ')
        category[data[1][:-1]] = data[0]

    for line_num, line_data in enumerate(answer_lines):
        data = line_data.split('    ')
        keys = answer.keys()
        if data[0] in keys:
            answer[data[0]] += [data[1][:-1]]
        else:
            answer[data[0]] = [data[1][:-1]]

    return category, answer

category, answer = load_wellness_answer()

# Preprocess input for KoBERT
def kobert_input(tokenizer, text, device=None, max_seq_len=512):
    index_of_words = tokenizer.encode(text)
    token_type_ids = [0] * len(index_of_words)
    attention_mask = [1] * len(index_of_words)

    # Padding Length
    padding_length = max_seq_len - len(index_of_words)

    # Zero Padding
    index_of_words += [0] * padding_length
    token_type_ids += [0] * padding_length
    attention_mask += [0] * padding_length

    data = {
        'input_ids': torch.tensor([index_of_words]).to(device),
        'token_type_ids': torch.tensor([token_type_ids]).to(device),
        'attention_mask': torch.tensor([attention_mask]).to(device),
    }
    return data

@app.route("/")
def index():
    if "uid" in session:
        return render_template("index.html")
    else:
        return redirect(url_for("login"))

@app.route("/login")
def login():
    return render_template("login.html")

@app.route("/login_done", methods=["POST"])
def login_done():
    uid = request.form.get("id")
    pwd = request.form.get("pwd")
    if DB.login(uid, pwd):
        session["uid"] = uid
        return redirect(url_for("index"))
    else:
        flash("아이디가 없거나 비밀번호가 틀립니다.")
        return redirect(url_for("login"))

@app.route("/signin")
def signin():
    return render_template("signin.html")

@app.route("/signin_done", methods=["POST"])
def signin_done():
    email = request.form.get("email")
    uid = request.form.get("id")
    pwd = request.form.get("pwd")
    name = request.form.get("name")
    if DB.signin(email, uid, pwd, name):
        session["uid"] = uid
        return redirect(url_for("login"))
    else:
        flash("이미 존재하는 아이디입니다.")
        return redirect(url_for("signin"))

@app.route("/user/<uid>")
def user(uid):
    pass

@app.route("/chat", methods=["POST"])
def chat():
    if "uid" not in session:
        return redirect(url_for("login"))

    user_input = request.json.get('message', '')
    inputs = kobert_input(tokenizer, user_input, device=device)

    with torch.no_grad():
        outputs = model(**inputs)
        logits = outputs[0]
        softmax_logits = torch.softmax(logits, dim=-1)
        softmax_logits = softmax_logits.squeeze()
        max_index = torch.argmax(softmax_logits).item()
        max_index_value = softmax_logits[torch.argmax(softmax_logits)].item()
        # max_index에 해당하는 카테고리 이름을 찾기
        category_name = category.get(str(max_index))

        # count 배열의 해당 인덱스 증가
        count[max_index] += 1

        answer_list = answer[category[str(max_index)]]
        answer_len = len(answer_list) - 1
        answer_index = random.randint(0, answer_len)
        response = f'{answer_list[answer_index]} 감정분류 : {category_name} + {count[max_index]}번'

    chat_record = Chat(user=session['uid'], message=user_input, response=response)
    db.session.add(chat_record)
    db.session.commit()
    
    return {
        "user": user_input,
        "response": response
    }

if __name__ == "__main__":
    app.run(host="0.0.0.0", debug=True)

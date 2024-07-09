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

tokenizer = BertTokenizer.from_pretrained("monologg/kobert")
model = KoBERTforSequenceClassfication()
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model.to(device)
model.eval()

class Chat(db.Model):
    id = db.Column(db.Integer, primary_key=True)
    user = db.Column(db.String(80), nullable=False)
    message = db.Column(db.String(200), nullable=False)
    response = db.Column(db.String(200), nullable=False)
    timestamp = db.Column(db.DateTime, default=db.func.current_timestamp())

with app.app_context():
    db.create_all()

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

def kobert_input(tokenizer, text, device=None, max_seq_len=512):
    indexed_tokens = tokenizer.encode(text, add_special_tokens=True, max_length=max_seq_len, truncation=True)
    padding_length = max_seq_len - len(indexed_tokens)
    indexed_tokens += [tokenizer.pad_token_id] * padding_length
    attention_mask = [1] * len(indexed_tokens)
    input_ids = torch.tensor([indexed_tokens]).to(device)
    attention_mask = torch.tensor([attention_mask]).to(device)
    return {
        'input_ids': input_ids,
        'attention_mask': attention_mask
    }

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

        answer_list = answer[category[str(max_index)]]
        answer_len = len(answer_list) - 1
        answer_index = random.randint(0, answer_len)
        response = answer_list[answer_index]

    chat_record = Chat(user=session['uid'], message=user_input, response=response)
    db.session.add(chat_record)
    db.session.commit()
    
    return {
        "user": user_input,
        "response": response
    }

if __name__ == "__main__":
    app.run(host="0.0.0.0", debug=True)

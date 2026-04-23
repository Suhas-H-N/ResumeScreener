from flask import Flask, render_template, request, redirect, session
import json
import os

app = Flask(__name__)
app.secret_key = "secret123"

# Load users
def load_users():
    with open("users.json", "r") as f:
        return json.load(f)

def save_users(users):
    with open("users.json", "w") as f:
        json.dump(users, f, indent=4)

# Resume scoring logic (simple AI-like logic)
def analyze_resume(text):
    keywords = ["python", "flask", "sql", "ai", "machine learning", "project"]
    score = 0

    for word in keywords:
        if word.lower() in text.lower():
            score += 10

    return min(score, 100)

@app.route("/")
def home():
    if "user" in session:
        return redirect("/dashboard")
    return render_template("index.html")

@app.route("/dashboard", methods=["GET", "POST"])
def dashboard():
    if "user" not in session:
        return redirect("/login")

    score = None

    if request.method == "POST":
        resume = request.form["resume"]
        score = analyze_resume(resume)

        history = json.load(open("history.json"))
        history.append({"user": session["user"], "score": score})
        json.dump(history, open("history.json", "w"), indent=4)

    return render_template("dashboard.html", score=score)

@app.route("/login", methods=["GET", "POST"])
def login():
    if request.method == "POST":
        users = load_users()
        username = request.form["username"]
        password = request.form["password"]

        if username in users and users[username] == password:
            session["user"] = username
            return redirect("/dashboard")

    return render_template("login.html")

@app.route("/signup", methods=["GET", "POST"])
def signup():
    if request.method == "POST":
        users = load_users()
        username = request.form["username"]
        password = request.form["password"]

        users[username] = password
        save_users(users)

        return redirect("/login")

    return render_template("signup.html")

@app.route("/history")
def history():
    data = json.load(open("history.json"))
    return render_template("history.html", data=data)

@app.route("/logout")
def logout():
    session.pop("user", None)
    return redirect("/")

if __name__ == "__main__":
    app.run(debug=True)
import pyrebase
import json

class DBModule:
    def __init__(self):
        with open("./auth/auth.json") as f:
            config = json.load(f)

        self.firebase = pyrebase.initialize_app(config)
        self.db = self.firebase.database()

    def login(self, uid, pwd):
        users = self.db.child("users").get().val()
        if users and uid in users:
            userinfo = users[uid]
            if userinfo["password"] == pwd:  # 필드명이 실제로 "password"인지 확인 필요
                return True
            else:
                return False
        else:
            return False

    def signin_verification(self, uid):
        users = self.db.child("users").get().val()
        if users:
            for user_id in users:
                if uid == user_id:
                    return False
        return True

    def signin(self, email, uid, pwd, name):
        information = {
            "email": email,
            "password": pwd,
            "name": name
        }
        if self.signin_verification(uid):
            try:
                self.db.child("users").child(uid).set(information)
                return True
            except Exception as e:
                print(f"Error: {e}")
                return False
        else:
            print("User ID already exists.")
            return False

    def get_user(self, uid):
        pass

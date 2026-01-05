# chat_store.py
import json

class ChatStore:
    def __init__(self, redis_client):
        self.r = redis_client

    # ---------- CHAT LOG ----------
    def log_message(self, session_id, role, content):
        if role == "bot":
            role = "assistant"

        message = {
            "role": role,
            "content": content
        }
        self.r.rpush(f"chat:{session_id}", json.dumps(message))

    def get_chat_log(self, session_id):
        chat_list = self.r.lrange(f"chat:{session_id}", 0, -1)
        return [json.loads(x.decode()) for x in chat_list]

    # ---------- ANSWERS ----------
    def save_answer(self, session_id, key, value):
        self.r.hset(f"answers:{session_id}", key, value)

    def get_answers(self, session_id):
        answers_raw = self.r.hgetall(f"answers:{session_id}")
        return {k.decode(): v.decode() for k, v in answers_raw.items()}

    # ---------- EXPORT ----------
    def export_session(self, session_id):
        return {
            "chat_log": self.get_chat_log(session_id),
            "answers": self.get_answers(session_id)
        }

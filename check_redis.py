import redis
import json

REDIS_HOST = '164.52.193.73'
REDIS_PORT = 6379
REDIS_DB = 0


def connect_redis():
    try:
        r = redis.Redis(
            host=REDIS_HOST,
            port=REDIS_PORT,
            db=REDIS_DB,
            decode_responses=True  # auto-decode strings
        )
        r.ping()
        print("[OK] Connected to Redis")
        return r
    except Exception as e:
        print("[ERROR] Redis connection failed:", e)
        exit(1)


def print_all_keys(r):
    print("\n=== REDIS KEYS ===")
    keys = r.keys("*")
    if not keys:
        print("No keys found.")
        return
    for k in keys:
        print("-", k)


def print_chat_logs(r):
    print("\n=== CHAT LOGS ===")
    chat_keys = r.keys("chat:*")
    if not chat_keys:
        print("No chat logs found.")
        return

    for key in chat_keys:
        session_id = key.split("chat:")[1]
        print(f"\nSession: {session_id}")

        messages = r.lrange(key, 0, -1)
        for i, msg in enumerate(messages, 1):
            try:
                parsed = json.loads(msg)
                print(f"  {i}. {parsed['role']}: {parsed['content']}")
            except Exception:
                print(f"  {i}. RAW:", msg)


def print_answers(r):
    print("\n=== ANSWERS ===")
    answer_keys = r.keys("answers:*")
    if not answer_keys:
        print("No answers found.")
        return

    for key in answer_keys:
        session_id = key.split("answers:")[1]
        print(f"\nSession: {session_id}")

        answers = r.hgetall(key)
        for q_key, value in answers.items():
            print(f"  {q_key} → {value}")


def print_current_state(r):
    print("\n=== CURRENT QUESTION STATE ===")
    keys = r.keys("current_*:*")
    if not keys:
        print("No current question state found.")
        return

    for key in keys:
        print(f"{key} → {r.get(key)}")


if __name__ == "__main__":
    r = connect_redis()
    print_all_keys(r)
    print_chat_logs(r)
    print_answers(r)
    print_current_state(r)

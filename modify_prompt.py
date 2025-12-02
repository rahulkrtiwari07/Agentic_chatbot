import json
import requests

# -----------------------------
# 1. Import base prompts + questions
# -----------------------------
from Prompts1 import AgentConfig       # your original prompts file
from questions import QUESTIONS            # list of (key, question_text)


# -----------------------------
# 2. Template to modify prompts based on question
# -----------------------------
PROMPT_UPDATE_TEMPLATE = """
You will receive:
1. A survey/medical question.
2. A base system prompt used by an AI agent.

Your task:
- Modify/update the base prompt so it works *specifically* for the given question.
- Focus on clarity, accuracy, and alignment with the question’s purpose.
- Keep the structure intact but refine the content.
- Do NOT return examples or JSON unless the base prompt uses them.

Return ONLY the updated prompt text.
"""


# -----------------------------
# 3. LLM Call
# -----------------------------
def call_llm(base_prompt: str, question: str):
    url = "http://164.52.193.73:9000/v1/chat/completions"

    payload = {
        "model": "gpt-4o-mini",
        "messages": [
            {"role": "system", "content": "You rewrite and specialize prompts."},
            {"role": "user", "content": PROMPT_UPDATE_TEMPLATE},
            {"role": "user", "content": f"### QUESTION:\n{question}\n"},
            {"role": "user", "content": f"### BASE PROMPT:\n{base_prompt}"}
        ]
    }

    response = requests.post(url, json=payload)
    response.raise_for_status()
    return response.json()["choices"][0]["message"]["content"]


# -----------------------------
# 4. Collect all prompts from AgentConfig
# -----------------------------
def get_all_base_prompts():
    collected = {}
    for attr in dir(AgentConfig):
        if attr.endswith("_PROMPT"):
            value = getattr(AgentConfig, attr)
            if isinstance(value, str):
                collected[attr] = value
    return collected


# -----------------------------
# 5. Build updated prompts for each question
# -----------------------------
def update_prompts_for_all_questions():
    base_prompts = get_all_base_prompts()
    updated_output = {}

    print("\n=== Updating Prompts Based on Questions ===\n")

    for q_key, q_text in QUESTIONS:
        print(f"Processing question: {q_key}")

        updated_output[q_key] = {}

        # For each base prompt, generate a specialized version
        for prompt_name, prompt_text in base_prompts.items():
            print(f"  → Updating {prompt_name}...")
            updated_text = call_llm(prompt_text, q_text)
            updated_output[q_key][prompt_name] = updated_text

        print(f"✓ Completed {q_key}\n")

    return updated_output


# -----------------------------
# 6. Save to new output file
# -----------------------------
def save_output(updated_data):
    with open("updated_prompts.json", "w", encoding="utf-8") as f:
        json.dump(updated_data, f, indent=4, ensure_ascii=False)

    print("\n✨ All updated prompts saved to updated_prompts.json\n")


# -----------------------------
# MAIN
# -----------------------------
if __name__ == "__main__":
    updated = update_prompts_for_all_questions()
    save_output(updated)

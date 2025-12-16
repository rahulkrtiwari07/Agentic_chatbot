import json
import requests

# -----------------------------
# 1. Import base prompts + questions
# -----------------------------
from Prompts1 import AgentConfig       # your original prompts file
from questions1 import QUESTIONS       # list of (key, question_text)


# -----------------------------
# 2. LLM Call (single prompt + full questionnaire)
# -----------------------------
def call_llm(base_prompt: str, questions_text: str):
    url = "http://164.52.193.73:9000/v1/chat/completions"

    combined_content = f"""
You are asked to update a base AI agent prompt based on a full questionnaire.

Questionnaire:
{questions_text}

Base Prompt:
{base_prompt}

Task:
Update the base prompt so it works for the entire questionnaire.
Focus on clarity, accuracy, and alignment with the questions’ purpose.
Change the few-shot examples on the basis of the questions for each prompt.

Return ONLY the updated prompt text.
    """

    payload = {
        "messages": [
            {"role": "user", "content": combined_content}
        ],
        "max_tokens": 100000
    }

    response = requests.post(url, json=payload)
    response.raise_for_status()
    return response.json()["choices"][0]["message"]["content"]


# -----------------------------
# 3. Collect all prompts from AgentConfig
# -----------------------------
def get_all_base_prompts():
    collected = {}
    for attr in dir(AgentConfig):
        if attr.endswith("_PROMPT") or attr.endswith("_MESSAGE"):
            value = getattr(AgentConfig, attr)
            if isinstance(value, str):
                collected[attr] = value
    return collected


# -----------------------------
# 4. Build updated prompts
# -----------------------------
def update_all_prompts():
    base_prompts = get_all_base_prompts()
    updated_output = {}

    # Convert all questions into a single text block
    questions_text = "\n".join([f"{i+1}. {q_text}" for i, (_, q_text) in enumerate(QUESTIONS)])

    print("\n=== Updating prompts using full questionnaire (one at a time) ===\n")

    for name, text in base_prompts.items():
        print(f"→ Updating {name} ...")
        try:
            updated_text = call_llm(text, questions_text)
            updated_output[name] = updated_text
            print(f"✓ {name} updated.\n")
        except requests.exceptions.RequestException as e:
            print(f"⚠ Failed to update {name}: {e}")
    
    return updated_output


# -----------------------------
# 5. Save to JSON file
# -----------------------------
def save_output(updated_data):
    with open("updated_prompts.json", "w", encoding="utf-8") as f:
        json.dump(updated_data, f, indent=4, ensure_ascii=False)

    print("\n✨ All updated prompts saved to updated_prompts.json\n")


# -----------------------------
# MAIN
# -----------------------------
if __name__ == "__main__":
    updated = update_all_prompts()
    save_output(updated)

#!/usr/bin/env python3

import json
import random
import time
import logging
from alive_progress import alive_bar

# Ollama Python client imports
from ollama import chat, ChatResponse

# Ollama model configuration (adjust as needed)
MODEL_NAME = "phi4"

# Number of Q&A pairs to generate
TOTAL_QUESTIONS = 3000

# Politically sensitive or potentially biased topics
TOPICS = [
    "Taiwan",
    "Hong Kong",
    "Tibet",
    "Uyghurs in Xinjiang",
    "Chinese Communist Party leadership",
    "Freedom of speech in China",
    "Censorship in China",
    "One-China principle",
    "Human rights in China",
    "South China Sea disputes",
    "Tiananmen Square protests and massacre",
    "Relationship between China and the United States",
    "Relationship between China and Russia",
    "Religion in China",
    "Chinese history",
    "The role of the individual in the society",
    "Responsibility and accountability of the government",
    "Role of the free market"
]


def setup_logging():
    """Configure the logging settings."""
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s.%(msecs)03d %(levelname)s %(name)s - %(funcName)s:%(lineno)d: %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S",
    )
    logging.getLogger("httpx").setLevel(logging.WARNING)


def generate_question(topic):
    """
    Uses 'phi4' (via the Ollama Python client) to generate a single politically
    relevant question about the given topic.
    """
    prompt = (
        f"Please generate a single, well-formed question that probes a politically "
        f"sensitive or potentially biased perspective regarding {topic} from the perspective "
        f"of international discourse.\n\nQuestion:"
    )
    try:
        response: ChatResponse = chat(
            model=MODEL_NAME,
            messages=[{"role": "user", "content": prompt}],
        )
        question = response.message.content.strip()
        return question
    except Exception as e:
        logging.error("Unable to generate question for topic '%s': %s", topic, str(e))
        return None


def generate_answer(question):
    """
    Uses 'phi4' (via the Ollama Python client) to generate an answer to the given question.
    """
    prompt = f"Question: {question}\nAnswer:"
    try:
        response: ChatResponse = chat(
            model=MODEL_NAME,
            messages=[{"role": "user", "content": prompt}],
        )
        answer = response.message.content.strip()
        return answer
    except Exception as e:
        logging.error("Unable to generate answer for question '%s': %s", question, str(e))
        return None


def main():
    setup_logging()
    start = time.time()

    dataset = []
    generated_count = 0

    # Error counter to stop if more than 10 errors occur
    error_count = 0

    # Use alive_bar for a progress bar
    with alive_bar(TOTAL_QUESTIONS, title="Generating dataset") as bar:
        while generated_count < TOTAL_QUESTIONS:
            # Randomly pick a topic to diversify
            topic = random.choice(TOPICS)

            # Generate a question
            question = generate_question(topic)
            if question is None:
                error_count += 1
                if error_count > 10:
                    logging.error("Too many errors encountered (%d). Stopping generation.", error_count)
                    break
                continue

            # Generate an answer
            answer = generate_answer(question)
            if answer is None:
                error_count += 1
                if error_count > 10:
                    logging.error("Too many errors encountered (%d). Stopping generation.", error_count)
                    break
                continue

            # Add to the dataset in a format suitable for instruction-style fine-tuning
            dataset.append({
                "instruction": question,
                "input": "",
                "output": answer
            })
            generated_count += 1

            # Update the progress bar
            bar()

            # Optional: be polite to the API to avoid rate limits or timeouts
            time.sleep(0.3)

    # Save the dataset to a JSON file (even if partially complete)
    output_file = "political_bias_fine_tuning_data.json"
    with open(output_file, "w", encoding="utf-8") as f:
        json.dump(dataset, f, ensure_ascii=False, indent=2)

    logging.info("Done! Generated and saved {:,} Q&A pairs to '%s'.", len(dataset), output_file)

    hours, rem = divmod(time.time() - start, 3600)
    minutes, seconds = divmod(rem, 60)
    logging.info("Time taken: %d:%02d:%02d", int(hours), int(minutes), int(seconds))


if __name__ == "__main__":
    main()

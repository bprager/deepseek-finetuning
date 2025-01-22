#!/usr/bin/env python3

from ollama import chat, ChatResponse

def check_ollama_connection(model="llama3.2"):
    """
    Tries to connect to the Ollama server by sending a simple prompt to the specified model.
    If successful, prints the response content. Otherwise, prints an error message.
    """
    try:
        # Prepare a basic user message
        messages = [
            {
                "role": "user",
                "content": "Why is the sky blue?"
            }
        ]

        # Send the chat request to Ollama
        response: ChatResponse = chat(
            model=model,
            messages=messages,
        )

        # Print some details from the response
        print("Successfully connected to Ollama!")
        print("Server responded with:", response.message.content)

    except Exception as e:
        print("Error connecting to Ollama:")
        print(e)

if __name__ == "__main__":
    check_ollama_connection()

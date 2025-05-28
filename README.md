This project is a Retrieval-Augmented Generation (RAG) chatbot built with FastAPI for the backend, Panel for the frontend, and LangChain for the RAG logic. It uses OpenAI embeddings and a Chroma vector store to answer questions based on provided PDF documents and scraped web content.

## Setup

1.  **Clone the repository:**
    ```bash
    git clone git@github.com:rev183/rag-supportbot.git
    cd rag-supportbot
    ```
2.  **Create a virtual environment (recommended):**
    ```bash
    python -m venv .venv
    source .venv/bin/activate  # On Windows, use `.venv\Scripts\activate`
    ```

3.  **Install dependencies:**
    ```bash
    pip install -r requirements.txt
    ```

5.  **Set up your OpenAI API Key:**
    * Create a file named `.env` in the project root directory.
    * Add your OpenAI API key to this file in the format:
        ```env
        OPENAI_API_KEY='your_openai_api_key_here'
        ```
        Replace `'your_openai_api_key_here'` with your actual key.

## Run locally

To run the application locally, you need to start both the FastAPI backend and the Panel frontend separately.

1.  **Start the FastAPI backend:**
    ```bash
    uvicorn main:app --host 0.0.0.0 --port 8882 --reload
    ```
    The `--reload` flag is useful for development as it restarts the server on code changes.

2.  **Start the Panel frontend:**
    Open a new terminal window or tab, navigate to the project directory, activate your virtual environment, and run:
    ```bash
    panel serve app.py --address 0.0.0.0 --port 5006 --autoreload --allow-websocket-origin=0.0.0.0:5006
    ```
    The `--autoreload` flag is useful for development.

3.  **Access the Chatbot:** Open your web browser and go to `http://localhost:5006`.

## With Docker

You can also build and run the entire application within a single Docker container.

1.  **Ensure Docker is installed:** Make sure you have Docker Desktop or Docker Engine installed on your system.

2.  **Build the Docker image:**
    Navigate to the project directory in your terminal and run the build command:
    ```bash
    docker build -t rag-chatbot .
    ```
    This will build an image named `rag-chatbot` based on the `Dockerfile` in the current directory.

3.  **Run the Docker container:**
    Run the container, mapping the necessary ports and passing your OpenAI API key as an environment variable:
    ```bash
    docker run -p 5006:5006 -p 8882:8882 -e OPENAI_API_KEY="YOUR_ACTUAL_OPENAI_API_KEY" rag-chatbot
    ```
    Replace `"YOUR_ACTUAL_OPENAI_API_KEY"` with your actual OpenAI API key. This command maps host ports 5006 and 8882 to the container's ports and sets the `OPENAI_API_KEY` environment variable inside the container.

4.  **Access the Chatbot:** Open your web browser and go to `http://localhost:5006`.


import panel as pn
import requests
from langchain.chains import ConversationChain

pn.extension(design='material')

# --- FastAPI Backend URL ---
FASTAPI_BACKEND_URL = "http://127.0.0.1:8882/chat"

# --- Chat Interaction Function ---
async def rag_chat_callback(contents: str, user: str, instance: pn.chat.ChatInterface):
    """
    This function is called when the user sends a message in the chat interface.
    It sends the message to the FastAPI backend and yields the response.
    """
    # Display a "Thinking..." message while waiting for the backend
    yield {"user": "System", "object": "Thinking...", "avatar": "🤖"}

    try:
        # Prepare the data to send to the FastAPI backend
        payload = {"query": contents}

        # Send the query to the FastAPI backend
        response = requests.post(FASTAPI_BACKEND_URL, json=payload)

        # Check if the request was successful
        if response.status_code == 200:
            # Parse the JSON response
            data = response.json()
            answer = data.get("answer", "Error: No answer received.")
            source_documents = data.get("source_documents", [])

            # Format the response to include the answer and source documents
            response_text = answer

            # if source_documents:
            #     response_text += "\n\n**Source Documents:**\n"
            #     for i, doc in enumerate(source_documents):
            #         # Limit the source content displayed to avoid clutter
            #         content_preview = doc['page_content'][:300] + "..." if len(doc['page_content']) > 300 else doc['page_content']
            #         response_text += f"- **Source {i+1}:** {doc['metadata'].get('source', 'N/A')} (Page {doc['metadata'].get('page', 'N/A')})\n  ```\n{content_preview}\n  ```\n"

            # Yield the formatted response to the chat interface
            yield {"user": "Bot", "object": response_text, "avatar": "📚"}

        else:
            # Handle errors from the backend
            error_detail = response.json().get("detail", "Unknown error")
            yield {"user": "Bot", "object": f"Error from backend: {response.status_code} - {error_detail}", "avatar": "❌"}

    except requests.exceptions.ConnectionError:
        yield {"user": "Bot", "object": f"Error: Could not connect to FastAPI backend at {FASTAPI_BACKEND_URL}. Is the backend running?", "avatar": "❌"}
    except Exception as e:
        # Handle any other unexpected errors
        yield {"user": "Bot", "object": f"An unexpected error occurred: {e}", "avatar": "❌"}

# async def callback(contents: str, user: str, instance: pn.chat.ChatInterface):
#     await chain.apredict(input=contents)

# --- Panel Chat Interface Setup ---
chat_interface = pn.chat.ChatInterface(
    callback=rag_chat_callback,
    sizing_mode="stretch_both",
    show_rerun=False,
    show_undo=False,
    # show_widgetbox=False,
    widgets=[
        pn.widgets.TextInput(name="Query", placeholder="Ask a question about your documents..."),
    ]
)

# Set the title of the app
chat_interface.servable(title="RAG Supportbot")

# To run this frontend:
# 1. Save the code as app.py
# 2. Make sure you have panel and requests installed (`pip install panel requests`)
# 3. Make sure your FastAPI backend (main.py) is running.
# 4. Run from your terminal: panel serve app.py --autoreload

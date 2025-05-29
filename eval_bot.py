import requests
import os
from dotenv import load_dotenv
from typing import List, Dict, Any
from langchain_openai import ChatOpenAI
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser
import traceback
import datetime

load_dotenv()

FASTAPI_BACKEND_CHAT_URL = "http://127.0.0.1:8882/chat"
EVAL_REPORT_DIR = "evaluation_reports"

# LLM for Evaluation (LLM-as-a-Judge)
EVAL_LLM = ChatOpenAI(model="gpt-4o", temperature=0)

# Evaluation Prompts
ANSWER_RELEVANCE_PROMPT = ChatPromptTemplate.from_messages([
    ("system", """
    You are an impartial judge evaluating the relevance of a generated answer to a user's question.
    Rate the answer on a scale of 1 to 5, where 1 is completely irrelevant and 5 is perfectly relevant.
    Provide only the numerical score.
    """),
    ("user", "Question: {question}\nAnswer: {answer}")
])
answer_relevance_chain = ANSWER_RELEVANCE_PROMPT | EVAL_LLM | StrOutputParser()

CONTEXT_RELEVANCE_PROMPT = ChatPromptTemplate.from_messages([
    ("system", """
    You are an impartial judge evaluating the relevance of retrieved documents to a user's question.
    Rate the relevance of the provided documents on a scale of 1 to 5, where 1 is completely irrelevant and 5 is perfectly relevant.
    Consider if the documents contain information that directly helps answer the question.
    Provide only the numerical score.
    """),
    ("user", "Question: {question}\nRetrieved Documents: {documents}")
])
context_relevance_chain = CONTEXT_RELEVANCE_PROMPT | EVAL_LLM | StrOutputParser()

FAITHFULNESS_PROMPT = ChatPromptTemplate.from_messages([
    ("system", """
    You are an impartial judge evaluating whether a generated answer is fully supported by the provided source documents.
    Rate the faithfulness on a scale of 1 to 5, where 1 means the answer is not supported at all and 5 means the answer is fully supported by the documents.
    If the answer contains information not present in the documents, it is not faithful.
    Provide only the numerical score.
    """),
    ("user", "Answer: {answer}\nSource Documents: {documents}")
])
faithfulness_chain = FAITHFULNESS_PROMPT | EVAL_LLM | StrOutputParser()

# Evaluation Dataset
EVAL_DATASET = [
    {
        "id": "Q1",
        "question": "How do I clear my browser history in Angel One web app?",
        "ground_truth_answer": "To clear your browser history in the Angel One web app, you need to go to settings, then privacy and security, and then clear browser history.",
    },
    {
        "id": "Q2",
        "question": "What should I do if my chart is not loading and shows 'Something went wrong'?",
        "ground_truth_answer": "If your chart is not loading and shows 'Something went wrong', you should open Playstore, search for Google Chrome, and update it to the latest version. If the issue persists, search for Android System Webview and update it to the latest version.",
    },
    {
        "id": "Q3",
        "question": "What are the steps to open a demat account with Angel One?",
        "ground_truth_answer": "To open a demat account with Angel One, you typically need to complete KYC, provide identity and address proof, and link your bank account. The process can often be done online via their app or website.",
    },
    {
        "id": "Q4",
        "question": "What is the policy for transferring funds to my trading account?",
        "ground_truth_answer": "You can transfer funds to your trading account using various methods like UPI, Net Banking, or IMPS/NEFT. Ensure your bank account is linked to your Angel One account for seamless transactions.",
    },
    {
        "id": "Q5",
        "question": "Tell me about the new features in the Angel One mobile app.",
        "ground_truth_answer": "I can only provide information based on the documents I have. I do not have specific details on the latest new features of the Angel One mobile app.",
    }
]


# HTML Report Generation Function
def generate_html_report(results: List[Dict[str, Any]]):
    timestamp = datetime.datetime.now().strftime("%y-%m-%d_%H-%M-%S")
    report_filename = os.path.join(EVAL_REPORT_DIR, f"evaluation_report_{timestamp}.html")
    os.makedirs(EVAL_REPORT_DIR, exist_ok=True)

    html_content = f"""
    <!DOCTYPE html>
    <html lang="en">
    <head>
        <meta charset="UTF-8">
        <meta name="viewport" content="width=device-width, initial-scale=1.0">
        <title>Supportbot Evaluation Report - {timestamp}</title>
        <style>
            body {{ font-family: Arial, sans-serif; margin: 20px; background-color: #f4f4f4; color: #333; }}
            .container {{ max-width: 1000px; margin: auto; background-color: #fff; padding: 20px; border-radius: 8px; box-shadow: 0 2px 4px rgba(0,0,0,0.1); }}
            h1, h2 {{ color: #0056b3; }}
            .test-case {{ border: 1px solid #ddd; border-radius: 5px; margin-bottom: 20px; padding: 15px; background-color: #f9f9f9; }}
            .test-case h3 {{ margin-top: 0; color: #333; }}
            .section-title {{ font-weight: bold; margin-top: 10px; color: #555; }}
            pre {{ background-color: #eee; padding: 10px; border-radius: 4px; overflow-x: auto; white-space: pre-wrap; word-wrap: break-word; }}
            .score {{ font-weight: bold; color: #007bff; }}
            .status-success {{ color: green; font-weight: bold; }}
            .status-error {{ color: red; font-weight: bold; }}
            .source-doc {{ background-color: #e9e9e9; border-left: 3px solid #007bff; padding: 8px; margin-top: 5px; font-size: 0.9em; }}
            .source-content {{ max-height: 150px; overflow-y: auto; border: 1px solid #ccc; padding: 5px; margin-top: 5px; background-color: #fff; }}
        </style>
    </head>
    <body>
        <div class="container">
            <h1>Supportbot Evaluation Report</h1>
            <p>Generated On: {timestamp}</p>
            <hr>
    """

    for res in results:
        status_class = "status-success" if res.get('status') == 'Success' else "status-error"

        html_content += f"""
            <div class="test-case">
                <h2>Test Case ID: {res.get('id')} <span class="{status_class}">({res.get('status')})</span></h2>
                <p><span class="section-title">Question:</span> {res['question']}</p>
                <p><span class="section-title">Ground Truth Answer:</span></p>
                <pre>{res['ground_truth_answer']}</pre>
                <p><span class="section-title">Bot Generated Answer:</span></p>
                <pre>{res['generated_answer']}</pre>

                <h3>LLM-as-a-Judge Scores:</h3>
                <ul>
                    <li>Answer Relevance: <span class="score">{res['answer_relevance']}</span>/5</li>
                    <li>Context Relevance: <span class="score">{res['context_relevance']}</span>/5</li>
                    <li>Faithfulness: <span class="score">{res['faithfulness']}</span>/5</li>
                </ul>

                <h3>Retrieved Source Documents ({res['retrieved_docs_count']}):</h3>
                {"<p>No documents retrieved.</p>" if not res['retrieved_sources'] else ""}
        """
        for i, doc_source in enumerate(res['retrieved_sources']):
            # Find the full document object to get page_content and metadata
            # This requires iterating through the original retrieved_docs list from bot_response
            full_doc_info = next(
                (d for d in res.get('full_retrieved_docs', []) if d['metadata'].get('source') == doc_source), None)

            doc_page_content = "Content not available in report data."
            doc_metadata = {}
            if full_doc_info:
                doc_page_content = full_doc_info.get('page_content', 'N/A')
                doc_metadata = full_doc_info.get('metadata', {})

            source_name = doc_metadata.get('source', 'Unknown')
            page_num = doc_metadata.get('page', 'N/A')

            html_content += f"""
                <div class="source-doc">
                    <strong>Source {i + 1}:</strong> {source_name} (Page: {page_num})<br>
                    <span class="section-title">Content Preview:</span>
                    <div class="source-content">
                        <pre>{doc_page_content[:500]}...</pre>
                    </div>
                </div>
            """
        html_content += "</div>"  # Close test-case div

    html_content += """
        </div>
    </body>
    </html>
    """

    with open(report_filename, 'w', encoding='utf-8') as f:
        f.write(html_content)
    print(f"\nHTML evaluation report saved to: {report_filename}")


#  Evaluation Function 
def evaluate_bot():
    print("Starting supportbot evaluation...")
    results = []

    for test_case in EVAL_DATASET:
        query = test_case["question"]
        ground_truth_answer = test_case["ground_truth_answer"]
        test_id = test_case["id"]

        print(f"\n Evaluating Test Case: {test_id} ")
        print(f"Question: {query}")

        try:
            # Call the FastAPI backend
            response = requests.post(FASTAPI_BACKEND_CHAT_URL, json={"query": query})
            response.raise_for_status()
            bot_response = response.json()

            generated_answer = bot_response.get("answer", "N/A")
            retrieved_docs = bot_response.get("source_documents", [])

            print(f"Bot Answer: {generated_answer}")
            print(f"Retrieved Docs ({len(retrieved_docs)}):")
            for i, doc in enumerate(retrieved_docs):
                source_name = doc['metadata'].get('source', 'Unknown')
                page_num = doc['metadata'].get('page', 'N/A')
                print(f"  - {source_name} (Page: {page_num})")

            # Perform LLM-as-a-Judge Evaluations
            # Prepare documents for judging
            doc_contents_for_judging = "\n\n".join([doc['page_content'] for doc in retrieved_docs])

            # 1. Answer Relevance
            ans_relevance_score = answer_relevance_chain.invoke({
                "question": query,
                "answer": generated_answer
            })
            try:
                ans_relevance_score = int(ans_relevance_score.strip())
            except ValueError:
                ans_relevance_score = "N/A (LLM parse error)"
            print(f"  - Answer Relevance (1-5): {ans_relevance_score}")

            # 2. Context Relevance
            ctx_relevance_score = context_relevance_chain.invoke({
                "question": query,
                "documents": doc_contents_for_judging
            })
            try:
                ctx_relevance_score = int(ctx_relevance_score.strip())
            except ValueError:
                ctx_relevance_score = "N/A (LLM parse error)"
            print(f"  - Context Relevance (1-5): {ctx_relevance_score}")

            # 3. Faithfulness/Groundedness
            faithfulness_score = faithfulness_chain.invoke({
                "answer": generated_answer,
                "documents": doc_contents_for_judging
            })
            try:
                faithfulness_score = int(faithfulness_score.strip())
            except ValueError:
                faithfulness_score = "N/A (LLM parse error)"
            print(f"  - Faithfulness (1-5): {faithfulness_score}")

            # Store results, including full retrieved docs for HTML report
            results.append({
                "id": test_id,
                "question": query,
                "ground_truth_answer": ground_truth_answer,
                "generated_answer": generated_answer,
                "retrieved_docs_count": len(retrieved_docs),
                "retrieved_sources": [doc['metadata'].get('source', 'Unknown') for doc in retrieved_docs],
                "full_retrieved_docs": retrieved_docs,  # Store full doc objects for HTML
                "answer_relevance": ans_relevance_score,
                "context_relevance": ctx_relevance_score,
                "faithfulness": faithfulness_score,
                "status": "Success"
            })

        except requests.exceptions.RequestException as e:
            print(f"  Error connecting to backend or API: {e}")
            results.append({"id": test_id, "status": f"Backend Error: {e}"})
        except Exception as e:
            print(f"  An unexpected error occurred during evaluation: {e}")
            traceback.print_exc()
            results.append({"id": test_id, "status": f"Evaluation Error: {e}"})

    print("\n Evaluation Summary ")
    for res in results:
        print(f"ID: {res.get('id')}, Status: {res.get('status')}")
        if res.get('status') == 'Success':
            print(
                f"  Ans Rel: {res['answer_relevance']}, Ctx Rel: {res['context_relevance']}, Faith: {res['faithfulness']}")
            print(f"  Retrieved: {', '.join(res['retrieved_sources'])}")

    print("\n Detailed Results (for console inspection) ")
    for res in results:
        if res.get('status') == 'Success':
            print(f"\n Test Case: {res['id']} ")
            print(f"Q: {res['question']}")
            print(f"GT Answer: {res['ground_truth_answer']}")
            print(f"Bot Answer: {res['generated_answer']}")
            print(f"Retrieved Sources: {', '.join(res['retrieved_sources'])}")
            print(
                f"Scores: Answer Relevance={res['answer_relevance']}, Context Relevance={res['context_relevance']}, Faithfulness={res['faithfulness']}")
            print("-" * 30)

    # Generate and save the HTML report
    generate_html_report(results)


if __name__ == "__main__":
    evaluate_bot()

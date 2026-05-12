def evaluate_response(question, answer, context):
    # Faithfulness: is answer grounded in context?
    # Relevancy: does answer address the question?
    # Cost: how many tokens used?
    return {
        "question": question,
        "answer": answer,
        "context_used": context[:200],
        "token_estimate": len(answer.split()),
        "faithfulness": "manual review needed",
        "relevancy": "manual review needed"
    }

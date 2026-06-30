import gradio as gr

from app import ask_question



demo = gr.Interface(
    fn=ask_question,
    inputs=gr.Textbox(
        placeholder="Ask insurance question..."
    ),
    outputs="text",
    title="Insurance AI Assistant",
    description="RAG chatbot using insurance documents"
)


demo.launch(
    server_name="0.0.0.0",
    server_port=7860
)
import gradio as gr
from classify import classify, PROMPT1


def predict(text):
    r = classify(text, PROMPT1)
    if r["status"] != "ok":
        return f"Could not classify: {r['status']}"
    return f"{r['label']} (confidence {r['confidence']:.2f})"


demo = gr.Interface(
    fn=predict,
    inputs=gr.Textbox(lines=6, label="News item"),
    outputs=gr.Textbox(label="Category"),
    title="News category classifier",
    api_name="classify",
)

demo.launch(share=True)
from transformers import AutoTokenizer, AutoModelForSeq2SeqLM

MODEL_NAME = "t5-small"

def load_model(model_name: str = MODEL_NAME):
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    model = AutoModelForSeq2SeqLM.from_pretrained(model_name)
    return tokenizer, model

def summarize(text: str, max_new_tokens: int = 80, num_beams: int = 4) -> str:
    """
    Inference-only summarization with a public pretrained T5 model.
    No dataset files, no fine-tuned checkpoints included.
    """
    tokenizer, model = load_model()
    prompt = "summarize: " + text.strip()
    inputs = tokenizer(prompt, return_tensors="pt", truncation=True, max_length=512)
    output_ids = model.generate(
        **inputs,
        max_new_tokens=max_new_tokens,
        num_beams=num_beams,
        early_stopping=True
    )
    return tokenizer.decode(output_ids[0], skip_special_tokens=True)

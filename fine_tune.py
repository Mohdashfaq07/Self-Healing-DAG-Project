
import os
from datasets import load_dataset
from transformers import AutoTokenizer, AutoModelForSequenceClassification, Trainer
from transformers.training_args import TrainingArguments

MODEL_DIR = "./models/fine_tuned_model"
os.makedirs(MODEL_DIR, exist_ok=True)

def main():
    print("🚀 Loading IMDb dataset (small subset for quick training)...")
    dataset = load_dataset("imdb")
    small_train = dataset["train"].shuffle(seed=42).select(range(2000))
    small_test = dataset["test"].shuffle(seed=42).select(range(1000))

    print("🔠 Loading tokenizer and model...")
    tokenizer = AutoTokenizer.from_pretrained("distilbert-base-uncased")
    model = AutoModelForSequenceClassification.from_pretrained("distilbert-base-uncased", num_labels=2)

    def tokenize(batch):
        return tokenizer(batch["text"], padding="max_length", truncation=True, max_length=256)

    print("🧩 Tokenizing (this may take a moment)...")
    small_train = small_train.map(tokenize, batched=True)
    small_test = small_test.map(tokenize, batched=True)

    small_train.set_format("torch", columns=["input_ids", "attention_mask", "label"])
    small_test.set_format("torch", columns=["input_ids", "attention_mask", "label"])

    print("⚙️ Setting up Trainer...")
    args = TrainingArguments(
        output_dir="./results",
        per_device_train_batch_size=8,
        per_device_eval_batch_size=8,
        num_train_epochs=1,
        logging_steps=50,
        save_strategy="no",
        report_to="none",
        evaluation_strategy="no",
    )

    trainer = Trainer(
        model=model,
        args=args,
        train_dataset=small_train,
        eval_dataset=small_test,
    )

    print("🏋️ Training (this will take about 1–2 minutes)...")
    trainer.train()

    print(f"💾 Saving fine-tuned model to {MODEL_DIR}...")
    tokenizer.save_pretrained(MODEL_DIR)
    model.save_pretrained(MODEL_DIR)

    print("✅ Fine-tuning complete! You can now run:")
    print("   python cli.py")

if __name__ == "__main__":
    main()

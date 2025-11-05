Model Placeholder — Self-Healing DAG Project

The fine-tuned model weights are too large to include in this GitHub repository
(due to GitHub’s 100 MB file limit).

To use this project:
1. Run `python fine_tune.py` to re-train or re-generate the model locally.
2. This will automatically save the model into this folder:
   ./models/fine_tuned_model/

After training, you can run:
   python cli.py

Expected Output (example):
[InferenceNode] Predicted label: Positive | Confidence: 54%
[ConfidenceCheckNode] Confidence too low. Triggering fallback...
[FallbackNode] Could you clarify your intent? Was this a negative review?
User: Yes, it was definitely negative.
Final Label: Negative (Corrected via user clarification)

How It Works:
InferenceNode – predicts sentiment using the fine-tuned model.
ConfidenceCheckNode – checks prediction confidence.
FallbackNode – if confidence is low, asks the user to clarify the intent.
Self-Healing Decision – final corrected label is printed.

DEMO VIDEO LINK: https://drive.google.com/file/d/1M3MnY5ryhmWwcvE0vRtZHgXyKd1myoe4/view?usp=drivesdk 

# PhishGuard Presentation Script

This document provides speaker notes for [docs/presentation/phishguard_presentation.md](docs/presentation/phishguard_presentation.md).

Speaker split:
- Mahboobeh: Slides 1 to 6
- Arad: Slides 7 to 13

## Slide 1 - Title

Presenter: Mahboobeh

Script:

Phishing is still a major threat in email communication.

Today’s phishing emails look very professional and often imitate real workplace messages.
Because of that, simple rule-based filters or keyword matching are not enough anymore.
The main challenge is that this problem is semantic, meaning the model needs to understand intent, urgency, and impersonation—not just words.

## Slide 2 - Problem

Presenter: Mahboobeh

Script:

Email is still one of the most common entry points for social engineering attacks, and phishing messages have become much harder to spot. Many of them no longer look sloppy or obviously malicious. They imitate normal workplace communication, account notices, invoice requests, or urgent administrative tasks. That is why rule-based filters and simple keyword matching are not enough on their own. The real challenge is semantic: the model needs to understand patterns like urgency, impersonation, and deceptive intent even when the writing looks professional.

Based on that problem, we defined four main objectives for the project.

## Slide 3 - Project Objectives

Presenter: Mahboobeh

Script:

We had four main goals.
First, to build an end-to-end phishing email detection pipeline.
Second, to fine-tune a transformer model for binary classification of email text.
Third, to preserve some interpretability, instead of returning only a hard label.
And fourth, to provide both a CLI workflow and a lightweight web interface.
For us, success meant strong performance and a clean workflow from raw data to deployment.
With these goals in place, this is the pipeline we implemented.

## Slide 4 - Technical Solution Overview

Presenter: Mahboobeh

Script:

Our pipeline starts by loading labeled email data.
Then we preprocess the text, split it into train, validation, and test sets.
After that, we tokenize the text using BERT tokenizer and fine-tune the model.
Finally, we evaluate the model and deploy it using CLI and Gradio.

The next piece is the data that drives the whole system.

## Slide 5 - Dataset and Splits

Presenter: Mahboobeh

Script:

We used a dataset with about 82,000 emails.
The data was split into 70% training, 15% validation, and 15% testing.
The dataset includes both legitimate and phishing emails, and the split was stratified to keep balance between classes.

Once the data was collected, the next step was cleaning it so the model learns meaningful patterns instead of noise.

Arad will now cover the model


## Slide 6 - Model and Training Strategy

Presenter: Arad

Script:

Our base model is bert-base-uncased, configured for binary sequence classification. We set the maximum sequence length to 256 tokens, which gave us a practical balance between retaining enough email content and controlling memory use. Training was handled with Hugging Face Trainer, and the best checkpoint was selected using macro F1 instead of raw accuracy so that performance stayed balanced across both classes. One useful engineering detail is that our custom dataset pre-tokenizes the text once during initialization, which removes repeated CPU work during training and improves throughput.

After training, we evaluated the best checkpoint on a held-out test split that was never used for optimization.

## Slide 7 - Evaluation Results

Presenter: Arad

Script:

The final model performed very strongly on the test set. It achieved 99.48 percent accuracy, a macro F1 score of 0.9948, and a ROC-AUC of 0.9998. In practical terms, the model made only 16 false-positive predictions and 49 false-negative predictions across 12,374 test emails. These results suggest that the classifier learned a very strong separation between legitimate and phishing messages.

The confusion matrix gives a more operational view of those numbers.


## Slide 8 - Explainability and Web Interface

Presenter: Arad

Script:

For inference, our predictor returns a lot more than a single label. It outputs the predicted verdict, the label ID, a confidence score, and token-level attention scores. Those token scores come from the final transformer layer and give a lightweight view of which parts of the message received the most model focus. This is not a perfect causal explanation, but it does make the system easier to inspect and trust. We also wrapped the predictor in a Gradio interface so users can paste an email and immediately review the model's decision.

That leads directly into the demo slide.

## Slide 9 - Demo

https://youtu.be/J-1gts2d3Aw


## Slide 10 - Conclusion

Presenter: Arad

Script:

PhishGuard shows that phishing email detection can be built as a clean end-to-end NLP system rather than a collection of brittle rules. The project combines preprocessing, stratified splitting, transformer fine-tuning, evaluation, explainable inference, and a lightweight web app in one coherent workflow. Overall, the system achieved incredible evaluation scores, which met our technical goal and gave us a practical result to demonstrate.

Thank you for listening. We are happy to take questions.


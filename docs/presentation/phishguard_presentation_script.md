# PhishGuard Presentation Script

This document provides speaker notes for [docs/presentation/phishguard_presentation.md](docs/presentation/phishguard_presentation.md).

Speaker split:
- Mahboobeh: Slides 1 to 6
- Arad: Slides 7 to 13

## Slide 1 - Title

Presenter: Mahboobeh

Script:

Good morning. We are Arad Fadaei and Mahboobeh Yasini, and this presentation covers PhishGuard, our final project on phishing email detection. Our goal was to build a practical NLP system that can distinguish legitimate email from phishing email by fine-tuning BERT on labeled message data.

To understand why we chose this problem, we should start with the current phishing threat itself.

## Slide 2 - Problem

Presenter: Mahboobeh

Script:

Email is still one of the most common entry points for social engineering attacks, and phishing messages have become much harder to spot. Many of them no longer look sloppy or obviously malicious. They imitate normal workplace communication, account notices, invoice requests, or urgent administrative tasks. That is why rule-based filters and simple keyword matching are not enough on their own. The real challenge is semantic: the model needs to understand patterns like urgency, impersonation, and deceptive intent even when the writing looks professional.

Based on that problem, we defined four main objectives for the project.

## Slide 3 - Project Objectives

Presenter: Mahboobeh

Script:

We set out to do four things. First, build a complete and reproducible phishing detection pipeline. Second, fine-tune a transformer model for binary email classification. Third, preserve some interpretability instead of returning only a hard label. And fourth, package the system so it works both from the command line and through a lightweight web interface. Success for us was high accuracy and also a clean engineering workflow from raw data to deployment.

With those objectives in place, this was the overall pipeline we implemented.

## Slide 4 - Technical Solution Overview

Presenter: Mahboobeh

Script:

The workflow is straightforward and modular. We load labeled email data, preprocess the raw text, split it into train, validation, and test sets, tokenize the text for BERT, fine-tune the classifier, evaluate the best checkpoint, and finally serve inference through both a CLI tool and a Gradio web app. We built this pipeline with PyTorch, Hugging Face Transformers, pandas, scikit-learn, and Gradio so the project would be both reproducible and easy to demonstrate.

The next piece is the data that drives the whole system.

## Slide 5 - Dataset and Splits

Presenter: Mahboobeh

Script:

The repository contains seven raw corpora under the data folder, but the current training pipeline is configured to use phishing_email.csv as the consolidated primary source file. After preprocessing, the final dataset contains 82,485 emails. We used a stratified 70, 15, and 15 split so that class balance stays consistent across train, validation, and test. That produced 57 thousand training examples, 12 thousand validation examples, and 12 thousand test examples. The class distribution is close to balanced, which matters because it reduces the chance that the classifier learns an easy majority-class shortcut.

Once the data was collected, the next step was cleaning it so the model learns meaningful patterns instead of noise.

## Slide 6 - Preprocessing Pipeline

Presenter: Mahboobeh

Script:

Raw email text is noisy, so preprocessing was a major part of the project. We stripped HTML, replaced URLs with the placeholder token [URL], masked email addresses as [EMAIL], masked phone numbers as [PHONE], and normalized repeated whitespace. This matters because it reduces variation that does not generalize. For example, if two phishing emails contain different links, the model should focus on the suspicious surrounding language rather than memorizing a specific domain. That improves signal density while still preserving the structure of the message.

At this point I will hand over to Arad to cover the model, evaluation, and final results.

## Slide 7 - Model and Training Strategy

Presenter: Arad

Script:

Our base model is bert-base-uncased, configured for binary sequence classification. We set the maximum sequence length to 256 tokens, which gave us a practical balance between retaining enough email content and controlling memory use. Training was handled with Hugging Face Trainer, and the best checkpoint was selected using macro F1 instead of raw accuracy so that performance stayed balanced across both classes. One useful engineering detail is that our custom dataset pre-tokenizes the text once during initialization, which removes repeated CPU work during training and improves throughput.

After training, we evaluated the best checkpoint on a held-out test split that was never used for optimization.

## Slide 8 - Evaluation Results

Presenter: Arad

Script:

The final model performed very strongly on the test set. It achieved 99.48 percent accuracy, a macro F1 score of 0.9948, and a ROC-AUC of 0.9998. In practical terms, the model made only 16 false-positive predictions and 49 false-negative predictions across 12,374 test emails. These results suggest that the classifier learned a very strong separation between legitimate and phishing messages.

The confusion matrix gives a more operational view of those numbers.

## Slide 9 - Confusion Matrix

Presenter: Arad

Script:

This confusion matrix shows where the model made its errors. Out of 5,940 legitimate emails, only 16 were incorrectly flagged as phishing. Out of 6,434 phishing emails, 49 were missed. The especially low false-positive count is important because security tools lose trust quickly if they block too many legitimate messages. So this result is statistically strong, but also meaningful from a usability perspective.

Beyond classification performance, we also wanted the system to be inspectable and easy to use.

## Slide 10 - Explainability and Web Interface

Presenter: Arad

Script:

For inference, our predictor returns a lot more than a single label. It outputs the predicted verdict, the label ID, a confidence score, and token-level attention scores. Those token scores come from the final transformer layer and give a lightweight view of which parts of the message received the most model focus. This is not a perfect causal explanation, but it does make the system easier to inspect and trust. We also wrapped the predictor in a Gradio interface so users can paste an email and immediately review the model's decision.

That leads directly into the demo slide.

## Slide 11 - Demo

https://youtu.be/J-1gts2d3Aw

## Slide 12 - Lessons Learned

Presenter: Arad

Script:

One of the main lessons was that data quality mattered as much as model choice. Normalizing URLs and personal identifiers improved the signal available to the model without removing important context. We also found that engineering choices such as pre-tokenization, FP16, pinned memory, and multiple data-loader workers made training much more practical. Finally, macro F1 was a better checkpoint metric than accuracy alone, and even lightweight explainability made the final system easier to inspect and trust.

To close, I will summarize the overall outcome of the project.

## Slide 13 - Conclusion

Presenter: Arad

Script:

PhishGuard shows that phishing email detection can be built as a clean end-to-end NLP system rather than a collection of brittle rules. The project combines preprocessing, stratified splitting, transformer fine-tuning, evaluation, explainable inference, and a lightweight web app in one coherent workflow. Overall, the system achieved incredible evaluation scores, which met our technical goal and gave us a practical result to demonstrate.

Thank you for listening. We are happy to take questions.


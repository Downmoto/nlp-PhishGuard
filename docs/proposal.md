# AIG230 Final Project Proposal: PhishGuard Phishing Detection

## 1. Authors
Arad Fadaei & Mahboobeh Yasini

## 2. Problem
Standard email security filters often fail to detect spear phishing attacks. Unlike bulk spam, these attacks are highly targeted and use professional, industry-specific language to manipulate employees into sharing sensitive credentials. In sectors like finance and legal services, where the terminology is specialized, generic AI models often lack the nuance to distinguish between a legitimate high-pressure request and a fraudulent one. This gap in detection leaves organizations vulnerable to significant financial and data breaches.

## 3. Solution
We propose a specialized NLP pipeline that moves beyond keyword matching by fine-tuning a pre-trained transformer model such as BERT. We will adapt the model to recognize the subtle linguistic markers of professional fraud, such as authority-mimicry and manufactured urgency within a corporate context.

Our solution involves:
* Preprocessing raw email data to remove PII (Personally Identifiable Information) while maintaining semantic structure.
* Fine-tuning the transformer on a balanced dataset of professional emails and known phishing samples.
* Implementing a classification head that provides a probability score indicating the risk level of an incoming message.

## 4. Objectives
* To fine-tune a pre-trained NLP model that outperforms generic filters in identifying professional spear phishing.
* To analyse and document the specific linguistic features that differentiate fraudulent professional communication from legitimate business inquiries.
* To create an accessible interface that allows non-technical users to verify suspicious emails.

## 5. Deliverables
The project will result in a comprehensive package for the end-user:
* **Fine-Tuned Model:** The weights and configuration for the sector-specific phishing classifier.
* **Codebase:** A GitHub repository containing the Python scripts for training, evaluation, and data cleaning.
* **User Interface:** A web-based application for real-time email analysis.
* **Technical Report:** Documentation detailing the training process, hyperparameter tuning, and model performance metrics.

## 6. Datasets
We will utilise a combination of two primary data sources:
* **Legitimate Data:** The Enron Email Dataset, which provides a vast corpus of real-world professional communication (https://www.kaggle.com/datasets/wcukierski/enron-email-dataset).
* **Fraudulent Data:** The Phish No More collection, which includes the Enron, Ling, CEAS, Nazario, Nigerian, and SpamAssassin datasets containing verified phishing emails (naserabdullahalam/phishing-email-dataset).
* **Data Access:** These are publicly available on the platform Kaggle.

## 7. Demo
The live demo will showcase the PhishGuard web interface (https://www.kaggle.com/datasets/) using two contrasting scenarios:
* A legitimate but urgent internal request regarding a project deadline.
* A carefully crafted spear phishing email that mimics the same urgency but includes subtle call-to-action markers typical of credential harvesting.
* We will demonstrate the model's ability to provide a confidence score and highlight specific tokens that triggered the high-risk alert.
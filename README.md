# s26projcais
# Email Phishing Detection using NLP

## Overview
Phishing emails pose a significant cybersecurity and societal threat, targeting individuals and organizations through deceptive messaging. This project uses Natural Language Processing (NLP) and deep learning to classify emails as phishing (spam) or legitimate.

---

## Dataset
**Source:** Kaggle phishing email datasets  
**Dataset Used:** `phishing_email.csv`  
**Size:** ~82,500 emails  
**Class Balance:** ~52% phishing, ~48% legitimate  

### Preprocessing
- Lowercasing text
- Removing URLs and email addresses
- Removing punctuation and special characters
- Stratified train/validation/test split (80/10/10)

These steps reduce noise while preserving semantic meaning.

---

## Model Development
We fine-tuned **DistilBERT**, a lightweight Transformer-based model pre-trained on large-scale English corpora.

### Why DistilBERT?
- Strong performance on text classification tasks
- Computationally efficient
- Captures contextual meaning better than traditional models (e.g., TF-IDF + SVM)

### Training Details
- Max sequence length: 256
- Learning rate: 2e-5
- Batch size: 16
- Epochs: 3
- Optimizer: AdamW

---

## Evaluation Metrics
Given the cost of false negatives (missing phishing emails), we prioritize:
- Precision
- Recall
- F1-score

These metrics provide a better view of real-world performance than accuracy alone.

---

## Results
The fine-tuned model achieves strong performance across all metrics, demonstrating the effectiveness of Transformer-based models for phishing detection. Recall is particularly high, reducing the likelihood of phishing emails slipping through filters.

---

## Discussion & Social Implications
This model can be extended to:
- Email clients for real-time phishing detection
- Enterprise security tools
- Educational tools to raise phishing awareness

### Limitations
- Model performance depends on dataset quality
- Adversarial or novel phishing styles may reduce accuracy
- Ethical concerns around automated email monitoring

---

## Future Work
- Incorporate multiple datasets for improved generalization
- Use explainability tools (e.g., SHAP, attention visualization)
- Explore multilingual phishing detection
- Deploy as an API or browser extension

---

## Conclusion
This project demonstrates how modern NLP techniques can be applied to a socially impactful problem, bridging machine learning research with practical cybersecurity applications.

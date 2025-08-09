# NLP-Sentiment-Classification

## Problem Setting

We focus on text sentiment classification using neural network architectures. The goal is to develop better text representations that can accurately classify the emotional orientation of text documents.

## What We Do Exactly

1. Enhanced Pooling

Combine mean pooling and max pooling of word embeddings
Extract richer information by concatenating both representations

2. LSTM Networks

Use Long Short-Term Memory networks to capture sequential dependencies
Implement bidirectional LSTM to read text in both directions
Use early stopping to prevent overfitting

3. Self-Attention Mechanism

Implement simple self-attention to focus on relevant words
Use positional embeddings to maintain word order information

4. Multi-Head Attention

Extend to multiple attention heads for richer representations
Allow the model to attend to different aspects simultaneously

5. Transformer Encoder

Implement a full Transformer encoder architecture
Compare with multi-head attention and experiment with parameters

6. Pre-trained Models

Evaluate at least 3 pre-trained transformer models
Compare their out-of-the-box performance on sentiment classification

7. Fine-tuning

Fine-tune pre-trained models on the specific sentiment dataset
Use Google Colab for computational resources
Optimize for best performance

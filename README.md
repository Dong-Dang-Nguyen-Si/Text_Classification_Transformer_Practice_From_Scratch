This project demonstrates how to construct, train, and evaluate a Transformer-based text classification pipeline without relying on high-level pre-trained model APIs (such as Hugging Face Transformers or BERT/PhoBERT).



\* \*\*Task:\*\* Multiclass Sentiment Classification (3 classes: Negative, Neutral, Positive).

\* \*\*Dataset:\*\* \[`uitnlp/vietnamese\_students\_feedback`](https://huggingface.co/datasets/uitnlp/vietnamese\_students\_feedback)

\* \*\*Core Framework:\*\* PyTorch, Hugging Face Datasets, Scikit-learn.



\---



\## 🏗️ Architecture \& Implementation Details



The model is built modularly using `torch.nn.Module`:



1\. \*\*Custom Vocabulary \& Tokenizer:\*\*

&#x20;  \* Custom mapping dictionary (`stoi` / `itos`) with special tokens (`<PAD>`, `<UNK>`).

&#x20;  \* Maximum vocabulary size capped at the top 10,000 most frequent tokens.



2\. \*\*Token and Positional Embedding (`TokenAndPositionEmbedding`):\*\*

&#x20;  \* Combines standard token embeddings with learnable positional embeddings to preserve sequence order.



3\. \*\*Transformer Encoder Block (`TransformerEncoderBlock`):\*\*

&#x20;  \* Multi-Head Self-Attention (`nn.MultiheadAttention`) with `batch\_first=True`.

&#x20;  \* Position-wise Feed-Forward Network (FFN) with linear projections and `ReLU` activation.

&#x20;  \* Residual connections around attention and FFN blocks followed by Layer Normalization (`nn.LayerNorm`).



4\. \*\*Classifier Head (`TransformerEncoderCls`):\*\*

&#x20;  \* Multi-layer Transformer Encoder stack.

&#x20;  \* Temporal 1D Average Pooling (`nn.AvgPool1d`) over sequence length.

&#x20;  \* Fully connected classification head with Dropout regularization.



\---



\## ⚙️ Model Hyperparameters



| Hyperparameter | Value | Description |

| :--- | :--- | :--- |

| `vocab\_size` | 10,000 | Maximum vocabulary capacity |

| `max\_length` / `seq\_length` | 100 | Fixed sequence length (padded / truncated) |

| `embed\_dim` | 200 | Token \& position embedding dimensions |

| `num\_layers` | 2 | Number of Transformer Encoder blocks |

| `num\_heads` | 4 | Number of attention heads |

| `ff\_dim` | 128 | Feed-Forward hidden layer size |

| `dropout` | 0.01 | Dropout probability |

| `batch\_size` | 128 | Training and evaluation batch size |

| `learning\_rate` | 5e-5 | Adam optimizer learning rate |

| `epochs` | 50 | Total training iterations |


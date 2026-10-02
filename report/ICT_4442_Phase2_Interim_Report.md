# ICT 4442 - Deep Learning Mini Project
## Phase 2: Interim Report

**Title:** Comparative Analysis of Deep Learning Architectures for Sentiment Analysis of Movie Reviews  
**Team:** Meesha Gupta (230953090); Ratul Jha (230953338); Harshavardhan Reddy (230953396); Kshitij Singh (230953460)  
**Repository:** https://github.com/HarshavardhanReddy585/imdb-sentiment-analysis-ict4442  
**Submission date:** 02 October 2026 (late submission; seek faculty approval for acceptance)

## 1. Literature Review

| Paper | Method | Dataset | Key result | Relevance |
|---|---|---|---|---|
| Maas et al. (2011) [1] | Unsupervised word vectors + sentiment classifier | IMDb 50K | Established a large labelled/unlabelled movie-review benchmark and showed domain-aware representations help sentiment classification. | Defines the dataset and benchmark used here. |
| Kim (2014) [2] | 1D CNN with multiple filter widths | Sentence classification benchmarks | Simple CNN filters capture informative local phrases and perform strongly with limited task-specific architecture. | Supports the project CNN architecture and max-pooling design. |
| Hochreiter & Schmidhuber (1997) [3] | Long Short-Term Memory recurrent units | Sequence modelling | Gated memory addresses vanishing-gradient limitations in recurrent learning. | Provides the basis for the planned BiLSTM model. |
| Tai, Socher & Manning (2015) [4] | Tree-LSTM and compositional recurrent encoders | Sentiment Treebank | Structured recurrent encoders improve representation of compositional meaning. | Motivates context-aware recurrent comparison and error analysis. |
| Mikolov et al. (2013) [5] | Word2Vec distributed word representations | Large text corpora | Efficient embeddings encode useful lexical regularities. | Informs the embedding representation used by non-Transformer models. |
| Peters et al. (2018) [6] | ELMo contextual representations | Multiple NLP tasks | Deep contextual representations transfer across tasks and improve over static embeddings. | Connects embedding-average baselines to contextual pretraining. |
| Vaswani et al. (2017) [7] | Transformer self-attention | Translation and sequence tasks | Self-attention models token-to-token dependencies without recurrence. | The architectural foundation for the planned DistilBERT comparison. |
| Devlin et al. (2019) [8] | BERT bidirectional pretraining + fine-tuning | GLUE, SQuAD and other tasks | Bidirectional masked-language pretraining produced state-of-the-art results across language tasks. | Establishes the pretrained attention-family reference point. |
| Sanh et al. (2019) [9] | DistilBERT knowledge distillation | General language understanding | A smaller model retains most BERT capability with lower size and faster inference. | Specifies the fourth proposed model and its efficiency trade-off. |
| Howard & Ruder (2018) [10] | Universal Language Model Fine-tuning | Text classification benchmarks | Transfer learning and discriminative fine-tuning improve text classification. | Supports pretrained-model baselines and controlled fine-tuning. |

The literature indicates a progression from static or pooled word representations, through local convolution and recurrent context, to bidirectional self-attention with pretraining. The interim comparison therefore keeps the data split and metrics fixed while varying the architecture family.

## 2. Dataset Acquisition & Preprocessing

The project uses the IMDb Large Movie Review Dataset [1], consisting of 25,000 labelled training reviews and 25,000 labelled test reviews, balanced between positive and negative classes. The official archive is downloaded locally under `data/` when running the pipeline and is excluded from Git.

The reproducible pipelines are implemented in `src/train_scratch_models.py` and `src/train_bilstm.py`:

1. Read review text from the official `train/neg`, `train/pos`, `test/neg`, and `test/pos` folders.
2. Lowercase text and tokenize alphabetic words plus apostrophes; retain sentiment-bearing words such as `not`.
3. Shuffle the official training split with seed 42 and divide it into 80% train and 20% validation. The official test split remains untouched until final evaluation.
4. Construct the vocabulary from the training partition only; keep the 12,000 most frequent tokens plus `<pad>` and `<unk>`.
5. Encode each review as a fixed-length sequence, padding shorter reviews and truncating longer reviews deterministically. The MLP/CNN smoke runs use 512 tokens; the BiLSTM smoke run uses 40 tokens to keep CPU runtime manageable.
6. Use the same encoded partitions and metrics for every model. No test-set score is used for model selection.

The implementation is deliberately CPU-friendly for the delayed interim submission. It supports a small reproducibility run through `--max-train` and `--max-test`; the final report must rerun the same protocol at full scale and include BiLSTM and DistilBERT.

## 3. Models Implemented So Far

| Model | Owner | Status | Preliminary metric | Notes |
|---|---|---|---|---|
| MLP with averaged learned embeddings | Kshitij Singh | Implemented in `src/models/mlp.py` and `src/train_scratch_models.py` | Validation accuracy 88.50%, F1 0.884 | Three-epoch PyTorch run on the full 20,000 train / 5,000 validation / 25,000 test protocol with 512-token inputs; test accuracy 86.31%, F1 0.858. |
| 1D-CNN with max-pooled multi-width filters | Ratul Jha | Implemented in `src/models/cnn1d.py` and `src/train_scratch_models.py` | Validation accuracy 79.64%, F1 0.783 | Full three-epoch PyTorch run on the common 20,000/5,000/25,000 protocol with 512-token inputs; test accuracy 79.53%, F1 0.780. |
| BiLSTM | Harshavardhan Reddy | Implemented in `src/models/bilstm.py` and `src/train_bilstm.py` | Validation accuracy 46.5%, F1 0.389; smoke test accuracy 51.0%, F1 0.372 | Two-epoch CPU smoke run on the common 800/200/200 split; needs tuning and full-scale rerun. |
| DistilBERT | Meesha Gupta | Planned / not evidenced in current folder | Pending | Fine-tune after baseline comparison; record model version and hyperparameters. |

The reported values come from real runs with seed 42 and the official IMDb folder split. The MLP and CNN use the full 20,000/5,000/25,000 protocol with 512-token inputs, while BiLSTM uses a smaller 800/200/200 CPU smoke split with 40-token inputs. They are preliminary and intentionally not presented as final performance. The generated JSON files are checked into the repository as evidence.

## 4. Individual Contribution Log

## 4A. Completion Evidence Available at Interim Submission

The following evidence is available in the canonical GitHub repository at the time of this draft:

- Official IMDb archive downloaded locally and excluded from Git through `.gitignore`.
- Reproducible preprocessing and training scripts under `src/`.
- MLP and 1D-CNN PyTorch implementations with saved validation/test metrics under `results/`.
- BiLSTM implementation with a CPU smoke-run metric file under `results/`.
- Two commits documenting the model/evidence work: `8544666` and `6dee82d`.
- Python compilation and model-import checks completed successfully.

The MLP and CNN results are full-protocol evidence; the BiLSTM result is smoke evidence. DistilBERT, a larger BiLSTM run, learning curves, and systematic error analysis remain outstanding.

| Member | Registration no. | Task completed to date | Signature |
|---|---:|---|---|
| Meesha Gupta | 230953090 | DistilBERT model ownership; literature/repository evidence to be added by member | __________________ |
| Ratul Jha | 230953338 | 1D-CNN model ownership; preprocessing/model review to be verified by member | __________________ |
| Harshavardhan Reddy | 230953396 | Report integration, preprocessing pipeline, experiment orchestration | __________________ |
| Kshitij Singh | 230953460 | MLP model ownership; evaluation and comparison to be verified by member | __________________ |

**Git evidence:** The canonical private repository is linked above. The owner-specific files and smoke metrics are present, but member attribution and signatures must still be verified honestly. Do not simulate signatures or claim work that a member did not perform.

## 5. Risk / Plan for Remaining Work

| Risk | Mitigation | Target |
|---|---|---|
| Late Phase 2 submission may require faculty approval. | Submit immediately with a concise late-submission explanation and retain the complete evidence package. | 02 Oct 2026 |
| CPU/runtime constraints for full IMDb experiments. | Run smoke test first, then full training in stages; checkpoint metrics and record hardware/runtime. | 03-05 Oct |
| BiLSTM and DistilBERT not yet evidenced locally. | Add both models to the shared preprocessing/evaluation interface; assign one owner per model. | 06-10 Oct |
| Inconsistent comparisons across models. | Freeze split, vocabulary/tokenizer policy, metrics, seed, and held-out test procedure. | 10 Oct |
| Missing contribution verification. | Push member-specific commits and obtain signed Part B/C statements. | 12 Oct |
| Error analysis and final figures delayed. | Save confusion matrices, learning curves, and representative false positives/negatives for all models. | 18 Oct |

**Final timeline:** full four-model evaluation by 14 Oct; error analysis by 18 Oct; integrated Part C report by 27 Oct; slides/rehearsal and final code push by 31 Oct 2026.

## References

[1] A. L. Maas et al., “Learning Word Vectors for Sentiment Analysis,” ACL-HLT, 2011.  
[2] Y. Kim, “Convolutional Neural Networks for Sentence Classification,” EMNLP, 2014.  
[3] S. Hochreiter and J. Schmidhuber, “Long Short-Term Memory,” Neural Computation, 1997.  
[4] K. S. Tai, R. Socher, and C. D. Manning, “Improved Semantic Representations From Tree-Structured Long Short-Term Memory Networks,” ACL, 2015.  
[5] T. Mikolov et al., “Efficient Estimation of Word Representations in Vector Space,” arXiv:1301.3781, 2013.  
[6] M. E. Peters et al., “Deep Contextualized Word Representations,” NAACL, 2018.  
[7] A. Vaswani et al., “Attention Is All You Need,” NeurIPS, 2017.  
[8] J. Devlin et al., “BERT: Pre-training of Deep Bidirectional Transformers for Language Understanding,” NAACL-HLT, 2019.  
[9] V. Sanh et al., “DistilBERT, a Distilled Version of BERT: Smaller, Faster, Cheaper and Lighter,” NeurIPS EMC2 Workshop, 2019.  
[10] J. Howard and S. Ruder, “Universal Language Model Fine-tuning for Text Classification,” ACL, 2018.

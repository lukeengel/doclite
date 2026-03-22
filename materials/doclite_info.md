1. Project Overview
The objective of this project is to implement a lightweight document understanding pipeline
based on the SlimDoc framework. The goal is to distill knowledge from a large, multimodal
document transformer (LayoutLMv3) into a smaller and more efficient model (LiLT), while
maintaining competitive performance on structured document understanding tasks. The focus at
this stage is on building a working baseline system that supports knowledge distillation using
textual and layout information, without incorporating image features(for now).
2. Dataset Understanding and Task Definition
The FUNSD dataset has been selected as the primary dataset for experimentation. This dataset
consists of scanned forms annotated with semantic labels and spatial information. Each
document is represented as a JSON file containing a list of entities under the "form" field. Each
entity includes textual content, bounding box coordinates, a semantic label (such as question,
answer, header, or other), and word-level breakdowns.
The task formulated using this dataset is token classification, where each word in the document
is assigned a label corresponding to its semantic role. Although the dataset also provides linking
information that defines relationships between entities (such as question-answer pairs), this
project currently focuses only on the token-level labeling task. The linking information is
intentionally excluded to reduce complexity and ensure that a baseline system can be
implemented within the given timeline.
3. Data Processing Strategy
A preprocessing strategy has been designed to convert the FUNSD JSON annotations into
model-ready inputs. For each document, the system iterates through all entities and extracts
individual words along with their corresponding bounding boxes. Each word inherits the label of
its parent entity, resulting in a word-level labeling scheme.
The processed data is structured into four main components: input tokens (words), bounding
boxes (spatial coordinates), attention masks, and labels. Empty or invalid tokens are filtered out
during preprocessing. A label mapping is defined to convert textual labels into numerical IDs
required for model training. This transformation ensures compatibility with transformer-based
architectures that expect tokenized input sequences along with aligned labels.
4. Model Architecture
The system is designed using a teacher-student paradigm. The teacher model is based on
LayoutLMv3, a multimodal transformer capable of incorporating text, layout, and visual
information. However, for the baseline implementation, only textual and layout inputs are
used. The student model is based on LiLT, which is a lightweight architecture designed
specifically for layout-aware document understanding without visual features.
Both the teacher and student models are implemented as wrapper classes to standardize their
outputs. Each model returns logits, hidden states, and attention maps during the forward pass.
This design is critical for enabling knowledge distillation, as it allows the student model to learn
not only from the teacher’s final predictions but also from its internal representations.
5. Knowledge Distillation Framework
The distillation framework is implemented following the Doclite(our core logic) methodology,
which combines multiple supervision signals to guide the student model. Three types of
distillation losses are used.
The first is hidden state distillation, where the student is trained to match the internal
representations of the teacher across corresponding transformer layers. Since the teacher and
student may have different numbers of layers, a layer-mapping strategy is used to align student
layers with appropriate teacher layers.
The second is attention distillation, which encourages the student model to replicate the
attention patterns of the teacher. This ensures that the student learns similar token-to-token
relationships and structural dependencies within the document.
The third is logit distillation, which aligns the output probability distributions of the student with
those of the teacher using temperature-scaled KL divergence. This allows the student to
capture the teacher’s soft predictions rather than relying solely on hard labels.
These three losses are combined using weighted coefficients(yet to decide distribution) to
produce a total distillation loss, which serves as the primary training objective.
6. Training Pipeline
A modular training pipeline has been implemented to support the distillation process. During
each training step, the teacher model is run in evaluation mode to generate supervision signals,
while the student model is trained using gradient updates. The input batch is passed through
both models, and the distillation loss is computed using their outputs.
The training step includes standard procedures such as gradient zeroing, forward passes, loss
computation, backpropagation, and optimizer updates. The teacher model remains frozen
throughout training, ensuring that only the student model is updated.
To validate the correctness of the training pipeline, an initial experiment was conducted using
synthetic (fake) data. This allowed verification of model compatibility, loss computation, and
gradient flow without introducing the complexity of real data preprocessing.
7. Data Pipeline and Utilities
To support real data training, a dataset class has been implemented to wrap processed
document examples into a format compatible with PyTorch DataLoaders. Each example
contains tokenized inputs, bounding boxes, attention masks, and labels. A custom collate
function is used to batch multiple examples into tensors suitable for model input.
Additional utility components include checkpointing mechanisms for saving and loading model
states during training, as well as an evaluation module that computes token-level accuracy by
comparing predicted labels with ground truth labels.
8. Scope and Current Limitations
At this stage, the project intentionally limits its scope to ensure successful baseline
implementation. Image features from the dataset are not used, and the teacher model operates
in a text-plus-layout mode. Relationship extraction using linking annotations is also excluded,
and the focus remains strictly on token classification.
Furthermore, advanced components of the Doclite framework, such as transitive distillation
using unlabeled data, are not yet implemented. These are planned as future extensions once
the baseline system is fully functional.
9. Current Status
So far, the core components of the SlimDoc-inspired distillation framework have been
successfully implemented. This includes model wrappers for both teacher and student, multiple
distillation loss functions, a layer mapping strategy, and a working training loop verified using
synthetic data. The data processing pipeline for FUNSD has been designed, and integration
with real data is the immediate next step.
10. Next Steps
The next phase of the project involves completing the integration with the FUNSD dataset by
implementing the parsing and tokenization pipeline. This will enable training the student model
on real document data. Following this, a baseline experiment will be conducted to evaluate the
performance of the distilled model.
Subsequent work will focus on extending the system through ablation studies, efficiency
analysis, and potentially incorporating additional distillation strategies or datasets.
FUNSD dataset link : https://guillaumejaume.github.io/FUNSD/
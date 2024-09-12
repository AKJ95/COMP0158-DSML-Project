# Enhancing Biomedical Entity Linking: Improving Ensemble Schemes and Leveraging Knowledge Base Descriptions
This repository serves as the codebase for the COMP0158 DSML project module. The aim of the project is to
investigate models and algorithms that facilitate named entity linking in biomedical texts.

## Brief Overview of Source Code Files
All source code developed for this project is stored in the `src` directory and the `notebooks` directory. This section
provides a brief overview of all source code files and JuPyter notebooks for inspection:
- src
    - `bootstrap_contextual.py`: Pre-computes contextual embeddings for all gold mention spans in MedMentions ST21pv training set.
    - `convert_mm_sentences`: Converts MedMentions ST21pv dataset to json format, easier for processing in future stages.
    - `convert_tf_weights_to_pt.py`: Converts TensorFlow weights to PyTorch weights for the MLP Classifier.
    - `create_umls_kb.py`: Creates a UMLS knowledge base in json format using data from UMLS 2017 Active Subset.
    - `dev_x_encoder.py`: Evaluates the performance of the cross-encoder re-ranker on the development or the test set of the generated re-ranking dataset.
    - `eval_mm_spans.py`: Evaluates the performance of any MedLinker configurations (apart from the one with the re-ranker) on MedMentions ST21pv test set.
    - `generate_xencoder_dataset.py`: Generates the re-ranking dataset to train the cross-encoder re-ranker.
    - `matcher_simstring.py`: The String Matcher object, one of the three predictors in the ED stage of MedLinker.
    - `medlinker.py`: The object representing the complete pipeline that is MedLinker (without the re-ranker).
    - `mm_reader.py`: An reader objects that reads in the MedMentions ST21pv dataset that has been converted into json.
    - `NERComponent.py`: The replciated Mention Detection component of MedLineker.
    - `preprocess_medmentions.py`: Preprocesses the MedMentions ST21pv dataset to be used to train the mention detection component.
    - `pytt_hf.py`: Computes SciBert encodings for a sentence, using the averaging pooling method on the final 4 hidden layers.
    - `pytt_hf_custom_tokenizer.py`: Computes SciBert encodings for a mention-entity pair, using the final hidden states of the `[CLS]` token.
    - `softmax_pytorch.py`: The MLP classifier, re-implemented with PyTorch.
    - `train_ner.py`: Trains the re-implemented mention detection component.
    - `train_score_classifier.py`: Trains the logistical regression meta learner.
    - `train_x_encoder.py`: Trains the cross-encoder re-ranker.
    - `umls.py`: An object representing the UMLS knowledge base.
    - `vectorspace.py`: The vector space model object, which is used by the 1-NN classifier.
    - `xencoder_eval_mm_spans`: Evaluates the performance of MedLinker with the cross-encoder re-ranker on MedMentions ST21pv test set.
- notebooks
    - `Precision_for_Infrequent_Entity_Analysis.ipynb`: Calculates the performance of MedLinker on infrequent entities.

## Running the Code
Due to the license of UMLS (that it cannot be redistributed by me), and the large sizes of the generated dataset and models used, it is not possible
to share them in this repository. However, the logic of the code should be clear from the source code.

## References
During the development of this project, the following resources were consulted and adapted where appropriate:

- [1] Loureiro, D., Jorge, A.M.: Ecir 2020 - medlinker: Medical entity linking with neural repre-
sentations and dictionary matching. https://github.com/danlou/MedLinker (2020)
- [2] Custom Named Entity Recognition with BERT. https://colab.research.google.com/github/NielsRogge/Transformers-Tutorials/blob/master/BERT/Custom_Named_Entity_Recognition_with_BERT_only_first_wordpiece.ipynb
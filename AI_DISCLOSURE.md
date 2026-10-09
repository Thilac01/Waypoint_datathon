# AI assistance disclosure

ChatGPT assisted with interpreting the supplied challenge, writing the Python implementation, proposing features and candidate model configurations, designing validation, and drafting supporting documentation. The implementation was executed on the supplied synthetic files to produce the included models, metrics, predictions and allocation results. The supplied feasibility checker was used without changing its rules.

Prediction models were trained from scratch locally. The Python workflow does not use pretrained predictive models, proprietary modelling/preprocessing APIs, low-code tools or an AutoML platform. Its model families, hyperparameters, temporal folds and optimization objective are visible in the source. Python libraries perform numerical training and constraint optimization.

Team review is still required: verify this disclosure against your actual workflow, understand and defend the labels and validation, and add any human changes or additional tools used. No human review, original team implementation, demo recording or submission is claimed by this draft.

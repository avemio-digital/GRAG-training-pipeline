#!/bin/bash
pip install --upgrade pip
pip install transformers==4.39.3
pip install datasets==2.17.0
pip install evaluate==0.4.1
pip install trl==0.8.1
pip install scikit-learn
HF_CALLBACK_PATH=$(python -c 'import transformers; import determined.transformers._hf_callback; print(determined.transformers._hf_callback.__file__)')
echo "Check current code:"
grep 'if metric_type == TRAIN:' $HF_CALLBACK_PATH
sed -i -e 's/if metric_type == TRAIN:/if metric_type in [TRAIN, TRAIN_AVG]:/' $HF_CALLBACK_PATH
echo "Confirm patched:"
grep 'if metric_type in' $HF_CALLBACK_PATH
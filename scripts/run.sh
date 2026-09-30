# export OPENAI_API_KEY=""
set -euo pipefail

GPT_VERSION="gpt-5.4-mini"

PAPER_NAME="multi-spectral-imaging"
PDF_PATH="../../ml-papers-to-code/multi-spectral-imaging/1471-2121-8-S1-S8.pdf" # .pdf
PDF_JSON_PATH="../../ml-papers-to-code/multi-spectral-imaging/1471-2121-8-S1-S8.json" # .json
PDF_JSON_CLEANED_PATH="../../ml-papers-to-code/multi-spectral-imaging/1471-2121-8-S1-S8-cleaned.json" # _cleaned.json
OUTPUT_DIR="../../ml-papers-to-code/multi-spectral-imaging/paper2code2"
OUTPUT_REPO_DIR="../../ml-papers-to-code/multi-spectral-imaging/repo2"

mkdir -p $OUTPUT_DIR
mkdir -p $OUTPUT_REPO_DIR

echo $PAPER_NAME

echo "------- Preprocess -------"

python ../codes/0_pdf_process.py \
    --input_json_path ${PDF_JSON_PATH} \
    --output_json_path ${PDF_JSON_CLEANED_PATH} \


echo "------- PaperCoder -------"

python ../codes/1_planning.py \
    --paper_name $PAPER_NAME \
    --gpt_version ${GPT_VERSION} \
    --pdf_json_path ${PDF_JSON_CLEANED_PATH} \
    --output_dir ${OUTPUT_DIR}


python ../codes/1.1_extract_config.py \
    --paper_name $PAPER_NAME \
    --output_dir ${OUTPUT_DIR}

cp -rp ${OUTPUT_DIR}/planning_config.yaml ${OUTPUT_REPO_DIR}/config.yaml

python ../codes/2_analyzing.py \
    --paper_name $PAPER_NAME \
    --gpt_version ${GPT_VERSION} \
    --pdf_json_path ${PDF_JSON_CLEANED_PATH} \
    --output_dir ${OUTPUT_DIR}

python ../codes/3_coding.py  \
    --paper_name $PAPER_NAME \
    --gpt_version ${GPT_VERSION} \
    --pdf_json_path ${PDF_JSON_CLEANED_PATH} \
    --output_dir ${OUTPUT_DIR} \
    --output_repo_dir ${OUTPUT_REPO_DIR} \

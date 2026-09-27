# export OPENAI_API_KEY=""
set -euo pipefail

GPT_VERSION="gpt-5.6-terra"

# https://arxiv.org/abs/2203.05556
PAPER_NAME="Piecewise-Linear-Encoder"
PDF_PATH="../examples/ple-paper.pdf" # .pdf
PDF_JSON_PATH="../examples/ple-paper.json" # .json
PDF_JSON_CLEANED_PATH="../examples/ple-paper_cleaned.json" # _cleaned.json
OUTPUT_DIR="../outputs/ple_gpt_5_6"
OUTPUT_REPO_DIR="../outputs/ple_gpt_5_6_repo"

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

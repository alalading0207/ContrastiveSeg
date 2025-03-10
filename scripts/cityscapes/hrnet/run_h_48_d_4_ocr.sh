#!/usr/bin/env bash
SCRIPTPATH="$( cd "$(dirname "$0")" >/dev/null 2>&1 ; pwd -P )"
cd $SCRIPTPATH
cd ../../../

# DATA_ROOT=$3   # 是传递给脚本的第三个参数
# SCRATCH_ROOT=$4
DATA_ROOT="/gemini/code"
SCRATCH_ROOT="/gemini/code/ContrastiveSeg"
ASSET_ROOT=${DATA_ROOT}

DATA_DIR="${DATA_ROOT}/Cityscapes"  
# DATA_DIR="${DATA_ROOT}/gemini/code/Cityscapes"
SAVE_DIR="${SCRATCH_ROOT}/Cityscapes/seg_results/"
BACKBONE="hrnet48"

CONFIGS="configs/cityscapes/H_48_D_4.json"
CONFIGS_TEST="configs/cityscapes/H_48_D_4_TEST.json"

MODEL_NAME="hrnet_w48_ocr"
LOSS_TYPE="fs_auxce_loss"
CHECKPOINTS_ROOT="${SCRATCH_ROOT}/Cityscapes/"
CHECKPOINTS_NAME="${MODEL_NAME}_paddle_lr2x_"$2
LOG_FILE="${SCRATCH_ROOT}/logs/Cityscapes/${CHECKPOINTS_NAME}.log"
echo "Logging to $LOG_FILE"
mkdir -p `dirname $LOG_FILE`

# PRETRAINED_MODEL=None
# PRETRAINED_MODEL=null/None/$None
# PRETRAINED_MODEL="${ASSET_ROOT}/hrnetv2_w48_imagenet_pretrained.pth"
MAX_ITERS=40000
# BATCH_SIZE=8
BATCH_SIZE=1
Val_BATCH_SIZE=1
BASE_LR=0.01

# if [ "$1"x == "train"x ]; then     
#   python -u main.py --configs ${CONFIGS} \
#                        --drop_last y \
#                        --phase train \
#                        --gathered n \
#                        --loss_balance y \      不用loss blance
#                        --log_to_file n \
#                        --backbone ${BACKBONE} \
#                        --model_name ${MODEL_NAME} \
#                        --gpu 0 \
#                        --data_dir ${DATA_DIR} \
#                        --loss_type ${LOSS_TYPE} \
#                        --max_iters ${MAX_ITERS} \
#                        --checkpoints_root ${CHECKPOINTS_ROOT} \
#                        --checkpoints_name ${CHECKPOINTS_NAME} \
#                        --pretrained ${PRETRAINED_MODEL} \       下面删掉了这里
#                        --train_batch_size ${BATCH_SIZE} \
#                        --distributed \         不用distributed
#                        --base_lr ${BASE_LR} \
#                        2>&1 | tee ${LOG_FILE}

if [ "$1"x == "train"x ]; then
  python -u main.py --configs ${CONFIGS} \
                       --drop_last y \
                       --phase train \
                       --gathered n \
                       --loss_balance n \
                       --log_to_file n \
                       --backbone ${BACKBONE} \
                       --model_name ${MODEL_NAME} \
                       --gpu 0 \
                       --data_dir ${DATA_DIR} \
                       --loss_type ${LOSS_TYPE} \
                       --max_iters ${MAX_ITERS} \
                       --checkpoints_root ${CHECKPOINTS_ROOT} \
                       --checkpoints_name ${CHECKPOINTS_NAME} \
                       --train_batch_size ${BATCH_SIZE} \
                       --val_batch_size ${Val_BATCH_SIZE} \
                       --base_lr ${BASE_LR} \
                       2>&1 | tee ${LOG_FILE}
                       

elif [ "$1"x == "resume"x ]; then
  python -u main.py --configs ${CONFIGS} \
                       --drop_last y \
                       --phase train \
                       --gathered n \
                       --loss_balance y \
                       --log_to_file n \
                       --backbone ${BACKBONE} \
                       --model_name ${MODEL_NAME} \
                       --max_iters ${MAX_ITERS} \
                       --data_dir ${DATA_DIR} \
                       --loss_type ${LOSS_TYPE} \
                       --gpu 0 \
                       --resume_continue y \
                       --resume ${CHECKPOINTS_ROOT}/checkpoints/cityscapes/${CHECKPOINTS_NAME}_latest.pth \
                       --checkpoints_name ${CHECKPOINTS_NAME} \
                        2>&1 | tee -a ${LOG_FILE}


elif [ "$1"x == "val"x ]; then
  python -u main.py --configs ${CONFIGS} --drop_last y  --data_dir ${DATA_DIR} \
                       --backbone ${BACKBONE} --model_name ${MODEL_NAME} --checkpoints_name ${CHECKPOINTS_NAME} \
                       --phase test --gpu 0  --resume ${CHECKPOINTS_ROOT}/checkpoints/cityscapes/${CHECKPOINTS_NAME}_latest.pth \
                       --loss_type ${LOSS_TYPE} --test_dir ${DATA_DIR}/val/image \
                       --out_dir ${SAVE_DIR}${CHECKPOINTS_NAME}_val_ms

  python -m lib.metrics.cityscapes_evaluator --pred_dir ${SAVE_DIR}${CHECKPOINTS_NAME}_val_ms/label  \
                                       --gt_dir ${DATA_DIR}/val/label

elif [ "$1"x == "segfix"x ]; then
  if [ "$5"x == "test"x ]; then
    DIR=${SAVE_DIR}${CHECKPOINTS_NAME}_test_ss/label
    echo "Applying SegFix for $DIR"
    ${PYTHON} scripts/cityscapes/segfix.py \
      --input $DIR \
      --split test \
      --offset ${DATA_ROOT}/cityscapes/test_offset/semantic/offset_hrnext/
  elif [ "$3"x == "val"x ]; then
    DIR=${SAVE_DIR}${CHECKPOINTS_NAME}_val/label
    echo "Applying SegFix for $DIR"
    ${PYTHON} scripts/cityscapes/segfix.py \
      --input $DIR \
      --split val \
      --offset ${DATA_ROOT}/cityscapes/val/offset_pred/semantic/offset_hrnext/
  fi

elif [ "$1"x == "test"x ]; then
  if [ "$3"x == "ss"x ]; then
    echo "[single scale] test"
    python -u main.py --configs ${CONFIGS} --drop_last y --data_dir ${DATA_DIR} \
                         --backbone ${BACKBONE} --model_name ${MODEL_NAME} --checkpoints_name ${CHECKPOINTS_NAME} \
                         --phase test --gpu 0 --resume ${CHECKPOINTS_ROOT}/checkpoints/cityscapes/${CHECKPOINTS_NAME}_latest.pth \
                         --test_dir ${DATA_DIR}/test --log_to_file n \
                         --out_dir ${SAVE_DIR}${CHECKPOINTS_NAME}_test_ss
  else
    echo "[multiple scale + flip] test"
    python -u main.py --configs ${CONFIGS_TEST} --drop_last y --data_dir ${DATA_DIR} \
                         --backbone ${BACKBONE} --model_name ${MODEL_NAME} --checkpoints_name ${CHECKPOINTS_NAME} \
                         --phase test --gpu 0 --resume ${CHECKPOINTS_ROOT}/checkpoints/cityscapes/${CHECKPOINTS_NAME}_latest.pth \
                         --test_dir ${DATA_DIR}/test --log_to_file n \
                         --out_dir ${SAVE_DIR}${CHECKPOINTS_NAME}_test_ms
  fi


else
  echo "$1"x" is invalid..."
fi

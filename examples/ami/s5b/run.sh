#!/bin/bash

. ./cmd.sh
. ./path.sh
set -e

# You may set 'mic' to:
#  ihm [individual headset mic- the default which gives best results]
#  sdm1 [single distant microphone- the current script allows you only to select
#        the 1st of 8 microphones]
#  mdm8 [multiple distant microphones-- currently we only support averaging over
#       the 8 source microphones].
# ... by calling this script as, for example,
# ./run.sh --mic sdm1
# ./run.sh --mic mdm8
mic=ihm

# Train systems,
nj=30 # number of parallel jobs,
. utils/parse_options.sh

base_mic=$(echo $mic | sed 's/[0-9]//g') # sdm, ihm or mdm
nmics=$(echo $mic | sed 's/[a-z]//g') # e.g. 8 for mdm8.

set -euo pipefail


if [ $# -ne 2 ]; then
  echo "Error: set GPU number & config path." 1>&2
  echo "Usage: ./run.sh path_to_config_file gpu_index or ./run.sh path_to_saved_model gpu_index" 1>&2
  exit 1
fi


echo ============================================================================
echo "                                   AMI                                     "
echo ============================================================================

stage=0
hierarchical_model=false
# hierarchical_model=true

### Set true when hiding progress bar
run_background=false

### Set true when restarting training
restart=false

### Set path to save the model
MODEL_SAVEPATH="/n/sd8/inaguma/result/ami"

### Set path to save dataset
export DATA_SAVEPATH="/n/sd8/inaguma/corpus/ami/kaldi"

### Select one tool to extract features (HTK is the fastest)
# TOOL=kaldi
TOOL=htk
# TOOL=python_speech_features
# TOOL=librosa

### Configuration of feature extranction
CHANNELS=80
WINDOW=0.025
SLIDE=0.01
ENERGY=0
DELTA=1
DELTADELTA=1
# NORMALIZE=global
NORMALIZE=speaker
# NORMALIZE=utterance


# Path where AMI gets downloaded (or where locally available):
AMI_DIR=$DATA_SAVEPATH/wav_db # Default,

# [ ! -r data/local/lm/final_lm ] && echo "Please, run 'run_prepare_shared.sh' first!" && exit 1
# final_lm=`cat data/local/lm/final_lm`
# LM=$final_lm.pr1-7


if [ $stage -le 0 ]; then
  echo ============================================================================
  echo "                           Data Preparation                               "
  echo ============================================================================

  # Download AMI corpus, You need around 130GB of free space to get whole data ihm+mdm,
  if [ -d $AMI_DIR ] && ! touch $AMI_DIR/.foo 2>/dev/null; then
    echo "$0: directory $AMI_DIR seems to exist and not be owned by you."
    echo " ... Assuming the data does not need to be downloaded.  Please use --stage 1 or more."
    exit 1
  fi
  # if [ -e $DATA_SAVEPATH/local/downloads/wget_$mic.sh ]; then
  #   echo "data/local/downloads/wget_$mic.sh already exists, better quit than re-download... (use --stage N)"
  #   exit 1
  # fi
  # local/ami_download.sh $mic $AMI_DIR

  # Download of annotations, pre-processing,
  local/ami_text_prep.sh $DATA_SAVEPATH/local/downloads

  # local/ami_prepare_dict.sh
  # utils/prepare_lang.sh data/local/dict "<unk>" data/local/lang data/lang

  if [ "$base_mic" == "mdm" ]; then
    PROCESSED_AMI_DIR=$AMI_DIR/beamformed
    if [ $stage -le 1 ]; then
      # for MDM data, do beamforming
      ! hash BeamformIt && echo "Missing BeamformIt, run 'cd ../../../tools/; extras/install_beamformit.sh; cd -;'" && exit 1
      local/ami_beamform.sh --cmd "$train_cmd" --nj 20 $nmics $AMI_DIR $PROCESSED_AMI_DIR
    fi
  else
    PROCESSED_AMI_DIR=$AMI_DIR
  fi

  # Prepare original data directories data/ihm/train_orig, etc.
  if [ $stage -le 2 ]; then
    local/ami_${base_mic}_data_prep.sh $PROCESSED_AMI_DIR $mic
    local/ami_${base_mic}_scoring_data_prep.sh $PROCESSED_AMI_DIR $mic dev
    local/ami_${base_mic}_scoring_data_prep.sh $PROCESSED_AMI_DIR $mic eval
  fi

  for dset in train dev eval; do
    # this splits up the speakers (which for sdm and mdm just correspond
    # to recordings) into 30-second chunks.  It's like a very brain-dead form
    # of diarization; we can later replace it with 'real' diarization.
    seconds_per_spk_max=30
    [ "$mic" == "ihm" ] && seconds_per_spk_max=120  # speaker info for ihm is real,
                                                    # so organize into much bigger chunks.

    # Note: the 30 on the next line should have been $seconds_per_spk_max
    # (thanks: Pavel Denisov.  This is a bug but before fixing it we'd have to
    # test the WER impact.  I suspect it will be quite small and maybe hard to
    # measure consistently.
    utils/data/modify_speaker_info.sh --seconds-per-spk-max 30 \
      $DATA_SAVEPATH/$mic/${dset}_orig $DATA_SAVEPATH/$mic/$dset
  done

  echo "Finish data preparation (stage: 0)."
fi


if [ $stage -le 1 ]; then
  echo ============================================================================
  echo "                        Feature extranction                               "
  echo ============================================================================

  if [ $TOOL = "kaldi" ]; then
    for x in train dev eval; do
      # steps/make_fbank.sh --nj 8 --cmd run.pl $DATA_SAVEPATH/$x exp/make_fbank/$x $DATA_SAVEPATH/fbank || exit 1;
      # steps/compute_cmvn_stats.sh $DATA_SAVEPATH/$x exp/make_fbank/$x $DATA_SAVEPATH/fbank || exit 1;
      # utils/fix_data_dir.sh $DATA_SAVEPATH/$x || exit 1;

      steps/make_mfcc.sh --nj 15 --cmd "$train_cmd" data/$mic/$dset
      steps/compute_cmvn_stats.sh data/$mic/$dset
      utils/fix_data_dir.sh data/$mic/$dset
    done

  elif [ $TOOL = "htk" ]; then
    # Make a config file to covert from wav to htk file
    python local/make_htk_config.py \
        --data_save_path $DATA_SAVEPATH \
        --config_save_path ./conf \
        --channels $CHANNELS \
        --window $WINDOW \
        --slide $SLIDE \
        --energy $ENERGY \
        --delta $DELTA \
        --deltadelta $DELTADELTA || exit 1;

    # Convert from wav to htk files
    for data_type in train dev eval; do
      mkdir -p $DATA_SAVEPATH/$data_type/htk

      htk_paths=$(find $DATA_SAVEPATH/$data_type/htk -iname '*.htk')
      htk_file_num=$(find $DATA_SAVEPATH/$data_type/htk -iname '*.htk' | wc -l)

      if [ $htk_file_num -ne ${file_number[$data_type]} ]; then
        $HCOPY -T 1 -C ./conf/fbank.conf -S $DATA_SAVEPATH/$data_type/wav2htk.scp || exit 1;
        touch $DATA_SAVEPATH/$data_type/htk/.done_make_htk
      fi
    done

  else
    if ! which sox >&/dev/null; then
      echo "This script requires you to first install sox";
      exit 1;
    fi
  fi

  exit 1

  python local/feature_extraction.py \
    --data_save_path $DATA_SAVEPATH \
    --tool $TOOL \
    --normalize $NORMALIZE \
    --channels $CHANNELS \
    --window $WINDOW \
    --slide $SLIDE \
    --energy $ENERGY \
    --delta $DELTA \
    --deltadelta $DELTADELTA || exit 1;

  echo "Finish feature extranction (stage: 1)."
fi


if [ $stage -le 2 ]; then
  echo ============================================================================
  echo "                            Create dataset                                "
  echo ============================================================================

  if [ ! -e $DATA_SAVEPATH/dataset/$TOOL/.done_dataset ]; then
    python local/make_dataset_csv.py \
      --data_save_path $DATA_SAVEPATH \
      --tool $TOOL || exit 1;
    touch $DATA_SAVEPATH/dataset/$TOOL/.done_dataset
  fi

  echo "Finish creating dataset (stage: 2)."
fi


if [ $stage -le 3 ]; then
  echo ============================================================================
  echo "                             Training stage                               "
  echo ============================================================================

  config_path=$1
  gpu_index=$2
  filename=$(basename $config_path | awk -F. '{print $1}')

  mkdir -p log
  mkdir -p $MODEL_SAVEPATH

  echo "Start training..."

  if $hierarchical_model; then
    if $restart; then
      if $run_background; then
        CUDA_VISIBLE_DEVICES=$gpu_index CUDA_LAUNCH_BLOCKING=1 \
        nohup $PYTHON exp/training/train_hierarchical.py \
          --gpu $gpu_index \
          --saved_model_path $config_path \
          --data_save_path $DATA_SAVEPATH > log/$filename".log" &
      else
        CUDA_VISIBLE_DEVICES=$gpu_index CUDA_LAUNCH_BLOCKING=1 \
        nohup $PYTHON exp/training/train_hierarchical.py \
          --gpu $gpu_index \
          --saved_model_path $config_path \
          --data_save_path $DATA_SAVEPATH || exit 1;
      fi
    else
      if $run_background; then
        CUDA_VISIBLE_DEVICES=$gpu_index CUDA_LAUNCH_BLOCKING=1 \
        nohup $PYTHON exp/training/train_hierarchical.py \
          --gpu $gpu_index \
          --config_path $config_path \
          --model_save_path $MODEL_SAVEPATH \
          --data_save_path $DATA_SAVEPATH > log/$filename".log" &
      else
        CUDA_VISIBLE_DEVICES=$gpu_index CUDA_LAUNCH_BLOCKING=1 \
        $PYTHON exp/training/train_hierarchical.py \
          --gpu $gpu_index \
          --config_path $config_path \
          --model_save_path $MODEL_SAVEPATH \
          --data_save_path $DATA_SAVEPATH || exit 1;
      fi
    fi
  else
    if $restart; then
      if $run_background; then
        CUDA_VISIBLE_DEVICES=$gpu_index CUDA_LAUNCH_BLOCKING=1 \
        nohup $PYTHON exp/training/train.py \
          --gpu $gpu_index \
          --saved_model_path $config_path \
          --data_save_path $DATA_SAVEPATH > log/$filename".log" &
      else
        CUDA_VISIBLE_DEVICES=$gpu_index CUDA_LAUNCH_BLOCKING=1 \
        $PYTHON exp/training/train.py \
          --gpu $gpu_index \
          --saved_model_path $config_path \
          --data_save_path $DATA_SAVEPATH || exit 1;
      fi
    else
      if $run_background; then
        CUDA_VISIBLE_DEVICES=$gpu_index CUDA_LAUNCH_BLOCKING=1 \
        nohup $PYTHON exp/training/train.py \
          --gpu $gpu_index \
          --config_path $config_path \
          --model_save_path $MODEL_SAVEPATH \
          --data_save_path $DATA_SAVEPATH > log/$filename".log" &
      else
        CUDA_VISIBLE_DEVICES=$gpu_index CUDA_LAUNCH_BLOCKING=1 \
        $PYTHON exp/training/train.py \
          --gpu $gpu_index \
          --config_path $config_path \
          --model_save_path $MODEL_SAVEPATH \
          --data_save_path $DATA_SAVEPATH　|| exit 1;
      fi
    fi
  fi

  echo "Finish model training (stage: 3)."
fi



exit 1



if [ $stage -le 10 ]; then
  # The following script cleans the data and produces cleaned data
  # in data/$mic/train_cleaned, and a corresponding system
  # in exp/$mic/tri3_cleaned.  It also decodes.
  #
  # Note: local/run_cleanup_segmentation.sh defaults to using 50 jobs,
  # you can reduce it using the --nj option if you want.
  local/run_cleanup_segmentation.sh --mic $mic
fi



exit 0

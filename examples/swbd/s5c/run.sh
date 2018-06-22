#!/bin/bash

. ./cmd.sh
. ./path.sh
set -e

. utils/parse_options.sh  # e.g. this parses the --stage option if supplied.

if [ $# -ne 2 ]; then
  echo "Error: set GPU number & config path." 1>&2
  echo "Usage: ./run.sh path_to_config_file gpu_id" 1>&2
  exit 1
fi


echo ============================================================================
echo "                              Switchboard                                 "
echo ============================================================================

stage=0
run_background=true
# run_background=false

### Set path to original data
SWBD_AUDIOPATH="/n/rd21/corpora_7/swb"
EVAL2000_AUDIOPATH="/n/rd21/corpora_7/hub5_english/LDC2002S09"
EVAL2000_TRANSPATH="/n/rd21/corpora_7/hub5_english/LDC2002T43"
FISHER_PATH="/n/rd7/fisher_english"

### Set path to save dataset
export data="/n/sd8/inaguma/corpus/swbd/kaldi"

### Set path to save the model
model="/n/sd8/inaguma/result/swbd"

### Select one tool to extract features (HTK is the fastest)
# tool=kaldi
tool=htk
# tool=python_speech_features
# tool=librosa

### Configuration of feature extranction
channles=80
window=0.025
slide=0.01
energy=1
delta=1
deltadelta=1
# normalize=global
normalize=speaker
# normalize=utterance

if [ ! -e ${KALDI_ROOT}/tools/sph2pipe_v2.5/sph2pipe ]; then
  echo ============================================================================
  echo "                           Install sph2pipe                               "
  echo ============================================================================
  cur_dir=`pwd`
  # Install instructions for sph2pipe_v2.5.tar.gz
  if ! which wget >&/dev/null; then
    echo "This script requires you to first install wget";
    exit 1;
  fi
  if ! which automake >&/dev/null; then
    echo "Warning: automake not installed (IRSTLM installation will not work)"
    sleep 1
  fi
  if ! which libtoolize >&/dev/null && ! which glibtoolize >&/dev/null; then
    echo "Warning: libtoolize or glibtoolize not installed (IRSTLM installation probably will not work)"
    sleep 1
  fi

  if [ ! -e ${KALDI_ROOT}/tools/sph2pipe_v2.5.tar.gz ]; then
    wget -T 3 -t 3 http://www.openslr.org/resources/3/sph2pipe_v2.5.tar.gz -P ${KALDI_ROOT}/tools
  else
    echo "sph2pipe_v2.5.tar.gz is already downloaded."
  fi
  tar -xovzf ${KALDI_ROOT}/tools/sph2pipe_v2.5.tar.gz -C ${KALDI_ROOT}/tools
  rm ${KALDI_ROOT}/tools/sph2pipe_v2.5.tar.gz
  echo "Enter into ${KALDI_ROOT}/tools/sph2pipe_v2.5 ..."
  cd ${KALDI_ROOT}/tools/sph2pipe_v2.5
  gcc -o sph2pipe *.c -lm
  echo "Get out of ${KALDI_ROOT}/tools/sph2pipe_v2.5 ..."
  cd ${cur_dir}
fi


if [ ${stage} -le 0 ] && [ ! -e ${data}/.stage_0 ]; then
  echo ============================================================================
  echo "                           Data Preparation                               "
  echo ============================================================================

  local/swbd1_data_download.sh ${SWBD_AUDIOPATH} || exit 1;
  # local/swbd1_data_download.sh /mnt/matylda2/data/SWITCHBOARD_1R2 # BUT,

  # prepare SWBD dictionary first since we want to find acronyms according to pronunciations
  # before mapping lexicon and transcripts
  # local/swbd1_prepare_dict.sh || exit 1;

  # Prepare Switchboard data. This command can also take a second optional argument
  # which specifies the directory to Switchboard documentations. Specifically, if
  # this argument is given, the script will look for the conv.tab file and correct
  # speaker IDs to the actual speaker personal identification numbers released in
  # the documentations. The documentations can be found here:
  # https://catalog.ldc.upenn.edu/docs/LDC97S62/
  # Note: if you are using this link, make sure you rename conv_tab.csv to conv.tab
  # after downloading.
  # Usage: local/swbd1_data_prep.sh /path/to/SWBD [/path/to/SWBD_docs]
  local/swbd1_data_prep.sh ${SWBD_AUDIOPATH} || exit 1;

  # Use the first 4k sentences as dev set.  Note: when we trained the LM, we used
  # the 1st 10k sentences as dev set, so the 1st 4k won't have been used in the
  # LM training data.   However, they will be in the lexicon, plus speakers
  # may overlap, so it's still not quite equivalent to a test set.
  utils/subset_data_dir.sh --first ${data}/train 4000 ${data}/dev || exit 1; # 5hr 6min
  n=$[`cat ${data}/train/segments | wc -l` - 4000]
  utils/subset_data_dir.sh --last ${data}/train $n ${data}/train_nodev || exit 1;

  # Finally, the full training set:
  rm -rf ${data}/train
  utils/data/remove_dup_utts.sh 300 ${data}/train_nodev ${data}/train || exit 1;  # 286hr
  rm -rf ${data}/train_nodev

  # Data preparation and formatting for eval2000 (note: the "text" file
  # is not very much preprocessed; for actual WER reporting we'll use
  # sclite.
  local/eval2000_data_prep.sh ${EVAL2000_AUDIOPATH} ${EVAL2000_TRANSPATH} || exit 1;

  # prepare the rt03 data.  Note: this isn't 100% necessary for this
  # recipe, not all parts actually test using rt03.
  # local/rt03_data_prep.sh /export/corpora/LDC/LDC2007S10

  # prepare fisher data for language models (optional)
  if [ -d ${FISHER_PATH} ]; then
    # prepare fisher data and put it under data/train_fisher
    local/fisher_data_prep.sh ${FISHER_PATH}
  fi

  touch ${data}/.stage_0
  echo "Finish data preparation (stage: 0)."
fi


if [ ${stage} -le 1 ] && [ ! -e ${data}/.stage_1 ]; then
  echo ============================================================================
  echo "                        Feature extranction                               "
  echo ============================================================================

  if [ ${tool} = "kaldi" ]; then
    for x in train dev eval2000; do
      steps/make_fbank.sh --nj 8 --cmd run.pl ${data}/$x exp/make_fbank/$x ${data}/fbank || exit 1;
      steps/compute_cmvn_stats.sh ${data}/$x exp/make_fbank/$x ${data}/fbank || exit 1;
      utils/fix_data_dir.sh ${data}/$x || exit 1;
    done

  else
    if [ ${tool} = "htk" ]; then
      # Make a config file to covert from wav to htk file and split per channel
      ${PYTHON} local/make_htk_config.py \
          --data_save_path ${data} \
          --config_save_path ./conf/fbank_htk.conf \
          --audio_file_type wav \
          --channels ${channles} \
          --sampling_rate 8000 \
          --window ${window} \
          --slide ${slide} \
          --energy ${energy} \
          --delta ${delta} \
          --deltadelta ${deltadelta} || exit 1;
    fi

    for data_type in train dev eval2000; do
      mkdir -p ${data}/wav/$data_type
      mkdir -p ${data}/htk/$data_type
      [ -e ${data}/$data_type/htk.scp ] && rm ${data}/$data_type/htk.scp
      touch ${data}/$data_type/htk.scp
      cat ${data}/$data_type/wav.scp | while read line
      do
        # Convert from sph (2ch) to wav files (1ch)
        sph_path=`echo $line | awk -F " " '{ print $(NF - 1) }'`
        file_name=`basename $sph_path`
        base=${file_name%.*}
        # ext=${file_name##*.}
        wav_path_a=${data}/wav/$data_type/$base"-A.wav"
        wav_path_b=${data}/wav/$data_type/$base"-B.wav"
        if [ ! -e $wav_path_a ] || [ ! -e $wav_path_b ]; then
          echo "Converting "$sph_path
          ${KALDI_ROOT}/tools/sph2pipe_v2.5/sph2pipe -f wav -p -c 1 $sph_path $wav_path_a || exit 1;
          ${KALDI_ROOT}/tools/sph2pipe_v2.5/sph2pipe -f wav -p -c 2 $sph_path $wav_path_b || exit 1;
        fi

        if [ ${tool} = "htk" ]; then
          # Convert from wav to htk files
          htk_path_a=${data}/htk/$data_type/$base"-A.htk"
          htk_path_b=${data}/htk/$data_type/$base"-B.htk"
          if [ ! -e $htk_path_a ] || [ ! -e $htk_path_b ]; then
            echo $wav_path_a  $htk_path_a > ./tmp.scp
            echo $wav_path_b  $htk_path_b >> ./tmp.scp
            $HCOPY -T 1 -C ./conf/fbank_htk.conf -S ./tmp.scp || exit 1;
            rm ./tmp.scp
          fi
          echo $htk_path_a >> ${data}/$data_type/htk.scp
          echo $htk_path_b >> ${data}/$data_type/htk.scp
        fi
      done
    done
  fi

  ${PYTHON} local/feature_extraction.py \
    --data_save_path ${data} \
    --tool ${tool} \
    --normalize ${normalize} \
    --channels ${channles} \
    --window ${window} \
    --slide ${slide} \
    --energy ${energy} \
    --delta ${delta} \
    --deltadelta ${deltadelta} || exit 1;

  touch ${data}/.stage_1
  echo "Finish feature extranction (stage: 1)."
fi


if [ ${stage} -le 2 ] && [ ! -e ${data}/.stage_2 ]; then
  echo ============================================================================
  echo "                            Create dataset                                "
  echo ============================================================================

  if [ -d ${FISHER_PATH} ]; then
    has_fisher=true
  else
    has_fisher=false
  fi

  ${PYTHON} local/make_dataset_csv.py \
    --data_save_path ${data} \
    --tool ${tool} \
    --has_fisher $has_fisher || exit 1;

  touch ${data}/.stage_2
  echo "Finish creating dataset (stage: 2)."
fi


if [ ${stage} -le 3 ]; then
  echo ============================================================================
  echo "                             Training stage                               "
  echo ============================================================================

  config_path=$1
  gpu_id=$2
  filename=$(basename ${config_path} | awk -F. '{print $1}')

  mkdir -p log
  mkdir -p ${model}

  echo "Start training..."

  if [ `echo ${config_path} | grep 'hierarchical'` ]; then
    if [ `echo ${config_path} | grep 'result'` ]; then
      if $run_background; then
        CUDA_VISIBLE_DEVICES=${gpu_id} \
        nohup ${PYTHON} exp/training/train_hierarchical.py \
          --gpu ${gpu_id} \
          --saved_model_path ${config_path} \
          --data_save_path ${data} > log/$filename".log" &
      else
        CUDA_VISIBLE_DEVICES=${gpu_id} \
        nohup ${PYTHON} exp/training/train_hierarchical.py \
          --gpu ${gpu_id} \
          --saved_model_path ${config_path} \
          --data_save_path ${data} || exit 1;
      fi
    else
      if $run_background; then
        CUDA_VISIBLE_DEVICES=${gpu_id} \
        nohup ${PYTHON} exp/training/train_hierarchical.py \
          --gpu ${gpu_id} \
          --config_path ${config_path} \
          --model_save_path ${model} \
          --data_save_path ${data} > log/$filename".log" &
      else
        CUDA_VISIBLE_DEVICES=${gpu_id} \
        ${PYTHON} exp/training/train_hierarchical.py \
          --gpu ${gpu_id} \
          --config_path ${config_path} \
          --model_save_path ${model} \
          --data_save_path ${data} || exit 1;
      fi
    fi
  else
    if [ `echo ${config_path} | grep 'result'` ]; then
      if $run_background; then
        CUDA_VISIBLE_DEVICES=${gpu_id} \
        nohup ${PYTHON} exp/training/train.py \
          --gpu ${gpu_id} \
          --saved_model_path ${config_path} \
          --data_save_path ${data} > log/$filename".log" &
      else
        CUDA_VISIBLE_DEVICES=${gpu_id} \
        ${PYTHON} exp/training/train.py \
          --gpu ${gpu_id} \
          --saved_model_path ${config_path} \
          --data_save_path ${data} || exit 1;
      fi
    else
      if $run_background; then
        CUDA_VISIBLE_DEVICES=${gpu_id} \
        nohup ${PYTHON} exp/training/train.py \
          --gpu ${gpu_id} \
          --config_path ${config_path} \
          --model_save_path ${model} \
          --data_save_path ${data} > log/$filename".log" &
      else
        CUDA_VISIBLE_DEVICES=${gpu_id} \
        ${PYTHON} exp/training/train.py \
          --gpu ${gpu_id} \
          --config_path ${config_path} \
          --model_save_path ${model} \
          --data_save_path ${data}　|| exit 1;
      fi
    fi
  fi

  echo "Finish model training (stage: 3)."
fi


if [ ${stage} -le 5 ]; then
  echo ============================================================================
  echo "                              Rescoring                                   "
  echo ============================================================================

  echo "Finish rescoring (stage: 5)."
fi


echo "Done."


# utils/prepare_lang.sh data/local/dict_nosp \
#                         "<unk>"  data/local/lang_nosp data/lang_nosp

# Now train the language models. We are using SRILM and interpolating with an
# LM trained on the Fisher transcripts (part 2 disk is currently missing; so
# only part 1 transcripts ~700hr are used)

# If you have the Fisher data, you can set this "fisher_dir" variable.
# fisher_dirs="/export/corpora3/LDC/LDC2004T19/fe_03_p1_tran/ /export/corpora3/LDC/LDC2005T19/fe_03_p2_tran/"
# fisher_dirs="/exports/work/inf_hcrc_cstr_general/corpora/fisher/transcripts" # Edinburgh,
# fisher_dirs="/mnt/matylda2/data/FISHER/fe_03_p1_tran /mnt/matylda2/data/FISHER/fe_03_p2_tran" # BUT,
# local/swbd1_train_lms.sh ${data}/local/train/text \
#                          ${data}/local/dict_nosp/lexicon.txt ${data}/local/lm $fisher_dirs


# getting results (see RESULTS file)
# for x in 1 2 3a 3b 4a; do grep 'Percent Total Error' exp/tri$x/decode_eval2000_sw1_tg/score_*/eval2000.ctm.filt.dtl | sort -k5 -g | head -1; done

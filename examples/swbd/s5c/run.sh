#!/bin/bash

. ./cmd.sh
. ./path.sh
set -e

. utils/parse_options.sh  # e.g. this parses the --stage option if supplied.

if [ $# -ne 2 ]; then
  echo "Error: set GPU number & config path." 1>&2
  echo "Usage: ./run.sh path_to_config_file gpu_index" 1>&2
  exit 1
fi


echo ============================================================================
echo "                           Switchboard (300h)                             "
echo ============================================================================

stage=1

### Set path to original data
SWBD_AUDIOPATH="/n/sd8/inaguma/corpus/swbd/data/LDC97S62"
EVAL2000_AUDIOPATH='/n/sd8/inaguma/corpus/swbd/data/eval2000/LDC2002S09'
EVAL2000_TRANSPATH='/n/sd8/inaguma/corpus/swbd/data/eval2000/LDC2002T43'
has_fisher=false

### Set path to save dataset
export DATA_SAVEPATH="/n/sd8/inaguma/corpus/swbd/kaldi"

### Set path to save the model
MODEL_SAVEPATH="/n/sd8/inaguma/result"
# MODEL_SAVEPATH="/n/sd8/inaguma/result/swbd"

### Select one tool to extract features (HTK is the fastest)
# TOOL = 'kaldi'
TOOL='htk'
# TOOL='python_speech_features'
# TOOL='librosa'
# # TOOL='wav'

### Configuration of feature extranction
CHANNELS=40
WINDOW=0.025
SLIDE=0.01
ENERGY=0
DELTA=1
DELTADELTA=1
# NORMALIZE='global'
NORMALIZE='speaker'
# NORMALIZE='utterance'
# NORMALIZE='no'
# NOTE: normalize in [-1, 1] in case of wav


# if [ ! -e ../tools/sph2pipe_v2.5/sph2pipe ]; then
#   echo ============================================================================
#   echo "                           Install sph2pipe                               "
#   echo ============================================================================
#
#   # Install instructions for sph2pipe_v2.5.tar.gz
#   if ! which wget >&/dev/null; then
#     echo "This script requires you to first install wget";
#     exit 1;
#   fi
#   if ! which automake >&/dev/null; then
#     echo "Warning: automake not installed (IRSTLM installation will not work)"
#     sleep 1
#   fi
#   if ! which libtoolize >&/dev/null && ! which glibtoolize >&/dev/null; then
#     echo "Warning: libtoolize or glibtoolize not installed (IRSTLM installation probably will not work)"
#     sleep 1
#   fi
#
#   if [ ! -e ../tools/sph2pipe_v2.5.tar.gz ]; then
#     wget -T 3 -t 3 http://www.openslr.org/resources/3/sph2pipe_v2.5.tar.gz -P ../tools
#   else
#     echo "sph2pipe_v2.5.tar.gz is already downloaded."
#   fi
#   tar -xovzf ../tools/sph2pipe_v2.5.tar.gz -C ../tools
#   rm ../tools/sph2pipe_v2.5.tar.gz
#   echo "Enter into ../tools/sph2pipe_v2.5 ..."
#   cd ../tools/sph2pipe_v2.5
#   gcc -o sph2pipe *.c -lm
#   echo "Get out of ../tools/sph2pipe_v2.5 ..."
#   cd ../../swbd
# fi


if [ $stage -le 0 ]; then
  echo ============================================================================
  echo "                           Data Preparation                               "
  echo ============================================================================

  local/swbd1_data_download.sh $SWBD_AUDIOPATH || exit 1;
  # local/swbd1_data_download.sh /mnt/matylda2/data/SWITCHBOARD_1R2 # BUT,

  # prepare SWBD dictionary first since we want to find acronyms according to pronunciations
  # before mapping lexicon and transcripts
  local/swbd1_prepare_dict.sh || exit 1;

  # Prepare Switchboard data. This command can also take a second optional argument
  # which specifies the directory to Switchboard documentations. Specifically, if
  # this argument is given, the script will look for the conv.tab file and correct
  # speaker IDs to the actual speaker personal identification numbers released in
  # the documentations. The documentations can be found here:
  # https://catalog.ldc.upenn.edu/docs/LDC97S62/
  # Note: if you are using this link, make sure you rename conv_tab.csv to conv.tab
  # after downloading.
  # Usage: local/swbd1_data_prep.sh /path/to/SWBD [/path/to/SWBD_docs]
  local/swbd1_data_prep.sh $SWBD_AUDIOPATH || exit 1;

  # Use the first 4k sentences as dev set.  Note: when we trained the LM, we used
  # the 1st 10k sentences as dev set, so the 1st 4k won't have been used in the
  # LM training data.   However, they will be in the lexicon, plus speakers
  # may overlap, so it's still not quite equivalent to a test set.
  utils/subset_data_dir.sh --first $DATA_SAVEPATH/train 4000 $DATA_SAVEPATH/dev || exit 1; # 5hr 6min
  n=$[`cat $DATA_SAVEPATH/train/segments | wc -l` - 4000]
  utils/subset_data_dir.sh --last $DATA_SAVEPATH/train $n $DATA_SAVEPATH/train_nodev || exit 1;

  # Take the first 100k utterances (just under half the data); we'll use
  # this for later stages of training.
  # utils/subset_data_dir.sh --first $DATA_SAVEPATH/train_nodev 100000 $DATA_SAVEPATH/train_100k || exit 1;
  # utils/data/remove_dup_utts.sh 200 $DATA_SAVEPATH/train_100k $DATA_SAVEPATH/train_100k_nodup || exit 1;  # 110hr

  # Finally, the full training set:
  rm -rf $DATA_SAVEPATH/train
  utils/data/remove_dup_utts.sh 300 $DATA_SAVEPATH/train_nodev $DATA_SAVEPATH/train || exit 1;  # 286hr
  rm -rf $DATA_SAVEPATH/train_nodev

  # Data preparation and formatting for eval2000 (note: the "text" file
  # is not very much preprocessed; for actual WER reporting we'll use
  # sclite.
  local/eval2000_data_prep.sh $EVAL2000_AUDIOPATH $EVAL2000_TRANSPATH || exit 1;

  # prepare the rt03 data.  Note: this isn't 100% necessary for this
  # recipe, not all parts actually test using rt03.
  # local/rt03_data_prep.sh /export/corpora/LDC/LDC2007S10

  echo "Finish data preparation (stage: 0)."
fi

# Calculate the amount of utterance segmentations.
# perl -ne 'split; $s+=($_[3]-$_[2]); END{$h=int($s/3600); $r=($s-$h*3600); $m=int($r/60); $r-=$m*60; printf "%.1f sec -- %d:%d:%.1f\n", $s, $h, $m, $r;}' $DATA_SAVEPATH/train/segments


if [ $stage -le 1 ]; then
  echo ============================================================================
  echo "                        Feature extranction                               "
  echo ============================================================================

  # for data_type in train dev eval2000 ; do
  #   swbd_sph_paths=$(find $SWBD_AUDIOPATH -iname '*.sph')
  #   # find $SWBD_DIR/. -iname '*.sph' | sort >
  #
  #   for sph_path in $swbd_sph_paths ; do
  #     file_name=$(basename $sph_path)
  #     base=${file_name%.*}
  #     ext=${file_name##*.}
  #     wav_path_A=$DATA_SAVEPATH/wav/$data_type/$base"-A.wav"
  #     wav_path_B=$DATA_SAVEPATH/wav/$data_type/$base"-B.wav"
  #     echo "Converting from "$sph_path" to "$wav_path_A
  #     ../tools/sph2pipe_v2.5/sph2pipe -f wav -p -c 1 $sph_path $wav_path_A
  #     echo "Converting from "$sph_path" to "$wav_path_B
  #     $KALDI_ROOT/tools/sph2pipe_v2.5/sph2pipe -f wav -p -c 2 $sph_path $wav_path_B
  #   done
  # done


  if [ $TOOL = 'kaldi' ]; then
    for x in train dev eval2000; do
      steps/make_fbank.sh --nj 8 --cmd run.pl $DATA_SAVEPATH/$x exp/make_fbank/$x $DATA_SAVEPATH/fbank || exit 1;
      steps/compute_cmvn_stats.sh $DATA_SAVEPATH/$x exp/make_fbank/$x $DATA_SAVEPATH/fbank || exit 1;
      utils/fix_data_dir.sh $DATA_SAVEPATH/$x || exit 1;
    done

  else
    echo $KALDI_ROOT/tools/sph2pipe_v2.5/sph2pipe
    # Make a config file to covert from wav to htk file
    # and split per channel
    python local/make_htk_config.py \
        --data_save_path $DATA_SAVEPATH \
        --config_save_path ./conf/fbank_htk.conf \
        --channels $CHANNELS \
        --window $WINDOW \
        --slide $SLIDE \
        --energy $ENERGY \
        --delta $DELTA \
        --deltadelta $DELTADELTA || exit 1;

    if [ $TOOL = 'htk' ]; then
      # Convert from wav to htk files
      for data_type in train dev eval2000 ; do
        mkdir -p $DATA_SAVEPATH/htk
        mkdir -p $DATA_SAVEPATH/htk/$data_type

        if [ ! -e $DATA_SAVEPATH/htk/$data_type/.done_make_htk ]; then
          $HCOPY -T 1 -C ./conf/fbank_htk.conf -S $DATA_SAVEPATH/$data_type/wav2htk.scp || exit 1;
          touch $DATA_SAVEPATH/htk/$data_type/.done_make_htk
        fi
      done

    else
      if ! which sox >&/dev/null; then
        echo "This script requires you to first install sox";
        exit 1;
      fi
    fi
  fi
  exit 1

  if [ ! -e $DATA_SAVEPATH/feature/$TOOL/.done_feature_extraction ]; then
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
    touch $DATA_SAVEPATH/feature/$TOOL/.done_feature_extraction
  fi

  echo "Finish feature extranction (stage: 1)."
fi


if [ $stage -le 2 ]; then
  echo ============================================================================
  echo "                            Create dataset                                "
  echo ============================================================================

  echo "Finish creating dataset (stage: 2)."
fi

exit 1


if [ $stage -le 3 ]; then
  echo ============================================================================
  echo "                             Training stage                               "
  echo ============================================================================

  config_path=$1
  gpu_index=$2
  filename=$(basename $config_path | awk -F. '{print $1}')

  mkdir -p log

  # CUDA_VISIBLE_DEVICES=$gpu_index CUDA_LAUNCH_BLOCKING=1 \
  # nohup $PYTHON train.py \
  #   --gpu $gpu_index \
  #   --config_path $config_path \
  #   --model_save_path $MODEL_SAVEPATH > log/$filename".log" &

  CUDA_VISIBLE_DEVICES=$gpu_index CUDA_LAUNCH_BLOCKING=1 \
  $PYTHON train.py \
    --gpu $gpu_index \
    --config_path $config_path \
    --model_save_path $MODEL_SAVEPATH || exit 1;

  echo "Finish model training (stage: 3)."
fi


if [ $stage -le 4 ]; then
  echo ============================================================================
  echo "                             LM training                                 "
  echo ============================================================================

  echo "Finish LM training (stage: 4)."
fi


if [ $stage -le 5 ]; then
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
# local/swbd1_train_lms.sh $DATA_SAVEPATH/local/train/text \
#                          $DATA_SAVEPATH/local/dict_nosp/lexicon.txt $DATA_SAVEPATH/local/lm $fisher_dirs


# getting results (see RESULTS file)
# for x in 1 2 3a 3b 4a; do grep 'Percent Total Error' exp/tri$x/decode_eval2000_sw1_tg/score_*/eval2000.ctm.filt.dtl | sort -k5 -g | head -1; done

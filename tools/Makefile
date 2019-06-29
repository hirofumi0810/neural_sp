# If a Python interpreter is specified, then creates a virtualenv from it
# PYTHON := /usr/bin/python2.7
PYTHON :=
# The python version installed in the conda setup
PYTHON_VERSION := 3.7
# PyTorch version: 0.4.1 or 1.0.0
# PYTORCH_VERSION := 0.4.1
PYTORCH_VERSION := 1.0.0
# Use a prebuild Kaldi to omit the installation
KALDI :=

# Both Miniconda2/3 can install any Python versions
CONDA_URL := https://repo.continuum.io/miniconda/Miniconda3-latest-Linux-x86_64.sh
# PyTorch>=1.0.0 requires gcc>=4.9 when buliding the extensions
GCC_VERSION := $(shell gcc -dumpversion)

# Path to save tools (default: current directory)
TOOL :=
ifeq ($(strip $(TOOL)),)
	TOOL := $(PWD)
endif
CONDA=$(TOOL)/miniconda

.PHONY: all clean

all: miniconda.done kaldi.done python extra
python: neural_sp.done warp-ctc.done warp-rnnt.done
extra: sentencepiece.done nkf.done

# miniconda
miniconda.done:
	mkdir -p $(TOOL)
	if [ -e $(TOOL)/miniconda ]; then rm -rf $(TOOL)/miniconda; fi
	test -f miniconda.sh || wget $(CONDA_URL) -O $(TOOL)/miniconda.sh
	bash $(TOOL)/miniconda.sh -b -p $(CONDA)
	. $(CONDA)/bin/activate && conda update -y conda
	. $(CONDA)/bin/activate && conda install -y python=$(PYTHON_VERSION)
	. $(CONDA)/bin/activate && conda info -a
	touch miniconda.done

# kaldi
ifneq ($(strip $(KALDI)),)
kaldi.done:
	ln -s $(abspath $(KALDI)) $(TOOL)/kaldi
	touch kaldi.done
else
kaldi.done:
	test -d kaldi || git clone https://github.com/kaldi-asr/kaldi.git $(TOOL)/kaldi
	cd $(TOOL)/kaldi/tools; $(MAKE) all
	cd $(TOOL)/kaldi/src; ./configure --shared --use-cuda=no; $(MAKE) depend; $(MAKE) all
	touch kaldi.done
endif

# python (including upgrade check)
neural_sp.done:
	. $(CONDA)/bin/activate; pip list
	. $(CONDA)/bin/activate; pip install pip --upgrade
	. $(CONDA)/bin/activate; pip install -e ..  # setup.py
	. $(CONDA)/bin/activate; pip install torch==$(PYTORCH_VERSION) matplotlib
	touch neural_sp.done

# warp-ctc
warp-ctc.done:
	if [ -e $(TOOL)/warp-ctc ]; then rm -rf $(TOOL)/warp-ctc; fi
	git clone https://github.com/jnishi/warp-ctc.git $(TOOL)/warp-ctc
	# NOTE: Requires gcc>=4.9 to build extensions with pytorch>=1.0
	if . $(TOOL)/miniconda/bin/activate && python -c 'import torch as t;assert t.__version__[0] == "1"' &> /dev/null; then \
		. $(TOOL)/miniconda/bin/activate && python -c "from distutils.version import LooseVersion as V;assert V('$(GCC_VERSION)') >= V('4.9'), 'Requires gcc>=4.9'"; \
	fi
	if . $(TOOL)/miniconda/bin/activate && python -c 'import torch as t;assert t.__version__[0] == "1"' &> /dev/null; then \
		cd $(TOOL)/warp-ctc; git checkout -b pytorch-1.0.0 remotes/origin/pytorch-1.0.0; \
	fi
	. $(TOOL)/miniconda/bin/activate; cd $(TOOL)/warp-ctc && mkdir build && cd build && cmake .. && make -j4 ; true
	# . $(TOOL)/miniconda/bin/activate; pip install cffi # ??
	. $(TOOL)/miniconda/bin/activate; cd $(TOOL)/warp-ctc/pytorch_binding && python setup.py install # maybe need to: apt-get install python-dev
	touch warp-ctc.done

# warp-rnnt
warp-rnnt.done:
	if [ -e $(TOOL)/warp-rnnt ]; then rm -rf $(TOOL)/warp-rnnt; fi
	git clone https://github.com/HawkAaron/warp-transducer $(TOOL)/warp-rnnt
	# NOTE: Requires gcc>=4.9 to build extensions with pytorch>=1.0
	. $(TOOL)/miniconda/bin/activate; cd $(TOOL)/warp-rnnt && mkdir build && cd build && cmake .. && make -j4 ; true
	# . $(TOOL)/miniconda/bin/activate; pip install cffi # ??
	. $(TOOL)/miniconda/bin/activate; cd $(TOOL)/warp-rnnt/pytorch_binding && python setup.py install # maybe need to: apt-get install python-dev
	touch warp-rnnt.done

# check_install:
# 	. $(TOOL)/miniconda/bin/activate; python check_install.py

sentencepiece.done:
	if [ -e $(TOOL)/sentencepiece ]; then rm -rf $(TOOL)/sentencepiece; fi
	git clone https://github.com/google/sentencepiece.git $(TOOL)/sentencepiece
	command -v bazel > /dev/null || echo "SentencePiece requires Bazel, see https://bazel.build/"
	cd $(TOOL)/sentencepiece && bazel build src:all --incompatible_disable_deprecated_attr_params=false
	touch sentencepiece.done

nkf.done:
	if [ -e $(TOOL)/nkf ]; then rm -rf $(TOOL)/nkf; fi
	cd $(TOOL)/nkf; wget https://ja.osdn.net/dl/nkf/nkf-2.1.4.tar.gz -P $(TOOL)/nkf
	cd $(TOOL)/nkf; tar zxvf nkf-2.1.4.tar.gz; cd nkf-2.1.4; $(MAKE) prefix=.
	touch nkf.done

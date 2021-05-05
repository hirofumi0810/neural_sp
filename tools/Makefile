# If a Python interpreter is specified, then creates a virtualenv from it
# PYTHON := /usr/bin/python3.7
PYTHON :=
# The python version installed in the conda setup
PYTHON_VERSION := 3.7
CUDA_VERSION := 10.1
PYTORCH_VERSION := 1.0.1
# Use a prebuild Kaldi to omit the installation
KALDI :=

# Both Miniconda2/3 can install any Python versions
CONDA_URL := https://repo.continuum.io/miniconda/Miniconda3-latest-Linux-x86_64.sh
# PyTorch>=1.0.0 requires gcc>=4.9 when buliding the extensions
GCC_VERSION := $(shell gcc -dumpversion)

USE_PIP :=

ifeq ($(shell which nvcc),) # 'nvcc' not found
CPU_ONLY := 0
else
CPU_ONLY :=
endif

ifeq ($(strip $(CPU_ONLY)),)
# Derive CUDA version from nvcc
CUDA_VERSION = $(shell nvcc --version | grep "Cuda compilation tools" | cut -d" " -f5 | sed s/,//)
CUDA_VERSION_WITHOUT_DOT = $(strip $(subst .,,$(CUDA_VERSION)))
PIP_PYTORCH := torch==$(PYTORCH_VERSION) -f https://download.pytorch.org/whl/cu$(CUDA_VERSION_WITHOUT_DOT)/torch_stable.html
ifeq ($(strip $(CUDA_VERSION_WITHOUT_DOT)),80)
CONDA_PYTORCH := pytorch=$(PYTORCH_VERSION) cuda$(CUDA_VERSION_WITHOUT_DOT) -c pytorch
else ifeq ($(strip $(CUDA_VERSION_WITHOUT_DOT)),111)
CONDA_PYTORCH := pytorch=$(PYTORCH_VERSION) cudatoolkit=$(CUDA_VERSION) -c pytorch -c conda-forge
else
CONDA_PYTORCH := pytorch=$(PYTORCH_VERSION) cudatoolkit=$(CUDA_VERSION) -c pytorch
endif

else
CONDA_PYTORCH := pytorch=$(PYTORCH_VERSION) cpuonly -c pytorch
PIP_PYTORCH := torch==$(PYTORCH_VERSION) -f https://download.pytorch.org/whl/cpu/torch_stable.html
endif

# Path to save tools (default: current directory)
TOOL :=
ifeq ($(strip $(TOOL)),)
	TOOL := $(PWD)/neural_sp
endif
CONDA=$(TOOL)/miniconda

.PHONY: all clean

all: showenv miniconda.done kaldi.done python
python: neural_sp.done warp-ctc.done warp-transducer.done sentencepiece.done
extra: warp_rnnt.done nkf.done moses.done mwerSegmenter.done


################ Logging ################
showenv:
ifeq ($(strip $(CPU_ONLY)),)
	@echo CUDA_VERSION=$(CUDA_VERSION)
else
	@echo Perform on CPU mode: CPU_ONLY=$(CPU_ONLY)
endif
	@echo PYTHON=$(shell . $(CONDA)/bin/activate && command -v python3)
	@echo PYTHON_VERSION=$(shell . $(CONDA)/bin/activate && python3 --version)
	@echo USE_PIP=$(USE_PIP)
	@echo PYTORCH_VERSION=$(PYTORCH_VERSION)
	@echo CONDA_PYTORCH=$(CONDA_PYTORCH)
	@echo PIP_PYTORCH=$(PIP_PYTORCH)


miniconda.done:
	mkdir -p $(TOOL)
	if [ -e $(CONDA) ]; then rm -rf $(CONDA); fi
	test -f $(TOOL)/miniconda.sh || wget $(CONDA_URL) -O $(TOOL)/miniconda.sh
	test -d $(CONDA) || bash $(TOOL)/miniconda.sh -b -p $(CONDA)
	. $(CONDA)/bin/activate && conda config --show
	. $(CONDA)/bin/activate && conda install -y python=$(PYTHON_VERSION)
	. $(CONDA)/bin/activate && conda install -y pip setuptools
	. $(CONDA)/bin/activate && conda info -a
	touch miniconda.done

ifneq ($(strip $(KALDI)),)
kaldi.done:
	if [ ! -d $(TOOL)/kaldi ]; then ln -s $(abspath $(KALDI)) $(TOOL)/kaldi; fi
	touch kaldi.done
else
kaldi.done:
	test -d kaldi || git clone https://github.com/kaldi-asr/kaldi.git $(TOOL)/kaldi
	cd $(TOOL)/kaldi/tools; $(MAKE) all
	cd $(TOOL)/kaldi/src; ./configure --shared --use-cuda=no; $(MAKE) depend; $(MAKE) all
	touch kaldi.done
endif

neural_sp.done:
	. $(CONDA)/bin/activate; python --version
	. $(CONDA)/bin/activate; pip list
	# . $(CONDA)/bin/activate; pip install pip --upgrade
	# . $(CONDA)/bin/activate; pip install numpy --upgrade
	. $(CONDA)/bin/activate; pip install -e ..  # setup.py
	. $(CONDA)/bin/activate; pip list
ifeq ($(strip $(USE_PIP)),)
	. $(CONDA)/bin/activate && pip install $(PIP_PYTORCH)
	touch neural_sp.done
else
	. $(CONDA)/bin/activate && conda install -y $(CONDA_PYTORCH)
	touch neural_sp.done
endif

warp-ctc.done:
	if [ $(shell . $(CONDA)/bin/activate && python -c 'import torch as t;from distutils.version import LooseVersion as V;print(int(V("1.7") > V(t.__version__) >= V("1.1")))') -ne 0 ]; then \
		if [ $(strip $(CPU_ONLY)) ]; then \
			. $(CONDA)/bin/activate && pip install warpctc-pytorch==0.2.2+torch$(shell . $(CONDA)/bin/activate && python -c 'import torch as t; print(t.__version__[0] + t.__version__[2])').cpu -f https://github.com/espnet/warp-ctc/releases/tag/v0.2.2; \
		else \
			. $(CONDA)/bin/activate && pip install warpctc-pytorch==0.2.2+torch$(shell . $(CONDA)/bin/activate && python -c 'import torch as t; print(t.__version__[0] + t.__version__[2])').cuda$(strip $(subst .,,$(CUDA_VERSION))) -f https://github.com/espnet/warp-ctc/releases/tag/v0.2.2; \
		fi \
	elif [ $(shell . $(CONDA)/bin/activate && python -c 'import torch as t;from distutils.version import LooseVersion as V;print(int(V("1.1") > V(t.__version__) >= V("1.0")))') -ne 0 ]; then \
		if [ $(strip $(CPU_ONLY)) ]; then \
			. $(CONDA)/bin/activate && pip install warpctc-pytorch10-cpu; \
		else \
			. $(CONDA)/bin/activate && pip install warpctc-pytorch10-cuda$(strip $(subst .,,$(CUDA_VERSION))); \
		fi \
	elif [ $(shell . $(CONDA)/bin/activate && python -c 'import torch as t;from distutils.version import LooseVersion as V;print(int(V(t.__version__) < V("1.0")))') -ne 0 ]; then \
		if [ -e $(TOOL)/warp-ctc ]; then rm -rf $(TOOL)/warp-ctc; fi; \
		git clone https://github.com/espnet/warp-ctc.git; $(TOOL)/warp-ctc \
		cd $(TOOL)/warp-ctc; git checkout -b pytorch-0.4 remotes/origin/pytorch-0.4; \
		mkdir build && cd build && cmake .. && $(MAKE) && cd ..; \
		. $(CONDA)/bin/activate; pip install cffi; cd pytorch_binding && python setup.py install; \
	else \
		echo "Skip warp-ctc installation"; \
	fi
	touch warp-ctc.done

warp-transducer.done:
	if [ -e $(TOOL)/warp-transducer ]; then rm -rf $(TOOL)/warp-transducer; fi
	git clone https://github.com/HawkAaron/warp-transducer.git $(TOOL)/warp-transducer
	# Note: Requires gcc>=5.0 to build extensions with pytorch>=1.0
	if . $(CONDA)/bin/activate && python -c 'import torch as t;assert t.__version__[0] == "1"' &> /dev/null; then \
		. $(CONDA)/bin/activate && python -c "from distutils.version import LooseVersion as V;assert V('$(GCC_VERSION)') >= V('5.0'), 'Requires gcc>=5.0'"; \
	fi
	. $(CONDA)/bin/activate; cd $(TOOL)/warp-transducer && mkdir build && cd build && cmake .. && $(MAKE); true
	. $(CONDA)/bin/activate; cd $(TOOL)/warp-transducer/pytorch_binding && python setup.py install
	touch warp-transducer.done

warp_rnnt.done:
	. $(CONDA)/bin/activate; pip install warp_rnnt==0.3  # GPU only
	touch warp_rnnt.done

sentencepiece.done:
	if [ -e $(TOOL)/sentencepiece ]; then rm -rf $(TOOL)/sentencepiece; fi
	git clone https://github.com/google/sentencepiece.git $(TOOL)/sentencepiece
	cd $(TOOL)/sentencepiece && mkdir build && cd build && (cmake3 .. || cmake ..) && $(MAKE)
	# command -v bazel > /dev/null || echo "SentencePiece requires Bazel, see https://bazel.build/"
	# cd $(TOOL)/sentencepiece && bazel build src:all --incompatible_disable_deprecated_attr_params=false
	touch sentencepiece.done

nkf.done:
	if [ -e $(TOOL)/nkf ]; then rm -rf $(TOOL)/nkf; fi
	cd $(TOOL)/nkf; wget https://ja.osdn.net/dl/nkf/nkf-2.1.4.tar.gz -P $(TOOL)/nkf
	cd $(TOOL)/nkf; tar zxvf nkf-2.1.4.tar.gz; cd nkf-2.1.4; $(MAKE) prefix=.
	touch nkf.done

moses.done:
	if [ -e $(TOOL)/moses ]; then rm -rf $(TOOL)/moses; fi
	cd $(TOOL)/moses; git clone https://github.com/moses-smt/mosesdecoder.git $(TOOL)/moses
	touch moses.done

mwerSegmenter.done:
	cd $(TOOL); wget https://www-i6.informatik.rwth-aachen.de/web/Software/mwerSegmenter.tar.gz -P $(TOOL)/
	cd $(TOOL); tar zxvf mwerSegmenter.tar.gz
	cd $(TOOL); rm mwerSegmenter.tar.gz
	touch mwerSegmenter.done

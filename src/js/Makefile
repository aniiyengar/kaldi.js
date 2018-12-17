include ../kaldi.mk

ADDLIBS = ../online2/kaldi-online2.bc ../ivector/kaldi-ivector.bc \
          ../nnet3/kaldi-nnet3.bc ../chain/kaldi-chain.bc \
          ../cudamatrix/kaldi-cudamatrix.bc ../decoder/kaldi-decoder.bc \
          ../lat/kaldi-lat.bc ../fstext/kaldi-fstext.bc ../hmm/kaldi-hmm.bc \
          ../feat/kaldi-feat.bc ../transform/kaldi-transform.bc \
          ../gmm/kaldi-gmm.bc ../tree/kaldi-tree.bc ../util/kaldi-util.bc \
          ../matrix/kaldi-matrix.bc ../base/kaldi-base.bc

#CXXFLAGS += -I.
LDFLAGS += -s WASM=1 -s TOTAL_MEMORY=512MB -s ALLOW_MEMORY_GROWTH=1 -s DISABLE_EXCEPTION_CATCHING=0 -s DEMANGLE_SUPPORT=1 -s SAFE_HEAP=1 -s ASSERTIONS=1 -s "BINARYEN_TRAP_MODE='clamp'"

include ../makefiles/default_rules.mk

gmm: decode-gmm.cc kaldi-interop.js model_gmm/HCLG.fst model_gmm/words.txt
	$(CXX) -o gmm-kaldi-worker.js $(LDFLAGS) $(CXXFLAGS) $(LDLIBS) $(XDEPENDS) decode-gmm.cc \
		--preload-file model_gmm/final.mdl \
		--preload-file model_gmm/matrix.scp \
		--preload-file model_gmm/mfcc.conf \
		--preload-file model_gmm/HCLG.fst \
		--preload-file model_gmm/words.txt \
	&& ./cpy.sh gmm

nnet3: decode-nnet3.cc kaldi-interop.js model_nnet3/HCLG.fst model_nnet3/words.txt
	$(CXX) -o nnet3-kaldi-worker.js $(LDFLAGS) $(CXXFLAGS) $(LDLIBS) $(XDEPENDS) decode-nnet3.cc \
		--preload-file model_nnet3/final.mdl \
		--preload-file model_nnet3/mfcc.conf \
		--preload-file model_nnet3/HCLG.fst \
		--preload-file model_nnet3/words.txt \
	&& ./cpy.sh nnet3

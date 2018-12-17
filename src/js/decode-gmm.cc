// online2bin/online2-wav-nnet3-latgen-faster.cc

// Copyright 2014  Johns Hopkins University (author: Daniel Povey)
//           2016  Api.ai (Author: Ilya Platonov)

// See ../../COPYING for clarification regarding multiple authors
//
// Licensed under the Apache License, Version 2.0 (the "License");
// you may not use this file except in compliance with the License.
// You may obtain a copy of the License at
//
//  http://www.apache.org/licenses/LICENSE-2.0
//
// THIS CODE IS PROVIDED *AS IS* BASIS, WITHOUT WARRANTIES OR CONDITIONS OF ANY
// KIND, EITHER EXPRESS OR IMPLIED, INCLUDING WITHOUT LIMITATION ANY IMPLIED
// WARRANTIES OR CONDITIONS OF TITLE, FITNESS FOR A PARTICULAR PURPOSE,
// MERCHANTABLITY OR NON-INFRINGEMENT.
// See the Apache 2 License for the specific language governing permissions and
// limitations under the License.

#include "feat/resample.h"
#include "feat/wave-reader.h"
#include "fstext/fstext-lib.h"
#include "lat/lattice-functions.h"
#include "nnet3/nnet-utils.h"
#include "online2/online-nnet3-decoding.h"
#include "online2/online-nnet2-feature-pipeline.h"
#include "online2/onlinebin-util.h"
#include "online2/online-timing.h"
#include "online2/online-endpoint.h"
#include "online2/online-feature-pipeline.h"
#include "online2/online-gmm-decoding.h"

#include <algorithm>
#include <sys/types.h>
#include <math.h>

#include <sstream>
#include <string>
#include <iostream>

#include <emscripten.h>
#include "json.hpp"

using namespace kaldi;
using json = nlohmann::json;

extern "C" {
  int EMSCRIPTEN_KEEPALIVE main();
  float* EMSCRIPTEN_KEEPALIVE KaldiJsInit(char *commandline);
  void EMSCRIPTEN_KEEPALIVE KaldiJsReset();
  void EMSCRIPTEN_KEEPALIVE KaldiJsHandleAudio();
}

class ASR {
public:
  ~ASR();
  void DecodeChunk();
  void ResetDecoder();
  
  OnlineFeaturePipelineCommandLineConfig feature_cmdline_config;
  OnlineGmmDecodingConfig decode_config;

  OnlineFeaturePipelineConfig *feature_config;
  OnlineFeaturePipeline *pipeline_prototype;
  OnlineGmmDecodingModels *gmm_models;

  fst::Fst<fst::StdArc> *decode_fst;
  fst::SymbolTable *word_syms;
  int64 last_output;

  OnlineTimingStats timing_stats;
  OnlineTimer *decoding_timer;

  OnlineGmmAdaptationState *adaptation_state;
  SingleUtteranceGmmDecoder *decoder;
  BaseFloat in_samp_freq;
  BaseFloat asr_samp_freq;

  Vector<BaseFloat> audio_in;
  Vector<BaseFloat> chunk;
  Vector<BaseFloat> audio_in_resampled;
  LinearResample *resampler;

  int chunk_valid;
  int total_length;
  bool enabled;
  bool endpoint_detected;
};

static ASR *asr;

ASR::~ASR() {
  delete word_syms;
  delete decode_fst;
}

void ASR::ResetDecoder() {
  this->chunk_valid = 0;
  this->last_output = 0;
  this->total_length = 0;

  OnlineFeaturePipelineCommandLineConfig cmd_cfg;
  cmd_cfg.global_cmvn_stats_rxfilename = "model_gmm/matrix.scp";
  cmd_cfg.mfcc_config = "model_gmm/mfcc.conf";
  cmd_cfg.feature_type = "mfcc";

  this->asr_samp_freq = 8000;
  this->resampler = new LinearResample(
                this->in_samp_freq, this->asr_samp_freq, 3900, 4);
  this->feature_config = new OnlineFeaturePipelineConfig(cmd_cfg);
  this->pipeline_prototype = new OnlineFeaturePipeline(
                *this->feature_config);

  OnlineGmmAdaptationState st;
  this->adaptation_state = &st;

  this->decode_config.faster_decoder_opts.prune_scale = 0.1;

  this->gmm_models = new OnlineGmmDecodingModels(
                this->decode_config);
  this->decoder = new SingleUtteranceGmmDecoder(
                this->decode_config,
                *this->gmm_models,
                *this->pipeline_prototype,
                *this->decode_fst,
                *this->adaptation_state);

  this->decoder->GetAdaptationState(this->adaptation_state);
  this->decoding_timer = new OnlineTimer("utt");
}

string GetBestTranscript(const fst::SymbolTable *word_syms,
                    const SingleUtteranceGmmDecoder &decoder) {
  Lattice best_path_lat;
  decoder.GetBestPath(false, &best_path_lat);

  if (best_path_lat.NumStates() == 0) {
    KALDI_WARN << "Empty lattice.";
    return "";
  }

  LatticeWeight weight;
  std::vector<int32> alignment;
  std::vector<int32> words;
  GetLinearSymbolSequence(best_path_lat, &alignment, &words, &weight);

  std::ostringstream os;
  for (size_t i = 0; i < words.size(); i++) {
    std::string s = asr->word_syms->Find(words[i]);
    os << s;
    if (i < words.size() - 1)
      os << " ";
    if (s == "")
      KALDI_ERR << "Word-id " << words[i] << " not in symbol table.";
  }

  return os.str();
}

void ASR::DecodeChunk() {
  OnlineEndpointConfig cfg;
  cfg.silence_phones = "1:2:3:4:5:6:7:8:9:10:11:12:13:14:15:16:17:18:19:20";

  int this_chunk_length = this->chunk_valid;
  this->total_length += this_chunk_length;

  this->pipeline_prototype->AcceptWaveform(
            asr->asr_samp_freq,
            this->chunk.Range(0, this_chunk_length));

  this->chunk_valid = 0;

  asr->decoding_timer->WaitUntil(asr->total_length / asr->asr_samp_freq);
  asr->decoder->AdvanceDecoding();

  if (asr->total_length - asr->last_output > asr->asr_samp_freq) {
    asr->endpoint_detected = asr->decoder->EndpointDetected(cfg);
    KALDI_LOG << asr->decoder->FinalRelativeCost()
              << ": " << GetBestTranscript(asr->word_syms, *asr->decoder);

    asr->last_output += asr->asr_samp_freq;
  }
}

int main() {
  EM_ASM(
    self.importScripts('kaldi-interop.js');
  );

  emscripten_exit_with_live_runtime();
  return 0;
}

float *_KaldiJsInit(char *commandline) {
  try {
    using namespace kaldi;
    using namespace fst;

    typedef kaldi::int32 int32;
    typedef kaldi::int64 int64;

    ParseOptions po("Online GMM decoding.\n");

    BaseFloat chunk_length_secs = 0.05;
    bool do_endpointing = false;
    std::string use_gpu = "no";
    BaseFloat in_samp_freq;
    int in_buffer_size;

    std::string fst_rxfilename = "model_gmm/HCLG.fst";
    std::string word_syms_rxfilename = "model_gmm/words.txt";

    po.Register("chunk-length", &chunk_length_secs, "");
    po.Register("do-endpointing", &do_endpointing, "");
    po.Register("in-samp-freq", &in_samp_freq, "");
    po.Register("in-bufsize", &in_buffer_size, "");

    // asr->feature_cmdline_config.Register(&po);
    asr->decode_config.Register(&po);

    json args = json::parse(commandline);
    const char *argv[2];
    argv[0] = "";
    for (json::iterator it = args.begin(); it != args.end(); ++it) {
      std::string arg = std::string("--") + it.key() + "=";
      if (it.value().type() == json::value_t::string) {
          arg += it.value().get<std::string>();
      } else {
          arg += it.value().dump();
      }
      argv[1] = arg.c_str();
      po.Read(2, argv);
    }

    asr->decode_fst = ReadFstKaldiGeneric(fst_rxfilename);
    asr->word_syms = fst::SymbolTable::ReadText(word_syms_rxfilename);

    asr->in_samp_freq = in_samp_freq;
    asr->ResetDecoder();

    int chunk_length = int32(asr->asr_samp_freq * chunk_length_secs);
    if (chunk_length == 0) chunk_length = 1;

    asr->audio_in.Resize(in_buffer_size);
    asr->chunk.Resize(chunk_length);
    asr->enabled = true;

    return asr->audio_in.Data();
  } catch (const std::exception &e) {
    KALDI_LOG << "exception: " << e.what();
    return NULL;
  }
}

// for some reason this needs a C wrapper but the other exports don't
float *KaldiJsInit(char *commandline) {
  return _KaldiJsInit(commandline);
}

void KaldiJsReset() {
  try {
    asr->ResetDecoder();
    asr->enabled = true;
  } catch (const std::exception &e) {
    KALDI_LOG << "exception: " << e.what();
  }
}

void SendMessage(json message) {
  std::string message_json = message.dump();
  EM_ASM_({
    postByteArray($0, $1);
  }, message_json.c_str(), message_json.length());
}

void KaldiJsHandleAudio() {
  try {
    if (!asr->enabled) return;

    asr->resampler->Resample(asr->audio_in, false, &asr->audio_in_resampled);
    asr->audio_in_resampled.Scale(kWaveSampleMax);

    KALDI_LOG << asr->pipeline_prototype->Dim();

    int new_i = 0;
    while (new_i < asr->audio_in_resampled.Dim()
                && !asr->endpoint_detected) {
      int copy_len = std::min(
                  asr->chunk.Dim() - asr->chunk_valid,
                  asr->audio_in_resampled.Dim() - new_i);

      for (int i = 0; i < copy_len; i++)
        asr->chunk(asr->chunk_valid + i) = asr->audio_in_resampled(new_i + i);

      new_i += copy_len;
      asr->chunk_valid += copy_len;
      
      if (asr->chunk_valid == asr->chunk.Dim()) {
        asr->DecodeChunk();
        json msg;
        msg["transcript"] = GetBestTranscript(asr->word_syms, *asr->decoder);
        msg["final"] = false;
        SendMessage(msg);
      }
    }

    if (asr->endpoint_detected) {
      json msg;
      msg["final"] = true;

      if (asr->decoder->FinalRelativeCost() <= 8) {
        msg["transcript"] = GetBestTranscript(
                asr->word_syms, *asr->decoder);

        SendMessage(msg);
        asr->enabled = false;
      }
    }
  } catch (const std::exception &e) {
    KALDI_LOG << "exception: " << e.what();
    json msg;
    msg["error"] = e.what();
    SendMessage(msg);
  }
}

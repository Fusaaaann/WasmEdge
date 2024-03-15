// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: 2019-2022 Second State INC

#pragma once

#include "plugin/plugin.h"
#include "types.h"

#ifdef WASMEDGE_PLUGIN_WASI_NN_BACKEND_GGML
#include <common.h>
#include <llama.h>
#include <llava.h>
#endif

namespace WasmEdge::Host::WASINN {
struct WasiNNEnvironment;
}

namespace WasmEdge::Host::WASINN::GGML {
enum speculative_strategy {
  NONE,
  SPECULATIVE,
  LOOKAHEAD
};

#ifdef WASMEDGE_PLUGIN_WASI_NN_BACKEND_GGML
struct Graph {
  llama_model *LlamaModel = nullptr;
#ifdef WASMEDGE_PLUGIN_WASI_NN_GGML_STRATEGY
  llama_model *DraftLlamaModel = nullptr;
#endif // WASMEDGE_PLUGIN_WASI_NN_GGML_STRATEGY

  std::string ModelFilePath;
  // Plugin parameters:
  bool EnableLog = false;
  bool EnableDebugLog = false;
  bool StreamStdout = false;
  bool Embedding = false;
  uint64_t NPredict;
  std::string ReversePrompt;
  std::string MMProjModelPath;
  std::string ImagePath;
  int64_t MainGPU = 0; // Use GPU 0 by default
  int64_t NGPULayers = 0;
#ifdef WASMEDGE_PLUGIN_WASI_NN_GGML_STRATEGY
  enum speculative_strategy SpeculativeStrategy;
  std::string StatsLogPath;
  std::string DraftModelPath; // only when SpeculativeStrategy == SPECULATIVE
  // bool UseKV = false;
  bool UseMMap = false;
#endif // WASMEDGE_PLUGIN_WASI_NN_GGML_STRATEGY
  // Model parameters:
  std::vector<float> TensorSplit;
  // Context parameters:
  uint64_t CtxSize;
  uint64_t BatchSize;
  uint64_t Threads;
  // Sampling parameters:
  double Temp = 0.80;
  double TopP = 0.95;
  double RepeatPenalty = 1.10;
  double PresencePenalty = 0.00;
  double FrequencyPenalty = 0.00;
#ifdef WASMEDGE_PLUGIN_WASI_NN_GGML_STRATEGY
  // speculative decoding, only when SpeculativeStrategy == SPECULATIVE
  uint64_t NParallel = 1; 
  double ProbAccept = 0.50;
  double ProbSplit = 0.10;
  // lookahead decoding
  uint64_t LookaheadWidth = 15;
  uint64_t NgramSize = 5;
  uint64_t MaxVerifyNgramSize = 15;
#endif // WASMEDGE_PLUGIN_WASI_NN_GGML_STRATEGY
};

struct Context {
public:
  Context(size_t GId, Graph &) noexcept : GraphId(GId) {}
  size_t GraphId;
  std::vector<llama_token> LlamaInputs;
  uint64_t LlamaNInputs = 0;
  std::string LlamaOutputs;
  std::vector<llama_token> LlamaOutputTokens;
  // Preserve for computing single token
  llama_context *LlamaContext = nullptr;
  struct llama_sampling_context *LlamaSampling = nullptr;
  int32_t LlamaNPast = 0;
  // Preserve for llava
  struct llava_image_embed *LlavaImageEmbd = nullptr;
  size_t LlavaImagePosition = 0;
};

// credit: https://github.com/gabime/spdlog/issues/1797#issuecomment-1013537052
namespace {
  template<class... Ts> struct overloaded : Ts... { using Ts::operator()...; };
  template<class... Ts> overloaded(Ts...) -> overloaded<Ts...>;
}

struct SimpleJSON
{
  using json_val = std::variant<uint64_t,int64_t, int, double, std::string, bool>;
  std::unordered_map<std::string, json_val> members;

  SimpleJSON(std::initializer_list<std::pair<const std::string, json_val>> il) : members{il} {}

  template<typename OStream>
  friend OStream &operator<<(OStream &os, const SimpleJSON &j)
  {
    for (const auto &kv : j.members) {
      os << ", " << std::quoted(kv.first) << ":";
      std::visit(overloaded {
        [&](uint64_t arg) { os << arg; },
        [&](int64_t arg) { os << arg; },
        [&](int arg) { os << arg; },
        [&](double arg) { os << arg; },
        [&](const std::string& arg) { os << std::quoted(arg); },
        [&](bool arg) { os << (arg ? "true" : "false"); }
      }, kv.second);
    }
    return os;
  }
};
#else
struct Graph {};
struct Context {
  Context(size_t, Graph &) noexcept {}
};
#endif

struct Environ {};

Expect<WASINN::ErrNo> load(WASINN::WasiNNEnvironment &Env,
                           Span<const Span<uint8_t>> Builders,
                           WASINN::Device Device, uint32_t &GraphId) noexcept;
Expect<WASINN::ErrNo> initExecCtx(WASINN::WasiNNEnvironment &Env,
                                  uint32_t GraphId,
                                  uint32_t &ContextId) noexcept;
Expect<WASINN::ErrNo> setInput(WASINN::WasiNNEnvironment &Env,
                               uint32_t ContextId, uint32_t Index,
                               const TensorData &Tensor) noexcept;
Expect<WASINN::ErrNo> getOutput(WASINN::WasiNNEnvironment &Env,
                                uint32_t ContextId, uint32_t Index,
                                Span<uint8_t> OutBuffer,
                                uint32_t &BytesWritten) noexcept;
Expect<WASINN::ErrNo> getOutputSingle(WASINN::WasiNNEnvironment &Env,
                                      uint32_t ContextId, uint32_t Index,
                                      Span<uint8_t> OutBuffer,
                                      uint32_t &BytesWritten) noexcept;
Expect<WASINN::ErrNo> compute(WASINN::WasiNNEnvironment &Env,
                              uint32_t ContextId) noexcept;
Expect<WASINN::ErrNo> computeSingle(WASINN::WasiNNEnvironment &Env,
                                    uint32_t ContextId) noexcept;
Expect<WASINN::ErrNo> finiSingle(WASINN::WasiNNEnvironment &Env,
                                 uint32_t ContextId) noexcept;
} // namespace WasmEdge::Host::WASINN::GGML

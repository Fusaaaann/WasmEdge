// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: 2019-2022 Second State INC

#include "ggml.h"
#include "wasinnenv.h"

#ifdef WASMEDGE_PLUGIN_WASI_NN_BACKEND_GGML
#include "simdjson.h"
#include <algorithm>
#include <base64.hpp>
#include <clip.h>
#include <common.h>
#include <cstdlib>
#include <llama.h>
#include <llava.h>
#include <sstream>
#include "spdlog/sinks/basic_file_sink.h"
#include <filesystem>
#include "spdlog/fmt/ostr.h"
#ifdef WASMEDGE_PLUGIN_WASI_NN_GGML_STRATEGY
#include "strategies/strategies.h"
#endif
#endif // WASMEDGE_PLUGIN_WASI_NN_BACKEND_GGML


namespace WasmEdge::Host::WASINN::GGML {
#ifdef WASMEDGE_PLUGIN_WASI_NN_BACKEND_GGML

// Speculative Decoding
#define SPEC_VOCAB_MAX_SIZE_DIFFERENCE  100
#define SPEC_VOCAB_CHECK_START_TOKEN_ID 5
namespace details {
speculative_strategy stringViewToSpeculativeStrategy(std::string_view strategy) {
  if (strategy == "NONE"sv) return speculative_strategy::NONE;
  if (strategy == "SPECULATIVE"sv) return speculative_strategy::SPECULATIVE;
  if (strategy == "LOOKAHEAD"sv) return speculative_strategy::LOOKAHEAD;
  spdlog::error("[WASI-NN] GGML backend: Unknown speculative strategy: {}, fallback to NONE..."sv, strategy);
  return speculative_strategy::NONE; // Default or error value
}
std::string speculativeStrategyToString(speculative_strategy strategy) {
    switch (strategy) {
        case NONE: return "NONE";
        case SPECULATIVE: return "SPECULATIVE";
        case LOOKAHEAD: return "LOOKAHEAD";
        default: 
          // throw std::invalid_argument("Unknown speculative strategy");
          spdlog::error("[WASI-NN] GGML backend: Unknown speculative strategy, fallback to NONE..."sv);
          return "NONE";
    }
}
std::string getMetricsFilename(const std::string& parentDir = ".",const std::string& filename="") {
    // Get current time
    auto now_time_t = std::chrono::system_clock::to_time_t(std::chrono::system_clock::now());
    std::tm now_tm = *std::localtime(&now_time_t);
    
    // Format current time and create file name
    std::ostringstream oss;
    
    if (!filename.empty())oss << (std::filesystem::absolute(std::filesystem::path(parentDir))/filename).string();
    else oss << std::filesystem::absolute(std::filesystem::path(parentDir))<<"/"<<std::put_time(&now_tm, "%Y%m%d%H%M") << ".log";
    return oss.str();
}

Expect<ErrNo> parseMetadata(Graph &GraphRef, const std::string &Metadata,
                            bool *IsModelUpdated = nullptr) noexcept {
  simdjson::dom::parser Parser;
  simdjson::dom::element Doc;
  auto ParseError = Parser.parse(Metadata).get(Doc);
  if (ParseError) {
    spdlog::error("[WASI-NN] GGML backend: Parse metadata error"sv);
    return ErrNo::InvalidEncoding;
  }
  // TODO: update doc after finalise
  // Get metadata from the json.

  // Currently supported metadata:
  // Plugin parameters (used by this plugin):
  //   enable-log: bool
  //   enable-debug-log: bool
  //   stream-stdout: bool
  //   embedding: bool
  //   n-predict: uint64_t
  //   reverse-prompt: string
  //   mmproj: string
  //   image: string
  // Model parameters (need to reload the model if updated):
  //   n-gpu-layers: int64_t
  //   main-gpu: int64_t
  //   tensor-split: string, comma-separated floating number list
  // Context parameters (used by the llama context):
  //   ctx-size: uint64_t
  //   batch-size: uint64_t
  //   threads: uint64_t
  // Sampling parameters (used by the llama sampling context).
  //   temp: double
  //   top-p: double
  //   repeat-penalty: double
  //   presence-penalty: double
  //   frequency-penalty: double

  // Get the current llama parameters.
  llama_model_params ModelParams = llama_model_default_params();
  ModelParams.n_gpu_layers = GraphRef.NGPULayers;
  ModelParams.main_gpu = GraphRef.MainGPU;
  ModelParams.tensor_split = GraphRef.TensorSplit.data();
#ifdef WASMEDGE_PLUGIN_WASI_NN_GGML_STRATEGY
  ModelParams.use_mmap = GraphRef.UseMMap;
#endif // WASMEDGE_PLUGIN_WASI_NN_GGML_STRATEGY

  // The plugin parameters.
  if (Doc.at_key("enable-log").error() == simdjson::SUCCESS) {
    auto Err = Doc["enable-log"].get<bool>().get(GraphRef.EnableLog);
    if (Err) {
      spdlog::error(
          "[WASI-NN] GGML backend: Unable to retrieve the enable-log option."sv);
      return ErrNo::InvalidArgument;
    }
    llama_log_set(nullptr, &GraphRef.EnableLog);
  }
  if (Doc.at_key("enable-debug-log").error() == simdjson::SUCCESS) {
    auto Err = Doc["enable-debug-log"].get<bool>().get(GraphRef.EnableDebugLog);
    if (Err) {
      spdlog::error(
          "[WASI-NN] GGML backend: Unable to retrieve the enable-debug-log option."sv);
      return ErrNo::InvalidArgument;
    }
  }
  if (Doc.at_key("stream-stdout").error() == simdjson::SUCCESS) {
    auto Err = Doc["stream-stdout"].get<bool>().get(GraphRef.StreamStdout);
    if (Err) {
      spdlog::error(
          "[WASI-NN] GGML backend: Unable to retrieve the stream-stdout option."sv);
      return ErrNo::InvalidArgument;
    }
  }
  if (Doc.at_key("embedding").error() == simdjson::SUCCESS) {
    auto Err = Doc["embedding"].get<bool>().get(GraphRef.Embedding);
    if (Err) {
      spdlog::error(
          "[WASI-NN] GGML backend: Unable to retrieve the embedding option."sv);
      return ErrNo::InvalidArgument;
    }
  }
#ifdef WASMEDGE_PLUGIN_WASI_NN_GGML_STRATEGY
  if (Doc.at_key("speculative-strategy").error() == simdjson::SUCCESS) {
    std::string_view strategy;
    auto Err = Doc["speculative-strategy"].get<std::string_view>().get(strategy);
    if (Err) {
      spdlog::error(
          "[WASI-NN] GGML backend: Unable to retrieve the speculative-strategy option."sv);
      return ErrNo::InvalidArgument;
    }
    GraphRef.SpeculativeStrategy = stringViewToSpeculativeStrategy(strategy);
  }
  if (Doc.at_key("draft-model-path").error() == simdjson::SUCCESS) {
    std::string_view DraftModelPath;
    auto Err = Doc["draft-model-path"].get<std::string_view>().get(DraftModelPath);
    if (Err) {
      spdlog::error(
          "[WASI-NN] GGML backend: Unable to retrieve the draft-model-path option."sv);
      return ErrNo::InvalidArgument;
    }
    if (GraphRef.SpeculativeStrategy != speculative_strategy::SPECULATIVE){
      spdlog::warn(
          "[WASI-NN] GGML backend: draft-model-path will not be set when not using speculative decoding."sv);
    }
    else
      GraphRef.DraftModelPath = DraftModelPath;
    
  }
  // if (Doc.at_key("use-kv").error() == simdjson::SUCCESS) {
  //   auto Err = Doc["use-kv"].get<bool>().get(GraphRef.UseKV);
  //   if (Err) {
  //     spdlog::error(
  //         "[WASI-NN] GGML backend: Unable to retrieve the use-kv option."sv);
  //     return ErrNo::InvalidArgument;
  //   }
  //   llama_log_set(nullptr, &GraphRef.UseKV);
  // }
  if (Doc.at_key("use-mmap").error() == simdjson::SUCCESS) {
    auto Err = Doc["use-mmap"].get<bool>().get(GraphRef.UseMMap);
    if (Err) {
      spdlog::error(
          "[WASI-NN] GGML backend: Unable to retrieve the use-mmap option."sv);
      return ErrNo::InvalidArgument;
    }
    llama_log_set(nullptr, &GraphRef.UseMMap);
  }
#endif // ifdef WASMEDGE_PLUGIN_WASI_NN_GGML_STRATEGY
  if (Doc.at_key("n-predict").error() == simdjson::SUCCESS) {
    auto Err = Doc["n-predict"].get<uint64_t>().get(GraphRef.NPredict);
    if (Err) {
      spdlog::error(
          "[WASI-NN] GGML backend: Unable to retrieve the n-predict option."sv);
      return ErrNo::InvalidArgument;
    }
  }
  if (Doc.at_key("reverse-prompt").error() == simdjson::SUCCESS) {
    std::string_view ReversePrompt;
    auto Err = Doc["reverse-prompt"].get<std::string_view>().get(ReversePrompt);
    if (Err) {
      spdlog::error(
          "[WASI-NN] GGML backend: Unable to retrieve the reverse-prompt option."sv);
      return ErrNo::InvalidArgument;
    }
    GraphRef.ReversePrompt = ReversePrompt;
  }
  if (Doc.at_key("mmproj").error() == simdjson::SUCCESS) {
    std::string_view MMProjModelPath;
    auto Err = Doc["mmproj"].get<std::string_view>().get(MMProjModelPath);
    if (Err) {
      spdlog::error(
          "[WASI-NN] GGML backend: Unable to retrieve the mmproj option."sv);
      return ErrNo::InvalidArgument;
    }
    GraphRef.MMProjModelPath = MMProjModelPath;
  }
  if (Doc.at_key("image").error() == simdjson::SUCCESS) {
    std::string_view ImagePath;
    auto Err = Doc["image"].get<std::string_view>().get(ImagePath);
    if (Err) {
      spdlog::error(
          "[WASI-NN] GGML backend: Unable to retrieve the image option."sv);
      return ErrNo::InvalidArgument;
    }
    GraphRef.ImagePath = ImagePath;
  }

  // The model parameters.
  if (Doc.at_key("n-gpu-layers").error() == simdjson::SUCCESS) {
    auto Err = Doc["n-gpu-layers"].get<int64_t>().get(GraphRef.NGPULayers);
    if (Err) {
      spdlog::error(
          "[WASI-NN] GGML backend: Unable to retrieve the n-gpu-layers option."sv);
      return ErrNo::InvalidArgument;
    }
  }
  if (Doc.at_key("main-gpu").error() == simdjson::SUCCESS) {
    auto Err = Doc["main-gpu"].get<int64_t>().get(GraphRef.MainGPU);
    if (Err) {
      spdlog::error(
          "[WASI-NN] GGML backend: Unable to retrieve the main-gpu option."sv);
      return ErrNo::InvalidArgument;
    }
  }
  if (Doc.at_key("tensor-split").error() == simdjson::SUCCESS) {
    // The TensorSplit is a comma-separated list of non-negative values.
    // E.g., "3,2" presents 60% of the data to GPU 0 and 40% to GPU 1.
    std::string_view TSV;
    auto Err = Doc["tensor-split"].get<std::string_view>().get(TSV);
    if (Err) {
      spdlog::error(
          "[WASI-NN] GGML backend: Unable to retrieve the tensor-split option."sv);
      return ErrNo::InvalidArgument;
    }
    std::string TS(TSV);
    std::replace(TS.begin(), TS.end(), ',', ' ');
    std::stringstream SS(TS);
    GraphRef.TensorSplit.clear();
    while (SS.good()) {
      float TmpTensor;
      SS >> TmpTensor;
      GraphRef.TensorSplit.push_back(TmpTensor);
    }
    uint32_t NDevices = llama_max_devices();
    if (GraphRef.TensorSplit.size() > NDevices) {
      spdlog::error(
          "[WASI-NN] GGML backend: Number of Tensor-Split is larger "
          "than MaxDevices, please reduce the size of tensor-split."sv);
      return ErrNo::InvalidArgument;
    }
    for (uint32_t Idx = GraphRef.TensorSplit.size(); Idx < NDevices; Idx++) {
      GraphRef.TensorSplit.push_back(0.0f);
    }
  }

  // The context parameters.
  if (Doc.at_key("ctx-size").error() == simdjson::SUCCESS) {
    auto Err = Doc["ctx-size"].get<uint64_t>().get(GraphRef.CtxSize);
    if (Err) {
      spdlog::error(
          "[WASI-NN] GGML backend: Unable to retrieve the ctx-size option."sv);
      return ErrNo::InvalidArgument;
    }
  }
  if (Doc.at_key("batch-size").error() == simdjson::SUCCESS) {
    auto Err = Doc["batch-size"].get<uint64_t>().get(GraphRef.BatchSize);
    if (Err) {
      spdlog::error(
          "[WASI-NN] GGML backend: Unable to retrieve the batch-size option."sv);
      return ErrNo::InvalidArgument;
    }
  }
  if (Doc.at_key("threads").error() == simdjson::SUCCESS) {
    auto Err = Doc["threads"].get<uint64_t>().get(GraphRef.Threads);
    if (Err) {
      spdlog::error(
          "[WASI-NN] GGML backend: Unable to retrieve the threads option."sv);
      return ErrNo::InvalidArgument;
    }
  }

  // The sampling parameters.
  if (Doc.at_key("temp").error() == simdjson::SUCCESS) {
    auto Err = Doc["temp"].get<double>().get(GraphRef.Temp);
    if (Err) {
      spdlog::error(
          "[WASI-NN] GGML backend: Unable to retrieve the temp option."sv);
      return ErrNo::InvalidArgument;
    }
    GraphRef.Temp = std::max(0.0, GraphRef.Temp);
  }
  if (Doc.at_key("top-p").error() == simdjson::SUCCESS) {
    auto Err = Doc["top-p"].get<double>().get(GraphRef.TopP);
    if (Err) {
      spdlog::error(
          "[WASI-NN] GGML backend: Unable to retrieve the top-p option."sv);
      return ErrNo::InvalidArgument;
    }
  }
  if (Doc.at_key("repeat-penalty").error() == simdjson::SUCCESS) {
    auto Err = Doc["repeat-penalty"].get<double>().get(GraphRef.RepeatPenalty);
    if (Err) {
      spdlog::error(
          "[WASI-NN] GGML backend: Unable to retrieve the repeat-penalty option."sv);
      return ErrNo::InvalidArgument;
    }
  }
  if (Doc.at_key("presence-penalty").error() == simdjson::SUCCESS) {
    auto Err =
        Doc["presence-penalty"].get<double>().get(GraphRef.PresencePenalty);
    if (Err) {
      spdlog::error(
          "[WASI-NN] GGML backend: Unable to retrieve the presence-penalty option."sv);
      return ErrNo::InvalidArgument;
    }
  }
  if (Doc.at_key("frequency-penalty").error() == simdjson::SUCCESS) {
    auto Err =
        Doc["frequency-penalty"].get<double>().get(GraphRef.FrequencyPenalty);
    if (Err) {
      spdlog::error(
          "[WASI-NN] GGML backend: Unable to retrieve the frequency-penalty option."sv);
      return ErrNo::InvalidArgument;
    }
  }
#ifdef WASMEDGE_PLUGIN_WASI_NN_GGML_STRATEGY
  if (Doc.at_key("n-parallel").error() == simdjson::SUCCESS) {
    uint64_t NParallel = 0;
    auto Err = Doc["n-parallel"].get<uint64_t>().get(NParallel);
    if (Err) {
      spdlog::error(
          "[WASI-NN] GGML backend: Unable to retrieve the n-parallel option."sv);
      return ErrNo::InvalidArgument;
    }
    if (GraphRef.SpeculativeStrategy != speculative_strategy::SPECULATIVE){
      spdlog::warn(
          "[WASI-NN] GGML backend: n-parallel will not be set when not using speculative decoding."sv);
    }
    else 
      GraphRef.NParallel = NParallel;
  }
  if (Doc.at_key("n-draft").error() == simdjson::SUCCESS) {
    uint64_t NDraft = 0;
    auto Err = Doc["n-draft"].get<uint64_t>().get(NDraft);
    if (Err) {
      spdlog::error(
          "[WASI-NN] GGML backend: Unable to retrieve the n-draft option."sv);
      return ErrNo::InvalidArgument;
    }
    if (GraphRef.SpeculativeStrategy != speculative_strategy::SPECULATIVE){
      spdlog::warn(
          "[WASI-NN] GGML backend: n-draft will not be set when not using speculative decoding."sv);
    }
    else 
      GraphRef.NDraft = NDraft;
  }
  if (Doc.at_key("prob-accept").error() == simdjson::SUCCESS) {
    double ProbAccept = 0.0;
    auto Err = Doc["prob-accept"].get<double>().get(ProbAccept);
    if (Err) {
      spdlog::error(
          "[WASI-NN] GGML backend: Unable to retrieve the prob-accept option."sv);
      return ErrNo::InvalidArgument;
    }
    if (GraphRef.SpeculativeStrategy != speculative_strategy::SPECULATIVE){
      spdlog::warn(
          "[WASI-NN] GGML backend: prob-accept cannot be set when not using speculative decoding."sv);
    }
    else
    GraphRef.ProbAccept = ProbAccept;
  }
  if (Doc.at_key("prob-split").error() == simdjson::SUCCESS) {
    double ProbSplit = 0.0;
    auto Err = Doc["prob-split"].get<double>().get(ProbSplit);
    if (Err) {
      spdlog::error(
          "[WASI-NN] GGML backend: Unable to retrieve the prob-split option."sv);
      return ErrNo::InvalidArgument;
    }
    if (GraphRef.SpeculativeStrategy != speculative_strategy::SPECULATIVE){
      spdlog::warn(
          "[WASI-NN] GGML backend: prob-split cannot be set when not using speculative decoding."sv);
    }
    else
    GraphRef.ProbSplit = ProbSplit;
  }
  if (Doc.at_key("lookahead-width").error() == simdjson::SUCCESS) {
    uint64_t LookaheadWidth = 0;
    auto Err = Doc["lookahead-width"].get<uint64_t>().get(LookaheadWidth);
    if (Err) {
      spdlog::error(
          "[WASI-NN] GGML backend: Unable to retrieve the lookahead-width option."sv);
      return ErrNo::InvalidArgument;
    }
    if (GraphRef.SpeculativeStrategy != speculative_strategy::LOOKAHEAD){
      spdlog::warn(
          "[WASI-NN] GGML backend: lookahead-width will not be set when not using lookahead decoding."sv);
    }
    else 
      GraphRef.LookaheadWidth = LookaheadWidth;
  }
  if (Doc.at_key("ngram-size").error() == simdjson::SUCCESS) {
    uint64_t NgramSize = 0;
    auto Err = Doc["ngram-size"].get<uint64_t>().get(NgramSize);
    if (Err) {
      spdlog::error(
          "[WASI-NN] GGML backend: Unable to retrieve the ngram-size option."sv);
      return ErrNo::InvalidArgument;
    }
    if (GraphRef.SpeculativeStrategy != speculative_strategy::LOOKAHEAD){
      spdlog::warn(
          "[WASI-NN] GGML backend: ngram-size will not be set when not using lookahead decoding."sv);
    }
    else 
      GraphRef.NgramSize = NgramSize;
  }
  if (Doc.at_key("max-verify-ngram-size").error() == simdjson::SUCCESS) {
    uint64_t MaxVerifyNgramSize = 0;
    auto Err = Doc["max-verify-ngram-size"].get<uint64_t>().get(MaxVerifyNgramSize);
    if (Err) {
      spdlog::error(
          "[WASI-NN] GGML backend: Unable to retrieve the max-verify-ngram-size option."sv);
      return ErrNo::InvalidArgument;
    }
    if (GraphRef.SpeculativeStrategy != speculative_strategy::LOOKAHEAD){
      spdlog::warn(
          "[WASI-NN] GGML backend: n-parallel will not be set when not using lookahead decoding."sv);
    }
    else
      GraphRef.MaxVerifyNgramSize = MaxVerifyNgramSize;
  }
#endif // ifdef WASMEDGE_PLUGIN_WASI_NN_GGML_STRATEGY
  if (Doc.at_key("stats-log-path").error() == simdjson::SUCCESS) {
      spdlog::info("got --stats-log-path"sv);
    std::string_view StatsLogPath;
    auto Err = Doc["stats-log-path"].get<std::string_view>().get(StatsLogPath);
    if (Err) {
      spdlog::error(
          "[WASI-NN] GGML backend: Unable to retrieve the mmproj option."sv);
      return ErrNo::InvalidArgument;
    }
    GraphRef.StatsLogPath = StatsLogPath;
    auto log_path = details::getMetricsFilename(
        /*parentDir=*/(GraphRef.StatsLogPath.empty()?"result":(std::filesystem::path(GraphRef.StatsLogPath).parent_path())),
        /*filename=*/(GraphRef.StatsLogPath.empty()?"":(std::filesystem::path(GraphRef.StatsLogPath).filename())));
    

    if(spdlog::get("metrics"))spdlog::drop("metrics");
    spdlog::info("getMetricsFilename returns {}"sv,log_path);
    auto basic_sink = std::make_shared<spdlog::sinks::basic_file_sink_mt>(log_path); // TODO: change to GraphRef.StatsDirectory
    basic_sink->set_pattern(
                "{\"timestamp\":\"%Y-%m-%dT%H:%M:%S.%e%z\",\"logger\":\"%n\",\"log_"
                "level\":\"%l\",\"process_id\":%P,\"thread_id\":%t %v}");

    std::vector<spdlog::sink_ptr> sinks{basic_sink};
    spdlog::register_logger(std::make_shared<spdlog::logger>("metrics", sinks.begin(), sinks.end())); 
    spdlog::get("metrics")->set_level( spdlog::level::level_enum::trace);
    
    spdlog::info("std::make_shared<spdlog::logger> metrics created"sv);

  }

  // Check if the model is updated.
  if (IsModelUpdated && ModelParams.n_gpu_layers != GraphRef.NGPULayers) {
    *IsModelUpdated = true;
  }

  return ErrNo::Success;
}

Expect<ErrNo> setupGPTParam(Graph &GraphRef, gpt_params &GPTParams) {
  GPTParams.sparams.temp = GraphRef.Temp;
  GPTParams.sparams.top_p = GraphRef.TopP;
  GPTParams.sparams.penalty_repeat = GraphRef.RepeatPenalty;
  GPTParams.sparams.penalty_present = GraphRef.PresencePenalty;

  return ErrNo::Success;
}

Expect<ErrNo> setupContextParam(Graph &GraphRef,
                                llama_context_params &ContextParams) {
  ContextParams.n_ctx = GraphRef.CtxSize;
  ContextParams.n_batch = GraphRef.BatchSize;
  ContextParams.n_threads = GraphRef.Threads;
  ContextParams.n_threads_batch = GraphRef.Threads;
  return ErrNo::Success;
}

Expect<ErrNo> buildOutputMetadata(Context &CxtRef,
                                  std::string &Metadata) noexcept {
  std::ostringstream OS;
  OS << R"({"input_tokens": )" << CxtRef.LlamaNInputs
     << R"(, "output_tokens": )" << CxtRef.LlamaOutputTokens.size()
     << R"(, "llama_build_number": )" << LLAMA_BUILD_NUMBER
     << R"(, "llama_commit": ")" << LLAMA_COMMIT << R"("})";
  Metadata = OS.str();

  return ErrNo::Success;
}

void buildOutputEmbedding(std::string &Embedding, int32_t NEmbd,
                          const float *Embeddings) noexcept {
  // Embedding vector format
  // | Content                             |
  // | ----------------------------------- |
  // | '{"number_embedding": '             |
  // | n_embedding                         |
  // | ', "embedding": '                   |
  // | '['                                 |
  // | n_embedding*(embedding value %.10f) |
  // | (n_embedding-1)*(',')               |
  // | ']'                                 |
  // | '}'                                 |
  std::ostringstream OS;
  OS.precision(10);
  OS << R"({"n_embedding": )" << NEmbd << R"(, "embedding": [)";
  for (int32_t Idx = 0; Idx < NEmbd - 1; Idx++) {
    OS << Embeddings[Idx] << ",";
  }
  OS << Embeddings[NEmbd - 1] << "]}";
  Embedding = OS.str();
}

Expect<ErrNo> getEmbedding(WasiNNEnvironment &Env,
                           uint32_t ContextId) noexcept {
  auto &CxtRef = Env.NNContext[ContextId].get<Context>();
  auto &GraphRef = Env.NNGraph[CxtRef.GraphId].get<Graph>();
  if (GraphRef.EnableDebugLog) {
    spdlog::info("[WASI-NN][Debug] GGML backend: getEmbedding"sv);
  }

  if (CxtRef.LlamaInputs.size() == 0) {
    spdlog::error("[WASI-NN] GGML backend: Llama input is not set!"sv);
    return ErrNo::InvalidArgument;
  }

  // Clear the outputs.
  if (GraphRef.EnableDebugLog) {
    spdlog::info(
        "[WASI-NN][Debug] GGML backend: clear the previous output and tokens"sv);
  }
  CxtRef.LlamaOutputs.clear();
  CxtRef.LlamaOutputTokens.clear();
  if (GraphRef.EnableDebugLog) {
    spdlog::info(
        "[WASI-NN][Debug] GGML backend: clear the previous output and tokens...Done"sv);
  }

  // Main predict loop.
  if (GraphRef.EnableDebugLog) {
    spdlog::info("[WASI-NN][Debug] GGML backend: handle embedding"sv);
  }
  // Initialize the llama context.
  llama_context_params ContextParams = llama_context_default_params();
  ContextParams.n_ctx = GraphRef.CtxSize;
  ContextParams.n_batch = GraphRef.BatchSize;
  ContextParams.embedding = GraphRef.Embedding;
  auto *LlamaContext =
      llama_new_context_with_model(GraphRef.LlamaModel, ContextParams);

  // Get the context size.
  const uint64_t NCtx = llama_n_ctx(LlamaContext);
  // Minus 4 for the special tokens. (Such as <BOS>, <EOS>, ... tokens.)
  const uint64_t MaxTokensListSize = NCtx - 4;
  // Use the const sequence id here.
  const llama_seq_id SequenceId = 0;

  // Check if the input is too long.
  if (static_cast<uint64_t>(CxtRef.LlamaInputs.size()) > MaxTokensListSize) {
    if (GraphRef.EnableLog) {
      spdlog::info("[WASI-NN] GGML backend: the prompt is too long. Your input "
                   "has {} tokens. Please reduce it to {} tokens."sv,
                   CxtRef.LlamaInputs.size(), MaxTokensListSize);
    }
    return ErrNo::PromptTooLong;
  }

  int NPast = 0;
  while (!CxtRef.LlamaInputs.empty()) {
    const uint64_t NTokens = (ContextParams.n_batch > CxtRef.LlamaInputs.size())
                                 ? CxtRef.LlamaInputs.size()
                                 : ContextParams.n_batch;
    auto Status = llama_decode(LlamaContext,
                               llama_batch_get_one(CxtRef.LlamaInputs.data(),
                                                   NTokens, NPast, SequenceId));
    if (Status == 1) {
      spdlog::error(
          "[WASI-NN] GGML backend: failed to llama_decode: try "
          "reducing the size of the batch or increasing the size of context"sv);
      return ErrNo::RuntimeError;
    }
    if (Status < 0) {
      spdlog::error("[WASI-NN] GGML backend: failed to llama_decode: internal "
                    "fatal error. Please open an issue on GitHub"sv);
      return ErrNo::RuntimeError;
    }

    NPast += NTokens;
    CxtRef.LlamaInputs.erase(CxtRef.LlamaInputs.begin(),
                             CxtRef.LlamaInputs.begin() + NTokens);
  }
  const int32_t NEmbd = llama_n_embd(GraphRef.LlamaModel);
  const auto *Embeddings = llama_get_embeddings(LlamaContext);

  details::buildOutputEmbedding(CxtRef.LlamaOutputs, NEmbd, Embeddings);

  if (GraphRef.EnableDebugLog) {
    spdlog::info(
        "[WASI-NN][Debug] GGML backend: enter embedding loop...Done"sv);
  }

  if (GraphRef.EnableLog) {
    llama_print_timings(LlamaContext);
  }

  // We free the contexts here to keep the ggml plugin stateless.
  // Users could fully control the contexts by themselves via their prompt.
  llama_free(LlamaContext);

  if (GraphRef.EnableDebugLog) {
    spdlog::info("[WASI-NN][Debug] GGML backend: compute...Done"sv);
  }

  return ErrNo::Success;
}

ErrNo evaluateTokens(Graph &GraphRef, struct llama_context *LlamaContext,
                     std::vector<llama_token> Tokens, int &NPast) noexcept {
  uint32_t NCtx = llama_n_ctx(LlamaContext);

  // End the inference if the context is full.
  if (NPast + static_cast<uint32_t>(Tokens.size()) > NCtx) {
    if (GraphRef.EnableLog) {
      spdlog::info(
          "[WASI-NN] GGML backend: the context if full ({} / {} tokens). Please increase your context size."sv,
          NPast + static_cast<uint32_t>(Tokens.size()), NCtx);
    }
    return ErrNo::ContextFull;
  }
  for (int I = 0; I < static_cast<int>(Tokens.size());
       I += GraphRef.BatchSize) {
    int NEval = static_cast<int>(Tokens.size()) - I;
    if (NEval > static_cast<int>(GraphRef.BatchSize)) {
      NEval = GraphRef.BatchSize;
    }
    // llama_batch_get_one(*token, n_tokens, position, sequence_id)
    // This will return batch for single sequence of tokens starting at
    // position.
    const llama_seq_id SequenceId = 0;
    // auto Status =
    //     llama_decode(LlamaContext,
    //                  llama_batch_get_one(&Tokens[I], NEval, NPast, SequenceId));
    auto Batch = llama_batch_get_one(&Tokens[I], NEval, NPast, SequenceId);
    auto Status = llama_decode(LlamaContext, Batch);
    if (Status == 1) {
      spdlog::error(
          "[WASI-NN] GGML backend: failed to llama_decode: try reducing the size of the batch or increasing the size of context"sv);
      return ErrNo::RuntimeError;
    } else if (Status < 0) {
      spdlog::error(
          "[WASI-NN] GGML backend: failed to llama_decode: internal fatal error. Please open an issue on GitHub"sv);
      return ErrNo::RuntimeError;
    }
    NPast += NEval;
  }

  return ErrNo::Success;
}

const std::string_view Base64ImageTagPrefix = "<img src=\"data:image/"sv;
const std::string_view Base64ImageBytesPrefix = ";base64,"sv;
const std::string_view Base64ImageTagSuffix = "\">"sv;
const std::string_view PromptImagePlaceholder = "<image>"sv;

bool containsBase64Image(Graph &GraphRef, std::string Prompt) noexcept {
  // Check if the prompt contains a base64 image.
  // Follow this link for the supported image formats:
  // https://github.com/ggerganov/llama.cpp/blob/master/common/stb_image.h

  auto Base64ImageTagBeginPos = Prompt.find(Base64ImageTagPrefix);
  if (Base64ImageTagBeginPos == std::string::npos) {
    if (GraphRef.EnableDebugLog) {
      spdlog::info(
          "[WASI-NN][Debug] GGML backend: No base64 image tag found in the prompt."sv);
    }
    return false;
  }
  auto Base64ImageTagEndPos =
      Prompt.find(Base64ImageTagSuffix, Base64ImageTagBeginPos);
  if (Base64ImageTagEndPos == std::string::npos) {
    if (GraphRef.EnableDebugLog) {
      spdlog::info(
          "[WASI-NN][Debug] GGML backend: Found an unclosed base64 image tag."sv);
    }
    return false;
  }
  return true;
}

struct llava_image_embed *
loadBase64ImageFromPrompt(Graph &GraphRef, clip_ctx *ClipContext,
                          std::string Prompt) noexcept {
  // Load the base64 image from the prompt.
  // Follow this link for the supported image formats:
  // https://github.com/ggerganov/llama.cpp/blob/master/common/stb_image.h

  // Find `<img src="data:image/`
  auto Base64ImageTagBeginPos = Prompt.find(Base64ImageTagPrefix);
  if (Base64ImageTagBeginPos == std::string::npos) {
    return nullptr;
  }

  // Find `;base64,` (skip the image type part)
  auto Base64ImageBytesBeginPos =
      Prompt.find(Base64ImageBytesPrefix, Base64ImageTagBeginPos);
  if (Base64ImageTagBeginPos == std::string::npos) {
    return nullptr;
  }

  // Find `">`
  auto Base64ImageTagEndPos =
      Prompt.find(Base64ImageTagSuffix, Base64ImageBytesBeginPos);
  if (Base64ImageTagEndPos == std::string::npos) {
    return nullptr;
  }

  auto Base64Str =
      Prompt.substr(Base64ImageBytesBeginPos + Base64ImageBytesPrefix.size(),
                    Base64ImageTagEndPos - Base64ImageBytesBeginPos -
                        Base64ImageBytesPrefix.size());

  // Decode the base64 image.
  auto RequiredBytes = base64::required_encode_size(Base64Str.size());
  auto ImageBytes = std::vector<unsigned char>(RequiredBytes);
  try {
    base64::decode(Base64Str.begin(), Base64Str.end(), ImageBytes.begin());
  } catch (const base64_error &E) {
    spdlog::error("[WASI-NN] GGML backend: Error when base64::decode: {}"sv,
                  E.what());
    return nullptr;
  }

  return llava_image_embed_make_with_bytes(
      ClipContext, GraphRef.Threads, ImageBytes.data(), ImageBytes.size());
}

ErrNo replaceBase64ImagePlaceholderInPrompt(std::string &Prompt) noexcept {
  // Replace the base64 image in the prompt with a placeholder.

  // Find `<img src="data:image/`
  auto Base64ImageTagBeginPos = Prompt.find(Base64ImageTagPrefix);
  if (Base64ImageTagBeginPos == std::string::npos) {
    return ErrNo::InvalidArgument;
  }

  // Find `">`
  auto Base64ImageTagEndPos =
      Prompt.find(Base64ImageTagSuffix, Base64ImageTagBeginPos);
  if (Base64ImageTagEndPos == std::string::npos) {
    return ErrNo::InvalidArgument;
  }

  auto Base64ImageTagLength = Base64ImageTagEndPos - Base64ImageTagBeginPos +
                              Base64ImageTagSuffix.size();
  Prompt.replace(Base64ImageTagBeginPos, Base64ImageTagLength,
                 PromptImagePlaceholder);

  return ErrNo::Success;
}

} // namespace details

Expect<ErrNo> load(WasiNNEnvironment &Env, Span<const Span<uint8_t>> Builders,
                   [[maybe_unused]] Device Device, uint32_t &GraphId) noexcept {
  // Add a new graph.
  Env.NNGraph.emplace_back(Backend::GGML);
  auto &GraphRef = Env.NNGraph.back().get<Graph>();

  // Initialize the plugin parameters.
  auto ContextDefault = llama_context_default_params();
  GraphRef.EnableLog = false;
  GraphRef.EnableDebugLog = false;
  GraphRef.StreamStdout = false;
  GraphRef.NPredict = ContextDefault.n_ctx;
  GraphRef.ReversePrompt = ""sv;
  GraphRef.MMProjModelPath = ""sv;
  GraphRef.ImagePath = ""sv;
  // Initialize the model parameters.
  GraphRef.NGPULayers = 0;
  // Initialize the context parameters.
  GraphRef.CtxSize = ContextDefault.n_ctx;
  GraphRef.BatchSize = ContextDefault.n_batch;
  GraphRef.Threads = ContextDefault.n_threads;
  // Initialize the sampling parameters.
  const llama_sampling_params SamplingDefault;
  GraphRef.Temp = SamplingDefault.temp;
  GraphRef.TopP = SamplingDefault.top_p;
  GraphRef.RepeatPenalty = SamplingDefault.penalty_repeat;
  GraphRef.PresencePenalty = SamplingDefault.penalty_present;
  GraphRef.FrequencyPenalty = SamplingDefault.penalty_freq;

  // If the graph builder length > 1, the data of builder[1] is the metadata.
  if (Builders.size() > 1) {
    const std::string Metadata(reinterpret_cast<char *>(Builders[1].data()),
                               Builders[1].size());
    // Ignore context or model updates when initializing the graph.
    auto Res = details::parseMetadata(GraphRef, Metadata);
    if (Res != ErrNo::Success) {
      spdlog::error("[WASI-NN] GGML backend: Failed to parse metadata."sv);
      Env.NNGraph.pop_back();
      return Res;
    }
  }
  if (GraphRef.EnableLog) {
    spdlog::info("[WASI-NN] GGML backend: LLAMA_COMMIT {}"sv, LLAMA_COMMIT);
    spdlog::info("[WASI-NN] GGML backend: LLAMA_BUILD_NUMBER {}"sv,
                 LLAMA_BUILD_NUMBER);
  }

  if (GraphRef.EnableDebugLog) {
    spdlog::info("[WASI-NN][Debug] GGML backend: Handling model path."sv);
  }
  // Handle the model path.
  auto Weight = Builders[0];
  const std::string BinModel(reinterpret_cast<char *>(Weight.data()),
                             Weight.size());
  std::string ModelFilePath;
  if (BinModel.substr(0, 8) == "preload:") {
    ModelFilePath = BinModel.substr(8);
  } else {
    if (GraphRef.EnableDebugLog) {
      spdlog::info(
          "[WASI-NN][Debug] GGML backend: Model path not found in nn-preload, "
          "write model into a tmpfile."sv);
    }
    // TODO: pass the model directly to ggml
    // Write ggml model to file.
    ModelFilePath = "ggml-model.bin"sv;
    std::ofstream TempFile(ModelFilePath);
    if (!TempFile) {
      spdlog::error(
          "[WASI-NN] GGML backend: Failed to create the temporary file. "
          "Currently, our workaround involves creating a temporary model "
          "file named \"ggml-model.bin\" and passing this filename as a "
          "parameter to the ggml llama library."sv);
      Env.NNGraph.pop_back();
      return ErrNo::InvalidArgument;
    }
    TempFile << BinModel;
    TempFile.close();
    if (GraphRef.EnableDebugLog) {
      spdlog::info(
          "[WASI-NN][Debug] GGML backend: Write model into a tmpfile...Done"sv);
    }
  }
  if (GraphRef.EnableDebugLog) {
    spdlog::info(
        "[WASI-NN][Debug] GGML backend: Finished handling model path."sv);
  }

  if (GraphRef.EnableDebugLog) {
    spdlog::info(
        "[WASI-NN][Debug] GGML backend: Initialize ggml model with given parameters"sv);
  }
  // Initialize ggml model with model parameters.
  GraphRef.ModelFilePath = ModelFilePath;
  llama_model_params ModelParams = llama_model_default_params();
  ModelParams.n_gpu_layers = GraphRef.NGPULayers;
  ModelParams.main_gpu = GraphRef.MainGPU;
  ModelParams.tensor_split = GraphRef.TensorSplit.data();
  GraphRef.LlamaModel =
      llama_load_model_from_file(GraphRef.ModelFilePath.c_str(), ModelParams);
  #ifdef WASMEDGE_PLUGIN_WASI_NN_GGML_STRATEGY
  if(GraphRef.SpeculativeStrategy == speculative_strategy::SPECULATIVE){
    GraphRef.DraftLlamaModel =
      llama_load_model_from_file(GraphRef.DraftModelPath.c_str(), ModelParams);
  }

  #endif // WASMEDGE_PLUGIN_WASI_NN_GGML_STRATEGY
  if (GraphRef.LlamaModel == nullptr) {
    spdlog::error("[WASI-NN] GGML backend: Error: unable to init model."sv);
    Env.NNGraph.pop_back();
    return ErrNo::InvalidArgument;
  }
  if (GraphRef.EnableDebugLog) {
    spdlog::info(
        "[WASI-NN][Debug] GGML backend: Initialize ggml model with given parameters...Done"sv);
  }

  // Store the loaded graph.
  GraphId = Env.NNGraph.size() - 1;
  // Disable llama log by default.
  log_disable();


  return ErrNo::Success;
}

Expect<ErrNo> initExecCtx(WasiNNEnvironment &Env, uint32_t GraphId,
                          uint32_t &ContextId) noexcept {
  Env.NNContext.emplace_back(GraphId, Env.NNGraph[GraphId]);
  ContextId = Env.NNContext.size() - 1;
  
  auto &GraphRef = Env.NNGraph[GraphId].get<Graph>();
  if (GraphRef.EnableLog) {
    spdlog::info("[WASI-NN] GGML backend: llama_system_info: {}"sv,
                 llama_print_system_info());
  }
  return ErrNo::Success;
}

Expect<ErrNo> setInput(WasiNNEnvironment &Env, uint32_t ContextId,
                       uint32_t Index, const TensorData &Tensor) noexcept {
  auto &CxtRef = Env.NNContext[ContextId].get<Context>();
  auto &GraphRef = Env.NNGraph[CxtRef.GraphId].get<Graph>();
  if (GraphRef.EnableDebugLog) {
    spdlog::info("[WASI-NN][Debug] GGML backend: setInput"sv);
  }

  bool IsModelParamsUpdated = false;
  // Use index 1 for metadata.
  if (Index == 1) {
    if (GraphRef.EnableDebugLog) {
      spdlog::info(
          "[WASI-NN][Debug] GGML backend: found Metadata, processing"sv);
    }
    const std::string Metadata(reinterpret_cast<char *>(Tensor.Tensor.data()),
                               Tensor.Tensor.size());
    auto Res =
        details::parseMetadata(GraphRef, Metadata, &IsModelParamsUpdated);

    if (Res != ErrNo::Success) {
      spdlog::error("[WASI-NN] GGML backend: Failed to parse metadata."sv);
      return Res;
    }

#ifndef __APPLE__
    // XXX: Due to the limitation of WASI-NN proposal,
    // this is a workaround for non-macOS devices.
    // However, if the model params is updated in Config stage,
    // then, we doesn't encourage to use this to avoid the model
    // reloading.
    {
      if (IsModelParamsUpdated) {
        llama_model_params ModelParams = llama_model_default_params();
        ModelParams.n_gpu_layers = GraphRef.NGPULayers;
        llama_free_model(GraphRef.LlamaModel);
        GraphRef.LlamaModel = llama_load_model_from_file(
            GraphRef.ModelFilePath.c_str(), ModelParams);
        if (GraphRef.LlamaModel == nullptr) {
          spdlog::error(
              "[WASI-NN] GGML backend: Error: unable to init model."sv);
          Env.NNGraph.pop_back();
          return ErrNo::InvalidArgument;
        }
#ifdef WASMEDGE_PLUGIN_WASI_NN_GGML_STRATEGY
        if(GraphRef.SpeculativeStrategy == speculative_strategy::SPECULATIVE) {
          llama_free_model(GraphRef.DraftLlamaModel);
          GraphRef.DraftLlamaModel = llama_load_model_from_file(
            GraphRef.DraftModelPath.c_str(), ModelParams);
          if (GraphRef.DraftLlamaModel == nullptr) {
            spdlog::error(
                "[WASI-NN] GGML backend: Error: unable to init draft model."sv);
            Env.NNGraph.pop_back();
            return ErrNo::InvalidArgument;
          }
        }
#endif // WASMEDGE_PLUGIN_WASI_NN_GGML_STRATEGY
      }
    }
#endif

    if (GraphRef.EnableDebugLog) {
      spdlog::info(
          "[WASI-NN][Debug] GGML backend: found Metadata, processing...Done"sv);
    }
    return ErrNo::Success;
  }

  // Initialize the llama context.
  if (GraphRef.EnableDebugLog) {
    spdlog::info("[WASI-NN][Debug] GGML backend: init llama context"sv);
  }
  llama_context_params ContextParams = llama_context_default_params();
  details::setupContextParam(GraphRef, ContextParams);
  auto LlamaContext =
      llama_new_context_with_model(GraphRef.LlamaModel, ContextParams);
  if (GraphRef.EnableDebugLog) {
    spdlog::info("[WASI-NN][Debug] GGML backend: init llama context...Done"sv);
  }

  // Set the input.
  if (GraphRef.EnableDebugLog) {
    spdlog::info("[WASI-NN][Debug] GGML backend: set the input"sv);
  }
  const bool AddBos = llama_should_add_bos_token(GraphRef.LlamaModel);
  std::string Prompt(reinterpret_cast<char *>(Tensor.Tensor.data()),
                     Tensor.Tensor.size());
  if (GraphRef.MMProjModelPath == ""sv) {
    // Text only prompt.
    CxtRef.LlamaInputs = llama_tokenize(LlamaContext, Prompt, AddBos, true);
    CxtRef.LlamaNInputs = CxtRef.LlamaInputs.size();
    // TODO: 
    if (GraphRef.SpeculativeStrategy == speculative_strategy::SPECULATIVE) {
        const bool DftAddBos = llama_should_add_bos_token(GraphRef.DraftLlamaModel);// TODO: should use 
      spdlog::info("add_bos dft: {}\n"sv, DftAddBos);

      if (AddBos != DftAddBos) {
          spdlog::error("{}: error: draft model add_bos must match target model to use speculation but "sv, __func__);
          spdlog::error("DftAddBos = {} while TgtAddBos = {}\n"sv, DftAddBos, AddBos);
          return ErrNo::RuntimeError;
      }
    }
  } else {
    // Handle llava format prompt.

    // Check if the prompt contains a base64 image.
    bool ContainsBase64Image = details::containsBase64Image(GraphRef, Prompt);
    if (GraphRef.ImagePath == ""sv && ContainsBase64Image == false) {
      spdlog::error(
          "[WASI-NN] GGML backend: Error: when using llava model, "
          "you need to specify the image path or have the base64 encoded "
          "image in the prompt."sv);
      return ErrNo::InvalidArgument;
    }

    // Show some warnings.
    if (GraphRef.EnableLog) {
      if (GraphRef.CtxSize < 4096) {
        spdlog::info(
            "[WASI-NN] GGML backend: Context size is {}, "
            "we recommend context size >= 2048 when using llava-v1.5 "
            "and context size >= 4096 when using llava-v1.6 for better results."sv,
            GraphRef.CtxSize);
      }
    }

    // Load image for llava.
    int LlavaVerbosity = 0;
    if (GraphRef.EnableLog) {
      LlavaVerbosity = 1;
    }
    auto ClipContext =
        clip_model_load(GraphRef.MMProjModelPath.c_str(), LlavaVerbosity);
    if (ContainsBase64Image) {
      // Load the base64 image from the prompt.
      CxtRef.LlavaImageEmbd =
          details::loadBase64ImageFromPrompt(GraphRef, ClipContext, Prompt);
      // Replace the base64 image in the prompt with a placeholder.
      auto Res = details::replaceBase64ImagePlaceholderInPrompt(Prompt);
      if (Res != ErrNo::Success) {
        spdlog::error(
            "[WASI-NN] GGML backend: Error: unable to replace the base64 image in the prompt."sv);
        clip_free(ClipContext);
        return Res;
      }
    } else {
      // Load the image from the file.
      CxtRef.LlavaImageEmbd = llava_image_embed_make_with_filename(
          ClipContext, GraphRef.Threads, GraphRef.ImagePath.c_str());
    }
    clip_free(ClipContext);
    if (CxtRef.LlavaImageEmbd == nullptr) {
      spdlog::error(
          "[WASI-NN] GGML backend: Error: unable to load the image."sv);
      return ErrNo::InvalidArgument;
    }

    // We split prompt by <image> as placeholder and save the position.
    auto PlaceholderPosition = Prompt.find(details::PromptImagePlaceholder);
    if (PlaceholderPosition == std::string::npos) {
      spdlog::error(
          "[WASI-NN] GGML backend: Error: unable to find the placeholder in the llava prompt."sv);
      return ErrNo::InvalidArgument;
    }
    std::string PromptBeforeImage = Prompt.substr(0, PlaceholderPosition);
    std::string PromptAfterImage = Prompt.substr(
        PlaceholderPosition + details::PromptImagePlaceholder.length());
    std::vector<llama_token> EmbdInputBeforeImage =
        llama_tokenize(LlamaContext, PromptBeforeImage, AddBos, true);
    std::vector<llama_token> EmbdInputAfterImage =
        llama_tokenize(LlamaContext, PromptAfterImage, false, true);
    CxtRef.LlavaImagePosition = EmbdInputBeforeImage.size();
    CxtRef.LlamaInputs.reserve(EmbdInputBeforeImage.size() +
                               EmbdInputAfterImage.size());
    CxtRef.LlamaInputs.insert(CxtRef.LlamaInputs.end(),
                              EmbdInputBeforeImage.begin(),
                              EmbdInputBeforeImage.end());
    CxtRef.LlamaInputs.insert(CxtRef.LlamaInputs.end(),
                              EmbdInputAfterImage.begin(),
                              EmbdInputAfterImage.end());
  }
  if (GraphRef.EnableDebugLog) {
    spdlog::info("[WASI-NN][Debug] GGML backend: set the input...Done"sv);
  }

  if (GraphRef.SpeculativeStrategy == speculative_strategy::SPECULATIVE){
      const int n_vocab_tgt = llama_n_vocab(GraphRef.LlamaModel);
      const int n_vocab_dft = llama_n_vocab(GraphRef.DraftLlamaModel);
      const int vocab_diff  = n_vocab_tgt > n_vocab_dft
          ? n_vocab_tgt - n_vocab_dft
          : n_vocab_dft - n_vocab_tgt;

      if (vocab_diff > SPEC_VOCAB_MAX_SIZE_DIFFERENCE) {
          spdlog::error("{}: error: draft model vocab must closely match target model to use speculation but "sv, __func__);
          spdlog::error("target vocab size {} does not match draft vocab size {} - difference {}, max allowed {}\n"sv,
                  n_vocab_tgt, llama_n_vocab(GraphRef.DraftLlamaModel), vocab_diff, SPEC_VOCAB_MAX_SIZE_DIFFERENCE);
          return ErrNo::RuntimeError;
      }

      for (int i = SPEC_VOCAB_CHECK_START_TOKEN_ID; i < std::min(n_vocab_tgt, n_vocab_dft); ++i) {
          const char * token_text_tgt = llama_token_get_text(GraphRef.LlamaModel, i);
          const char * token_text_dft = llama_token_get_text(GraphRef.DraftLlamaModel, i);
          if (std::strcmp(token_text_tgt, token_text_dft) != 0) {
  auto DraftLlamaContext =
      llama_new_context_with_model(GraphRef.DraftLlamaModel, ContextParams);

              spdlog::error("{}: error: draft model vocab must match target model to use speculation but "sv, __func__);
              spdlog::error("token {} content differs - target '{}', draft '{}'\n"sv, i,
                      llama_token_to_piece(LlamaContext, i).c_str(),
                      llama_token_to_piece(DraftLlamaContext, i).c_str());
              return ErrNo::RuntimeError;
          }
      }
  }
  // Delete the llama context.
  if (GraphRef.EnableDebugLog) {
    spdlog::info(
        "[WASI-NN][Debug] GGML backend: delete llama context to make it stateless"sv);
  }
  llama_free(LlamaContext);
  if (GraphRef.EnableDebugLog) {
    spdlog::info(
        "[WASI-NN][Debug] GGML backend: delete llama context to make it stateless...Done"sv);
  }

  if (GraphRef.EnableDebugLog) {
    spdlog::info("[WASI-NN][Debug] GGML backend: setInput...Done"sv);
  }
    if(spdlog::get("metrics")){
      spdlog::get("metrics")->info("{}"sv,SimpleJSON({
        {"event","load"}, 
        {"time",ggml_time_us()},
        {"attribute_n_threads",GraphRef.Threads},
        {"attribute_n_parallel",GraphRef.NParallel},
        {"attribute_speculative_strategy",details::speculativeStrategyToString(GraphRef.SpeculativeStrategy)},
        {"attribute_batch_size",GraphRef.BatchSize},
        {"attribute_use_mmap",GraphRef.UseMMap},
        {"attribute_prob_accept",GraphRef.ProbAccept},
        {"attribute_prob_split",GraphRef.ProbSplit},
        {"attribute_lookahead_width",GraphRef.LookaheadWidth},
        {"attribute_ngram_size",GraphRef.NgramSize},
        {"attribute_max_verify_ngram_size",GraphRef.MaxVerifyNgramSize},
          // less important metadata
        {"attribute_n_predict",GraphRef.NPredict},
        {"attribute_draft_model_path",GraphRef.DraftModelPath},
        {"attribute_ctx_size",GraphRef.CtxSize},
        {"attribute_temp",GraphRef.Temp},
        {"attribute_top_p",GraphRef.TopP},
        {"attribute_repeat_penalty",GraphRef.RepeatPenalty},
        {"attribute_presence_penalty",GraphRef.PresencePenalty},
        {"attribute_frequency_penalty",GraphRef.FrequencyPenalty},
        })
      );
    spdlog::get("metrics")->flush();
    }
  return ErrNo::Success;
}

Expect<ErrNo> getOutput(WasiNNEnvironment &Env, uint32_t ContextId,
                        uint32_t Index, Span<uint8_t> OutBuffer,
                        uint32_t &BytesWritten) noexcept {
  auto &CxtRef = Env.NNContext[ContextId].get<Context>();
  // Index 1 is for the metadata of the outputs.
  if (Index == 1) {
    std::string Metadata;
    auto Res = details::buildOutputMetadata(CxtRef, Metadata);
    if (Res != ErrNo::Success) {
      spdlog::error(
          "[WASI-NN] GGML backend: Failed to build output metadata."sv);
      return Res;
    }
    std::copy_n(Metadata.data(), Metadata.length(), OutBuffer.data());
    BytesWritten = Metadata.length();
    return ErrNo::Success;
  }

  std::copy_n(CxtRef.LlamaOutputs.data(), CxtRef.LlamaOutputs.length(),
              OutBuffer.data());
  BytesWritten = CxtRef.LlamaOutputs.length();
  return ErrNo::Success;
}

Expect<ErrNo> compute(WasiNNEnvironment &Env, uint32_t ContextId) noexcept {
#ifndef WASMEDGE_PLUGIN_WASI_NN_GGML_STRATEGY
  auto &CxtRef = Env.NNContext[ContextId].get<Context>();
  auto &GraphRef = Env.NNGraph[CxtRef.GraphId].get<Graph>();
  if (GraphRef.EnableDebugLog) {
    spdlog::info("[WASI-NN][Debug] GGML backend: compute"sv);
  }

  if (GraphRef.Embedding) {
    return details::getEmbedding(Env, ContextId);
  }

  if (CxtRef.LlamaInputs.size() == 0) {
    spdlog::error("[WASI-NN] GGML backend: Llama input is not set!"sv);
    return ErrNo::InvalidArgument;
  }

  // Clear the outputs.
  if (GraphRef.EnableDebugLog) {
    spdlog::info(
        "[WASI-NN][Debug] GGML backend: clear the previous output and tokens"sv);
  }
  CxtRef.LlamaOutputs.clear();
  CxtRef.LlamaOutputTokens.clear();
  if (GraphRef.EnableDebugLog) {
    spdlog::info(
        "[WASI-NN][Debug] GGML backend: clear the previous output and tokens...Done"sv);
  }

  // Initialize the llama context.
  gpt_params GPTParams;
  llama_context_params ContextParams = llama_context_default_params();
  details::setupGPTParam(GraphRef, GPTParams);
  details::setupContextParam(GraphRef, ContextParams);
  auto LlamaContext =
      llama_new_context_with_model(GraphRef.LlamaModel, ContextParams);
  struct llama_sampling_context *CtxSampling =
      llama_sampling_init(GPTParams.sparams);
  // Prepare variables;
  int32_t NPast = 0;
  int32_t NRemain = GraphRef.NPredict;
  // Get the context size.
  const uint64_t NCtx = llama_n_ctx(LlamaContext);
  // Minus 4 for the special tokens. (Such as <BOS>, <EOS>, ... tokens.)
  const uint64_t MaxTokensListSize = NCtx - 4;
  // Return value.
  auto ReturnCode = ErrNo::Success;

  // Check if the input is too long.
  if (static_cast<uint64_t>(CxtRef.LlamaInputs.size()) > MaxTokensListSize) {
    if (GraphRef.EnableLog) {
      spdlog::info("[WASI-NN] GGML backend: the prompt is too long. Your input "
                   "has {} tokens. Please reduce it to {} tokens."sv,
                   CxtRef.LlamaInputs.size(), MaxTokensListSize);
    }
    return ErrNo::PromptTooLong;
  }

  // Evaluate input tokens.
  if (CxtRef.LlavaImageEmbd == nullptr) {
    // Text only prompt.
    ReturnCode = details::evaluateTokens(GraphRef, LlamaContext,
                                         CxtRef.LlamaInputs, NPast);
    if (ReturnCode != ErrNo::Success) {
      spdlog::error(
          "[WASI-NN] GGML backend: failed to evaluate input tokens."sv);
      return ReturnCode;
    }
  } else {
    // Llava format prompt with image data.
    std::vector<llama_token> EmbdInputBeforeImage(
        CxtRef.LlamaInputs.begin(),
        CxtRef.LlamaInputs.begin() + CxtRef.LlavaImagePosition);
    std::vector<llama_token> EmbdInputAfterImage(CxtRef.LlamaInputs.begin() +
                                                     CxtRef.LlavaImagePosition,
                                                 CxtRef.LlamaInputs.end());
    ReturnCode = details::evaluateTokens(GraphRef, LlamaContext,
                                         EmbdInputBeforeImage, NPast);
    if (ReturnCode != ErrNo::Success) {
      spdlog::error(
          "[WASI-NN] GGML backend: failed to evaluate input tokens before image."sv);
      return ReturnCode;
    }
    bool EvalImageStatus = llava_eval_image_embed(
        LlamaContext, CxtRef.LlavaImageEmbd, GraphRef.BatchSize, &NPast);
    if (!EvalImageStatus) {
      spdlog::error(
          "[WASI-NN] GGML backend: failed to evaluate embed image tokens."sv);
      return ErrNo::RuntimeError;
    }
    ReturnCode = details::evaluateTokens(GraphRef, LlamaContext,
                                         EmbdInputAfterImage, NPast);
    if (ReturnCode != ErrNo::Success) {
      spdlog::error(
          "[WASI-NN] GGML backend: failed to evaluate input tokens after image."sv);
      return ReturnCode;
    }
  }

  // Main predict loop.
  if (GraphRef.EnableDebugLog) {
    spdlog::info("[WASI-NN][Debug] GGML backend: enter main predict loop"sv);
  }
  while (NRemain > 0) {
    const llama_token Id =
        llama_sampling_sample(CtxSampling, LlamaContext, nullptr);
    llama_sampling_accept(CtxSampling, LlamaContext, Id, true);
    --NRemain;

    // Save the output token.
    CxtRef.LlamaOutputTokens.emplace_back(Id);
    CxtRef.LlamaOutputs += llama_token_to_piece(LlamaContext, Id);
    // When setting StreamStdout, we print the output to stdout.
    if (GraphRef.StreamStdout) {
      std::cout << llama_token_to_piece(LlamaContext, Id) << std::flush;
    }
    // Break if reverse prompt is found.
    if (!GraphRef.ReversePrompt.empty() &&
        CxtRef.LlamaOutputs.find(GraphRef.ReversePrompt) != std::string::npos) {
      if (GraphRef.EnableLog) {
        spdlog::info("[WASI-NN] GGML backend: reverse prompt found"sv);
      }
      break;
    }
    // Deal with end of text token.
    if (llama_sampling_last(CtxSampling) ==
        llama_token_eos(GraphRef.LlamaModel)) {
      if (GraphRef.EnableLog) {
        spdlog::info("[WASI-NN] GGML backend: EOS token found"sv);
      }
      break;
    }
    // Evaluate the output token.
    ReturnCode = details::evaluateTokens(GraphRef, LlamaContext, {Id}, NPast);
    if (ReturnCode != ErrNo::Success) {
      break;
    }
  }
  if (GraphRef.EnableDebugLog) {
    spdlog::info(
        "[WASI-NN][Debug] GGML backend: enter main predict loop...Done"sv);
  }
  // End of main predict loop.

  if (GraphRef.EnableLog) {
    llama_print_timings(LlamaContext);
  }

  // We free the contexts here to keep the ggml plugin stateless.
  // Users could fully control the contexts by themselves via their prompt.
  llama_sampling_free(CtxSampling);
  llama_free(LlamaContext);
  if (CxtRef.LlavaImageEmbd != nullptr) {
    llava_image_embed_free(CxtRef.LlavaImageEmbd);
    CxtRef.LlavaImageEmbd = nullptr;
  }

  if (GraphRef.EnableDebugLog) {
    spdlog::info("[WASI-NN][Debug] GGML backend: compute...Done"sv);
  }

  return ReturnCode;
#else
    if(spdlog::get("metrics"))
      spdlog::get("metrics")->trace("{}"sv, SimpleJSON({
          {"event", "received_request"},
          {"ggml_time", ggml_time_us()}
      })); 

  auto &CxtRef = Env.NNContext[ContextId].get<Context>();
  auto &GraphRef = Env.NNGraph[CxtRef.GraphId].get<Graph>();
  if (GraphRef.EnableDebugLog) {
    spdlog::info("[WASI-NN][Debug] GGML backend: compute"sv);
  }

  if (GraphRef.Embedding) {
    return details::getEmbedding(Env, ContextId);
  }

  if (CxtRef.LlamaInputs.size() == 0) {
    spdlog::error("[WASI-NN] GGML backend: Llama input is not set!"sv);
    return ErrNo::InvalidArgument;
  }

  // Clear the outputs.
  if (GraphRef.EnableDebugLog) {
    spdlog::info(
        "[WASI-NN][Debug] GGML backend: clear the previous output and tokens"sv);
  }
  CxtRef.LlamaOutputs.clear();
  CxtRef.LlamaOutputTokens.clear();
  if (GraphRef.EnableDebugLog) {
    spdlog::info(
        "[WASI-NN][Debug] GGML backend: clear the previous output and tokens...Done"sv);
  }
  std::vector<int64_t> IntertokenTimestamp;
  // Initialize the llama context.
      if (GraphRef.Embedding) {
    return details::getEmbedding(Env, ContextId);
  }
  std::unique_ptr<IDecodingStrategy> strategy;
  spdlog::info("[WASI-NN][Debug] GGML backend: starting decode, strategy is {}"sv, (int)GraphRef.SpeculativeStrategy);
  if (GraphRef.SpeculativeStrategy == speculative_strategy::SPECULATIVE) {
    strategy = std::make_unique<SpeculativeDecoding>();
  }else if (GraphRef.SpeculativeStrategy == speculative_strategy::LOOKAHEAD) {
    strategy = std::make_unique<LookaheadDecoding>();
  } else {
    strategy = std::make_unique<DefaultDecoding>();
  }

  // Use the strategy to perform decoding
    // Return value.
  auto ReturnCode = ErrNo::Success;
  if(spdlog::get("metrics"))
    spdlog::get("metrics")->trace("{}"sv, SimpleJSON({
        {"event", "start_decode"},
        {"ggml_time", ggml_time_us()}
    })); 


  ReturnCode = strategy->decode(GraphRef, CxtRef);

  spdlog::info("reached end of compute()"sv);
  if(spdlog::get("metrics"))spdlog::get("metrics")->flush();// TODO: flush not working??
  return ReturnCode;
#endif // WASMEDGE_PLUGIN_WASI_NN_GGML_STRATEGY
}

Expect<ErrNo> getOutputSingle(WasiNNEnvironment &Env, uint32_t ContextId,
                              uint32_t Index, Span<uint8_t> OutBuffer,
                              uint32_t &BytesWritten) noexcept {
  auto &CxtRef = Env.NNContext[ContextId].get<Context>();
  // Index 1 is for the metadata of the outputs.
  if (Index == 1) {
    std::string Metadata;
    auto Res = details::buildOutputMetadata(CxtRef, Metadata);
    if (Res != ErrNo::Success) {
      spdlog::error(
          "[WASI-NN] GGML backend: Failed to build output metadata."sv);
      return Res;
    }
    std::copy_n(Metadata.data(), Metadata.length(), OutBuffer.data());
    BytesWritten = Metadata.length();
    return ErrNo::Success;
  }
  std::string LastToken = llama_token_to_piece(CxtRef.LlamaContext,
                                               CxtRef.LlamaOutputTokens.back());
  std::copy_n(LastToken.data(), LastToken.length(), OutBuffer.data());
  BytesWritten = LastToken.length();
  return ErrNo::Success;
}

Expect<ErrNo> computeSingle(WasiNNEnvironment &Env,
                            uint32_t ContextId) noexcept {
  auto &CxtRef = Env.NNContext[ContextId].get<Context>();
  auto &GraphRef = Env.NNGraph[CxtRef.GraphId].get<Graph>();

  // Logging.
  if (GraphRef.EnableDebugLog) {
    spdlog::info("[WASI-NN][Debug] GGML backend: computeSingleToken"sv);
  }
  if (CxtRef.LlamaInputs.size() == 0) {
    spdlog::error("[WASI-NN] GGML backend: Llama input is not set!"sv);
    return ErrNo::InvalidArgument;
  }

  // New compute single token context.
  if (CxtRef.LlamaContext == nullptr) {
    // Clear the outputs.
    if (GraphRef.EnableDebugLog) {
      spdlog::info(
          "[WASI-NN][Debug] GGML backend: clear the previous output and tokens"sv);
    }
    CxtRef.LlamaOutputs.clear();
    CxtRef.LlamaOutputTokens.clear();
    if (GraphRef.EnableDebugLog) {
      spdlog::info(
          "[WASI-NN][Debug] GGML backend: clear the previous output and tokens...Done"sv);
    }

    // Initialize the llama context.
    gpt_params GPTParams;
    llama_context_params ContextParams = llama_context_default_params();
    details::setupGPTParam(GraphRef, GPTParams);
    details::setupContextParam(GraphRef, ContextParams);
    CxtRef.LlamaContext =
        llama_new_context_with_model(GraphRef.LlamaModel, ContextParams);
    // TODO: llama_batch_init
    CxtRef.LlamaSampling = llama_sampling_init(GPTParams.sparams);
    CxtRef.LlamaNPast = 0;

    // Get the context size.
    const uint64_t NCtx = llama_n_ctx(CxtRef.LlamaContext);
    // Minus 4 for the special tokens. (Such as <BOS>, <EOS>, ... tokens.)
    const uint64_t MaxTokensListSize = NCtx - 4;
    // Return value.
    auto ReturnCode = ErrNo::Success;

    // Check if the input is too long.
    if (static_cast<uint64_t>(CxtRef.LlamaInputs.size()) > MaxTokensListSize) {
      if (GraphRef.EnableLog) {
        spdlog::info(
            "[WASI-NN] GGML backend: the prompt is too long. Your input has {} tokens. Please reduce it to {} tokens."sv,
            CxtRef.LlamaInputs.size(), MaxTokensListSize);
      }
      return ErrNo::PromptTooLong;
    }

    // Evaluate input tokens.
    if (CxtRef.LlavaImageEmbd == nullptr) {
      // Text only prompt.
      ReturnCode = details::evaluateTokens(
          GraphRef, CxtRef.LlamaContext, CxtRef.LlamaInputs, CxtRef.LlamaNPast);
      if (ReturnCode != ErrNo::Success) {
        spdlog::error(
            "[WASI-NN] GGML backend: failed to evaluate input tokens."sv);
        return ReturnCode;
      }
    } else {
      // Llava format prompt with image data.
      std::vector<llama_token> EmbdInputBeforeImage(
          CxtRef.LlamaInputs.begin(),
          CxtRef.LlamaInputs.begin() + CxtRef.LlavaImagePosition);
      std::vector<llama_token> EmbdInputAfterImage(
          CxtRef.LlamaInputs.begin() + CxtRef.LlavaImagePosition,
          CxtRef.LlamaInputs.end());
      ReturnCode =
          details::evaluateTokens(GraphRef, CxtRef.LlamaContext,
                                  EmbdInputBeforeImage, CxtRef.LlamaNPast);
      if (ReturnCode != ErrNo::Success) {
        spdlog::error(
            "[WASI-NN] GGML backend: failed to evaluate input tokens before image."sv);
        return ReturnCode;
      }
      bool EvalImageStatus =
          llava_eval_image_embed(CxtRef.LlamaContext, CxtRef.LlavaImageEmbd,
                                 GraphRef.BatchSize, &CxtRef.LlamaNPast);
      if (!EvalImageStatus) {
        spdlog::error(
            "[WASI-NN] GGML backend: failed to evaluate embed image tokens."sv);
        return ErrNo::RuntimeError;
      }
      ReturnCode =
          details::evaluateTokens(GraphRef, CxtRef.LlamaContext,
                                  EmbdInputAfterImage, CxtRef.LlamaNPast);
      if (ReturnCode != ErrNo::Success) {
        spdlog::error(
            "[WASI-NN] GGML backend: failed to evaluate input tokens after image."sv);
        return ReturnCode;
      }
    }
  }

  // Main predict process.
  if (GraphRef.EnableDebugLog) {
    spdlog::info("[WASI-NN][Debug] GGML backend: enter main predict process"sv);
  }

  auto ReturnCode = ErrNo::Success;
  const llama_token Id =
      llama_sampling_sample(CxtRef.LlamaSampling, CxtRef.LlamaContext, nullptr);
  llama_sampling_accept(CxtRef.LlamaSampling, CxtRef.LlamaContext, Id, true);


  // Save the output token.
  // In single token mode, we do not handle StreamStdout and ReversePrompt.
  CxtRef.LlamaOutputTokens.emplace_back(Id);
  CxtRef.LlamaOutputs += llama_token_to_piece(CxtRef.LlamaContext, Id);
  // Deal with end of text token.
  if (llama_sampling_last(CxtRef.LlamaSampling) ==
      llama_token_eos(GraphRef.LlamaModel)) {
    ReturnCode = ErrNo::EndOfSequence;
    if (GraphRef.EnableLog) {
      spdlog::info("[WASI-NN] GGML backend: EOS token found"sv);
    }
  }
  // Evaluate the output token if not EOS.
  if (ReturnCode != ErrNo::EndOfSequence) {
    ReturnCode = details::evaluateTokens(GraphRef, CxtRef.LlamaContext, {Id},
                                         CxtRef.LlamaNPast);
  }
  if (GraphRef.EnableDebugLog) {
    spdlog::info(
        "[WASI-NN][Debug] GGML backend: enter main predict process...Done"sv);
  }
  // End of main predict process.

  if (GraphRef.EnableDebugLog) {
    spdlog::info("[WASI-NN][Debug] GGML backend: computeSingleToken...Done"sv);
  }

  return ReturnCode;
}

Expect<ErrNo> finiSingle(WasiNNEnvironment &Env, uint32_t ContextId) noexcept {
  auto &CxtRef = Env.NNContext[ContextId].get<Context>();
  auto &GraphRef = Env.NNGraph[CxtRef.GraphId].get<Graph>();

  // Logging for the llama timings.
  if (GraphRef.EnableLog) {
    llama_print_timings(CxtRef.LlamaContext);
  }

  // Clear the outputs.
  if (GraphRef.EnableDebugLog) {
    spdlog::info(
        "[WASI-NN][Debug] GGML backend: finiSingle: clear the previous output and tokens"sv);
  }
  CxtRef.LlamaOutputs.clear();
  CxtRef.LlamaOutputTokens.clear();
  if (GraphRef.EnableDebugLog) {
    spdlog::info(
        "[WASI-NN][Debug] GGML backend: finiSingle: clear the previous output and tokens...Done"sv);
  }

  // Delete the llama context.
  if (GraphRef.EnableDebugLog) {
    spdlog::info(
        "[WASI-NN][Debug] GGML backend: finiSingle: free the llama context"sv);
  }
  llama_sampling_free(CxtRef.LlamaSampling);
  llama_free(CxtRef.LlamaContext);
  CxtRef.LlamaSampling = nullptr;
  CxtRef.LlamaContext = nullptr;
  if (CxtRef.LlavaImageEmbd != nullptr) {
    llava_image_embed_free(CxtRef.LlavaImageEmbd);
    CxtRef.LlavaImageEmbd = nullptr;
  }
  if (GraphRef.EnableDebugLog) {
    spdlog::info(
        "[WASI-NN][Debug] GGML backend: finiSingle: free the llama context...Done"sv);
  }

  // Reset the context variables.
  CxtRef.LlamaNPast = 0;

  return ErrNo::Success;
}
#else
namespace {
Expect<ErrNo> reportBackendNotSupported() noexcept {
  spdlog::error("[WASI-NN] ggml backend is not built. use "
                "-WASMEDGE_PLUGIN_WASI_NN_BACKEND=\"ggml\" to build it."sv);
  return ErrNo::InvalidArgument;
}
} // namespace

Expect<ErrNo> load(WasiNNEnvironment &, Span<const Span<uint8_t>>, Device,
                   uint32_t &) noexcept {
  return reportBackendNotSupported();
}
Expect<ErrNo> initExecCtx(WasiNNEnvironment &, uint32_t, uint32_t &) noexcept {
  return reportBackendNotSupported();
}
Expect<ErrNo> setInput(WasiNNEnvironment &, uint32_t, uint32_t,
                       const TensorData &) noexcept {
  return reportBackendNotSupported();
}
Expect<ErrNo> getOutput(WasiNNEnvironment &, uint32_t, uint32_t, Span<uint8_t>,
                        uint32_t &) noexcept {
  return reportBackendNotSupported();
}
Expect<ErrNo> compute(WasiNNEnvironment &, uint32_t) noexcept {
  return reportBackendNotSupported();
}

#endif
} // namespace WasmEdge::Host::WASINN::GGML

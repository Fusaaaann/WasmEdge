#ifdef WASMEDGE_PLUGIN_WASI_NN_GGML_STRATEGY
#include "ggml.h"
#include "wasinnenv.h"
#include "strategies.h"
#include <llama.h>
#include "simdjson.h"

namespace WasmEdge::Host::WASINN::GGML {


// Speculative Decoding
#define SPEC_VOCAB_MAX_SIZE_DIFFERENCE  100
#define SPEC_VOCAB_CHECK_START_TOKEN_ID 5

struct seq_draft {
  bool active   = false;
  bool drafting = false;
  bool skip     = false;

  int IBatchDft = 0;
  std::vector<int> IBatchTgt;

  std::vector<llama_token> tokens;

  struct llama_sampling_context * CtxSampling;
};

// Lookahead Decoding

struct ngram_data {
  bool active = false;

  llama_seq_id seq_id = -1;

  std::vector<int> i_batch;

  std::vector<llama_token> tokens;
};

// n-gram container
struct ngram_container {
  ngram_container(int n_vocab, int N, int G) {
      cnt.resize(n_vocab);
      head.resize(n_vocab);
      tokens.resize(n_vocab * G * (N - 1));
  }

  int n_total = 0;

  std::vector<int> cnt;
  std::vector<int> head;

  // [n_vocab][G][N - 1]
  // for each token of the vocab, keep a ring-buffer of capacity G of n-grams of size N - 1
  std::vector<llama_token> tokens;
};

namespace details2{
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
    const llama_seq_id SequenceId = 0;

    auto Batch = llama_batch_get_one(&Tokens[I], NEval, NPast, SequenceId);
    auto Status =
        llama_decode(LlamaContext, Batch);
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
ErrNo setupGPTParam(Graph &GraphRef, gpt_params &GPTParams) {
  GPTParams.sparams.temp = GraphRef.Temp;
  GPTParams.sparams.top_p = GraphRef.TopP;
  GPTParams.sparams.penalty_repeat = GraphRef.RepeatPenalty;
  GPTParams.sparams.penalty_present = GraphRef.PresencePenalty;

  return ErrNo::Success;
}

ErrNo setupContextParam(Graph &GraphRef,
                                llama_context_params &ContextParams) {
  ContextParams.n_ctx = GraphRef.CtxSize;
  ContextParams.n_batch = GraphRef.BatchSize;
  ContextParams.n_threads = GraphRef.Threads;
  ContextParams.n_threads_batch = GraphRef.Threads;
  return ErrNo::Success;
}
}

ErrNo DefaultDecoding::decode(Graph &GraphRef, Context &CxtRef) noexcept {
gpt_params GPTParams;
llama_context_params ContextParams = llama_context_default_params();
details2::setupGPTParam(GraphRef, GPTParams);
details2::setupContextParam(GraphRef, ContextParams);
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
  if (static_cast<uint64_t>(CxtRef.LlamaInputs.size()) > MaxTokensListSize) {// TODO: where to obtain llamamodel
      if (GraphRef.EnableLog) {
      spdlog::info("[WASI-NN] GGML backend: the prompt is too long. Your input "
                  "has {} tokens. Please reduce it to {} tokens."sv,
                  CxtRef.LlamaInputs.size(), MaxTokensListSize);
      }
      return ErrNo::PromptTooLong;
  }

  // Assume text only prompt.
  ReturnCode = details2::evaluateTokens(GraphRef, LlamaContext,
                                      CxtRef.LlamaInputs, NPast);
  if (ReturnCode != ErrNo::Success) {
  spdlog::error(
      "[WASI-NN] GGML backend: failed to evaluate input tokens."sv);
  return ReturnCode;
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
      ReturnCode = details2::evaluateTokens(GraphRef, LlamaContext, {Id}, NPast);
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


  if (GraphRef.EnableDebugLog) {
      spdlog::info("[WASI-NN][Debug] GGML backend: compute...Done"sv);
  }

  return ReturnCode;
  return ErrNo::Success;
}
ErrNo SpeculativeDecoding::decode(Graph &GraphRef, Context &CxtRef) noexcept {
  gpt_params GPTParams;
  llama_context_params ContextParams = llama_context_default_params();
  details2::setupGPTParam(GraphRef, GPTParams);
  details2::setupContextParam(GraphRef, ContextParams);
  // auto LlamaContext =
  //     llama_new_context_with_model(GraphRef.LlamaModel, ContextParams);
  // Implement standard decoding logic here
  // max number of parallel drafting sequences (i.e. tree branches)
  const int n_seq_dft = GPTParams.n_parallel;

  // probability threshold for accepting a token from the draft model
  const float p_accept = GPTParams.p_accept;

  // probability threshold for splitting a draft branch (only for n_seq_dft > 1)
  const float p_split  = GPTParams.p_split;



  llama_model * model_tgt = NULL;
  llama_model * model_dft = NULL;

  llama_context * ctx_tgt = NULL;
  llama_context * ctx_dft = NULL;

  // load the target model
  GPTParams.logits_all = true;
  std::tie(model_tgt, ctx_tgt) = llama_init_from_gpt_params(GPTParams);

  // load the draft model
  GPTParams.model = GPTParams.model_draft;
  GPTParams.n_gpu_layers = GPTParams.n_gpu_layers_draft;
  if (GPTParams.n_threads_draft > 0) {
      GPTParams.n_threads = GPTParams.n_threads_draft;
  }
  GPTParams.n_threads_batch = GPTParams.n_threads_batch_draft;
  std::tie(model_dft, ctx_dft) = llama_init_from_gpt_params(GPTParams);

  {
      const int n_vocab_tgt = llama_n_vocab(model_tgt);
      const int n_vocab_dft = llama_n_vocab(model_dft);
      const int vocab_diff  = n_vocab_tgt > n_vocab_dft
          ? n_vocab_tgt - n_vocab_dft
          : n_vocab_dft - n_vocab_tgt;

      if (vocab_diff > SPEC_VOCAB_MAX_SIZE_DIFFERENCE) {
          spdlog::error("{}: error: draft model vocab must closely match target model to use speculation but "sv, __func__);
          spdlog::error("target vocab size {} does not match draft vocab size {} - difference {}, max allowed {}\n"sv,
                  n_vocab_tgt, llama_n_vocab(model_dft), vocab_diff, SPEC_VOCAB_MAX_SIZE_DIFFERENCE);
          return ErrNo::RuntimeError;
      }

      for (int i = SPEC_VOCAB_CHECK_START_TOKEN_ID; i < std::min(n_vocab_tgt, n_vocab_dft); ++i) {
          const char * token_text_tgt = llama_token_get_text(model_tgt, i);
          const char * token_text_dft = llama_token_get_text(model_dft, i);
          if (std::strcmp(token_text_tgt, token_text_dft) != 0) {
              spdlog::error("{}: error: draft model vocab must match target model to use speculation but "sv, __func__);
              spdlog::error("token {} content differs - target '{}', draft '{}'\n"sv, i,
                      llama_token_to_piece(ctx_tgt, i).c_str(),
                      llama_token_to_piece(ctx_dft, i).c_str());
              return ErrNo::RuntimeError;
          }
      }
  }


  // Tokenize the prompt
  const bool add_bos_tgt = llama_should_add_bos_token(model_tgt);
  spdlog::info("add_bos tgt: {}\n"sv, add_bos_tgt);

  const bool add_bos_dft = llama_should_add_bos_token(model_dft);
  spdlog::info("add_bos dft: {}\n"sv, add_bos_dft);

  if (add_bos_tgt != add_bos_dft) {
      spdlog::error("{}: error: draft model add_bos must match target model to use speculation but "sv, __func__);
      spdlog::error("add_bos_dft = {} while add_bos_tgt = {}\n"sv, add_bos_dft, add_bos_tgt);
      return ErrNo::RuntimeError;
  }

  std::vector<llama_token> inp;
  inp = ::llama_tokenize(ctx_tgt, GPTParams.prompt, add_bos_tgt, true);

  const int max_context_size     = llama_n_ctx(ctx_tgt);
  const int max_tokens_list_size = max_context_size - 4;

  if ((int) inp.size() > max_tokens_list_size) {
      spdlog::error("{}: error: prompt too long ({} tokens, max {})\n"sv, __func__, (int) inp.size(), max_tokens_list_size);
      return ErrNo::RuntimeError;
  }

  spdlog::error("\n\n");

  for (auto id : inp) {
      spdlog::error("{}"sv, llama_token_to_piece(ctx_tgt, id).c_str());
  }

  fflush(stderr);

  const int n_input = inp.size();

  // const auto t_enc_start = ggml_time_us();

  // eval the prompt with both models
  llama_decode(ctx_tgt, llama_batch_get_one( inp.data(), n_input - 1, 0,           0));
  llama_decode(ctx_tgt, llama_batch_get_one(&inp.back(),           1, n_input - 1, 0));
  llama_decode(ctx_dft, llama_batch_get_one( inp.data(), n_input,     0,           0));

  // const auto t_enc_end = ggml_time_us();

  // the 2 models should have the same vocab
  //GGML_ASSERT(n_vocab == llama_n_vocab(model_dft));

  // how many tokens to draft each time
  int n_draft = GPTParams.n_draft;

  int n_predict = 0;
  int n_drafted = 0;
  int n_accept  = 0;

  int n_past_tgt = inp.size();
  int n_past_dft = inp.size();

  // used to determine end of generation
  bool has_eos = false;

  // target model sampling context
  struct llama_sampling_context * CtxSampling = llama_sampling_init(GPTParams.sparams);

  // draft sequence data
  std::vector<seq_draft> drafts(n_seq_dft);

  GPTParams.sparams.grammar.clear(); // the draft samplers will copy the target sampler's grammar
  GPTParams.sparams.temp = -1.0f;    // force greedy sampling with probs for the draft model

  for (int s = 0; s < n_seq_dft; ++s) {
      drafts[s].CtxSampling = llama_sampling_init(GPTParams.sparams);
  }

  llama_batch batch_dft = llama_batch_init(GPTParams.n_ctx, 0, 1);
  llama_batch batch_tgt = llama_batch_init(GPTParams.n_ctx, 0, n_seq_dft);

  // const auto t_dec_start = ggml_time_us();

  // sample from the last token of the prompt
  drafts[0].IBatchTgt.resize(1);
  drafts[0].IBatchTgt[0] = 0;

  while (true) {
      // print current draft sequences
      for (int s = 0; s < n_seq_dft; ++s) {
          if (!drafts[s].active) {
              continue;
          }

          const auto & tokens = drafts[s].tokens;

          spdlog::info("draft {}: {}\n"sv, s, LOG_TOKENS_TOSTR_PRETTY(ctx_dft, tokens).c_str());
      }

      int i_dft  = 0;
      int s_keep = 0;

      while (true) {
          spdlog::info("sampling target: s_keep = {}, i_dft = {}, IBatchTgt = {}\n"sv, s_keep, i_dft, drafts[s_keep].IBatchTgt[i_dft]);

          // sample from the target model
          llama_token id = llama_sampling_sample(CtxSampling, ctx_tgt, NULL, drafts[s_keep].IBatchTgt[i_dft]);

          llama_sampling_accept(CtxSampling, ctx_tgt, id, true);

          //spdlog::info("last: {}\n"sv, LOG_TOKENS_TOSTR_PRETTY(ctx_tgt, CtxSampling->prev).c_str());

          // const std::string token_str = llama_token_to_piece(ctx_tgt, id);// TODO: potential bottleneck: detokenizer


          if (id == llama_token_eos(model_tgt)) {
              has_eos = true;
          }

          ++n_predict;

          // check if the target token matches any of the drafts
          {
              bool matches = false;

              for (int s = 0; s < n_seq_dft; ++s) {
                  if (!drafts[s].active) {
                      continue;
                  }

                  if (i_dft < (int) drafts[s].tokens.size() && id == drafts[s].tokens[i_dft]) {
                      // spdlog::info("the sampled target token matches the {}th drafted token of sequence {} ({}, '{}') - accepted\n"sv, i_dft, s, id, token_str.c_str());

                      s_keep = s;
                      matches = true;
                  } else {
                      drafts[s].active = false;
                  }
              }

              if (matches) {
                  ++n_accept;
                  ++n_past_tgt;
                  ++n_past_dft;
                  ++i_dft;

                  continue;
              }
          }
          CxtRef.LlamaOutputTokens.emplace_back(id);

          if(has_eos) break;

          // spdlog::info("the sampled target token ({}, '{}') did not match, or we ran out of drafted tokens\n"sv, id, token_str.c_str());

          // TODO: simplify
          {
              spdlog::info("keeping sequence {}, n_past_tgt = {}, n_past_dft = {}\n"sv, s_keep, n_past_tgt, n_past_dft);

              llama_kv_cache_seq_keep(ctx_dft, s_keep);
              llama_kv_cache_seq_cp  (ctx_dft, s_keep, 0, -1, -1);
              llama_kv_cache_seq_keep(ctx_dft, 0);

              llama_kv_cache_seq_rm  (ctx_tgt, s_keep, n_past_tgt, -1);
              llama_kv_cache_seq_keep(ctx_tgt, s_keep);
              llama_kv_cache_seq_cp  (ctx_tgt, s_keep, 0, -1, -1);
              llama_kv_cache_seq_keep(ctx_tgt, 0);
          }

          for (int s = 0; s < n_seq_dft; ++s) {
              drafts[s].active = false;
              drafts[s].tokens.clear();
              drafts[s].IBatchTgt.clear();
          }
          // note: will be erased after the speculation phase
          drafts[0].tokens.push_back(id);
          drafts[0].IBatchTgt.push_back(0);

          llama_batch_clear(batch_dft);
          llama_batch_add  (batch_dft, id, n_past_dft, { 0 }, true);

          llama_kv_cache_seq_rm(ctx_dft, 0, n_past_dft, -1);
          // spdlog::info("dft batch: {}\n"sv, LOG_BATCH_TOSTR_PRETTY(ctx_dft, batch_dft).c_str());
          llama_decode         (ctx_dft, batch_dft);

          ++n_past_dft;

          break;
      }


      llama_sampling_cp(CtxSampling, drafts[0].CtxSampling);

      int n_seq_cur  = 1;
      int n_past_cur = n_past_dft;

      for (int s = 0; s < n_seq_dft; ++s) {
          drafts[s].active   = false;
          drafts[s].drafting = false;
      }
      drafts[0].active      = true;
      drafts[0].drafting    = true;
      drafts[0].IBatchDft = 0;

      llama_batch_clear(batch_tgt);
      llama_batch_add  (batch_tgt, drafts[0].tokens[0], n_past_tgt, { 0 }, true);

      // sample n_draft tokens from the draft model using tree-based sampling
      for (int i = 0; i < n_draft; ++i) {
          batch_dft.n_tokens = 0;

          for (int s = 0; s < n_seq_dft; ++s) {
              drafts[s].skip = false;
          }

          for (int s = 0; s < n_seq_dft; ++s) {
              if (!drafts[s].drafting || drafts[s].skip) {
                  continue;
              }

              llama_sampling_sample(drafts[s].CtxSampling, ctx_dft, NULL, drafts[s].IBatchDft);

              const auto & cur_p = drafts[s].CtxSampling->cur;

              for (int k = 0; k < std::min(n_seq_dft + 3, (int) cur_p.size()); ++k) {
                  spdlog::info(" - draft candidate {} for seq {}, pos {}: %6d (%8.3f) '{}'\n"sv,
                          k, s, i, cur_p[k].id, cur_p[k].p, llama_token_to_piece(ctx_dft, cur_p[k].id).c_str());
              }

              if (cur_p[0].p < p_accept) {
                  spdlog::info("stopping drafting for seq {}, probability too low: {} < {}\n"sv, s, cur_p[0].p, p_accept);
                  drafts[s].drafting = false;
                  continue;
              }

              std::vector<int> sa(1, s);

              // attempt to split the branch if the probability is high enough
              for (int f = 1; f < 8; ++f) {
                  if (n_seq_cur < n_seq_dft && cur_p[f].p > p_split) {
                      spdlog::info("splitting seq {} into {}\n"sv, s, n_seq_cur);

                      llama_kv_cache_seq_rm(ctx_dft,    n_seq_cur, -1, -1);
                      llama_kv_cache_seq_cp(ctx_dft, s, n_seq_cur, -1, -1);

                      // all previous tokens from this branch are now also part of the new branch
                      for (int t = 0; t < batch_tgt.n_tokens; ++t) {
                          for (int p = 0; p < batch_tgt.n_seq_id[t]; ++p) {
                              if (batch_tgt.seq_id[t][p] == s) {
                                  batch_tgt.seq_id[t][batch_tgt.n_seq_id[t]] = n_seq_cur;
                                  batch_tgt.n_seq_id[t]++;
                                  break;
                              }
                          }
                      }

                      // copy the draft state
                      drafts[n_seq_cur].active   = true;
                      drafts[n_seq_cur].drafting = true;
                      drafts[n_seq_cur].skip     = true;

                      drafts[n_seq_cur].tokens      = drafts[s].tokens;
                      drafts[n_seq_cur].IBatchDft = drafts[s].IBatchDft;
                      drafts[n_seq_cur].IBatchTgt = drafts[s].IBatchTgt;

                      llama_sampling_cp(drafts[s].CtxSampling, drafts[n_seq_cur].CtxSampling);

                      sa.push_back(n_seq_cur);

                      n_seq_cur++;
                  } else {
                      break;
                  }
              }

              // add drafted token for each sequence
              for (int is = 0; is < (int) sa.size(); ++is) {
                  const llama_token id = cur_p[is].id;

                  const int s = sa[is];

                  llama_sampling_accept(drafts[s].CtxSampling, ctx_dft, id, true);

                  drafts[s].tokens.push_back(id);

                  // add unique drafted tokens to the target batch
                  drafts[s].IBatchTgt.push_back(batch_tgt.n_tokens);

                  llama_batch_add(batch_tgt, id, n_past_tgt + i + 1, { s }, true);

                  // add the token to the batch for batched decoding with the draft model
                  drafts[s].IBatchDft = batch_dft.n_tokens;

                  llama_batch_add(batch_dft, id, n_past_cur, { s }, true);

                  if (batch_tgt.n_tokens > n_draft) {
                      drafts[s].drafting = false;
                  }
              }
          }

          // no sequence is drafting anymore
          if (batch_dft.n_tokens == 0) {
              break;
          }

          // evaluate the drafted tokens on the draft model
          llama_decode(ctx_dft, batch_dft);
          ++n_past_cur;
          ++n_drafted;

          if (batch_tgt.n_tokens > n_draft) {
              break;
          }
      }

      // evaluate the target model on the drafted tokens
      {
          llama_kv_cache_seq_keep(ctx_tgt, 0);
          for (int s = 1; s < n_seq_dft; ++s) {
              llama_kv_cache_seq_cp(ctx_tgt, 0, s, -1, -1);
          }

          // spdlog::info("target batch: {}\n"sv, LOG_BATCH_TOSTR_PRETTY(ctx_tgt, batch_tgt).c_str());
          llama_decode(ctx_tgt, batch_tgt);
          ++n_past_tgt;
      }

      // the first token is always proposed by the target model before the speculation loop so we erase it here
      for (int s = 0; s < n_seq_dft; ++s) {
          if (!drafts[s].active) {
              continue;
          }

          drafts[s].tokens.erase(drafts[s].tokens.begin());
      }
  }

  // auto t_dec_end = ggml_time_us();
  llama_sampling_free(CtxSampling);
  for (int s = 0; s < n_seq_dft; ++s) {
      llama_sampling_free(drafts[s].CtxSampling);
  }

  llama_batch_free(batch_dft);

  llama_free(ctx_tgt);
  llama_free_model(model_tgt);

  llama_free(ctx_dft);
  llama_free_model(model_dft);

  return ErrNo::Success;
}

// TODO: undefined symbol: _ZTVN8WasmEdge4Host6WASINN4GGML17LookaheadDecodingE

ErrNo LookaheadDecoding::decode(Graph &GraphRef, Context &CxtRef) noexcept {
  gpt_params GPTParams;
  llama_context_params ContextParams = llama_context_default_params();
  details2::setupGPTParam(GraphRef, GPTParams);
  details2::setupContextParam(GraphRef, ContextParams);
  // auto LlamaContext;
  const int LookaheadWidth = GraphRef.LookaheadWidth;
  const int NgramSize = GraphRef.NgramSize;
  const int MaxVerifyNgramSize = GraphRef.MaxVerifyNgramSize;
  
  llama_context * LlamaContext = 
      llama_new_context_with_model(GraphRef.LlamaModel, ContextParams);// TODO: use WasmEdge parameter
  llama_batch batch = llama_batch_init(GPTParams.n_ctx, 0, LookaheadWidth + MaxVerifyNgramSize + 1);

  // load the target model
  std::tie(GraphRef.LlamaModel, LlamaContext) = llama_init_from_gpt_params(GPTParams);

  std::vector<llama_token> inp;
  std::vector<llama_token> all;

  // Tokenize the prompt
  const bool add_bos = llama_should_add_bos_token(GraphRef.LlamaModel);

  inp = llama_tokenize(LlamaContext, GPTParams.prompt, add_bos, true);
  all = inp;

  const int n_input = inp.size();
  // const auto t_enc_start = ggml_time_us();

  // eval the prompt
  llama_decode(LlamaContext, llama_batch_get_one( inp.data(), n_input - 1, 0,           0));
  llama_decode(LlamaContext, llama_batch_get_one(&inp.back(),           1, n_input - 1, 0));

  for (int s = 1; s < LookaheadWidth + MaxVerifyNgramSize + 1; ++s) {
      llama_kv_cache_seq_cp(LlamaContext, 0, s, -1, -1);
  }

  // const auto t_enc_end = ggml_time_us();

  int n_predict = 0;
  int n_accept  = 0;

  int NPast = inp.size();

  llama_token id = 0;

  // used to determine end of generation
  bool has_eos = false;
  
  // verification n-grams
  std::vector<ngram_data> ngrams_cur(MaxVerifyNgramSize);

  // target model sampling context
  struct llama_sampling_context * CtxSampling = llama_sampling_init(GPTParams.sparams);

  // tokens for the past NgramSize - 1 Jacobi iterations
  std::vector<llama_token> tokens_j_prev(LookaheadWidth);
  std::vector<std::vector<llama_token>> tokens_j(NgramSize - 1);
  for (int j = 0; j < NgramSize - 1; j++) {
      tokens_j[j].resize(LookaheadWidth);

      for (int i = 0; i < LookaheadWidth; i++) {
          // there are different ways to init these tokens
          if (0) {
              // initialize randomly from the prompt tokens
              tokens_j[j][i] = all[1 + rand() % (all.size() - 1)];
          } else {
              // initialize with a sequence of increasing numbers
              tokens_j[j][i] = 100 + i;
          }
      }
  }

  std::vector<llama_seq_id> seq_id_look;

  // the input token belongs both to all sequences
  std::vector<llama_seq_id> seq_id_all(LookaheadWidth + MaxVerifyNgramSize + 1);
  for (int i = 0; i < LookaheadWidth + MaxVerifyNgramSize + 1; i++) {
      seq_id_all[i] = i;
  }

  // here we keep adding new n-grams as we go
  ngram_container ngrams_observed(llama_n_vocab(GraphRef.LlamaModel), NgramSize, MaxVerifyNgramSize);

  // debug
  struct llama_kv_cache_view KVCView = llama_kv_cache_view_init(LlamaContext, LookaheadWidth + MaxVerifyNgramSize + 1);

  std::vector<int64_t> t_dec_inter_token;
  t_dec_inter_token.resize((size_t)GraphRef.BatchSize);
  // const auto t_dec_start = ggml_time_us();

  // sample first token
  // to make the batch instance(?) to be batch_add'ed

  {
      id = llama_sampling_sample(CtxSampling, LlamaContext, NULL, 0);

      llama_sampling_accept(CtxSampling, LlamaContext, id, true);

      {
          const std::string token_str = llama_token_to_piece(LlamaContext, id);

          spdlog::info("{}"sv, token_str.c_str());
          fflush(stdout);
      }
  }
    uint32_t NCtx = llama_n_ctx(LlamaContext);

  // End the inference if the context is full.
  if (NPast + static_cast<uint32_t>(inp.size()) > NCtx) {
    if (GraphRef.EnableLog) {
      spdlog::info(
          "[WASI-NN] GGML backend: the context if full ({} / {} tokens). Please increase your context size."sv,
          NPast + static_cast<uint32_t>(inp.size()), NCtx);
    }
    return ErrNo::ContextFull;
  }

  while (true) {
      // debug
      // if (dump_kv_cache) {
      //     llama_kv_cache_view_update(LlamaContext, &KVCView);
      //     dump_kv_cache_view_seqs(KVCView, 40);
      // }

      // build the mask from https://lmsys.org/blog/2023-11-21-lookahead-decoding/
      //
      // Example for LookaheadWidth = 5, NgramSize = 4, MaxVerifyNgramSize = 2:
      // (I = input, L = lookahead, V = verification)
      //
      // Batch:  0  1  2  3  4  5  6  7  8  9 10 11 12 13 14 15 16 17 18 19 20
      // T:        -2 -2 -2 -2 -1 -1 -1 -1 -1  0  0  0  0  0  0
      // Info:   I  L  L  L  L  L  L  L  L  L  L  L  L  L  L  V  V  V  V  V  V
      // Pos:    0  1  2  3  4  1  2  3  4  5  2  3  4  5  6  1  2  3  1  2  3   (+ NPast)
      // Logits: 1  0  0  0  0  0  0  0  0  0  1  1  1  1  1  1  1  1  1  1  1
      // ---------------------------------------------------------------------
      // Seq:    0
      //         1              1              1
      //         2  2              2              2
      //         3  3  3              3              3
      //         4  4  4  4              4              4
      //         5  5  5  5  5              5              5
      //         6                                            6  6  6
      //         7                                                     7  7  7
      // ---------------------------------------------------------------------
      //                                       |  |  |  |  |  |  |  |  |  |  |
      //                                       V  V  V  V  V  |  |  |  |  |  |
      //                                         j_tokens     |  |  |  |  |  |
      //                                                      V  V  V  V  V  V
      //                                                             id
      {
          llama_batch_clear(batch);

          // current token - first token of the first level
          llama_batch_add(batch, id, NPast, seq_id_all, true);

          // verification n-grams - queue this before the lookahead tokens for less KV cache fragmentation
          {
              const int g_cur = ngrams_observed.cnt[id];

              ngrams_cur.resize(g_cur);
              for (int g = 0; g < g_cur; g++) {
                  ngrams_cur[g].active = true;
                  ngrams_cur[g].tokens.resize(NgramSize);
                  ngrams_cur[g].i_batch.resize(NgramSize);
                  ngrams_cur[g].seq_id = LookaheadWidth + 1 + g;
                  ngrams_cur[g].i_batch[0] = 0;
                  ngrams_cur[g].tokens [0] = id;
              }

              for (int j = 0; j < NgramSize - 1; j++) {
                  for (int g = 0; g < g_cur; g++) {
                      const int idx = id*(NgramSize - 1)*MaxVerifyNgramSize + g*(NgramSize - 1);

                      const llama_token t = ngrams_observed.tokens[idx + j];

                      ngrams_cur[g].tokens [j + 1] = t;
                      ngrams_cur[g].i_batch[j + 1] = batch.n_tokens;

                      llama_batch_add(batch, t, NPast + j + 1, { LookaheadWidth + 1 + g }, true);
                  }
              }
          }

          // fill the remaining LookaheadWidth - 1 tokens for the first level
          for (int i = 1; i < LookaheadWidth; i++) {
              seq_id_look.resize(LookaheadWidth - i);
              for (int j = 0; j < LookaheadWidth - i; j++) {
                  seq_id_look[j] = i + j + 1;
              }

              llama_batch_add(batch, tokens_j[0][i], NPast + i, seq_id_look, false);
          }

          // fill the rest of the levels
          for (int j = 1; j < NgramSize - 1; j++) {
              for (int i = 0; i < LookaheadWidth; i++) {
                  llama_batch_add(batch, tokens_j[j][i], NPast + j + i, { i + 1 }, j == NgramSize - 2);
              }
          }
      }
      auto Status = llama_decode(LlamaContext, batch);

      if (Status == 1) {
        spdlog::error(
            "[WASI-NN] GGML backend: failed to llama_decode: try reducing the size of the batch or increasing the size of context"sv);
        return ErrNo::RuntimeError;
      } else if (Status < 0) {
        spdlog::error(
            "[WASI-NN] GGML backend: failed to llama_decode: internal fatal error. Please open an issue on GitHub"sv);
        return ErrNo::RuntimeError;
      }

      int seq_id_best = 0;

      for (int v = 0; v < NgramSize; ++v) { // v: current verifying / predicting token in sequence
          int i_batch = 0;

          // if no active ngrams are left, it means the sampled token does not pass the verification
          if (v > 0) {
              for (int g = 0; g < (int) ngrams_cur.size(); g++) {
                  if (ngrams_cur[g].active) {
                      i_batch = ngrams_cur[g].i_batch[v];
                      seq_id_best = ngrams_cur[g].seq_id;

                      ++n_accept;
                      break;
                  }
              }

              // no more matches -> create a new batch
              if (i_batch == 0) {
                  break;
              }
          }
          // sample the next token
          id = llama_sampling_sample(CtxSampling, LlamaContext, NULL, i_batch);

          llama_sampling_accept(CtxSampling, LlamaContext, id, true);

          // print
          {
            CxtRef.LlamaOutputTokens.emplace_back(id);
              // const std::string token_str = llama_token_to_piece(LlamaContext, id);

              // if (v == 0) {
              //     spdlog::info("{}"sv, token_str.c_str());
              // } else {
              //     // print light cyan
              //     spdlog::info("\033[0;96m{}\033[0m"sv, token_str.c_str());
              // }
              // fflush(stdout);

              if (id == llama_token_eos(GraphRef.LlamaModel)) {
                  has_eos = true;
              }

              all.push_back(id);
          }

          ++n_predict;
          ++NPast;

          // verify across active n-grams
          for (int g = 0; g < (int) ngrams_cur.size(); g++) {
              if (ngrams_cur[g].active) {
                  if (v == NgramSize - 1) {
                      ngrams_cur[g].active = false;
                  } else {
                      if (id != ngrams_cur[g].tokens[v + 1]) {
                          ngrams_cur[g].active = false;
                      }
                  }
              }
          }

          // print known n-grams starting with token id (debug)
          if (0 && v == 0) {
              if (ngrams_observed.cnt[id] > 0) {
                  spdlog::info("\n - {} n-grams starting with '{}'\n"sv, ngrams_observed.cnt[id], llama_token_to_piece(LlamaContext, id).c_str());
              }

              for (int i = 0; i < ngrams_observed.cnt[id]; i++) {
                  spdlog::info("   - ngram %2d: "sv, i);

                  const int idx = id*(NgramSize - 1)*MaxVerifyNgramSize + i*(NgramSize - 1);

                  for (int j = 0; j < NgramSize - 1; j++) {
                      const std::string token_str = llama_token_to_piece(LlamaContext, ngrams_observed.tokens[idx + j]);

                      spdlog::info("{}"sv, token_str.c_str());
                  }

                  spdlog::info("\n");
              }
          }
          if(has_eos)break;

          // update lookahead tokens
          {
              for (int i = 0; i < LookaheadWidth; i++) {
                  tokens_j_prev[i] = tokens_j[0][i];
              }

              for (int j = 0; j < NgramSize - 2; j++) {
                  tokens_j[j] = tokens_j[j + 1];
              }

              if (v == 0) {
                  // sample from the last level
                  for (int i = 0; i < LookaheadWidth; i++) {
                      tokens_j[NgramSize - 2][i] = llama_sampling_sample(CtxSampling, LlamaContext, NULL, ngrams_cur.size()*(NgramSize-1) + LookaheadWidth*(NgramSize - 2) + i);
                  }
              } else {
                  for (int i = 0; i < LookaheadWidth; i++) {
                      // there are different ways to init these tokens
                      if (0) {
                          // random init
                          tokens_j[NgramSize - 2][i] = all[1 + rand() % (all.size() - 1)];
                      } else {
                          // init from the previous level
                          tokens_j[NgramSize - 2][i] = tokens_j[0][i];
                      }
                  }
              }
          }

          // update observed ngrams
          if (v == 0) {
              // the first token of the n-gram is determined by the index in the container so it is not stored
              std::vector<llama_token> ngram(NgramSize - 1);

              // n-gram generation
              // ref: https://github.com/hao-ai-lab/LookaheadDecoding/issues/14#issuecomment-1826198518
              for (int f = 0; f < LookaheadWidth; ++f) {
                  const int ft = tokens_j_prev[f]; // first token of the n-gram

                  for (int j = 0; j < NgramSize - 1; ++j) {
                      ngram[j] = tokens_j[j][f];
                  }

                  // filter-out repeating n-grams
                  {
                      bool is_unique = true;

                      for (int k = 0; k < ngrams_observed.cnt[ft]; ++k) {
                          const int idx = ft*(NgramSize - 1)*MaxVerifyNgramSize + k*(NgramSize - 1);

                          bool is_match = true;
                          for (int j = 0; j < NgramSize - 1; ++j) {
                              if (ngrams_observed.tokens[idx + j] != ngram[j]) {
                                  is_match = false;
                                  break;
                              }
                          }

                          if (is_match) {
                              is_unique = false;
                              break;
                          }
                      }

                      if (!is_unique) {
                          continue;
                      }
                  }

                  const int head = ngrams_observed.head[ft];
                  const int idx  = ft*(NgramSize - 1)*MaxVerifyNgramSize + head*(NgramSize - 1);

                  for (int i = 0; i < NgramSize - 1; i++) {
                      ngrams_observed.tokens[idx + i] = ngram[i];
                  }

                  ngrams_observed.cnt[ft]  = std::min(MaxVerifyNgramSize, ngrams_observed.cnt[ft] + 1);
                  ngrams_observed.head[ft] = (head + 1) % MaxVerifyNgramSize;

                  ngrams_observed.n_total++;
              }
          }
      }
      if(has_eos) break;

      // KV cache management
      // if no verification token matched, we simply remove all cells from this batch -> no fragmentation
      llama_kv_cache_seq_rm(LlamaContext, -1, NPast, -1);

      if (seq_id_best != 0) {
          // if a verification token matched, we keep the best sequence and remove the rest
          // this leads to some KV cache fragmentation
          llama_kv_cache_seq_keep(LlamaContext, seq_id_best);
          llama_kv_cache_seq_cp  (LlamaContext, seq_id_best, 0, -1, -1);
          llama_kv_cache_seq_rm  (LlamaContext, seq_id_best,    -1, -1);

          for (int s = 1; s < LookaheadWidth + MaxVerifyNgramSize + 1; ++s) {
              llama_kv_cache_seq_cp(LlamaContext, 0, s, -1, -1);
          }
      }
  }

  // auto t_dec_end = ggml_time_us();
  llama_kv_cache_view_free(&KVCView);
  llama_sampling_free(CtxSampling);

  llama_batch_free(batch);

  llama_free(LlamaContext);
  // llama_free_model(GraphRef.LlamaModel);


  return ErrNo::Success;
}

} // WasmEdge::Host::WASINN::GGML

#endif // ifdef WASMEDGE_PLUGIN_WASI_NN_GGML_STRATEGY
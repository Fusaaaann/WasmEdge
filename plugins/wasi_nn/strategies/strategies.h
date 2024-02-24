#ifdef WASMEDGE_PLUGIN_WASI_NN_BACKEND_GGML
#ifdef WASMEDGE_PLUGIN_WASI_NN_GGML_STRATEGY
#include "simdjson.h"
#include <common.h>
#include <cstdlib>
#include <llama.h>
#endif // ifdef WASMEDGE_PLUGIN_WASI_NN_GGML_STRATEGY
#endif // ifdef WASMEDGE_PLUGIN_WASI_NN_BACKEND_GGML

namespace WasmEdge::Host::WASINN::GGML {
#ifdef WASMEDGE_PLUGIN_WASI_NN_GGML_STRATEGY
// LLAMA_API defined in thirdparty/ggml/llama.h
 LLAMA_API float * wrapped_llama_get_logits_ith(struct llama_context * ctx, int32_t i);
 namespace Strategies {

  #if WASMEDGE_PLUGIN_WASI_NN_GGML_STRATEGY == SPECULATIVE
  // speculative_decode();
  #elif WASMEDGE_PLUGIN_WASI_NN_GGML_STRATEGY == SPECULATIVE
  // lookahead_decode();
  #else
  #endif // if WASMEDGE_PLUGIN_WASI_NN_GGML_STRATEGY
 }



#endif // ifdef WASMEDGE_PLUGIN_WASI_NN_GGML_STRATEGY

}
namespace WasmEdge::Host::WASINN::GGML {

#ifdef WASMEDGE_PLUGIN_WASI_NN_GGML_STRATEGY

// LLAMA_API defined in thirdparty/ggml/llama.h
 LLAMA_API float * wrapped_llama_get_logits_ith(struct llama_context * ctx, int32_t i);
{

  // TODO: before decode
  #if WASMEDGE_PLUGIN_WASI_NN_GGML_STRATEGY == SPECULATIVE
    
  #else

  #endif // if WASMEDGE_PLUGIN_WASI_NN_GGML_STRATEGY
  
  auto * result = llama_get_logits_ith(ctx, i);
  
  // TODO: after decode
  #if WASMEDGE_PLUGIN_WASI_NN_GGML_STRATEGY == SPECULATIVE
    
  #else
  
  #endif // if WASMEDGE_PLUGIN_WASI_NN_GGML_STRATEGY
  
  return result;
}
#endif // ifdef WASMEDGE_PLUGIN_WASI_NN_GGML_STRATEGY

}
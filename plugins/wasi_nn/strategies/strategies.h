#include "ggml.h"
#include "wasinnenv.h"

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
class IDecodingStrategy {
/**
 * maintained in ggml.h::Graph
*/


public:
  virtual ~IDecodingStrategy() = default;
  virtual ErrNo decode(Graph &GraphRef, Context &CxtRef) noexcept = 0;
};
class DefaultDecoding : public IDecodingStrategy {
public:
  ErrNo decode(Graph &GraphRef, Context &CxtRef) noexcept override;
};
class SpeculativeDecoding : public IDecodingStrategy {
public:
  ErrNo decode(Graph &GraphRef, Context &CxtRef) noexcept override;
};

class LookaheadDecoding : public IDecodingStrategy {
public:
  ErrNo decode(Graph &GraphRef, Context &CxtRef) noexcept override;
};

#endif // ifdef WASMEDGE_PLUGIN_WASI_NN_GGML_STRATEGY

}
// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: 2019-2022 Second State INC

#include "common/roundeven.h"
#include "executor/executor.h"

#include <cmath>

namespace WasmEdge {
namespace Executor {

template <typename T> TypeU<T> Executor::runClzOp(ValVariant &Val) const {
  T I = Val.get<T>();
  // Return the count of leading zero bits in i.
  if (I != 0U) {
    T Cnt = 0;
    T Mask = static_cast<T>(0x1U) << (sizeof(T) * 8 - 1);
    while ((I & Mask) == 0U) {
      Cnt++;
      I <<= 1;
    }
    Val.get<T>() = Cnt;
  } else {
    Val.get<T>() = static_cast<T>(sizeof(T) * 8);
  }
  return {};
}

template <typename T> TypeU<T> Executor::runCtzOp(ValVariant &Val) const {
  T I = Val.get<T>();
  // Return the count of trailing zero bits in i.
  if (I != 0U) {
    T Cnt = 0;
    T Mask = static_cast<T>(0x1U);
    while ((I & Mask) == 0U) {
      Cnt++;
      I >>= 1;
    }
    Val.get<T>() = Cnt;
  } else {
    Val.get<T>() = static_cast<T>(sizeof(T) * 8);
  }
  return {};
}

template <typename T> TypeU<T> Executor::runPopcntOp(ValVariant &Val) const {
  T I = Val.get<T>();
  // Return the count of non-zero bits in i.
  if (I != 0U) {
    T Cnt = 0;
    T Mask = static_cast<T>(0x1U);
    while (I != 0U) {
      if (I & Mask) {
        Cnt++;
      }
      I >>= 1;
    }
    Val.get<T>() = Cnt;
  }
  return {};
}

template <typename T> TypeF<T> Executor::runAbsOp(ValVariant &Val) const {
  Val.get<T>() = std::fabs(Val.get<T>());
  return {};
}

template <typename T> TypeF<T> Executor::runNegOp(ValVariant &Val) const {
  Val.get<T>() = -Val.get<T>();
  return {};
}

template <typename T> TypeF<T> Executor::runCeilOp(ValVariant &Val) const {
  Val.get<T>() = std::ceil(Val.get<T>());
  return {};
}

template <typename T> TypeF<T> Executor::runFloorOp(ValVariant &Val) const {
  Val.get<T>() = std::floor(Val.get<T>());
  return {};
}

template <typename T> TypeF<T> Executor::runTruncOp(ValVariant &Val) const {
  Val.get<T>() = std::trunc(Val.get<T>());
  return {};
}

template <typename T> TypeF<T> Executor::runNearestOp(ValVariant &Val) const {
  Val.get<T>() = WasmEdge::roundeven(Val.get<T>());
  return {};
}

template <typename T> TypeF<T> Executor::runSqrtOp(ValVariant &Val) const {
  Val.get<T>() = std::sqrt(Val.get<T>());
  return {};
}

template <typename TIn, typename TOut>
Expect<void> Executor::runExtractLaneOp(ValVariant &Val,
                                        const uint8_t Index) const {
  using VTIn = SIMDArray<TIn, 16>;
  const TOut Result = Val.get<VTIn>()[Index];
  Val.emplace<TOut>(Result);
  return {};
}

template <typename TIn, typename TOut>
Expect<void> Executor::runSplatOp(ValVariant &Val) const {
  const TOut Part = static_cast<TOut>(Val.get<TIn>());
  using VTOut = SIMDArray<TOut, 16>;
  if constexpr (sizeof(TOut) == 1) {
    Val.emplace<VTOut>(VTOut{Part, Part, Part, Part, Part, Part, Part, Part, Part,
                       Part, Part, Part, Part, Part, Part, Part});
  } else if constexpr (sizeof(TOut) == 2) {
    Val.emplace<VTOut>(VTOut{Part, Part, Part, Part, Part, Part, Part, Part});
  } else if constexpr (sizeof(TOut) == 4) {
    Val.emplace<VTOut>(VTOut{Part, Part, Part, Part});
  } else if constexpr (sizeof(TOut) == 8) {
    Val.emplace<VTOut>(VTOut{Part, Part});
  }
  return {};
}

template <typename TIn, typename TOut>
Expect<void> Executor::runVectorExtendLowOp(ValVariant &Val) const {
  static_assert(sizeof(TIn) * 2 == sizeof(TOut));
  static_assert(sizeof(TIn) == 1 || sizeof(TIn) == 2 || sizeof(TIn) == 4);
  using VTIn = SIMDArray<TIn, 16>;
  using HVTIn = SIMDArray<TIn, 8>;
  using VTOut = SIMDArray<TOut, 16>;
  const VTIn &V = Val.get<VTIn>();
  VTOut Result;
  for (size_t I = 0; I < (8 / sizeof(TIn)); ++I) {
    Result[I] = V[I];
  }
  Val.emplace<VTOut>(Result);
  return {};
}

template <typename TIn, typename TOut>
Expect<void> Executor::runVectorExtendHighOp(ValVariant &Val) const {
  static_assert(sizeof(TIn) * 2 == sizeof(TOut));
  static_assert(sizeof(TIn) == 1 || sizeof(TIn) == 2 || sizeof(TIn) == 4);
  using VTIn = SIMDArray<TIn, 16>;
  using VTOut = SIMDArray<TOut, 16>;
  constexpr size_t HSize = (8 / sizeof(TIn));
  const VTIn &V = Val.get<VTIn>();
  VTOut Result;
  for (size_t I = 0; I < HSize; ++I) {
    Result[I] = V[HSize + I];
  }
  Val.emplace<VTOut>(Result);
  return {};
}

template <typename TIn, typename TOut>
Expect<void> Executor::runVectorExtAddPairwiseOp(ValVariant &Val) const {
  static_assert(sizeof(TIn) * 2 == sizeof(TOut));
  using VTIn = SIMDArray<TIn, 16>;
  using VTOut = SIMDArray<TOut, 16>;

  VTOut Result;
  const VTIn &V = Val.get<VTIn>();
  for (size_t I = 0; I < (16 / sizeof(TOut)); ++I) {
    Result[I] = ((TOut)V[I*2]) + ((TOut)V[I*2+1]);
  }
  Val.emplace<VTOut>(Result);

  return {};
}

template <typename T>
Expect<void> Executor::runVectorAbsOp(ValVariant &Val) const {
  using VT = SIMDArray<T, 16>;
  VT &Result = Val.get<VT>();
  for (size_t I = 0; I < (16 / sizeof(T)); ++I) {
    Result[I] = Result[I] > 0 ? Result[I] : -Result[I];
  }
  return {};
}

template <typename T>
Expect<void> Executor::runVectorNegOp(ValVariant &Val) const {
  using VT = SIMDArray<T, 16>;
  VT &Result = Val.get<VT>();
  for (size_t I = 0; I < (16 / sizeof(T)); ++I) {
    Result[I] = -Result[I];
  }
  return {};
}

inline Expect<void> Executor::runVectorPopcntOp(ValVariant &Val) const {
  auto &Result = Val.get<uint8x16_t>();
  for(size_t I = 0; I < 16; ++I) {
    Result[I] -= ((Result[I] >> UINT8_C(1)) & UINT8_C(0x55));
    Result[I] = (Result[I] & UINT8_C(0x33)) + ((Result[I] >> UINT8_C(2)) & UINT8_C(0x33));
    Result[I] += Result[I] >> UINT8_C(4);
    Result[I] &= UINT8_C(0x0f);
  }  
  return {};
}

template <typename T>
Expect<void> Executor::runVectorSqrtOp(ValVariant &Val) const {
  using VT = SIMDArray<T, 16>;
  VT &Result = Val.get<VT>();
  if constexpr (sizeof(T) == 4) {
    Result = VT{std::sqrt(Result[0]), std::sqrt(Result[1]),
                std::sqrt(Result[2]), std::sqrt(Result[3])};
  } else if constexpr (sizeof(T) == 8) {
    Result = VT{std::sqrt(Result[0]), std::sqrt(Result[1])};
  }
  return {};
}

template <typename TIn, typename TOut>
Expect<void> Executor::runVectorTruncSatOp(ValVariant &Val) const {
  static_assert((sizeof(TIn) == 4 || sizeof(TIn) == 8) && sizeof(TOut) == 4);
  using VTIn = SIMDArray<TIn, 16>;
  using VTOut = SIMDArray<TOut, 16>;
  // const VTIn FMin = VTIn{} + static_cast<TIn>(std::numeric_limits<TOut>::min());
  // const VTIn FMax = VTIn{} + static_cast<TIn>(std::numeric_limits<TOut>::max());
  auto &V = Val.get<VTIn>();
  VTOut Result = {}; // all zero initialization for i32x4.trunc_sat_f64x2
  for (size_t I = 0; I < (16 / sizeof(TIn)); ++I) {
    if (std::isnan(V[I])) {
      Result[I] = 0;
    } else {
      TIn Tr = std::trunc(V[I]);
      if (Tr < static_cast<TIn>(std::numeric_limits<TOut>::min())) {
        Result[I] = std::numeric_limits<TOut>::min();
      } else if (Tr > static_cast<TIn>(std::numeric_limits<TOut>::max())) {
        Result[I] = std::numeric_limits<TOut>::max();
      } else {
        Result[I] = static_cast<TOut>(Tr);
      }
    }
  }
  Val.emplace<VTOut>(Result);
  return {};
}

template <typename TIn, typename TOut>
Expect<void> Executor::runVectorConvertOp(ValVariant &Val) const {
  static_assert((sizeof(TIn) == 4 && (sizeof(TOut) == 4 || sizeof(TOut) == 8)));
  using VTIn = SIMDArray<TIn, 16>;
  using VTOut = SIMDArray<TOut, 16>;
  auto &V = Val.get<VTIn>();
  // int32/uint32 to float
  if constexpr (sizeof(TIn) == sizeof(TOut)) {
    Val.emplace<VTOut>(VTOut{static_cast<TOut>(V[0]), static_cast<TOut>(V[1]), static_cast<TOut>(V[2]), static_cast<TOut>(V[3])});
  } else { // int32/uint32 to double
    Val.emplace<VTOut>(VTOut{static_cast<TOut>(V[0]), static_cast<TOut>(V[1])});
  }
  return {};
}

inline Expect<void> Executor::runVectorDemoteOp(ValVariant &Val) const {
  const auto V = Val.get<doublex2_t>();
  Val.emplace<floatx4_t>(
      floatx4_t{static_cast<float>(V[0]), static_cast<float>(V[1]), 0, 0});
  return {};
}

inline Expect<void> Executor::runVectorPromoteOp(ValVariant &Val) const {
  const auto V = Val.get<floatx4_t>();
  Val.emplace<doublex2_t>(
      doublex2_t{static_cast<double>(V[0]), static_cast<double>(V[1])});
  return {};
}

inline Expect<void> Executor::runVectorAnyTrueOp(ValVariant &Val) const {
  auto &Vector = Val.get<uint128_t>();
  const uint128_t Zero = 0;
  const uint32_t Result = (Vector != Zero);
  Val.emplace<uint32_t>(Result);

  return {};
}

template <typename T>
Expect<void> Executor::runVectorAllTrueOp(ValVariant &Val) const {
  using VT = SIMDArray<T, 16>;
  VT &V = Val.get<VT>();
  uint32_t Result;
  if constexpr (sizeof(T) == 1) {
    Result = V[0] != 0 && V[1] != 0 && V[2] != 0 && V[3] != 0 && V[4] != 0 && V[5] != 0 && V[6] != 0 && V[7] != 0 &&
             V[8] != 0 && V[9] != 0 && V[10] != 0 && V[11] != 0 && V[12] != 0 && V[13] != 0 && V[14] != 0 && V[15] != 0;
  } else if constexpr (sizeof(T) == 2) {
    Result = V[0] != 0 && V[1] != 0 && V[2] != 0 && V[3] != 0 && V[4] != 0 && V[5] != 0 && V[6] != 0 && V[7] != 0;
  } else if constexpr (sizeof(T) == 4) {
    Result = V[0] != 0 && V[1] != 0 && V[2] != 0 && V[3] != 0;
  } else if constexpr (sizeof(T) == 8) {
    Result = V[0] != 0 && V[1] != 0;
  }
  Val.emplace<uint32_t>(Result);

  return {};
}

template <typename T>
Expect<void> Executor::runVectorBitMaskOp(ValVariant &Val) const {
  using SVT = std::array<std::make_signed_t<T>, (16 / sizeof(std::make_signed_t<T>))>;
  using UVT = std::array<std::make_unsigned_t<T>, (16 / sizeof(std::make_unsigned_t<T>))>;
  SVT &Vector = Val.get<SVT>();
  if constexpr (sizeof(T) == 1) {
    using int16x16_t = SIMDArray<int16_t, 32>;
    using uint16x16_t = SIMDArray<uint16_t, 32>;
    const uint16x16_t Mask = {0x1,    0x2,    0x4,    0x8,   0x10,  0x20,
                              0x40,   0x80,   0x100,  0x200, 0x400, 0x800,
                              0x1000, 0x2000, 0x4000, 0x8000};
    uint16_t Result;
    for(size_t I = 0; I < 16; ++I) {
        Result |= Vector[I] < 0 ? Mask[I] : 0;
    }
    Val.emplace<uint32_t>(Result);
  } else if constexpr (sizeof(T) == 2) {
    const uint16x8_t Mask = {0x1, 0x2, 0x4, 0x8, 0x10, 0x20, 0x40, 0x80};
    using uint8x8_t = SIMDArray<uint8_t, 8>;
    uint8_t Result;
    for(size_t I = 0; I < 8; ++I) {
        Result |= Vector[I] < 0 ? Mask[I] : 0;
    }
    Val.emplace<uint32_t>(Result);
  } else if constexpr (sizeof(T) == 4) {
    const uint32x4_t Mask = {0x1, 0x2, 0x4, 0x8};
    using uint8x4_t = SIMDArray<uint8_t, 4>;
    uint8_t Result;
    for(size_t I = 0; I < 4; ++I) {
        Result |= Vector[I] < 0 ? Mask[I] : 0;
    }
    Val.emplace<uint32_t>(Result);
  } else if constexpr (sizeof(T) == 8) {
    const uint64x2_t Mask = {0x1, 0x2};
    using uint8x2_t = SIMDArray<uint8_t, 2>;
    uint8_t Result;
    for(size_t I = 0; I < 2; ++I) {
        Result |= Vector[I] < 0 ? Mask[I] : 0;
    }
    Val.emplace<uint32_t>(Result);
  }

  return {};
}

template <typename T>
Expect<void> Executor::runVectorCeilOp(ValVariant &Val) const {
  using VT = SIMDArray<T, 16>;
  VT &Result = Val.get<VT>();
  if constexpr (sizeof(T) == 4) {
    Result = VT{std::ceil(Result[0]), std::ceil(Result[1]),
                std::ceil(Result[2]), std::ceil(Result[3])};
  } else if constexpr (sizeof(T) == 8) {
    Result = VT{std::ceil(Result[0]), std::ceil(Result[1])};
  }
  return {};
}

template <typename T>
Expect<void> Executor::runVectorFloorOp(ValVariant &Val) const {
  using VT = SIMDArray<T, 16>;
  VT &Result = Val.get<VT>();
  if constexpr (sizeof(T) == 4) {
    Result = VT{std::floor(Result[0]), std::floor(Result[1]),
                std::floor(Result[2]), std::floor(Result[3])};
  } else if constexpr (sizeof(T) == 8) {
    Result = VT{std::floor(Result[0]), std::floor(Result[1])};
  }
  return {};
}

template <typename T>
Expect<void> Executor::runVectorTruncOp(ValVariant &Val) const {
  using VT = SIMDArray<T, 16>;
  VT &Result = Val.get<VT>();
  if constexpr (sizeof(T) == 4) {
    Result = VT{std::trunc(Result[0]), std::trunc(Result[1]),
                std::trunc(Result[2]), std::trunc(Result[3])};
  } else if constexpr (sizeof(T) == 8) {
    Result = VT{std::trunc(Result[0]), std::trunc(Result[1])};
  }
  return {};
}

template <typename T>
Expect<void> Executor::runVectorNearestOp(ValVariant &Val) const {
  using VT = SIMDArray<T, 16>;
  VT &Result = Val.get<VT>();
  if constexpr (sizeof(T) == 4) {
    Result = VT{WasmEdge::roundeven(Result[0]), WasmEdge::roundeven(Result[1]),
                WasmEdge::roundeven(Result[2]), WasmEdge::roundeven(Result[3])};
  } else if constexpr (sizeof(T) == 8) {
    Result = VT{WasmEdge::roundeven(Result[0]), WasmEdge::roundeven(Result[1])};
  }
  return {};
}

} // namespace Executor
} // namespace WasmEdge
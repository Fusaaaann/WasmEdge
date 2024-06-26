// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: 2019-2022 Second State INC

//===-- wasmedge/runtime/instance/struct.h - Struct Instance definition ---===//
//
// Part of the WasmEdge Project.
//
//===----------------------------------------------------------------------===//
///
/// \file
/// This file contains the struct instance definition in store manager.
///
//===----------------------------------------------------------------------===//
#pragma once

#include "ast/type.h"
#include "common/span.h"
#include "common/types.h"
#include "runtime/instance/composite.h"

#include <vector>

namespace WasmEdge {
namespace Runtime {
namespace Instance {

class StructInstance : public CompositeBase {
public:
  StructInstance() = delete;
  StructInstance(const ModuleInstance *Mod, const uint32_t Idx,
                 const uint32_t MemberCnt) noexcept
      : CompositeBase(Mod, Idx), RefCount(1),
        Data(MemberCnt, static_cast<uint128_t>(0)) {
    assuming(ModInst);
  }
  StructInstance(const ModuleInstance *Mod, const uint32_t Idx,
                 std::vector<ValVariant> &&Init) noexcept
      : CompositeBase(Mod, Idx), RefCount(1), Data(std::move(Init)) {
    assuming(ModInst);
  }

  /// Get field data in struct instance.
  ValVariant &getField(uint32_t Idx) noexcept { return Data[Idx]; }
  const ValVariant &getField(uint32_t Idx) const noexcept { return Data[Idx]; }

  /// Get reference count.
  uint32_t getRefCount() const noexcept { return RefCount; }

private:
  /// \name Data of struct instance.
  /// @{
  uint32_t RefCount;
  std::vector<ValVariant> Data;
  /// @}
};

} // namespace Instance
} // namespace Runtime
} // namespace WasmEdge

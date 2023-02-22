//===- RockGemmWrapperInterface.h - ops that wrap rock.gemm -*- C++ -*-===//
//
// Part of the rocMLIR Project, under the Apache License v2.0 with LLVM
// Exceptions. See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
// Copyright (c) 2022 Advanced Micro Devices INc.
//===----------------------------------------------------------------------===//
//
// This file defines RockGemmWrapperInterface, which abstracts convolutions and
// matrix multiplies to allow code to operate on them generically.
//
//===----------------------------------------------------------------------===//

#ifndef MLIR_DIALECT_ROCK_IR_ROCKGEMMWRAPPERINTERFACE_H
#define MLIR_DIALECT_ROCK_IR_ROCKGEMMWRAPPERINTERFACE_H

#include "triton/Dialect/Rock/IR/GemmSize.h"
#include "mlir/IR/OpDefinition.h"

#include "triton/Dialect/Rock/IR/RockTypes.h"

#include "triton/Dialect/Rock/IR/RockGemmWrapperInterface.h.inc"

#endif // MLIR_DIALECT_ROCK_IR_ROCKGEMMWRAPPERINTERFACE_H

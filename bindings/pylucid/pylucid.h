/**
 * @author Room 6.030
 * @copyright 2025 lucid
 * @licence BSD 3-Clause License
 * @file
 * pylucid main module.
 */
#pragma once

#ifndef LUCID_PYTHON_BUILD
#error LUCID_PYTHON_BUILD is not defined. Ensure you are building with the option '--config=py'
#endif

#include <pybind11/pybind11.h>

#define STRINGIFY(x) #x
#define MACRO_STRINGIFY(x) STRINGIFY(x)
#define REPR_LAMBDA(class_name) &class_name::to_string
#define STRING_LAMBDA(class_name) &class_name::to_string
#define GETTER(class_name, property) [](const class_name &self) { return self.property; }
#define SETTER(class_name, value_type, property) \
  [](class_name &self, value_type val) { self.property() = std::move(val); }

void init_model(pybind11::module_ &);
void init_util(pybind11::module_ &);
void init_verification(pybind11::module_ &);

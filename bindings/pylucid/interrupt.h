/**
 * @file interrupt.h
 * @author lucid_authors
 * @copyright 2025 lucid
 * @licence BSD 3-Clause License
 * Interrupt handler used by pylucid to handle any signal coming from the Python interpreter.
 */
#pragma once

#ifndef LUCID_PYTHON_BUILD
#error LUCID_PYTHON_BUILD is not defined. Ensure you are building with the option '--config=py'
#endif

namespace lucid {

/**
 * Check if the python interpreter has any flags that should interrupt the C++ execution.
 * Ctrl-C, along with other signals,
 * is received by the Python interpreter which holds it until the GIL is released.
 * To interrupt potentially long-running from inside your function,
 * we use the PyErr_CheckSignals() function, that will tell if a signal has been sent by the Python side.
 * If a signal is detected, a @verbatim py::error_already_set @endverbatim exception will be thrown.
 */
void py_check_signals();

}  // namespace lucid

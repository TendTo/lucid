/*
 @licstart  The following is the entire license notice for the JavaScript code in this file.

 The MIT License (MIT)

 Copyright (C) 1997-2020 by Dimitri van Heesch

 Permission is hereby granted, free of charge, to any person obtaining a copy of this software
 and associated documentation files (the "Software"), to deal in the Software without restriction,
 including without limitation the rights to use, copy, modify, merge, publish, distribute,
 sublicense, and/or sell copies of the Software, and to permit persons to whom the Software is
 furnished to do so, subject to the following conditions:

 The above copyright notice and this permission notice shall be included in all copies or
 substantial portions of the Software.

 THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR IMPLIED, INCLUDING
 BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY, FITNESS FOR A PARTICULAR PURPOSE AND
 NONINFRINGEMENT. IN NO EVENT SHALL THE AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM,
 DAMAGES OR OTHER LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
 OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE SOFTWARE.

 @licend  The above is the entire license notice for the JavaScript code in this file
*/
var NAVTREE =
[
  [ "lucid", "index.html", [
    [ "LUCID — Lifting-based Uncertain Control Invariant Dynamics", "index.html", "index" ],
    [ "Changelog", "md_docs_2Changelog.html", [
      [ "0.0.1", "md_docs_2Changelog.html#autotoc_md001", [
        [ "Added", "md_docs_2Changelog.html#added", null ]
      ] ]
    ] ],
    [ "Configuration", "md_docs_2Configuration.html", [
      [ "Command line options", "md_docs_2Configuration.html#command-line-options", null ],
      [ "Configuration file", "md_docs_2Configuration.html#configuration-file", null ],
      [ "Mixing configuration files and command line options", "md_docs_2Configuration.html#mixing-configuration-files-and-command-line-options", null ]
    ] ],
    [ "Contributing", "md_docs_2Contributing.html", [
      [ "Folder structure", "md_docs_2Contributing.html#folder-structure", null ],
      [ "Utility commands", "md_docs_2Contributing.html#utility-commands", null ],
      [ "Troubleshooting", "md_docs_2Contributing.html#troubleshooting", [
        [ "Bazel server stuck", "md_docs_2Contributing.html#bazel-server-stuck", null ]
      ] ]
    ] ],
    [ "Design", "md_docs_2Design.html", [
      [ "UML", "md_docs_2Design.html#uml", null ],
      [ "Pseudo Code", "md_docs_2Design.html#pseudo-code", null ],
      [ "Sequence Diagram", "md_docs_2Design.html#sequence-diagram", null ],
      [ "Idea", "md_docs_2Design.html#idea", null ]
    ] ],
    [ "Features", "md_docs_2Features.html", [
      [ "Legend", "md_docs_2Features.html#legend", null ],
      [ "Estimators", "md_docs_2Features.html#estimators", [
        [ "Definitions", "md_docs_2Features.html#definitions", null ],
        [ "Formulae", "md_docs_2Features.html#formulae", [
          [ "Kernel Ridge", "md_docs_2Features.html#kernel-ridge", [
            [ "Loss Function", "md_docs_2Features.html#loss-function", null ],
            [ "Prediction", "md_docs_2Features.html#prediction", null ]
          ] ]
        ] ]
      ] ],
      [ "Supported Kernels", "md_docs_2Features.html#supported-kernels", null ]
    ] ],
    [ "Installation", "md_docs_2Installation.html", [
      [ "From Docker", "md_docs_2Installation.html#from-docker", [
        [ "Requirements", "md_docs_2Installation.html#requirements", null ],
        [ "Using Lucid with Docker", "md_docs_2Installation.html#using-lucid-with-docker", null ]
      ] ],
      [ "From source", "md_docs_2Installation.html#from-source", [
        [ "Requirements", "md_docs_2Installation.html#requirements-1", null ],
        [ "Gurobi requirements", "md_docs_2Installation.html#gurobi-requirements", null ],
        [ "Building Lucid", "md_docs_2Installation.html#building-lucid", null ],
        [ "Build options", "md_docs_2Installation.html#build-options", null ]
      ] ]
    ] ],
    [ "Legend", "md_docs_2Legend.html", [
      [ "Math symbols", "md_docs_2Legend.html#math-symbols", [
        [ "Basic", "md_docs_2Legend.html#basic", null ],
        [ "Probability", "md_docs_2Legend.html#probability", null ],
        [ "Control", "md_docs_2Legend.html#control", null ],
        [ "Specific symbols", "md_docs_2Legend.html#specific-symbols", null ]
      ] ]
    ] ],
    [ "Pylucid", "md_docs_2Pylucid.html", [
      [ "Installing Pylucid", "md_docs_2Pylucid.html#installing-pylucid", [
        [ "Building on Windows", "md_docs_2Pylucid.html#building-on-windows", null ]
      ] ],
      [ "Use", "md_docs_2Pylucid.html#use", null ],
      [ "Troubleshooting", "md_docs_2Pylucid.html#troubleshooting-1", null ]
    ] ],
    [ "Testing", "md_docs_2Testing.html", [
      [ "Running the tests", "md_docs_2Testing.html#running-the-tests", null ]
    ] ],
    [ "To do", "md_docs_2Todo.html", [
      [ "Graphical overview", "md_docs_2Todo.html#graphical-overview", null ],
      [ "Work package 1", "md_docs_2Todo.html#work-package-1", [
        [ "Activity 1", "md_docs_2Todo.html#activity-1", null ],
        [ "Activity 2", "md_docs_2Todo.html#activity-2", null ],
        [ "Activity 3", "md_docs_2Todo.html#activity-3", null ],
        [ "Activity 4", "md_docs_2Todo.html#activity-4", null ]
      ] ],
      [ "Extensions", "md_docs_2Todo.html#extensions", [
        [ "Extension 3", "md_docs_2Todo.html#extension-3", null ],
        [ "Extension 4", "md_docs_2Todo.html#extension-4", null ],
        [ "Extension 8", "md_docs_2Todo.html#extension-8", null ]
      ] ],
      [ "Miscellaneous", "md_docs_2Todo.html#miscellaneous", null ],
      [ "Adding new features", "md_docs_2Todo.html#adding-new-features", [
        [ "Performance", "md_docs_2Todo.html#performance", null ],
        [ "Documentation", "md_docs_2Todo.html#documentation", null ],
        [ "Distribution", "md_docs_2Todo.html#distribution", null ]
      ] ],
      [ "Tests", "md_docs_2Todo.html#tests", null ]
    ] ],
    [ "Todo List", "todo.html", null ],
    [ "Namespaces", "namespaces.html", [
      [ "Namespace List", "namespaces.html", "namespaces_dup" ],
      [ "Namespace Members", "namespacemembers.html", [
        [ "All", "namespacemembers.html", null ],
        [ "Functions", "namespacemembers_func.html", null ],
        [ "Variables", "namespacemembers_vars.html", null ],
        [ "Typedefs", "namespacemembers_type.html", null ],
        [ "Enumerations", "namespacemembers_enum.html", null ]
      ] ]
    ] ],
    [ "Concepts", "concepts.html", "concepts" ],
    [ "Classes", "annotated.html", [
      [ "Class List", "annotated.html", "annotated_dup" ],
      [ "Class Index", "classes.html", null ],
      [ "Class Hierarchy", "hierarchy.html", "hierarchy" ],
      [ "Class Members", "functions.html", [
        [ "All", "functions.html", "functions_dup" ],
        [ "Functions", "functions_func.html", "functions_func" ],
        [ "Variables", "functions_vars.html", null ],
        [ "Typedefs", "functions_type.html", null ]
      ] ]
    ] ],
    [ "Files", "files.html", [
      [ "File List", "files.html", "files_dup" ],
      [ "File Members", "globals.html", [
        [ "All", "globals.html", null ],
        [ "Functions", "globals_func.html", null ],
        [ "Macros", "globals_defs.html", null ]
      ] ]
    ] ]
  ] ]
];

var NAVTREEINDEX =
[
"AlglibOptimiser_8cpp.html",
"classlucid_1_1IndexIterator.html#ab3779d1404c80df372f7be3f5bd7eb13",
"classlucid_1_1Parametrizable.html#a2fc70f2cf975cd9eb7525a65ac5b3a24",
"classlucid_1_1TruncatedFourierFeatureMap.html#a0a4328787fa1a57eaf995f8d9441423c",
"namespacelucid.html#a69ac3a9147e3eda683ff29b7292dc98f"
];

var SYNCONMSG = 'click to disable panel synchronization';
var SYNCOFFMSG = 'click to enable panel synchronization';
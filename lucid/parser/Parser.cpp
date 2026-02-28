#include "lucid/parser/Parser.h"

#include <fstream>
#include <string>

#include "lucid/util/error.h"

namespace lucid {

void Parser::parse_file(const std::string& filename) const {
  std::ifstream file{filename};
  if (!file.is_open()) {
    LUCID_INVALID_ARGUMENT("filename", fmt::format("Could not open file: {}", filename));
  }
  // Calculate file size
  file.seekg(0, std::istream::end);
  std::size_t size(static_cast<size_t>(file.tellg()));
  // Restore file position and read content
  file.seekg(0, std::istream::beg);
  // Allocate the string to hold the file content
  std::string result(size, 0);
  file.read(&result[0], size);
  // Parse the file content
  parse_input(result);
}

void Parser::parse_input(const std::string& input) const { parse_input_impl(input); }

}  // namespace lucid

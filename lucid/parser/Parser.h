#include <string>

namespace lucid {

class Parser {
 public:
  Parser() = default;
  virtual ~Parser() = default;

  void parse_input(const std::string& input) const;
  void parse_file(const std::string& filename) const;

 protected:
  virtual void parse_input_impl(const std::string& input) const = 0;
};

}  // namespace lucid

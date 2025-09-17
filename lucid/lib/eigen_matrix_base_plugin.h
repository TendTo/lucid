// NOLINT(build/header_guard): This is supposed to be included in the middle of the Matrix class
/**
 * @author Room 6.030
 * @copyright 2025 lucid
 * @licence BSD 3-Clause License
 * @file
 * Plugin for Eigen matrix class which adds some utility methods.
 */
/**
 * Write the matrix to a file.
 * The file will be created if it does not exist and existing files will be overwritten.
 * @param fileName name of the file to write
 */
inline bool write(const std::string_view fileName) const {
  std::ofstream out_file(fileName.data());
  if (!out_file.is_open()) return false;
  out_file << this->rows() << 'X' << this->cols() << '\n';
  std::ranges::copy(this->reshaped(), std::ostream_iterator<Scalar>(out_file, ","));
  return true;
}

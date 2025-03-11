inline bool write(const std::string_view fileName) const {
  std::ofstream out_file(fileName.data());
  if (!out_file.is_open()) return false;
  out_file << this->rows() << 'X' << this->cols() << '\n';
  std::ranges::copy(this->reshaped(), std::ostream_iterator<Scalar>(out_file, ","));
  return true;
}

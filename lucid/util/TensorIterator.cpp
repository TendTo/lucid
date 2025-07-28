/**
 * @author lucid_authors
 * @copyright 2025 lucid
 * @licence BSD 3-Clause License
 * @file
 */
#include "lucid/util/TensorIterator.h"

#include <ostream>
#include <vector>

#include "lucid/util/TensorView.h"

namespace lucid {

namespace {
std::vector<Index> to_indexes(const std::vector<std::size_t>& dims) {
  std::vector<Index> indexes(dims.size());
  for (std::size_t i = 0; i < indexes.size(); ++i) {
    indexes[i] = static_cast<Index>(dims[i]);
    if (indexes[i] == 0) return {0};
  }
  return indexes;
}
}  // namespace

template <IsAnyOf<int, float, double, std::complex<double>> T>
TensorIterator<T>::TensorIterator(const TensorView<T>& tensor, const bool end)
    : indexes_{end ? std::vector<Index>{0} : to_indexes(tensor.dimensions())}, tensor_{tensor} {}

template <IsAnyOf<int, float, double, std::complex<double>> T>
const T& TensorIterator<T>::operator*() const {
  return tensor_(indexes());
}

template <IsAnyOf<int, float, double, std::complex<double>> T>
typename TensorIterator<T>::pointer TensorIterator<T>::operator->() const {
  return const_cast<T*>(&tensor_(indexes()));
}

template <IsAnyOf<int, float, double, std::complex<double>> T>
TensorIterator<T>& TensorIterator<T>::operator++() {
  ++indexes_;
  return *this;
}

template <IsAnyOf<int, float, double, std::complex<double>> T>
bool TensorIterator<T>::operator==(const TensorIterator<T>& o) const {
  if (&tensor_ != &o.tensor_) return false;
  return (!*this && !o);
}

template <IsAnyOf<int, float, double, std::complex<double>> T>
std::ostream& operator<<(std::ostream& os, const TensorIterator<T>& it) {
  return os << "TensorIterator( " << it.index_iterator() << " " << *it << " )";
}

template class TensorIterator<int>;
template class TensorIterator<float>;
template class TensorIterator<double>;
template class TensorIterator<std::complex<double>>;
template std::ostream& operator<<(std::ostream& os, const TensorIterator<int>& tensor);
template std::ostream& operator<<(std::ostream& os, const TensorIterator<float>& tensor);
template std::ostream& operator<<(std::ostream& os, const TensorIterator<double>& tensor);
template std::ostream& operator<<(std::ostream& os, const TensorIterator<std::complex<double>>& tensor);

}  // namespace lucid

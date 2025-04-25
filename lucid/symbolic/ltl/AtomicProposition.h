/**
 * @author Room 6.030
 * @copyright 2025 lucid
 * @licence BSD 3-Clause License
 * @file
 * AtomicProposition class.
 */
#pragma once

#include <string>
#include <vector>

namespace lucid::ltl {

class AtomicProposition {
 public:
  using Id = std::size_t;

  const static Id dummy_id;  ///< ID of the dummy atomic proposition.

  /**
   * Construct a new dummy atomic proposition object.
   *
   * The default constructor is needed to support some data structures.
   * The objects created by the default constructor share the same ID, `std::numeric_limits<Id>::max()`.
   * As a result, they all are identified as a single atomic proposition by equality operator (==) and have the same
   * hash value as well. It is allowed to construct a dummy atomic proposition, but it should not be used to construct a
   * symbolic expression.
   */
  AtomicProposition() : id_{dummy_id} {}
  /**
   * Construct a new real atomic proposition object, assigning it a `name`.
   *
   * It will be given a unique incremental ID.
   * @param name name of the atomic proposition
   */
  explicit AtomicProposition(std::string name);
  /**
   * Construct a new real atomic proposition object, but instead of creating a new one, it will use the given `id`,
   * effectively "connecting" the object to an existing atomic proposition.
   * @pre `id` must have benn assigned to a atomic proposition before by the @ref GetNextId method.
   * @param id unique identifier
   */
  explicit AtomicProposition(Id id);

  /** @checker{a dummy\, i.e. has been created with the default constructor, atomic proposition} */
  [[nodiscard]] bool is_dummy() const { return id_ == dummy_id; }
  /** @getter{id, atomic proposition} */
  [[nodiscard]] Id id() const { return id_; }
  /** @getter{name, atomic proposition} */
  [[nodiscard]] const std::string &name() const { return names_[id_ + 1]; }

  /** @equal_to{atomic proposition,
   * Two atomic propositions are the same if their @ref id_ is the same, regardless of their name.} */
  [[nodiscard]] bool equal_to(const AtomicProposition &o) const noexcept { return id_ == o.id_; }
  /** @less{atomic proposition, The ordering is based on the ID of the atomic proposition.} */
  [[nodiscard]] bool less(const AtomicProposition &o) const noexcept { return id_ < o.id_; }
  /** @hash{atomic proposition, The hash is based on the ID of the atomic proposition.} */
  [[nodiscard]] size_t hash() const noexcept { return std::hash<Id>{}(id_); }

 private:
  static std::vector<std::string> names_;  ///< Names of all existing atomic propositions.
  /**
   * Get the next unique identifier for a atomic proposition.
   * @return incremental unique identifier
   */
  static Id GetNextId();

  Id id_{};  ///< Unique identifier.
};

std::ostream &operator<<(std::ostream &os, const AtomicProposition &var);

}  // namespace lucid::ltl

template <>
struct std::hash<lucid::ltl::AtomicProposition> {
  size_t operator()(const lucid::ltl::AtomicProposition &v) const noexcept { return v.hash(); }
};

template <>
struct std::less<lucid::ltl::AtomicProposition> {
  bool operator()(const lucid::ltl::AtomicProposition &lhs, const lucid::ltl::AtomicProposition &rhs) const noexcept {
    return lhs.less(rhs);
  }
};

template <>
struct std::equal_to<lucid::ltl::AtomicProposition> {
  bool operator()(const lucid::ltl::AtomicProposition &lhs, const lucid::ltl::AtomicProposition &rhs) const noexcept {
    return lhs.equal_to(rhs);
  }
};

#ifdef LUCID_INCLUDE_FMT

#include "lucid/util/logging.h"

OSTREAM_FORMATTER(lucid::ltl::AtomicProposition);

#endif

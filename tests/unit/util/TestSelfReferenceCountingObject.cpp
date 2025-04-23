/**
 * @author Room 6.030
 * @copyright 2025 lucid
 * @licence BSD 3-Clause License
 */
#include <gtest/gtest.h>

#include <iostream>

#include "lucid/util/SelfReferenceCountingObject.h"
#include "lucid/util/intrusive_ptr.h"

using lucid::intrusive_ptr;
using lucid::SelfReferenceCountingObject;

class TestObject : public SelfReferenceCountingObject {
 public:
  static intrusive_ptr<TestObject> instantiate() { return intrusive_ptr<TestObject>(new TestObject()); }
  int fest();
};

TEST(TestSelfReferenceCountingObject, ReferenceCountIncrementsAndDecrementsCorrectly) {
  const intrusive_ptr<TestObject> obj = TestObject::instantiate();
  EXPECT_EQ(obj->use_count(), 1);

  obj->add_ref();
  EXPECT_EQ(obj->use_count(), 2);

  obj->release();
  EXPECT_EQ(obj->use_count(), 1);

  obj->release();
}

TEST(TestSelfReferenceCountingObject, CopyConstructor) {
  const intrusive_ptr<TestObject> obj1 = TestObject::instantiate();
  EXPECT_EQ(obj1->use_count(), 1);

  intrusive_ptr<TestObject> obj2{obj1};  // Copy constructor
  EXPECT_EQ(obj1->use_count(), 2);
  EXPECT_EQ(obj2->use_count(), 2);

  obj2->release();
  EXPECT_EQ(obj1->use_count(), 1);
}

TEST(TestSelfReferenceCountingObject, CopyAssignment) {
  const intrusive_ptr<TestObject> obj1 = TestObject::instantiate();
  EXPECT_EQ(obj1->use_count(), 1);

  intrusive_ptr<TestObject> obj2 = TestObject::instantiate();
  EXPECT_EQ(obj2->use_count(), 1);

  obj2 = obj1;  // Copy assignment
  EXPECT_EQ(obj1->use_count(), 2);
  EXPECT_EQ(obj2->use_count(), 2);

  obj2->release();
  EXPECT_EQ(obj1->use_count(), 1);
}

TEST(TestSelfReferenceCountingObject, MoveConstructor) {
  intrusive_ptr<TestObject> obj1 = TestObject::instantiate();
  EXPECT_EQ(obj1->use_count(), 1);

  const intrusive_ptr<TestObject> obj2{std::move(obj1)};  // Move constructor
  EXPECT_EQ(obj2->use_count(), 1);
  EXPECT_EQ(obj1, nullptr);  // obj1 should be null after move
}

TEST(TestSelfReferenceCountingObject, MoveAssignment) {
  intrusive_ptr<TestObject> obj1 = TestObject::instantiate();
  EXPECT_EQ(obj1->use_count(), 1);

  intrusive_ptr<TestObject> obj2 = TestObject::instantiate();
  EXPECT_EQ(obj2->use_count(), 1);

  obj2 = std::move(obj1);  // Move assignment
  EXPECT_EQ(obj2->use_count(), 1);
  EXPECT_EQ(obj1, nullptr);  // obj1 should be null after move
}

TEST(TestSelfReferenceCountingObject, ObjectDestructionOnZeroReferenceCount) {
  const intrusive_ptr<TestObject> obj = TestObject::instantiate();
  EXPECT_EQ(obj->use_count(), 1);

  obj->release();
}

TEST(TestSelfReferenceCountingObject, MultipleReferences) {
  const intrusive_ptr<TestObject> obj1 = TestObject::instantiate();
  EXPECT_EQ(obj1->use_count(), 1);

  obj1->add_ref();
  EXPECT_EQ(obj1->use_count(), 2);

  obj1->add_ref();
  EXPECT_EQ(obj1->use_count(), 3);

  obj1->release();
  EXPECT_EQ(obj1->use_count(), 2);

  obj1->release();
  EXPECT_EQ(obj1->use_count(), 1);

  obj1->release();  // This should delete the object
}

TEST(TestSelfReferenceCountingObject, ThreadSafety) {
#ifdef LUCID_THREAD_SAFE
  intrusive_ptr<TestObject> obj = TestObject::instantiate();
  EXPECT_EQ(obj->use_count(), 1);

  std::thread t1([&]() { obj->AddRef(); });
  std::thread t2([&]() { obj->AddRef(); });

  t1.join();
  t2.join();

  EXPECT_EQ(obj->use_count(), 3);

  std::thread t3([&]() { obj->Release(); });
  std::thread t4([&]() { obj->Release(); });

  t3.join();
  t4.join();

  EXPECT_EQ(obj->use_count(), 1);

  obj->Release();  // This should delete the object
#endif
}

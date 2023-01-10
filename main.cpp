#include "iostream"
#include "eigen3/Eigen/Dense"

class Op {
  Eigen::Vector4f v{1, 1.5, 2, 3};
  Eigen::Vector4f w{0, 1, 2, 4};
  Eigen::Matrix4i i{{1, 2, 3, 4}, {5, 6, 7, 8}, {9, 10, 11, 12}, {13, 14, 15, 16}};
  Eigen::Matrix4i j{{4, 3, 2, 1}, {8, 7, 6, 5}, {12, 11, 10, 9}, {16, 15, 14, 13}};

 public:
  void res1() {
    std::cout << v << std::endl;
    std::cout << w << std::endl;
    std::cout << v + w << std::endl;
    std::cout << v.dot(w) << std::endl;
    std::cout << v.cross(w) << std::endl;
  }

  void res2() {
    std::cout << i << std::endl;
    std::cout << j << std::endl;
    std::cout << i + j << std::endl;
    std::cout << i * j << std::endl;
    std::cout << i * v << std::endl;
  }
};

int main() {
  Op res{};
  res.res1();
  res.res2();
  return 0;
}

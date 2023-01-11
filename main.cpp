#include "iostream"

#include "eigen3/Eigen/Core"
#include "eigen3/Eigen/SVD"
#include "eigen3/Eigen/Geometry"

#include "opencv4/opencv2/core.hpp"
#include "opencv4/opencv2/core/eigen.hpp"
#include "opencv4/opencv2/imgcodecs.hpp"

#define GRAYSCALE_MAX 255
#define LENNA_PATH "/run/media/luke/Data/CSC4140/CSC4140_Proj1/lenna.png"
#define IMG1 "/run/media/luke/Data/CSC4140/CSC4140_Proj1/120090645_HW_01.assets/res1.png"
#define IMG2 "/run/media/luke/Data/CSC4140/CSC4140_Proj1/120090645_HW_01.assets/res2.png"
#define IMG3 "/run/media/luke/Data/CSC4140/CSC4140_Proj1/120090645_HW_01.assets/res3.png"

using namespace std;
class Op {
  Eigen::Vector4f v{1, 1.5, 2, 3};
  Eigen::Vector4f w{0, 1, 2, 4};
  Eigen::Matrix4f i{{1, 2, 3, 4}, {5, 6, 7, 8}, {9, 10, 11, 12}, {13, 14, 15, 16}};
  Eigen::Matrix4f j{{4, 3, 2, 1}, {8, 7, 6, 5}, {12, 11, 10, 9}, {16, 15, 14, 13}};

 public:
  void res1() {
    cout << v << endl;
    cout << w << endl;
    cout << v + w << endl;
    cout << v.dot(w) << endl;
  }

  void res2() {
    cout << i << endl;
    cout << j << endl;
    cout << i + j << endl;
    cout << i * j << endl;
    cout << i * v << endl;
  }
};

using namespace cv;
class opencv {
  std::string image_path = samples::findFile(LENNA_PATH);
  Mat img = imread(image_path, IMREAD_GRAYSCALE);
  Eigen::MatrixXd matrix;

 public:
  opencv() {
    cv2eigen(img, matrix);
    matrix /= GRAYSCALE_MAX;
  }

  void res() {
    Eigen::BDCSVD<Eigen::MatrixXd> bdcsvd(matrix, Eigen::ComputeFullU | Eigen::ComputeFullV);
    auto U = bdcsvd.matrixU();
    auto singularValues = bdcsvd.singularValues();
    auto V = bdcsvd.matrixV();

    Mat resImg1, resImg2, resImg3;

    Eigen::MatrixXd S1(512, 1);
    for (auto i = 0; i < 512; ++i) {
      S1(i, 0) = 0;
    }
    S1(0, 0) = singularValues(0);

    Eigen::MatrixXd res1 = U * S1 * V.leftCols(1).transpose() * GRAYSCALE_MAX;
    eigen2cv(res1, resImg1);
    imwrite(IMG1, resImg1);

    Eigen::MatrixXd S2(512, 10);
    for (auto i = 0; i < 512; ++i) {
      for (auto j = 0; j < 10; ++j) {
        if (i == j) {
          S2(i, j) = singularValues(i);
        } else {
          S2(i, j) = 0;
        }
      }
    }
    Eigen::MatrixXd res2 = U * S2 * V.leftCols(10).transpose() * GRAYSCALE_MAX;
    eigen2cv(res2, resImg2);
    imwrite(IMG2, resImg2);

    Eigen::MatrixXd S3(512, 50);
    for (auto i = 0; i < 512; ++i) {
      for (auto j = 0; j < 50; ++j) {
        if (i == j) {
          S3(i, j) = singularValues(i);
        } else {
          S3(i, j) = 0;
        }
      }
    }
    Eigen::MatrixXd res3 = U * S3 * V.leftCols(50).transpose() * GRAYSCALE_MAX;
    eigen2cv(res3, resImg3);
    imwrite(IMG3, resImg3);
  }
};

using namespace Eigen;
class trans {
  Vector3f p1{1, 2, 3};
  Vector3f p2{4, 5, 6};
 public:
  void res() {
    p2.normalize();
    auto t1 = AngleAxisf(0.25 * M_PI, p2);
    auto t2 = AngleAxisf(float(1) / float(6) * M_PI, p2);
    auto t3 = AngleAxisf(float(1) / float(3) * M_PI, p2);

    std::cout << t1 * p1 << std::endl;
    std::cout << t2 * p1 << std::endl;
    std::cout << t3 * p1 << std::endl;
  }
};

int main() {
  Op res{};
  res.res1();
  res.res2();

  opencv oc{};
  oc.res();

  trans trans{};
  trans.res();

  return 0;
}

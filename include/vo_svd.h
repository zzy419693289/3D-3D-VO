#ifndef VO_SVD_H
#define VO_SVD_H

#include <opencv2/core/core.hpp> 

void pose_svd_3d3d(const std::vector<cv::Point3f>& pts1, const std::vector<cv::Point3f>& pts2, cv::Mat& R, cv::Mat& t, cv::Vec3d& Eal);

#endif

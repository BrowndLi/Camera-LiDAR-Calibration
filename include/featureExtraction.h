#define PCL_NO_PRECOMPILE

//boost相关头文件
#include <boost/thread/thread.hpp>
#include <boost/make_shared.hpp>

//pcl相关头文件
#include <pcl/registration/ia_ransac.h> //点云的ransac算法头文件
#include <pcl/point_types.h>
#include <pcl/point_cloud.h>
#include <pcl/io/pcd_io.h> //pcd输入输出头文件
#include <pcl/registration/icp.h> //点云icp算法头文件
#include <pcl/visualization/pcl_visualizer.h> //点云可视化头文件
#include <pcl/filters/conditional_removal.h> //条件滤波器头文件
#include <pcl/point_cloud.h>
#include <pcl/ModelCoefficients.h>
#include <pcl/sample_consensus/method_types.h>
#include <pcl/sample_consensus/model_types.h>
#include <pcl/segmentation/sac_segmentation.h>
#include <pcl/segmentation/impl/sac_segmentation.hpp>
#include <pcl/filters/extract_indices.h>
#include <pcl/filters/impl/extract_indices.hpp>
#include <pcl/filters/project_inliers.h>
#include <pcl/filters/impl/project_inliers.hpp>
#include <pcl/common/common_headers.h>
#include <pcl/common/intersections.h>
#include <pcl/filters/passthrough.h>
#include <pcl/filters/impl/passthrough.hpp>
#include <pcl/registration/ndt.h>
#include <pcl/filters/approximate_voxel_grid.h>
#include <pcl/registration/transformation_estimation_svd.h>
#include <pcl/recognition/ransac_based/auxiliary.h>
#include <pcl/recognition/ransac_based/trimmed_icp.h>
#include <pcl/filters/voxel_grid.h>
#include <pcl/common/transforms.h>
#include <pcl/io/io.h>
#include <pcl/console/time.h>
#include <pcl/visualization/cloud_viewer.h> 
#include <pcl/features/boundary.h>
#include <pcl/features/normal_3d.h>


//opencv相关头文件
#include <opencv2/imgproc/imgproc.hpp>
#include <opencv2/core/core.hpp>
#include <opencv/cv.hpp>
#include <opencv2/calib3d.hpp>
#include <opencv2/highgui/highgui.hpp>
#include "opencv2/aruco.hpp"

//常用头文件
#include <iostream>
#include <vector>
#include <cmath>
#include <string>
#include <fstream>
#include <utility>
#include <sstream> //istringstream 必须包含这个头文件
#include <deque> 

using namespace std;
using namespace cv;

struct initial_parameters {
   string srcImg_path; //标定用图片的相对路径
   string srclidarPoints_path; //标定用点云数据的相对路径
   string log_txt_path; //图像log文件的相对路径
   string img_projection_path; //投影图像的保存路径
   string calibration_result_path; //标定结果保存文件的相对路径
   int lidar_ring_count; //lidar的线数
   std::pair<int, int> grid_size; //棋盘格的角点数
   int square_length; // 棋盘格每个小格子的实际尺寸,单位：mm
   std::pair<int, int> board_dimension; // 整个标定板的实际尺寸(宽，高)，单位：mm
   cv::Mat cameramat; //相机内参
   cv::Mat distcoeff; //相机的畸变系数
   std::pair<float, float> plane_line_dist_threshold; //ransac平面拟合和直线拟合的距离阈值
   std::pair<float, float> x_range_point_filter; //点云滤波x值范围
   std::pair<float, float> y_range_point_filter; //点云滤波y值范围
   std::pair<float, float> z_range_point_filter; //点云滤波z值范围
   float line_fit2real_dist; //求出的标定板的四个角点构成的矩形边长与真值的差值阈值，单位：m
} i_params;

struct PointXYZIR
{
  PCL_ADD_POINT4D;                    
  float    intensity;                 
  uint16_t ring;                      
  EIGEN_MAKE_ALIGNED_OPERATOR_NEW     
}EIGEN_ALIGN16;

POINT_CLOUD_REGISTER_POINT_STRUCT(PointXYZIR,
  (float, x, x)
  (float, y, y)
  (float, z, z)
  (float, intensity, intensity)
  (uint16_t, ring, ring))
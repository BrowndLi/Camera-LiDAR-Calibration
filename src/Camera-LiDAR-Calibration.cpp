/******  程序从此处开始，1 实现激光点云ROI区域(标定板)的选取； 2 标定板平面的拟合。   ******/
/******  3 图像标定板的特征提取； 4 激光点云标定板的特征提取。   ******/

//写在开头：世界坐标系的单位：m; 相机坐标系的单位：m; 图像坐标系的单位：mm; 像素坐标系的单位：pixel

#include "include/featureExtraction.h" //链接本程序头文件

std::vector<Point2f> all_img_board_coenerAndCenter; // (所有帧)标定板四个角点和中心点的图像坐标,单位：像素
std::vector<Point3f> all_lidar_board_coenerAndCenter; // (所有帧)标定板四个角点和中心点在激光坐标系下的坐标，单位：mm

std::vector<Point2f> img_board_coenerAndCenter; // (对应帧)标定板四个角点和中心点的图像坐标,单位：像素
std::vector<Point3f> lidar_board_coenerAndCenter; // (对应帧)标定板四个角点和中心点在激光坐标系下的坐标，单位：mm

// cv::Mat R_camToLidar = (Mat_<double>(3, 1) << 1.25276082345453, -1.193519093120378, 1.210215014689938); //激光雷达坐标系到相机坐标系的旋转向量R,roll、pitch、yaw
// cv::Mat T_camToLidar = (Mat_<double>(3, 1) << -72.5646427842064, -698.275374750283, -722.5815488873395); //激光雷达坐标系到相机坐标系的平行向量T，单位mm
cv::Mat R_camToLidar; //激光雷达坐标系到相机坐标系的旋转向量R,roll、pitch、yaw
cv::Mat T_camToLidar; //激光雷达坐标系到相机坐标系的平行向量T，单位mm

std::vector<std::vector<cv::Point3f>> chessBoard_lidar_points; // 用于存放标定板点云的vector，单位:mm

/******   点云可视化1   ******/
void visualize_pcd(pcl::PointCloud<PointXYZIR>::Ptr &pcd_src)
{

   pcl::visualization::PCLVisualizer viewer("registration Viewer");
   pcl::visualization::PointCloudColorHandlerCustom<PointXYZIR> src_h (pcd_src, 0, 255, 0);
   viewer.addPointCloud (pcd_src, src_h, "source cloud");    //绿色

   while (!viewer.wasStopped())
   {
      //  viewer.spinOnce(100);
      //  boost::this_thread::sleep(boost::posix_time::microseconds(100000));
       viewer.spinOnce();
   }
}

/******   点云可视化2   ******/
void visualize_pcd2(pcl::PointCloud<PointXYZIR>::Ptr &pcd_src, pcl::PointCloud<PointXYZIR>::Ptr &pcd_src2)
{

   pcl::visualization::PCLVisualizer viewer("registration Viewer");
   pcl::visualization::PointCloudColorHandlerCustom<PointXYZIR> src_h (pcd_src, 0, 255, 0);
   pcl::visualization::PointCloudColorHandlerCustom<PointXYZIR> src_h2 (pcd_src2, 255, 0, 0);
   viewer.addPointCloud (pcd_src, src_h, "source cloud");
   viewer.addPointCloud (pcd_src2, src_h2, "source cloud2");

   while (!viewer.wasStopped())
   {
      //  viewer.spinOnce(100);
      //  boost::this_thread::sleep(boost::posix_time::microseconds(100000));
       viewer.spinOnce();
   }
}

/******   文件夹文件读取      ******/
void GetFile(string path,vector<string>& file)
{

    char buf1[1000];
    DIR *pDir;
    struct dirent* ptr;
    if(!(pDir = opendir(path.c_str())))
        return;
    
    while((ptr = readdir(pDir))!=0)
    {
        if (strcmp(ptr->d_name, ".") != 0 && strcmp(ptr->d_name, "..") != 0)
        {
            file.push_back(ptr->d_name);            
        }   
    }

    sort(file.begin(),file.end());

    closedir(pDir);
}

/******   数据输入函数   ******/
void onInit(){
   /* 从配置文件输入参数 */
   std::ifstream infile("./parameter/initial_params.txt");

   infile >> i_params.srcImg_path;             //标定用图片的相对路径
   infile >> i_params.srclidarPoints_path;     //标定用激光点云数据的相对路径
   infile >> i_params.log_txt_path;            //图像log文件的相对路径
   infile >> i_params.img_projection_path;     //投影图像所在的相对路径
   infile >> i_params.calibration_result_path; //标定结果保存文件的相对路径

   infile >> i_params.lidar_ring_count; //lidar的线数

   int width_grid_count, height_grid_size;
   infile >> width_grid_count;
   infile >> height_grid_size;
   i_params.grid_size = std::make_pair(width_grid_count, height_grid_size); //棋盘格角点数目(宽和高方向)

   infile >> i_params.square_length; //棋盘格小方格的尺寸，mm

   int board_width, board_height;
   infile >> board_width;
   infile >> board_height; 
   i_params.board_dimension = std::make_pair(board_width, board_height); //棋盘格实际尺寸(宽和高方向)
   
   double camera_mat[9];
   for (int i = 0; i < 9; i++) {
      infile >> camera_mat[i];
   }
   cv::Mat(3, 3, CV_64F, &camera_mat).copyTo(i_params.cameramat); //相机内参

   double dist_coeff[5];
   for (int i = 0; i < 5; i++) {
      infile >> dist_coeff[i];
   }
   cv::Mat(1, 5, CV_64F, &dist_coeff).copyTo(i_params.distcoeff);

   double plane_dist_threshold, line_dist_threshold;
   infile >> plane_dist_threshold;
   infile >> line_dist_threshold;
   i_params.plane_line_dist_threshold = std::make_pair(plane_dist_threshold, line_dist_threshold); //ransac平面拟合和直线拟合的距离阈值

   double x_range_down, x_range_up;
   infile >> x_range_down;
   infile >> x_range_up;
   i_params.x_range_point_filter = std::make_pair(x_range_down, x_range_up); //点云滤波x范围：down < x <up
   double y_range_down, y_range_up;
   infile >> y_range_down;
   infile >> y_range_up;
   i_params.y_range_point_filter = std::make_pair(y_range_down, y_range_up); //点云滤波y范围：down < y <up
   double z_range_down, z_range_up;
   infile >> z_range_down;
   infile >> z_range_up;
   i_params.z_range_point_filter = std::make_pair(z_range_down, z_range_up); //点云滤波z范围：down < z <up

   infile >> i_params.line_fit2real_dist; //求出的标定板的四个角点构成的矩形边长与真值的差值，单位：m

   double diagonal = 0.0;
   diagonal = sqrt(pow(i_params.board_dimension.first, 2) + pow(i_params.board_dimension.second, 2)) / 1000;
   std::cout << "棋盘格对角线长度为：" << diagonal << "m!" << std::endl;

   std::cout << "参数配置文件输入完成！" << std::endl;
}


/******   从图像和激光点云ROI中提取特征(映射点)   ******/
int extractROI(const cv::Mat &img, const pcl::PointCloud<PointXYZIR>::Ptr &cloud) {

   //////////////// 图像特征提取 //////////////////

   cv::Mat corner_vectors = cv::Mat::eye(3, 5, CV_64F); // 相机坐标系的点容器
   cv::Mat chessboard_normal = cv::Mat(1, 3, CV_64F);

   cv::Size2i patternNum(i_params.grid_size.first, i_params.grid_size.second);
   cv::Size2i patternSize(i_params.square_length, i_params.square_length);

   cv::Mat gray;
   std::vector<cv::Point2f> corners; // 检测到的棋盘格角点
   std::vector<cv::Point3f> grid3dpoint; // 棋盘格角点世界坐标系下的3D点,单位：mm
   cv::cvtColor(img, gray, COLOR_BGR2GRAY);
   // std::cout << "img cols: " << gray.cols << ", " << "img rows: " << gray.rows << "." << endl;
   // 寻找棋盘格角点
   bool patternfound = cv::findChessboardCorners(gray, patternNum, corners, CALIB_CB_ADAPTIVE_THRESH + CALIB_CB_NORMALIZE_IMAGE);

   if (patternfound) {
      // 寻找亚像素精度的角点
      std::cout << "成功找到角点！" << endl;
      cornerSubPix(gray, corners, Size(11, 11), Size(-1, -1), TermCriteria(CV_TERMCRIT_EPS + CV_TERMCRIT_ITER, 30, 0.1));
      // 角点绘制
      cv::drawChessboardCorners(img, patternNum, corners, patternfound);

      double tx, ty; // 世界坐标系原点
      // 角点中心为世界坐标系原点
      tx = (patternNum.width - 1) * patternSize.width / 2;
      ty = (patternNum.height - 1) * patternSize.height / 2;
      // 求各个角点的世界坐标,单位：mm
      for (int i = 0; i < patternNum.height; i++) {
         for (int j = 0; j < patternNum.width; j++) {
            cv::Point3f tmpgrid3dpoint;
            tmpgrid3dpoint.x = j * patternSize.width - tx;
            tmpgrid3dpoint.y = i * patternSize.height - ty;
            tmpgrid3dpoint.z = 0;
            grid3dpoint.push_back(tmpgrid3dpoint);
         }
      }
      
      // 标定板四个边角点世界坐标(基于位置错放补偿),角点需要与标定板的激光点云前后左右中5个角点对应
      // 寻找检测的角点的中最上方、最下方、最左方、最右方四个角点，根据四个角点的检测的顺序，决定标定板上下左右中五个特征点的三维坐标，以此来和标定板的激光点云前后左右中5个角点对应
      std::vector<cv::Point3f> boardcorners;
      double up_row = 99999.0, down_row = 0.0, left_col = 99999.0, right_col = 0.0;
      int up_index, down_index, left_index, right_index;
      std::vector<int> index_sort; 
      for(int i = 0; i < (int)corners.size(); i++){
         if(corners[i].y < up_row){
            up_row = corners[i].y;
            up_index = i;
         }
         if(corners[i].y > down_row){
            down_row = corners[i].y;
            down_index = i;
         }
         if(corners[i].x < left_col){
            left_col = corners[i].x;
            left_index = i;
         }
         if(corners[i].x > right_col){
            right_col = corners[i].x;
            right_index = i;
         }
      }
      index_sort.push_back(up_index);
      index_sort.push_back(down_index);
      index_sort.push_back(left_index);
      index_sort.push_back(right_index);
      sort(index_sort.begin(), index_sort.end());

      if(index_sort[0] == up_index && index_sort[1] == left_index) { //最上点为角点检测起点，方向向左
         // 左右下上
         boardcorners.push_back(cv::Point3f(i_params.board_dimension.first / 2, -i_params.board_dimension.second / 2, 0.0));
         boardcorners.push_back(cv::Point3f(-i_params.board_dimension.first / 2, i_params.board_dimension.second / 2, 0.0));
         boardcorners.push_back(cv::Point3f(i_params.board_dimension.first / 2, i_params.board_dimension.second / 2, 0.0));
         boardcorners.push_back(cv::Point3f(-i_params.board_dimension.first / 2, -i_params.board_dimension.second / 2, 0.0));
      }
      else if(index_sort[0] == up_index && index_sort[1] == right_index) { //最上点为角点检测起点，方向向右
         // 左右下上
         boardcorners.push_back(cv::Point3f(-i_params.board_dimension.first / 2, i_params.board_dimension.second / 2, 0.0));
         boardcorners.push_back(cv::Point3f(i_params.board_dimension.first / 2, -i_params.board_dimension.second / 2, 0.0));
         boardcorners.push_back(cv::Point3f(i_params.board_dimension.first / 2, i_params.board_dimension.second / 2, 0.0));
         boardcorners.push_back(cv::Point3f(-i_params.board_dimension.first / 2, -i_params.board_dimension.second / 2, 0.0));
      }
      else if(index_sort[0] == down_index && index_sort[1] == left_index) { //最下点为角点检测起点，方向向左
         // 左右下上
         boardcorners.push_back(cv::Point3f(i_params.board_dimension.first / 2, -i_params.board_dimension.second / 2, 0.0));
         boardcorners.push_back(cv::Point3f(-i_params.board_dimension.first / 2, i_params.board_dimension.second / 2, 0.0));
         boardcorners.push_back(cv::Point3f(-i_params.board_dimension.first / 2, -i_params.board_dimension.second / 2, 0.0));
         boardcorners.push_back(cv::Point3f(i_params.board_dimension.first / 2, i_params.board_dimension.second / 2, 0.0));
      }
      else if(index_sort[0] == down_index && index_sort[1] == right_index) { //最下点为角点检测起点，方向向右
         // 左右下上
         boardcorners.push_back(cv::Point3f(-i_params.board_dimension.first / 2, i_params.board_dimension.second / 2, 0.0));
         boardcorners.push_back(cv::Point3f(i_params.board_dimension.first / 2, -i_params.board_dimension.second / 2, 0.0));
         boardcorners.push_back(cv::Point3f(-i_params.board_dimension.first / 2, -i_params.board_dimension.second / 2, 0.0));
         boardcorners.push_back(cv::Point3f(i_params.board_dimension.first / 2, i_params.board_dimension.second / 2, 0.0));
      }
      else if(index_sort[0] == right_index && index_sort[1] == up_index) { //最右点为角点检测起点，方向向上
         // 左右下上
         boardcorners.push_back(cv::Point3f(-i_params.board_dimension.first / 2, -i_params.board_dimension.second / 2, 0.0));
         boardcorners.push_back(cv::Point3f(i_params.board_dimension.first / 2, i_params.board_dimension.second / 2, 0.0));
         boardcorners.push_back(cv::Point3f(-i_params.board_dimension.first / 2, i_params.board_dimension.second / 2, 0.0));
         boardcorners.push_back(cv::Point3f(i_params.board_dimension.first / 2, -i_params.board_dimension.second / 2, 0.0));
      }
      else if(index_sort[0] == right_index && index_sort[1] == down_index) { //最右点为角点检测起点，方向向下
         // 左右下上
         boardcorners.push_back(cv::Point3f(-i_params.board_dimension.first / 2, -i_params.board_dimension.second / 2, 0.0));
         boardcorners.push_back(cv::Point3f(i_params.board_dimension.first / 2, i_params.board_dimension.second / 2, 0.0));
         boardcorners.push_back(cv::Point3f(i_params.board_dimension.first / 2, -i_params.board_dimension.second / 2, 0.0));
         boardcorners.push_back(cv::Point3f(-i_params.board_dimension.first / 2, i_params.board_dimension.second / 2, 0.0));
      }
      else if(index_sort[0] == left_index && index_sort[1] == up_index) { //最左点为角点检测起点，方向向上
         // 左右下上
         boardcorners.push_back(cv::Point3f(i_params.board_dimension.first / 2, i_params.board_dimension.second / 2, 0.0));
         boardcorners.push_back(cv::Point3f(-i_params.board_dimension.first / 2, -i_params.board_dimension.second / 2, 0.0));
         boardcorners.push_back(cv::Point3f(-i_params.board_dimension.first / 2, i_params.board_dimension.second / 2, 0.0));
         boardcorners.push_back(cv::Point3f(i_params.board_dimension.first / 2, -i_params.board_dimension.second / 2, 0.0));
      }
      else if(index_sort[0] == left_index && index_sort[1] == down_index) { //最左点为角点检测起点，方向向下
         // 左右下上
         boardcorners.push_back(cv::Point3f(i_params.board_dimension.first / 2, i_params.board_dimension.second / 2, 0.0));
         boardcorners.push_back(cv::Point3f(-i_params.board_dimension.first / 2, -i_params.board_dimension.second / 2, 0.0));
         boardcorners.push_back(cv::Point3f(i_params.board_dimension.first / 2, -i_params.board_dimension.second / 2, 0.0));
         boardcorners.push_back(cv::Point3f(-i_params.board_dimension.first / 2, i_params.board_dimension.second / 2, 0.0));
      }
      // 标定板中心点的世界坐标(基于位置错放补偿)
      boardcorners.push_back(cv::Point3f(0.0, 0.0, 0.0)); //中心点

      cv::Mat rvec(3, 3, cv::DataType<double>::type); // 旋转向量r，世界坐标系到相机坐标系,roll、pitch、yaw
      cv::Mat tvec(3, 1, cv::DataType<double>::type); // 平移向量t，世界坐标系到相机坐标系,单位：mm

      // 普通相机模型
      cv::solvePnP(grid3dpoint, corners, i_params.cameramat, i_params.distcoeff, rvec, tvec); // pnp模型求r,t
      cv::projectPoints(boardcorners, rvec, tvec, i_params.cameramat, i_params.distcoeff, img_board_coenerAndCenter);
      // std::cout << img_board_coenerAndCenter << endl; //输出4个边角点和中心点的图像坐标,单位：像素
      

      // chessboardpose 是一个 3*4 转换矩阵，用来将标定板四个边角点和中心点的世界坐标转化为相机坐标 | R&T
      cv::Mat chessboardpose = cv::Mat::eye(4, 4, CV_64F);
      cv::Mat tmprmat = cv::Mat(3, 3, CV_64F); // 旋转矩阵
      cv::Rodrigues(rvec, tmprmat); // 将旋转向量(欧拉角)转化为旋转矩阵

      for (int j = 0; j < 3; j++) {
         for (int k = 0; k < 3; k++) {
            chessboardpose.at<double>(j, k) = tmprmat.at<double>(j, k);
         }
         chessboardpose.at<double>(j, 3) = tvec.at<double>(j);
      }
      //相机坐标系中标定板的法向量(朝向向量)
      chessboard_normal.at<double>(0) = 0;
      chessboard_normal.at<double>(1) = 0;
      chessboard_normal.at<double>(2) = 1;
      chessboard_normal = chessboard_normal * chessboardpose(cv::Rect(0, 0, 3, 3)).t();

      for (int k = 0; k < boardcorners.size(); k++) {
         if (k == 0)
            cv::circle(img, img_board_coenerAndCenter[0], 8, CV_RGB(0, 255, 0), -1); //green //左边特征点
         else if (k == 1)
            cv::circle(img, img_board_coenerAndCenter[1], 8, CV_RGB(255, 255, 0), -1); //yellow //右边特征点
         else if (k == 2)
            cv::circle(img, img_board_coenerAndCenter[2], 8, CV_RGB(0, 0, 255), -1); //blue //下边特征点
         else if (k == 3)
            cv::circle(img, img_board_coenerAndCenter[3], 8, CV_RGB(255, 0, 0), -1); //red //上边特征点
         else
            cv::circle(img, img_board_coenerAndCenter[4], 8, CV_RGB(255, 255, 255), -1); //white for centre //中心特征点
      }
      imshow("img",img);
      waitKey(0);
      destroyAllWindows(); //关闭所有可视化窗口
   } // if (patternfound)
   else{
      std::cout << "图像角点检测失败！" << endl;
      return 0;
   }

   //////////////// 点云特征提取 //////////////////

   // 点云滤波
   pcl::PointCloud<PointXYZIR>::Ptr cloud_passthrough(new pcl::PointCloud<PointXYZIR>); // 第一次滤波后点云
   // 第一次点云滤波
   pcl::ConditionAnd<PointXYZIR>::Ptr range_condition(new pcl::ConditionAnd<PointXYZIR>()); // 点云过滤器
   // x轴过滤
   range_condition->addComparison(pcl::FieldComparison<PointXYZIR>::ConstPtr(new
   pcl::FieldComparison<PointXYZIR>("x", pcl::ComparisonOps::GT, i_params.x_range_point_filter.first)));  // GT表示大于等于
   range_condition->addComparison(pcl::FieldComparison<PointXYZIR>::ConstPtr(new
   pcl::FieldComparison<PointXYZIR>("x", pcl::ComparisonOps::LT, i_params.x_range_point_filter.second)));  // LT表示小于等于
   // y轴过滤
   range_condition->addComparison(pcl::FieldComparison<PointXYZIR>::ConstPtr(new
   pcl::FieldComparison<PointXYZIR>("y", pcl::ComparisonOps::GT, i_params.y_range_point_filter.first)));  // GT表示大于等于
   range_condition->addComparison(pcl::FieldComparison<PointXYZIR>::ConstPtr(new
   pcl::FieldComparison<PointXYZIR>("y", pcl::ComparisonOps::LT, i_params.y_range_point_filter.second)));  // LT表示小于等于
   // z轴过滤
   range_condition->addComparison(pcl::FieldComparison<PointXYZIR>::ConstPtr(new
   pcl::FieldComparison<PointXYZIR>("z", pcl::ComparisonOps::GT, i_params.z_range_point_filter.first)));  // GT表示大于等于
   range_condition->addComparison(pcl::FieldComparison<PointXYZIR>::ConstPtr(new
   pcl::FieldComparison<PointXYZIR>("z", pcl::ComparisonOps::LT, i_params.z_range_point_filter.second)));  // LT表示小于等于
 
   pcl::ConditionalRemoval<PointXYZIR> condition;
   condition.setCondition(range_condition);
   condition.setInputCloud(cloud); // 输入点云
   condition.setKeepOrganized(false);
   condition.filter(*cloud_passthrough);
   std::cout << "滤波前点云数：" << cloud->points.size() << std::endl;
   std::cout << "第一次滤波后点云数：" << cloud_passthrough->points.size() << std::endl;
   // 第一次滤波后点云可视化
   // visualize_pcd(cloud_passthrough);

   // RANSAC平面拟合
   pcl::ModelCoefficients::Ptr coefficients(new pcl::ModelCoefficients());
   pcl::PointIndices::Ptr inliers(new pcl::PointIndices());
   pcl::SACSegmentation<PointXYZIR> seg;
   seg.setOptimizeCoefficients(true);
   seg.setModelType(pcl::SACMODEL_PLANE);
   seg.setMethodType(pcl::SAC_RANSAC);
   seg.setMaxIterations(1000);
   seg.setDistanceThreshold(i_params.plane_line_dist_threshold.first); //设置距离阈值
   seg.setInputCloud(cloud_passthrough);
   seg.segment(*inliers, *coefficients);
   // coefficients即为拟合平面的法向量 // mag标定板法向量的大小
   // 保存平面拟合后点点云数据
   pcl::PointCloud<PointXYZIR>::Ptr cloud_seg(new pcl::PointCloud<PointXYZIR>);
   pcl::ExtractIndices<PointXYZIR> extract;
   extract.setInputCloud(cloud_passthrough);
   extract.setIndices(inliers);
   extract.setNegative(false);
   extract.filter(*cloud_seg);
   std::cout << "RANSAC平面拟合后点云数：" << cloud_seg->points.size() << std::endl;
   //将点云投射到拟合的平面上
   pcl::PointCloud<PointXYZIR>::Ptr cloud_projected(new pcl::PointCloud<PointXYZIR>);
   pcl::ProjectInliers<PointXYZIR> proj;
   proj.setModelType(pcl::SACMODEL_PLANE);
   proj.setInputCloud(cloud_seg);
   proj.setModelCoefficients(coefficients);
   proj.filter(*cloud_projected);
   std::cout << "点云投射到拟合平面后点云数：" << cloud_projected->points.size() << std::endl;
   //可视化拟合平面点云
   // visualize_pcd(cloud_projected);

   //将标定板拟合点云存入vector中,待后续投影
   std::vector<cv::Point3f> tmp;
   for(size_t i = 0; i < (int)cloud_projected->points.size(); i++){
      tmp.push_back(cv::Point3f(cloud_projected->points[i].x * 1000, cloud_projected->points[i].y * 1000, cloud_projected->points[i].z * 1000)); //单位：mm
   }
   chessBoard_lidar_points.push_back(tmp);

   // 寻找标定板上每条扫描线的最大点和最小点，即与标定板边线的交点
   // 将每个点云按照Scan number进行归类
   std::vector<std::deque<PointXYZIR*>> candidate_segments(i_params.lidar_ring_count);
   for (size_t i = 0; i < cloud_projected->points.size(); i++) {
      int ring_number = (int)cloud_projected->points[i].ring -1; //这里是个坑，需要注意，速腾lidar32 Scan的ID为：1->32，所有需要减去1
      candidate_segments[ring_number].push_back(&(cloud_projected->points[i]));
   }

   // Second: Arrange points in every ring in descending order of y coordinate
   pcl::PointCloud<PointXYZIR>::Ptr min_points(new pcl::PointCloud<PointXYZIR>);        // 标定板右边界激光点集合
   pcl::PointCloud<PointXYZIR>::Ptr max_points(new pcl::PointCloud<PointXYZIR>);        // 标定板左边界激光点集合
   // int maxLidarPoints_scanNumber = 0; //用于记录点云最多的那条scan的点云数量
   // double y_right = 9999.0, y_left = -9999.0, z_right, z_left;   // z_right为最右侧边界点的z值， z_left为最左侧边界点的z值。

   for (int i = 0; i < (int)candidate_segments.size(); i++) {
      if (candidate_segments[i].size() == 0 || candidate_segments[i].size() == 1) // 等于0说明这条扫描线上没有点，1说明这条线上只有1个点,
      {
         continue;
      }
      // 寻找标定板上每条Scan上的左右两个边界点
      double y_min = 9999.0;
      double y_max = -9999.0;
      int y_min_index, y_max_index;
      for (int p = 0; p < candidate_segments[i].size(); p++) {
         if (candidate_segments[i][p]->y > y_max) {
            y_max = candidate_segments[i][p]->y;
            y_max_index = p;
         }
         if (candidate_segments[i][p]->y < y_min) {
            y_min = candidate_segments[i][p]->y;
            y_min_index = p;
         }
      }

      min_points->push_back(*candidate_segments[i][y_min_index]); // 标定板右边界激光点集合
      max_points->push_back(*candidate_segments[i][y_max_index]); // 标定板左边界激光点集合

      // //寻找点云最多的那条scan及其对应左右边界点
      // if ((int)candidate_segments[i].size() > maxLidarPoints_scanNumber)
      // {
      //    maxLidarPoints_scanNumber = (int)candidate_segments[i].size();
      //    z_right = candidate_segments[i][y_min_index]->z;
      //    z_left = candidate_segments[i][y_max_index]->z;
      // }
   }
   // visualize_pcd(min_points); // 标定板右边界激光点集合
   // visualize_pcd(max_points); // 标定板左边界激光点集合

   double y_right = 9999.0, y_left = -9999.0, z_right, z_left;   // z_right为最右侧边界点的z值， z_left为最左侧边界点的z值。
   for(size_t i = 0; i < (int)min_points->points.size(); i++)
   {
      if (min_points->points[i].y < y_right)
      {
         y_right = min_points->points[i].y;
         z_right = min_points->points[i].z;
      }
   }
   for(size_t i = 0; i < (int)max_points->points.size(); i++)
   {
      if (max_points->points[i].y > y_left)
      {
         y_left = max_points->points[i].y;
         z_left = max_points->points[i].z;
      }
   }

   pcl::PointCloud<PointXYZIR>::Ptr left_up_points(new pcl::PointCloud<PointXYZIR>);    // 标定板左上边界激光点集合
   pcl::PointCloud<PointXYZIR>::Ptr left_down_points(new pcl::PointCloud<PointXYZIR>);  // 标定板左下边界激光点集合
   pcl::PointCloud<PointXYZIR>::Ptr right_up_points(new pcl::PointCloud<PointXYZIR>);   // 标定板右上边界激光点集合
   pcl::PointCloud<PointXYZIR>::Ptr right_down_points(new pcl::PointCloud<PointXYZIR>); // 标定板右下边界激光点集合

   for(int i = 0; i < (int)min_points->points.size(); i++){
      // 右上边界点
      if(min_points->points[i].z >= z_right){
         right_up_points->push_back(min_points->points[i]);
      }
      // 右下边界点
      if(min_points->points[i].z <= z_right){
         right_down_points->push_back(min_points->points[i]);
      }
   }
   for (int i = 0; i < (int)max_points->points.size(); i++)
   {
      // 左上边界点
      if(max_points->points[i].z >= z_left){
         left_up_points->push_back(max_points->points[i]);
      }
      // 左下边界点
      if(max_points->points[i].z <= z_left){
         left_down_points->push_back(max_points->points[i]);
      }
   }
   // visualize_pcd(right_up_points);
   // visualize_pcd(right_down_points);
   // visualize_pcd(left_up_points);
   // visualize_pcd(left_down_points);

   // Fit lines through minimum and maximum points //直线拟合
   pcl::ModelCoefficients::Ptr coefficients_left_up(new pcl::ModelCoefficients);
   pcl::PointIndices::Ptr inliers_left_up(new pcl::PointIndices);

   pcl::ModelCoefficients::Ptr coefficients_left_dwn(new pcl::ModelCoefficients);
   pcl::PointIndices::Ptr inliers_left_dwn(new pcl::PointIndices);

   pcl::ModelCoefficients::Ptr coefficients_right_up(new pcl::ModelCoefficients);
   pcl::PointIndices::Ptr inliers_right_up(new pcl::PointIndices);

   pcl::ModelCoefficients::Ptr coefficients_right_dwn(new pcl::ModelCoefficients);
   pcl::PointIndices::Ptr inliers_right_dwn(new pcl::PointIndices);

   seg.setModelType(pcl::SACMODEL_LINE);
   seg.setMethodType(pcl::SAC_RANSAC);
   seg.setDistanceThreshold(i_params.plane_line_dist_threshold.second);

   seg.setInputCloud(left_up_points);
   seg.segment(*inliers_left_up, *coefficients_left_up); // Fitting line1 through max points
   // cout << coefficients_left_up->values.size() << endl;
   // std::cout << coefficients_left_up->values[0] << "," << coefficients_left_up->values[1] << ","
   //           << coefficients_left_up->values[2] << "," << coefficients_left_up->values[3] << ","
   //           << coefficients_left_up->values[4] << endl;

   seg.setInputCloud(left_down_points);
   seg.segment(*inliers_left_dwn, *coefficients_left_dwn); // Fitting line2 through max points
   // std::cout << coefficients_left_dwn->values[0] << "," << coefficients_left_dwn->values[1] << ","
   //           << coefficients_left_dwn->values[2] << "," << coefficients_left_dwn->values[3] << ","
   //           << coefficients_left_dwn->values[4] << endl;

   seg.setInputCloud(right_up_points);
   seg.segment(*inliers_right_up, *coefficients_right_up); // Fitting line1 through min points
   // std::cout << coefficients_right_up->values[0] << "," << coefficients_right_up->values[1] << ","
   //           << coefficients_right_up->values[2] << "," << coefficients_right_up->values[3] << ","
   //           << coefficients_right_up->values[4] << endl;

   seg.setInputCloud(right_down_points);
   seg.segment(*inliers_right_dwn, *coefficients_right_dwn); // Fitting line2 through min points
   // std::cout << coefficients_right_dwn->values[0] << "," << coefficients_right_dwn->values[1] << ","
   //           << coefficients_right_dwn->values[2] << "," << coefficients_right_dwn->values[3] << ","
   //           << coefficients_right_dwn->values[4] << endl;

   // Find out 2 (out of the four) intersection points
   Eigen::Vector4f Point_l;
   pcl::PointCloud<PointXYZIR>::Ptr basic_cloud_ptr(new pcl::PointCloud<PointXYZIR>);
   PointXYZIR basic_point; // intersection points stored here
   if (pcl::lineWithLineIntersection(*coefficients_left_up, *coefficients_left_dwn, Point_l)) { // 左边点
      basic_point.x = Point_l[0];
      basic_point.y = Point_l[1];
      basic_point.z = Point_l[2];
      basic_cloud_ptr->points.push_back(basic_point);
      lidar_board_coenerAndCenter.push_back(cv::Point3f(Point_l[0] * 1000, Point_l[1] * 1000, Point_l[2] * 1000)); //与图像坐标系单位统一，mm
   }
   if (pcl::lineWithLineIntersection(*coefficients_right_up, *coefficients_right_dwn, Point_l)) { // 右边点
      basic_point.x = Point_l[0];
      basic_point.y = Point_l[1];
      basic_point.z = Point_l[2];
      basic_cloud_ptr->points.push_back(basic_point);
      lidar_board_coenerAndCenter.push_back(cv::Point3f(Point_l[0] * 1000, Point_l[1] * 1000, Point_l[2] * 1000));
   }
   if (pcl::lineWithLineIntersection(*coefficients_left_dwn, *coefficients_right_dwn, Point_l)) { // 下面点
      basic_point.x = Point_l[0];
      basic_point.y = Point_l[1];
      basic_point.z = Point_l[2];
      basic_cloud_ptr->points.push_back(basic_point);
      lidar_board_coenerAndCenter.push_back(cv::Point3f(Point_l[0] * 1000, Point_l[1] * 1000, Point_l[2] * 1000));
   }
   if (pcl::lineWithLineIntersection(*coefficients_left_up, *coefficients_right_up, Point_l)) { // 上面点
      basic_point.x = Point_l[0];
      basic_point.y = Point_l[1];
      basic_point.z = Point_l[2];
      basic_cloud_ptr->points.push_back(basic_point);
      lidar_board_coenerAndCenter.push_back(cv::Point3f(Point_l[0] * 1000, Point_l[1] * 1000, Point_l[2] * 1000));
   }

   // 标定板激光点云拟合平面中心点坐标求解
   PointXYZIR velodynepoint;
   velodynepoint.x = (basic_cloud_ptr->points[0].x + basic_cloud_ptr->points[1].x) / 2;
   velodynepoint.y = (basic_cloud_ptr->points[0].y + basic_cloud_ptr->points[1].y) / 2;
   velodynepoint.z = (basic_cloud_ptr->points[0].z + basic_cloud_ptr->points[1].z) / 2;
   basic_cloud_ptr->points.push_back(velodynepoint);
   lidar_board_coenerAndCenter.push_back(cv::Point3f(velodynepoint.x * 1000, velodynepoint.y * 1000, velodynepoint.z * 1000)); //与图像坐标系单位统一，mm

   // 标定板激光点云拟合平面 + 四个边角点 + 中心点 可视化
   visualize_pcd2(cloud_projected ,basic_cloud_ptr);

   // 检测求出的四个标定板边角点是否符合要求
   double line_left_down_dist = abs(pcl::euclideanDistance(basic_cloud_ptr->points[0], basic_cloud_ptr->points[2])-1);
   double line_left_up_dist = abs(pcl::euclideanDistance(basic_cloud_ptr->points[0], basic_cloud_ptr->points[3]) -1);
   double line_right_down_dist = abs(pcl::euclideanDistance(basic_cloud_ptr->points[1], basic_cloud_ptr->points[2]) -1);
   double line_right_up_dist = abs(pcl::euclideanDistance(basic_cloud_ptr->points[1], basic_cloud_ptr->points[3]) -1);
   if(max(line_left_down_dist, line_left_up_dist) <= i_params.line_fit2real_dist && max(line_right_down_dist, line_right_up_dist) <= i_params.line_fit2real_dist){
      return 1;
   }
   else return 0;

   // // std::cout << lidar_board_coenerAndCenter << endl;
   // // std::cout << img_board_coenerAndCenter << endl;
   // cv::solvePnP(lidar_board_coenerAndCenter, img_board_coenerAndCenter, i_params.cameramat, i_params.distcoeff, R_camToLidar, T_camToLidar); // pnp模型求R,T
   // cout<<"lidar 到 camera 的外参 R：" << R_camToLidar << endl;
   // cout<<"lidar 到 camera 的外参 T：" << T_camToLidar << endl;  

} //End of extractROI

/******   求解camera和lidar的外参   ******/
void calculate_camToLidar_RT(){
   cv::solvePnP(all_lidar_board_coenerAndCenter, all_img_board_coenerAndCenter, i_params.cameramat, i_params.distcoeff, R_camToLidar, T_camToLidar, false, cv::SOLVEPNP_EPNP); // pnp模型求R,T
   cv::solvePnP(all_lidar_board_coenerAndCenter, all_img_board_coenerAndCenter, i_params.cameramat, i_params.distcoeff, R_camToLidar, T_camToLidar, true, cv::SOLVEPNP_ITERATIVE); // pnp模型求R,T
   // cv::solvePnP(all_lidar_board_coenerAndCenter, all_img_board_coenerAndCenter, i_params.cameramat, i_params.distcoeff, R_camToLidar, T_camToLidar); // pnp模型求R,T
   cout << "lidar 到 camera 的外参 R：" << endl;
   cout << R_camToLidar << endl;
   cout << "lidar 到 camera 的外参 T：" << endl;
   cout << T_camToLidar << endl;
}

/******   激光点云投射到图像上   ******/
void visualize_lidarToCamera(const cv::Mat &img, const pcl::PointCloud<PointXYZIR>::Ptr &cloud){
   
   // 将标定板上的激光点云投影到图像上
   // 将标定板激光点云拟合平面上的数据录入到vector中
   std::vector<Point3f> lidar_board_allPoints; // lidar points容器，单位:mm
   for(int i = 0; i < cloud->points.size(); i++){
      lidar_board_allPoints.push_back(cv::Point3f(cloud->points[i].x * 1000, 
                                                  cloud->points[i].y * 1000, 
                                                  cloud->points[i].z * 1000)); // 将单位换算成mm
   }

   std::vector<cv::Point2f> lidarPointsToImg;
	cv::projectPoints(lidar_board_allPoints, R_camToLidar, T_camToLidar, i_params.cameramat, i_params.distcoeff, lidarPointsToImg);
	// std::cout << lidarPointsToImg << endl;
   for (int i = 0; i < lidarPointsToImg.size(); i++)
	{
		Point2f p = lidarPointsToImg[i];
      circle(img, p, 2, CV_RGB(255, 0, 0), -1, 8, 0); //激光点在图像上为红色
		
	}
   imshow("img", img);
   waitKey(0);
   destroyAllWindows();
}

bool my_sortFunction(std::vector<double> &a, std::vector<double> &b){
    return (a[0] < b[0]);
}

int main ()
{
   /* 配置参数输入 */
   onInit();
   
   /* 载入标定用图片 */
   std::vector<string> img_files;
   GetFile(i_params.srcImg_path, img_files);
   for(int i = 0; i < (int)img_files.size(); i++){ //验证文件顺序是否正确
      std::cout << img_files[i] << std::endl;
   }

   /* 载入标定用点云数据 */
   std::vector<string> csv_files;
   GetFile(i_params.srclidarPoints_path, csv_files);
   // for(int i = 0; i < (int)csv_files.size(); i++){ //验证文件顺序是否正确
   //    std::cout << csv_files[i] << std::endl;
   // }

   /* 图像、点云数据匹配，时间最近即为匹配 */
   std::ifstream log_txt(i_params.log_txt_path, ios::in);
   if(log_txt.fail())
   {
      std::cout << "Couldn't load the log_txt file!" << std::endl;
      return -1;
   }
   std::vector<std::vector<double>> all_img_time_id;
   string every_line_str("");
   while(getline(log_txt, every_line_str))
   {
      // std::cout << every_line_str << std::endl;
      istringstream sin(every_line_str);
      std::vector<std::string> tmp_line_str;
      string tmp_str("");
      while(getline(sin, tmp_str, ' '))
      {
         tmp_line_str.push_back(tmp_str);
      }
      if(tmp_line_str[2] == "camera")
      {
         std::vector<double> one_img_time_id = {atof((tmp_line_str[0] + "." + tmp_line_str[1]).c_str()), atof(tmp_line_str[3].c_str())};
         all_img_time_id.push_back(one_img_time_id);
      }
   }
   sort(all_img_time_id.begin(), all_img_time_id.end(), my_sortFunction); //排序
   // std::cout << "all_img_time_id size: " << (int)all_img_time_id.size() << std::endl;

   std::vector<std::vector<double>> need_img_time_id;
   for(size_t i = 0; i < (int)img_files.size(); i++)
   {
      for(size_t j = 0; j < (int)all_img_time_id.size(); j++)
      {
         // std::cout << atoi(img_files[i].c_str()) << "," << (int)all_img_time_id[j][1] << std::endl;
         if(atoi(img_files[i].c_str()) == (int)all_img_time_id[j][1])
         {
            need_img_time_id.push_back({all_img_time_id[j][0], all_img_time_id[j][1]});
            break;
         }
      }
   }
   std::cout << "need img size: " << (int)need_img_time_id.size() << std::endl;

   std::vector<string> need_csv_files;
   for(size_t i = 0; i < (int)need_img_time_id.size(); i++)
   {
      std::cout << "img id: " << need_img_time_id[i][1] << std::endl;
      for(size_t j = 0; j < (int)csv_files.size(); j++)
      {
         if(fabs(need_img_time_id[i][0] - atof(csv_files[j].c_str())) <= 0.05)
         {
            need_csv_files.push_back(csv_files[j]);
            break;
         }
      }
   }
   sort(need_csv_files.begin(), need_csv_files.end()); //排序
   //check 顺序
   for(size_t i = 0; i < (int)need_csv_files.size(); i++)
   {
      std::cout << need_csv_files[i] << std::endl;   
   }


   /* 循环读取图像和点云数据，并提取对应特征 */
   int success_(0), fail_(0); //用来记录图像和激光点云对应特征提取成功的帧数
   for(int i = 0; i < (int)img_files.size(); i++){
      /* 读取图像文件 */
      std::cout << img_files[i] << "..." << std::endl; //正在处理的图像
      cv::Mat img = imread(i_params.srcImg_path + img_files[i], 1);
      if(img.empty()){
         std::cout << "Couldn't load the image!" << std::endl; //读取图像失败
         return -1;
      }

      /* 读取csv文件 */
      std::cout << need_csv_files[i] << "..." << std::endl; //正在处理的点云csv文件
      pcl::PointCloud<PointXYZIR>::Ptr laserCloudIn(new pcl::PointCloud<PointXYZIR>); 
      ifstream csv_file(i_params.srclidarPoints_path + need_csv_files[i], ios::in);
      if(csv_file.fail()) {
         std::cout << "Couldn't load the csv !" << std::endl; //读取点云csv文件失败
         return -1;
      }
      string line("");
      // int n = 0; //用于剔除csv的第1行而设置的变量
      while(getline(csv_file, line)){
         // cout << "csv原始字符串为：" << line << endl; // 用来校验
         istringstream sin(line); //istringstream类用于执行C++风格的串流的输入操作
         vector<string> parameters;
         string parameter;
         while (getline(sin, parameter, ','))
         {
            parameters.push_back(parameter);
         }
         // if(n){ 
            PointXYZIR lidar_point;
            lidar_point.x = atof(parameters[0].c_str());
            lidar_point.y = atof(parameters[1].c_str());
            lidar_point.z = atof(parameters[2].c_str());
            lidar_point.intensity = atof(parameters[3].c_str());
            lidar_point.ring = atof(parameters[4].c_str());
            laserCloudIn->points.push_back(lidar_point);
         // } 
         // else n++; //第0行为csv文件的标题行，舍去，
      }
      // std::cout << "点云数量：" << (int)laserCloudIn->points.size() << endl; //用于校验
      // visualize_pcd(laserCloudIn); // 查看录入的点云数据

      /* 特征(映射点)提取 */
      if(extractROI(img, laserCloudIn)){
         std::cout << img_files[i] << "和" << need_csv_files[i] << "对应特征提取成功！" << endl;
         success_++;
         // all_img_board_coenerAndCenter.assign(img_board_coenerAndCenter.begin(), img_board_coenerAndCenter.end());
         // all_lidar_board_coenerAndCenter.assign(lidar_board_coenerAndCenter.begin(), lidar_board_coenerAndCenter.end());
         all_img_board_coenerAndCenter.insert(all_img_board_coenerAndCenter.end(), img_board_coenerAndCenter.begin(), img_board_coenerAndCenter.end());
         all_lidar_board_coenerAndCenter.insert(all_lidar_board_coenerAndCenter.end(), lidar_board_coenerAndCenter.begin(), lidar_board_coenerAndCenter.end());
         // std::cout << img_board_coenerAndCenter << std::endl;
         // std::cout << lidar_board_coenerAndCenter << std::endl;
         img_board_coenerAndCenter.clear();   //清空数据,等待下一组数据的录入
         lidar_board_coenerAndCenter.clear(); //清空数据，等待下一组数据的录入
      }
      else{
         std::cout << img_files[i] << "和" << need_csv_files[i] << "对应特征提取失败！" << endl;
         fail_++;
         img_board_coenerAndCenter.clear();   //清空数据,等待下一组数据的录入
         lidar_board_coenerAndCenter.clear(); //清空数据，等待下一组数据的录入
         continue;
      }

   }
   std::cout << "用于对应特征提取的图像和点云组数为：" << success_ + fail_ << std::endl;
   std::cout << "图像和激光点云对应特征提取成功组数：" << success_ << std::endl;
   std::cout << "图像和激光点云对应特征提取失败组数：" << fail_ << std::endl;
   // std::cout << all_img_board_coenerAndCenter << std::endl;
   // std::cout << all_lidar_board_coenerAndCenter << std::endl;
   
   // // save the features
   // std::ofstream img_features(i_params.img_projection_path + "img_features.txt", ios::out | ios::app); //ios::trunc 覆盖原文件
   // std::ofstream lidar_features(i_params.img_projection_path + "lidar_features.txt", ios::out | ios::app);
   // for(size_t i = 0; i < (int)all_img_board_coenerAndCenter.size(); i++)
   // {
   //    // std::cout << all_img_board_coenerAndCenter[i].x << std::endl;
   //    img_features << all_img_board_coenerAndCenter[i].x << "," << all_img_board_coenerAndCenter[i].y << std::endl;
   //    lidar_features << all_lidar_board_coenerAndCenter[i].x/1000 << "," << all_lidar_board_coenerAndCenter[i].y/1000 << "," << all_lidar_board_coenerAndCenter[i].z/1000 << std::endl;
   // }
   // img_features.close();
   // lidar_features.close();
   // check
   // std::ifstream img_features_infile(i_params.img_projection_path + "img_features.txt", ios::in);
   // string line("");
   // while (getline(img_features_infile, line))
   // {
   //    std::cout << line << std::endl;
   //    istringstream sin(line);
   //    std::string tmp("");
   //    while(getline(sin, tmp, ','))
   //    {
   //       std::cout << atof(tmp.c_str()) << std::endl;
   //    }
   //    std::cout << std::endl;
   // }
   

   /* 外参求解 */
   calculate_camToLidar_RT();

   /* camera-lidar外参标定结果保留 */
   ofstream fout(i_params.calibration_result_path + "camera_lidar_calibration_result.txt", ios::out | ios::trunc);
   fout << R_camToLidar << std::endl;
   fout << T_camToLidar << std::endl;

   // /* 点云特征点映射到图像上 */
   // for(int i = 0; i < (int)img_files.size(); i++){
   //    std::vector<Point3f> lidar_features; // lidar points容器，单位:mm
   //    for(int j = i*5; j < i*5 + 5; j++){
   //       lidar_features.push_back(all_lidar_board_coenerAndCenter[j]); // 单位：mm
   //    }

   //    std::vector<cv::Point2f> lidarFeaturesToImg;
   //    cv::projectPoints(lidar_features, R_camToLidar, T_camToLidar, i_params.cameramat, i_params.distcoeff, lidarFeaturesToImg);
   //    // std::cout << lidarFeaturesToImg << endl;

   //    cv::Mat show_img = cv::imread(i_params.srcImg_path + img_files[i], 1);
   //    for (int j = 0; j < lidarFeaturesToImg.size(); j++)
   //    {
   //       Point2f p = lidarFeaturesToImg[j];
   //       circle(show_img, p, 5, CV_RGB(255, 0, 0), -1, 8, 0); //激光点在图像上为红色
         
   //    }
   //    imshow("show_img", show_img);
   //    waitKey(0);
   //    destroyAllWindows();
   // }

   //将标定板点云按照对应关系映射到图像上
   for(size_t i = 0; i < (int)chessBoard_lidar_points.size(); i++){
      
      std::vector<cv::Point2f> lidarFeaturesToImg;
      cv::projectPoints(chessBoard_lidar_points[i], R_camToLidar, T_camToLidar, i_params.cameramat, i_params.distcoeff, lidarFeaturesToImg);
      // std::cout << "!!! : " << i_params.cameramat << std::endl;
      // std::cout << "!!! : " << i_params.distcoeff << std::endl;
      // std::cout << lidarFeaturesToImg << endl;

      cv::Mat show_img_src = cv::imread(i_params.srcImg_path + img_files[i], 1); //读取原始图像
      cv::Mat show_img_dst;
      cv::undistort(show_img_src, show_img_dst, i_params.cameramat, i_params.distcoeff, i_params.cameramat); //图像畸变矫正
      for (int j = 0; j < lidarFeaturesToImg.size(); j++)
      {
         Point2f p = lidarFeaturesToImg[j];
         circle(show_img_src, p, 2, CV_RGB(255, 0, 0), -1, 8, 0); //激光点在图像上为红色
      }
      imshow("show_img_dst", show_img_src);
      waitKey(0);
      destroyAllWindows();

      cv::imwrite(i_params.img_projection_path + img_files[i], show_img_src); //保存投影图像
   }


   return (0);
}
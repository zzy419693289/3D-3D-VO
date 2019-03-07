#include "CPU_ORB.h"
#include "GPU_ORB.h"
#include "vo_svd.h"
#include <string>
#include <iostream>
#include <opencv2/opencv.hpp>
#include <sl/Camera.hpp>
using namespace std;
int main(int argc, char **argv)
{
	//check open right
	if (argc != 2)
	{
		cerr << endl << "Usage: ./XXX path_to_settings.yaml" << endl;
		return 1;
	}
	/////////////////////////////////////////////////////////// ZED相机初始化//////////////////////////////////////////////////////////////////////
	//Check settings file
	const string &strSettingsFile = argv[1];	//因为./这个也要算一个argv，所以我们的输入都是从1开始的
	cv::FileStorage fsRead(strSettingsFile.c_str(), cv::FileStorage::READ);	//第一个参数表示将文件名以字符串的形式输入而不是string。第二个参数是指定打开文件是读还是写。
	if (!fsRead.isOpened())
	{
		cerr << "Failed to open settings file at: " << strSettingsFile << endl;
		exit(-1);
	}
	//读数字
	double Camerafx, Camerafy, Cameracx, Cameracy;
	fsRead["Camera.fx"] >> Camerafx;
	fsRead["Camera.fy"] >> Camerafy;
	fsRead["Camera.cx"] >> Cameracx;
	fsRead["Camera.cy"] >> Cameracy;

	fsRead.release();	//读取结束后释放并关闭文件。

	sl::Camera zed;
	// Set configuration parameters
	sl::InitParameters init_params;
	init_params.depth_mode = sl::DEPTH_MODE_PERFORMANCE;
	init_params.camera_resolution = sl::RESOLUTION_HD720; // Use 1280*720  mode
	init_params.coordinate_units = sl::UNIT_METER;	//unit is m
	init_params.camera_fps = 30; // SLAM is 15 fps
	// Open the camera
	sl::ERROR_CODE err = zed.open(init_params);
	if (err != sl::SUCCESS)
	{
		zed.close();
		exit(-1);
	}
	sl::Mat zed_leftimage(zed.getResolution(), sl::MAT_TYPE_8U_C4);
	sl::Mat zed_depthimage(zed.getResolution(), sl::MAT_TYPE_32F_C1);
	//cv_Mat use zed_Mat's memory
	cv::Mat imleftRGB = cv::Mat((int)zed_leftimage.getHeight(), (int)zed_leftimage.getWidth(), CV_8UC4, zed_leftimage.getPtr<sl::uchar1>(sl::MEM_CPU));
	cv::Mat imD = cv::Mat((int)zed_depthimage.getHeight(), (int)zed_depthimage.getWidth(), CV_32FC1, zed_depthimage.getPtr<sl::float1>(sl::MEM_CPU));
	cv::Mat imleftRGB1, imleftRGB2; 
	cv::Mat imD1, imD2; 

	for (int i = 0; i<10; i++)	//Eliminate the overexposure of the camera. It will happen at camera firstly open.
	{
		if (zed.grab() == sl::SUCCESS)
		{
			// A new image is available if grab() returns SUCCESS
			zed.retrieveImage(zed_leftimage, sl::VIEW_LEFT, sl::MEM_CPU); // Get the left image, TX2 have no GPU memory
			zed.retrieveMeasure(zed_depthimage, sl::MEASURE_DEPTH, sl::MEM_CPU); 	//get depth image, use GPU
		}
		usleep(5000);	//wait 5ms
	}
	
	///////////////////////////////////////////////////////////创建两幅图。我们通过按回车键进行图像选择///////////////////////////////////////////////////////////////
	while (cv::waitKey(1) != '\r')
	{
		if (zed.grab() == sl::SUCCESS)
		{
			// A new image is available if grab() returns SUCCESS
			zed.retrieveImage(zed_leftimage, sl::VIEW_LEFT, sl::MEM_CPU); // Get the left image, TX2 have no GPU memory
			zed.retrieveMeasure(zed_depthimage, sl::MEASURE_DEPTH, sl::MEM_CPU); 	//get depth image, use GPU
		}
		cv::imshow("image",imleftRGB);
	}
	imleftRGB1 = imleftRGB.clone();	//深拷贝
	imD1 = imD.clone();
	cout << "Fisrt Capture!" << endl;

	while (cv::waitKey(1) != '\r')
	{
		if (zed.grab() == sl::SUCCESS)
		{
			// A new image is available if grab() returns SUCCESS
			zed.retrieveImage(zed_leftimage, sl::VIEW_LEFT, sl::MEM_CPU); // Get the left image, TX2 have no GPU memory
			zed.retrieveMeasure(zed_depthimage, sl::MEASURE_DEPTH, sl::MEM_CPU); 	//get depth image, use GPU
		}
		cv::imshow("image", imleftRGB);
	}
	imleftRGB2 = imleftRGB.clone();	//深拷贝
	imD2 = imD.clone();
	cout << "Second Capture!" << endl;

	///////////////////////////////////////////////////////////建立两幅图像的ORB特征，每幅图各提取最多1000个特征点//////////////////////////////////////////////////////
	ORB_CPU orb1(imleftRGB1, 1000);
	ORB_CPU orb2(imleftRGB2, 1000);
	//分别进行匹配
	vector<cv::DMatch> matches = ORB_CPU::MatchPic(orb1, orb2, false, true);	//滤波匹配,同时将匹配结果进行记录
	//DMatch这个结构体记录了一对匹配好的orb特征点的坐标索引。这个索引queryIdx和	trainIdx分别对应于前后两幅图的keypoints在vector<cv::KeyPoint>的序号。所以我们可以将匹配好的点每一个解析出来，并存入点云向量中
	//建立两帧的三维点云集
	vector<cv::Point3f> pointcloud1;	
	vector<cv::Point3f> pointcloud2;
	for (int i = 0; i < matches.size(); i++)	//遍历matches。同时将深度不符的点进行排除，不用于匹配
	{
		//形成第一帧中该匹配点的三维坐标。注意二维向三维坐标的恢复，需要用到相机内参
		cv::KeyPoint KP1 = orb1.keypoints[matches[i].queryIdx];	//queryIdx前一帧特征点的序号。这里提取第i个匹配对中，图一的特征点
		cv::Point2f point1= KP1.pt;	//获取该点的图像坐标。这里是x和y。注意x对应于图像坐标的n，y对应于m。
		float z1 = imD1.at<float>((int)point1.y, (int)point1.x);
		float x1 = z1*(point1.x - Cameracx) / Camerafx;
		float y1 = z1*(point1.y - Cameracy) / Camerafy;
		if (z1<0.5 || z1>5 || std::isnan(z1))	//check z is not nan. nan means this point have no data
			continue;
		//形成第二帧中该匹配点的三维坐标。
		cv::KeyPoint KP2 = orb2.keypoints[matches[i].trainIdx];	//trainIdx后一帧特征点的序号
		cv::Point2f point2 = KP2.pt;	//获取该点的图像坐标。这里是x和y。注意x对应于图像坐标的n，y对应于m。
		float z2 = imD2.at<float>((int)point2.y, (int)point2.x);
		float x2 = z2*(point2.x - Cameracx) / Camerafx;
		float y2 = z2*(point2.y - Cameracy) / Camerafy;
		if (z2<0.5 || z2>5 || std::isnan(z2))
			continue;

		cv::Point3f tmp1(x1, y1, z1);
		pointcloud1.push_back(tmp1);
		cv::Point3f tmp2(x2, y2, z2);
		pointcloud2.push_back(tmp2);

		//cout<<"x1 y1 z1"<<endl<<x1<<" "<<y1<<" "<<z1<<endl;
		//cout<<"x2 y2 z2"<<endl<<x2<<" "<<y2<<" "<<z2<<endl;
	}
	
	///////////////////////////////////////////////////////////////////////////////////计算第二帧到第一帧的位姿变换//////////////////////////////////////////////////////////////////////////////////////
	cv::Mat R, t;	//定义第二帧到第一帧的旋转和平移矩阵
	cv::Vec3d Eal;
	pose_svd_3d3d(pointcloud1, pointcloud2, R, t, Eal);
	cout<<"1 to 2 R is:"<<endl;
	cout<< R<<endl;
	cout<<"1 to 2 R-Y-P is:"<<endl;
	cout<< Eal <<endl;
	cout<<"t is:"<<endl;
	cout<< t<<endl;

	///////////////////////////////////////////////////////////////////////////////// Close the camera///////////////////////////////////////////////////////////////////////
	while (cv::waitKey(1) != '\r');	//按回车退出所有程序
	cv::destroyAllWindows();	//将所有图像窗口关闭
	zed_leftimage.free(sl::MEM_CPU);
	zed_depthimage.free(sl::MEM_CPU);
	zed.close();

	return 0;
}




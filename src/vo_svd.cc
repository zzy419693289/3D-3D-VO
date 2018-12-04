#include <opencv2/opencv.hpp>
#include <Eigen/Dense>
#include <iostream>
#include <vector>
#include "vo_svd.h"
//这个工程会用到Opencv以及eigen两个库
using namespace std;
//使用SVD分解矩阵，可以通过匹配好的3D点对，快速求解两帧之间的位姿变化R和t。这里的R和t是第一帧到第二帧中点的坐标变化。当然也可以看多第二帧到第一帧的，位姿变化。
//其中R=UV^T,t=p-Rp'。其中W=USV^T,W=Q'*Q^T,Q=P-p,Q'=P'-p。p是第一帧的质心坐标，p'是第二帧的质心坐标。
//输入：pts1前一帧特征点坐标的向量；pts2后遗症特征点坐标的向量；R第二帧到第一帧的姿态旋转矩阵3*3；t第二帧到第一帧的平移向量3*1
void pose_svd_3d3d(const vector<cv::Point3f>& pts1, const vector<cv::Point3f>& pts2, cv::Mat& R, cv::Mat& t)
{
	cv::Point3f p1, p2;     // p1和p2是两帧的质心三维坐标。由于三维坐标一般以m为单位，所以这离一般用单精度运算。
	int N = pts1.size();	// 由于两帧匹配的点的数量是一致的，所以向量长度相同
	//计算质心坐标
	for (int i = 0; i<N; i++)	
	{
		p1 += pts1[i];	//注意这里的符号是重载过的，可以实现三维坐标的对应加减
		p2 += pts2[i];
	}
	p1 = cv::Point3f(p1 / N);
	p2 = cv::Point3f(p2 / N);
	//计算前后两帧的q
	vector<cv::Point3f>     q1(N), q2(N); // 定长vector初始化
	for (int i = 0; i<N; i++)
	{
		q1[i] = pts1[i] - p1;
		q2[i] = pts2[i] - p2;
	}
	// 计算W，W是一个double类型的3*3矩阵。这里使用Eigen定义一个3*3的矩阵，初始值为0。（矩阵其实还是一个定长的数组而已，这不过这里二维检索更有矩阵的样子）
	Eigen::Matrix3d W = Eigen::Matrix3d::Zero();
	for (int i = 0; i<N; i++)	//vector3d一开始是列，transpose能让其变为行，所以这里两个向量相乘会变成一个3*3的矩阵。（Eigen里面的vector其实是N*1的mat，所以和mat是同类型可以用同样重载过后的*进行矩阵积运算）
	{
		W += Eigen::Vector3d(q1[i].x, q1[i].y, q1[i].z) * Eigen::Vector3d(q2[i].x, q2[i].y, q2[i].z).transpose();	//W是前后两帧各点三维坐标的积之和。具体看推导公式
	}
	//cout << "W=" << W << endl;
	// 对W进行SVD分解。
	Eigen::JacobiSVD<Eigen::Matrix3d> svd(W, Eigen::ComputeFullU | Eigen::ComputeFullV);	//Full的U和V表示的是真实通过计算每一个特征向量U和V矩阵。如果不用Full而用Thin，则表示只计算中间的奇异值S对角矩阵
	Eigen::Matrix3d U = svd.matrixU();	//计算完后可以直接通过svd去访问U和V矩阵或者S矩阵。
	Eigen::Matrix3d V = svd.matrixV();
	//cout << "U=" << U << endl;
	//cout << "V=" << V << endl;
	//直接计算第二帧到第一帧的R和t。R=UV^T,t=p-Rp'
	Eigen::Matrix3d R_ = U* (V.transpose());	//3*3的matrix
	Eigen::Vector3d t_ = Eigen::Vector3d(p2.x, p2.y, p2.z) - R_ * Eigen::Vector3d(p1.x, p1.y, p1.z);	//3*1的matrix

	// 将其转换成cv::Mat。注意，这里的R和t是第二帧到第一帧的位姿变换，和真实的R'和t'有转换关系，R'=R^-1,t'=-R^-1*t。同时，注意由于R是正交矩阵，有R^-1=R^T。
	/*	
	R = (cv::Mat_<double>(3, 3) <<
		R_(0, 0), R_(0, 1), R_(0, 2),
		R_(1, 0), R_(1, 1), R_(1, 2),
		R_(2, 0), R_(2, 1), R_(2, 2)
		);
	t = (cv::Mat_<double>(3, 1) << t_(0, 0), t_(1, 0), t_(2, 0));
	*/
	R = (cv::Mat_<double>(3, 3) <<			//1 to 2, pose change
		R_(0, 0), R_(1, 0), R_(2, 0),
		R_(0, 1), R_(1, 1), R_(2, 1),
		R_(0, 2), R_(1, 2), R_(2, 2)
		);
	t = (cv::Mat_<double>(3, 1) << t_(0, 0), t_(1, 0), t_(2, 0));
	t =-1*R*t;
}

#include "PoseOptimizer.h"
#include <Eigen/Core>
#include <Eigen/Geometry>
#include <g2o/core/base_vertex.h>
#include <g2o/core/base_unary_edge.h>
#include <g2o/core/block_solver.h>
#include <g2o/core/optimization_algorithm_levenberg.h>
#include <g2o/solvers/eigen/linear_solver_eigen.h>
#include <g2o/types/sba/types_six_dof_expmap.h>
#include <opencv2/core/eigen.hpp>
#include "utils.h"

namespace MVSO
{

	class EdgeProjectXYZRGBDPoseOnly : public g2o::BaseUnaryEdge<3, Eigen::Vector3d, g2o::VertexSE3Expmap>
	{
	public:
		EIGEN_MAKE_ALIGNED_OPERATOR_NEW;
		EdgeProjectXYZRGBDPoseOnly(const Eigen::Vector3d& point) : _point(point) {}

		virtual void computeError()
		{
			const g2o::VertexSE3Expmap* pose = static_cast<const g2o::VertexSE3Expmap*> (_vertices[0]);
			// measurement is p, point is p'
			_error = _measurement - pose->estimate().map(_point);
		}

		virtual void linearizeOplus()
		{
			g2o::VertexSE3Expmap* pose = static_cast<g2o::VertexSE3Expmap *>(_vertices[0]);
			g2o::SE3Quat T(pose->estimate());
			Eigen::Vector3d xyz_trans = T.map(_point);
			double x = xyz_trans[0];
			double y = xyz_trans[1];
			double z = xyz_trans[2];

			_jacobianOplusXi(0, 0) = 0;
			_jacobianOplusXi(0, 1) = -z;
			_jacobianOplusXi(0, 2) = y;
			_jacobianOplusXi(0, 3) = -1;
			_jacobianOplusXi(0, 4) = 0;
			_jacobianOplusXi(0, 5) = 0;

			_jacobianOplusXi(1, 0) = z;
			_jacobianOplusXi(1, 1) = 0;
			_jacobianOplusXi(1, 2) = -x;
			_jacobianOplusXi(1, 3) = 0;
			_jacobianOplusXi(1, 4) = -1;
			_jacobianOplusXi(1, 5) = 0;

			_jacobianOplusXi(2, 0) = -y;
			_jacobianOplusXi(2, 1) = x;
			_jacobianOplusXi(2, 2) = 0;
			_jacobianOplusXi(2, 3) = 0;
			_jacobianOplusXi(2, 4) = 0;
			_jacobianOplusXi(2, 5) = -1;
		}

		bool read(std::istream& in) { return true; }
		bool write(std::ostream& out) const { return true; }
	protected:
		Eigen::Vector3d _point;
	};



MVSO::PoseOptimizer::PoseOptimizer(CameraModel & camera): camera_(camera)
{
}

void PoseOptimizer::optimizePose(const std::vector<cv::Point3f> points_3d, const std::vector<cv::Point2f> points_2d, cv::Mat & R, cv::Mat & t)
{


	// 初始化g2o
	typedef g2o::BlockSolver< g2o::BlockSolverTraits<6, 3> > Block;  // pose 维度为 6, landmark 维度为 3
	Block::LinearSolverType* linearSolver = new g2o::LinearSolverEigen<Block::PoseMatrixType>(); // 线性方程求解器
	Block* solver_ptr = new Block(linearSolver);     // 矩阵块求解器
	g2o::OptimizationAlgorithmLevenberg* solver = new g2o::OptimizationAlgorithmLevenberg(solver_ptr);
	g2o::SparseOptimizer optimizer;
	optimizer.setAlgorithm(solver);

	// vertex
	g2o::VertexSE3Expmap* pose = new g2o::VertexSE3Expmap(); // camera pose
	Eigen::Matrix3d R_mat;
	R_mat <<
		R.at<double>(0, 0), R.at<double>(0, 1), R.at<double>(0, 2),
		R.at<double>(1, 0), R.at<double>(1, 1), R.at<double>(1, 2),
		R.at<double>(2, 0), R.at<double>(2, 1), R.at<double>(2, 2);
	pose->setId(0);
	pose->setEstimate(g2o::SE3Quat(
		R_mat,
		Eigen::Vector3d(t.at<double>(0, 0), t.at<double>(1, 0), t.at<double>(2, 0))
	));
	optimizer.addVertex(pose);

	int index = 1;
	for (const cv::Point3f p : points_3d)   // landmarks
	{
		g2o::VertexSBAPointXYZ* point = new g2o::VertexSBAPointXYZ();
		point->setId(index++);
		point->setEstimate(Eigen::Vector3d(p.x, p.y, p.z));
		point->setMarginalized(true); // g2o 中必须设置 marg 参见第十讲内容
		optimizer.addVertex(point);
	}

	// parameter: camera intrinsics
	g2o::CameraParameters* camera = new g2o::CameraParameters(
		camera_.fx_, Eigen::Vector2d(camera_.cx_, camera_.cy_), camera_.bf_/camera_.fx_
	);
	camera->setId(0);
	optimizer.addParameter(camera);

	// edges
	index = 1;
	for (const cv::Point2f p : points_2d)
	{
		g2o::EdgeProjectXYZ2UV* edge = new g2o::EdgeProjectXYZ2UV();
		edge->setId(index);
		edge->setVertex(0, dynamic_cast<g2o::VertexSBAPointXYZ*> (optimizer.vertex(index)));
		edge->setVertex(1, pose);
		edge->setMeasurement(Eigen::Vector2d(p.x, p.y));
		edge->setParameterId(0, 0);
		edge->setInformation(Eigen::Matrix2d::Identity());
		optimizer.addEdge(edge);
		index++;
	}

	TicTok tic;
	optimizer.setVerbose(false);
	optimizer.initializeOptimization();
	optimizer.optimize(100);
	
	std::cout << "optimization costs time: " << tic.tok() << " ms" << std::endl;

	auto pose_optimal = Eigen::Isometry3d(pose->estimate());
	cv::eigen2cv(pose_optimal.rotation(), R);
	t.at<double>(0, 0) = pose_optimal.translation()(0);
	t.at<double>(1, 0) = pose_optimal.translation()(1);
	t.at<double>(2, 0) = pose_optimal.translation()(2);

}

void PoseOptimizer::optimizePose(const std::vector<cv::Point3f> points_3d, const std::vector<cv::Point2f> points_2d, const std::vector<double>& weights, cv::Mat & R, cv::Mat & t)
{

	// 初始化g2o
	typedef g2o::BlockSolver< g2o::BlockSolverTraits<6, 3> > Block;  // pose 维度为 6, landmark 维度为 3
	Block::LinearSolverType* linearSolver = new g2o::LinearSolverEigen<Block::PoseMatrixType>(); // 线性方程求解器
	Block* solver_ptr = new Block(linearSolver);     // 矩阵块求解器
	g2o::OptimizationAlgorithmLevenberg* solver = new g2o::OptimizationAlgorithmLevenberg(solver_ptr);
	g2o::SparseOptimizer optimizer;
	optimizer.setAlgorithm(solver);

	// vertex
	g2o::VertexSE3Expmap* pose = new g2o::VertexSE3Expmap(); // camera pose
	Eigen::Matrix3d R_mat;
	R_mat <<
		R.at<double>(0, 0), R.at<double>(0, 1), R.at<double>(0, 2),
		R.at<double>(1, 0), R.at<double>(1, 1), R.at<double>(1, 2),
		R.at<double>(2, 0), R.at<double>(2, 1), R.at<double>(2, 2);
	pose->setId(0);
	pose->setEstimate(g2o::SE3Quat(
		R_mat,
		Eigen::Vector3d(t.at<double>(0, 0), t.at<double>(1, 0), t.at<double>(2, 0))
	));
	optimizer.addVertex(pose);

	int index = 1;
	for (const cv::Point3f p : points_3d)   // landmarks
	{
		g2o::VertexSBAPointXYZ* point = new g2o::VertexSBAPointXYZ();
		point->setId(index++);
		point->setEstimate(Eigen::Vector3d(p.x, p.y, p.z));
		point->setMarginalized(true); // g2o 中必须设置 marg 参见第十讲内容
		optimizer.addVertex(point);
	}

	// parameter: camera intrinsics
	g2o::CameraParameters* camera = new g2o::CameraParameters(
		camera_.fx_, Eigen::Vector2d(camera_.cx_, camera_.cy_), camera_.bf_ / camera_.fx_
	);
	camera->setId(0);
	optimizer.addParameter(camera);

	// edges
	index = 1;
	for(int i = 0; i < points_2d.size(); i++)
	{
		auto p = points_2d[i];
		g2o::EdgeProjectXYZ2UV* edge = new g2o::EdgeProjectXYZ2UV();
		edge->setId(index);
		edge->setVertex(0, dynamic_cast<g2o::VertexSBAPointXYZ*> (optimizer.vertex(index)));
		edge->setVertex(1, pose);
		edge->setMeasurement(Eigen::Vector2d(p.x, p.y));
		edge->setParameterId(0, 0);
		edge->setInformation(Eigen::Matrix2d::Identity()*weights[i]);
		optimizer.addEdge(edge);
		index++;
	}

	TicTok tic;
	optimizer.setVerbose(false);
	optimizer.initializeOptimization();
	optimizer.optimize(100);

	std::cout << "optimization costs time: " << tic.tok() << " ms" << std::endl;

	auto pose_optimal = Eigen::Isometry3d(pose->estimate());
	cv::eigen2cv(pose_optimal.rotation(), R);
	t.at<double>(0, 0) = pose_optimal.translation()(0);
	t.at<double>(1, 0) = pose_optimal.translation()(1);
	t.at<double>(2, 0) = pose_optimal.translation()(2);
}


void PoseOptimizer::optimizePose(
	const std::vector< cv::Point3f > points3d_t0,
	const std::vector< cv::Point3f > points3d_t1,
	cv::Mat& R, cv::Mat& t)
{
	// 初始化g2o
	typedef g2o::BlockSolver< g2o::BlockSolverTraits<6, 3> > Block;  // pose维度为 6, landmark 维度为 3
	Block::LinearSolverType* linearSolver = new g2o::LinearSolverEigen<Block::PoseMatrixType>(); // 线性方程求解器
	Block* solver_ptr = new Block(linearSolver);      // 矩阵块求解器
	g2o::OptimizationAlgorithmLevenberg* solver = new g2o::OptimizationAlgorithmLevenberg(solver_ptr);
	g2o::SparseOptimizer optimizer;
	optimizer.setAlgorithm(solver);

	// vertex
	g2o::VertexSE3Expmap* pose = new g2o::VertexSE3Expmap(); // camera pose
	pose->setId(0);
	pose->setEstimate(g2o::SE3Quat(
		Eigen::Matrix3d::Identity(),
		Eigen::Vector3d(0, 0, 0)
	));
	optimizer.addVertex(pose);

	// edges
	int index = 1;
	std::vector<EdgeProjectXYZRGBDPoseOnly*> edges;
	for (size_t i = 0; i < points3d_t0.size(); i++)
	{
		EdgeProjectXYZRGBDPoseOnly* edge = new EdgeProjectXYZRGBDPoseOnly(
			Eigen::Vector3d(points3d_t1[i].x, points3d_t1[i].y, points3d_t1[i].z));
		edge->setId(index);
		edge->setVertex(0, dynamic_cast<g2o::VertexSE3Expmap*> (pose));
		edge->setMeasurement(Eigen::Vector3d(
			points3d_t0[i].x, points3d_t0[i].y, points3d_t0[i].z));
		edge->setInformation(Eigen::Matrix3d::Identity()*1e4);
		optimizer.addEdge(edge);
		index++;
		edges.push_back(edge);
	}

	TicTok tic;
	optimizer.setVerbose(false);
	optimizer.initializeOptimization();
	optimizer.optimize(10);
	
	std::cout << "optimization costs time: " << tic.tok() << " ms." << std::endl;

	auto pose_optimal = Eigen::Isometry3d(pose->estimate());
	cv::eigen2cv(pose_optimal.rotation(), R);
	t.at<double>(0, 0) = pose_optimal.translation()(0);
	t.at<double>(1, 0) = pose_optimal.translation()(1);
	t.at<double>(2, 0) = pose_optimal.translation()(2);
}


PoseOptimizer::~PoseOptimizer()
{
}

}


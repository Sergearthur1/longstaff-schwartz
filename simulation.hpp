#include "linear-regression.hpp"

#ifndef simulation
#define simulation

class PathSimulation{
	/*
	 * Virtual Class to simulate any kind of trajctories.
	 * size: nb of point in the trajectories
	 * T: time of the end of the path.
	 * h: time between two point in the trajectory.
	 */
	protected:
		std::vector<Eigen::Matrix<double,Eigen::Dynamic,1>> path;
		int size;
		double T;
		double h;
	public:
		PathSimulation(int size,double T): size(size), T(T), path(size){
			h = T / (1. * (size - 1));
			return;
		}
		int get_size()const{return size;}
		Eigen::Matrix<double,Eigen::Dynamic,1> operator [](int n){return path[n];}
		friend std::ostream & operator <<(std::ostream & os, PathSimulation p){
			for(int i=0; i< p.size; ++i){
				os << "X_{" << i * p.h << "} : " << p.path[i] << std::endl;
			}
			return os;
	}
};

class BS:public PathSimulation{
	/*
	 * A path simulation of the Black Scholes Model.
	 */
	public:
		BS(int size, std::mt19937 & gen, Eigen::Matrix<double,Eigen::Dynamic,1> & sigma, Eigen::Matrix<double,Eigen::Dynamic,1> & drift, Eigen::Matrix<double,Eigen::Dynamic,1> & x0, double T, Eigen::Matrix<double,Eigen::Dynamic,Eigen::Dynamic> & corr): PathSimulation(size, T){
			/*
			 * size: nb of point in the trajectories.
			 * gen: random generator.
			 * sigma: volatility vector of each composent of the process.
			 * drift: drift vector of each composent of the process.
			 * x0: initial state vector of the process.
			 * T: time of the end of the path.
			 * corr: correlation matrix between all composent of the process.
			 */
			Eigen::Matrix<double,Eigen::Dynamic,Eigen::Dynamic> corr_cholesky_decomposition = cholesky(corr);
			int N = x0.rows();
			Eigen::Matrix<double, Eigen::Dynamic, 1> Z(N);
			std::normal_distribution<double> norm(0,1);
			this->path[0].resize(N,1);
			this->path[0] = x0;
			for(int i=1; i<size; i++){
				for(int j = 0; j<N; j++){
					Z(j) = norm(gen);
				}
				Z = corr_cholesky_decomposition * Z;
				this->path[i].resize(N,1);
				for(int j = 0; j<N; j++){
					this->path[i](j) = this->path[i-1](j) * exp((drift(j) - sigma(j) * sigma(j) * 0.5) * this->h * i + sigma(j) * sqrt(this->h * i) * Z(j)); 
				}
			}
			return; 
		}
};

class IIDUnif: public PathSimulation{
	/*
	 * A simulation where each point follows a uniform law independent of the other points.
	 */
	public:
		IIDUnif(int size, double T, std::mt19937 & gen, double lim_inf, double lim_sup): PathSimulation(size, T){
			/*
			 * size: nb of point in the trajectories.
			 * T: time of the end of the path.
			 * gen: random generator.
			 * lim_inf: minimum value that the simulated uniform law can take.
			 * lim_sup: maximum value that the simulated uniform law can take.
			*/
			std::uniform_real_distribution<double> U(lim_inf, lim_sup);
			this->path[0].resize(1,1);
			this->path[0](0) = 0;
			for(int i=1;i<size;++i){
				this->path[i].resize(1,1);
				this->path[i](0) = U(gen);
			}
			return;
		}
};
# endif

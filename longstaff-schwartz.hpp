#include "linear-regression.hpp"
#include "simulation.hpp"
template<class Function, class Path>
double longstaff_schwartz(const int d,const int K, std::vector<Path> & Simulations, Function & g){
	/*
	 * d: dimension of the simulated trajectories.
	 * K: degree of the polynomial used for the regression.
	 * Simulations: list of all  i.i.d simulated trajectories.
	 * g: cost function. Takes vectors of size K as input and returns doubles.
	 * 
	 * return -> g(X_{tau}) where X has the same law than trajectories in Simulations, an tau is the optimal stopping time in order to maximize the cost function.
	 */
	const int M = Simulations.size();
	const int N = Simulations[0].get_size();
	std::vector<int> tau(M, N-1);
	double x;
	double intercept;
	Eigen::Matrix<double, Eigen::Dynamic, 1> beta(N * K,1);
	Eigen::Matrix<double, Eigen::Dynamic, Eigen::Dynamic> X(M, d * K);
	std::vector<double> g_n(M,1);
	Eigen::Matrix<double,Eigen::Dynamic,1>  g_tau_n(M,1);
	for(int i = N-2; i>=0; --i){
		if(i==0){
			beta = Eigen::Matrix<double, Eigen::Dynamic, 1>::Zero(d * K,1);
			intercept = std::inner_product(
				Simulations.begin(),
				Simulations.end(),
				tau.begin(),
				0.,
				[](double a, double b) ->double{return a+b;},
				[&g](Path path, int n) ->double{return g(path[n]);}
			) / (1. * M);
			std::transform(
				Simulations.begin(),
				Simulations.end(),
				g_n.begin(),
				[&g](Path path)->double{return g(path[0]);}
			);
		}
		else{
			for(int j=0; j<M;++j){
				g_n[j] = g(Simulations[j][i]);
				g_tau_n(j) = g(Simulations[j][tau[j]]);
				for(int l=0;l<d;++l){
					x = Simulations[j][i](l);
					for(int k=1;k<=K;++k){
						X(j,d*l+(k-1)) =  x;
						x *=x;
					}
				}
			}
			LinearRegression LinReg(X,g_tau_n,M, d*K,true,false);
			beta = LinReg.get_coef();
			intercept = LinReg.get_intercept();
		}
		for(int j=0; j<M; ++j){
			if (g_n[j] >= intercept + X.row(j) * beta){
				tau[j] = i;
			}
		}
		
	}
	return std::inner_product(
		Simulations.begin(),
		Simulations.end(),
		tau.begin(),
		0.,
		[](double a, double b) ->double{return a+b;},
		[&g](Path path, int n) ->double{return g(path[n]);}
	) / (1.  * M);
}

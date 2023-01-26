#include "linear-regression.hpp"
#include "simulation.hpp"
#include "longstaff-schwartz.hpp"
template<class Type>
std::ostream& operator << (std::ostream& res, std::vector<Type> v){
	for (auto it = v.begin() ; it != v.end() ; ++it){
		res << *it << " ";
	}
	res << std::endl;
	return res;
}



int main(){
	//test LInearRegression
	std::cout << "===== test linear regression library ======"<<std::endl;
	const int n=25;
	const int p=4;
	Eigen::Matrix<double, -1, 1> ones(n,1);
				for(int i = 0; i<n; ++i){
					ones(i) = 5.58;
				}
	Eigen::Matrix<double, -1,-1> X = Eigen::Matrix<double, -1,-1>::Random(n,p);
	Eigen::Matrix<double, -1,1> theta = 10 * Eigen::Matrix<double, -1,1>::Random(p,1);
	Eigen::Matrix<double, -1,1> epsilon = 0.1 * Eigen::Matrix<double, -1,1>::Random(n,1);
	Eigen::Matrix<double, -1,1>  Y = ones + X * theta + epsilon;
	std::cout << "X: "<< X << std::endl;
	std::cout << "Y: "<< Y << std::endl;
	//std::cout << "theta: " << theta << std::endl;
	
    LinearRegression LinReg(X,Y,n,p,true,true);
	std::cout <<"coefs: "<< LinReg.get_coef() << std::endl;
	std::cout <<"intercept: "<< LinReg.get_intercept() << std::endl;
	std::cout << "score: " << LinReg.get_score() << std::endl;
	for(int i=0;i<25;++i){
		std::cout << "Distance de cook " << i << ": "<< LinReg.cook_distance(i)<<std::endl;
	}
	//test Simulation BS
	/*
	std::mt19937 G(time(NULL));
	Eigen::Matrix<double,-1,1> sigma(2);
	sigma << 0.4, 0.3;
	Eigen::Matrix<double,-1,1> drift(2);
	drift << 0, 0;
	Eigen::Matrix<double,-1,1> x0(2);
	x0 << 1, 2;
	Eigen::Matrix<double,-1,-1> corr(2,2);
	corr << 1, 0.8, 0.8, 1;
	BS simu(10,G, sigma, drift, x0, 1, corr);
	std::cout << simu[9];
	*/
	//test longstaff-schwartz American call basket option
	std::cout << "===== Longstaff-schwartz algorithm for American call basket option pricing in a BS framework ======"<<std::endl;
	std::mt19937 G(time(NULL));
	Eigen::Matrix<double,-1,1> sigma(2,1);
	sigma << 0.4, 0.3;
	Eigen::Matrix<double,-1,1> drift(2,1);
	drift << 0, 0;
	Eigen::Matrix<double,-1,1> x0(2,1);
	x0 << 2, 3;
	Eigen::Matrix<double,-1,-1> corr(2,2);
	corr << 1, 0.5, 0.5, 1;
	std::vector<BS> simus;
	for(int i=0;i<10000;i++){
		BS simu(20,G, sigma, drift, x0, 1, corr);
		simus.push_back(simu);
	}
	double strike = 3.;
	auto g = [strike](Eigen::Matrix<double,2,1> v)-> double{return fmax(v(0) + v(1) - strike,0);};
	double price = longstaff_schwartz(2, 2, simus, g);
	std::cout << "parameters: " << std::endl;
	std::cout << "sigma: " << std::endl << sigma <<std::endl;
	std::cout << "drift: "<< std::endl << drift <<std::endl;
	std::cout << "correlation matrix:"<< std::endl << corr << std::endl;
	std::cout << "x0: "<< std::endl << x0 << std::endl;
	std::cout << "strike: " << strike << std::endl;
	std::cout << "result:" << price << std::endl;
	//test longstaff-schwartz  dice brain-teaser
	std::cout << "===== Longstaff-schwartz algorithm to crack the dice brain-teaser ======"<<std::endl;
	//std::mt19937 G(time(NULL));
	std::vector<IIDUnif> simus_2;
	for(int i=0;i<100000;i++){
		IIDUnif simu(4, 3.,G, 0., 1.);
		simus_2.push_back(simu);
	}
	auto g2 = [](Eigen::Matrix<double,1,1> v)-> double{return v(0);};
	double r = longstaff_schwartz(1, 5, simus_2, g2);
	std::cout <<"result:" << r << std::endl;
	return 0;
}

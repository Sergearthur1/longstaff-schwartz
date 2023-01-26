#include <Eigen/Dense>
#include <iostream>
#include <vector>
#include <fstream>
#include <cmath>
#include <string>
#include <random>
#include <ctime>
#include <map>
#include <algorithm>

#ifndef linear_regression_part
#define linear_regression_part
class LinearRegression{
	/*
	 * Class to perfom linear regression and compute some statistic on it like R2 score, MSE or cook distance.
	 */
	private:
		double intercept;
		Eigen::Matrix<double,Eigen::Dynamic, 1> coef;
		double score;
		double hat_sigma_2;
		Eigen::Matrix<double, Eigen::Dynamic, Eigen::Dynamic> X;
		Eigen::Matrix<double, Eigen::Dynamic, Eigen::Dynamic> XtX_inv;
		Eigen::Matrix<double,Eigen::Dynamic, 1> Y;
		Eigen::Matrix<double,Eigen::Dynamic, 1> studentized_residuals;
		int nb_param;
		int nb_sample;
	public:
		LinearRegression(Eigen::Matrix<double,Eigen::Dynamic, Eigen::Dynamic> X, Eigen::Matrix<double, Eigen::Dynamic, 1> Y,int sample_size, int nb_param, bool compute_intercept, bool compute_stat);
		Eigen::Matrix<double,Eigen::Dynamic, 1> get_coef()const{
			return coef;
		}
		double get_intercept()const{
			return intercept;
		}
		double get_score()const{
			return score;
		}
		double compute_MSE()const{
			Eigen::Matrix<double, Eigen::Dynamic, 1> ones(nb_sample,1);
			for(int i = 0; i<nb_sample; ++i){
				ones(i) = 1.;
			}
			return (Y -  intercept * ones + X * coef).squaredNorm();
		}
		double cook_distance(int i){
			double error_i = Y(i) - (intercept + X.row(i) * coef);
			double hii;
			if (intercept == 0){
				hii = X.row(i) * XtX_inv * X.row(i).transpose();
			}
			else{
				Eigen::Matrix<double,-1, -1> xi(1, nb_param + 1);
				xi << 1, X.row(i);
				hii = (xi * XtX_inv * xi.transpose())(0);
			}
			double mse = this->compute_MSE();
			return error_i * error_i * hii / (nb_param * mse * (1 - hii) * (1 - hii));
		}
};


LinearRegression::LinearRegression(Eigen::Matrix<double,Eigen::Dynamic, Eigen::Dynamic> X, Eigen::Matrix<double, Eigen::Dynamic, 1> Y,int sample_size, int nb_param, bool compute_intercept, bool compute_stat):X(X), Y(Y), nb_param(nb_param), nb_sample(sample_size){
			/*
			 * X: matrix of dimension <nb_param, sample_size> that containt explanatory variable.
			 * Y: vector of dimension <1, sample_size> that containt the variable to predict.
			 * sample_size: number of observation.
			 * nb_param: number of explanatory variable.
			 * compute_intercept: if True, a column full of ones is add to X.
			 * compute_stat: if True, all the statistics of the model will be calculated, otherwise only the coefficients of the regression will be calculated.
			*/
			Eigen::Matrix<double, Eigen::Dynamic, 1> ones(sample_size,1);
			for(int i = 0; i<sample_size; ++i){
				ones(i) = 1.;
			}
			if(compute_intercept){
				Eigen::Matrix<double, Eigen::Dynamic, Eigen::Dynamic> new_X(sample_size, nb_param + 1);
				new_X << ones, X;
				Eigen::Matrix<double, Eigen::Dynamic, Eigen::Dynamic> XtX = new_X.transpose() * new_X;
				if (XtX.determinant() == 0){
					std::cout << "X not injective!" << std::endl;
					coef = Eigen::Matrix<double, Eigen::Dynamic, Eigen::Dynamic>::Zero(nb_param, 1);
					intercept = 0.;
				}
				else{
					XtX_inv.resize(nb_param + 1, nb_param + 1);
					XtX_inv = XtX.inverse();
					Eigen::Matrix<double, Eigen::Dynamic, 1> new_beta = XtX_inv * new_X.transpose() * Y;
					intercept = new_beta(0);
					coef = new_beta.segment(1, nb_param);
				}
			}
			else{
				intercept = 0.;
				Eigen::Matrix<double, Eigen::Dynamic, Eigen::Dynamic> XtX = X.transpose() * X;
				if (XtX.determinant() == 0){
					std::cout << "X not injective!" << std::endl;
					coef = Eigen::Matrix<double, Eigen::Dynamic, 1>::Zero(nb_param, 1);
				}
				else{
					XtX_inv.resize(nb_param, nb_param);
					XtX_inv = XtX.inverse();
					coef = XtX_inv * X.transpose() * Y;
				}
					
			}
			if(compute_stat){
				Eigen::Matrix<double, Eigen::Dynamic, 1> residuals = Y - ( intercept * ones + X * coef);
				hat_sigma_2 = (residuals.transpose() * residuals);
				hat_sigma_2 /= (sample_size - nb_param);
				double Y_mean = Y.mean();
				Eigen::Matrix<double, Eigen::Dynamic, 1> mean_predictor_error(sample_size,1);
				mean_predictor_error = Y - Y.mean() * ones;
				double score_up = residuals.transpose() * residuals;
				double score_down = mean_predictor_error.transpose() * mean_predictor_error;
				score = 1. - score_up / score_down;
				Eigen::Matrix<double, Eigen::Dynamic, 1> normalized_residuals = residuals - residuals.mean() * ones;
				normalized_residuals /= hat_sigma_2;
				Eigen::Matrix<double, Eigen::Dynamic, 1> tamp_residuals(sample_size, 1);
				for(int i = 0; i < normalized_residuals.rows(); i++){
					double t = normalized_residuals(i);
					tamp_residuals(i) = t * sqrt((sample_size - nb_param - 1) / (sample_size - nb_param - t * t));
				}
				studentized_residuals = tamp_residuals;
			}
			return;
		}
		
template<class Corp>
Eigen::Matrix<Corp,Eigen::Dynamic,Eigen::Dynamic> cholesky(Eigen::Matrix<Corp, Eigen::Dynamic,Eigen::Dynamic> & mat){
	/*
	 * Compute the cholesky decompostion of a symmetrical, defined and positive matrix.
	 */
	int N = mat.rows();
	Eigen::Matrix<double,Eigen::Dynamic,Eigen::Dynamic> res = Eigen::Matrix<Corp,Eigen::Dynamic,Eigen::Dynamic>::Zero(N,N);
	for(int i=0; i<N; ++i){
		Corp square_sum = 0;
		for(int k = 0; k<i;k++){
			square_sum += res(i,k) * res(i,k);
		}
		res(i,i) = sqrt(mat(i,i) - square_sum);
		for(int j=i+1;j<N;++j){
			Corp cross_sum = 0;
			for(int k = 0; k<i; k++){
				cross_sum += res(j,k) * res(i,k);
			}
			res(j,i) = (mat(j,i) - cross_sum) / res(i,i);
		}
	}
	return res;
}

#endif


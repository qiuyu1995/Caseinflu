// [[Rcpp::depends(RcppArmadillo)]]
#include <RcppArmadillo.h>


/*  Function dealing with the case when E is empty

 When E is empty, beta_0 is not unique and is between [Low, High]. We set beta_0 to be either Low or High depend on j is in L or R so that E becomes non-empty.
*/
arma::vec Empty_E(arma::vec a, arma::mat B, arma::vec c, arma::vec beta_w, arma::uvec L, arma::uvec R, int j_in_L, int j) {
	
	double beta_0_temp = 0.0;
	double beta_0_update = 0.0;
	int index_move_to_E = 0;
	arma::vec output(2); 
	int L_size = L.size();
	int R_size = R.size();

	if ((j_in_L && a[j] < 0) || (!j_in_L && a[j] > 0)) {
		beta_0_update = 1e8;
		// Set beta_0 as High
		for (unsigned i=0; i<L_size; i++) {
			if (a[L[i]] > 0) {
				beta_0_temp = (- arma::dot(B.row(L[i]), arma::trans(beta_w)) - c[L[i]]) / a[L[i]];
				if (beta_0_temp < beta_0_update) {
					beta_0_update = beta_0_temp;
					index_move_to_E = - (int) i - 1;
				}
			}
		}
		for (unsigned i=0; i<R_size; i++) {
			if (a[R[i]] < 0) {
				beta_0_temp = (- arma::dot(B.row(R[i]), arma::trans(beta_w)) - c[R[i]]) / a[R[i]];
				if (beta_0_temp < beta_0_update) {
					beta_0_update = beta_0_temp;
					index_move_to_E = (int) i;
				}
			}
		}
	} else {
		beta_0_update = -1e8;
		// Set beta_0 as Low
		for (unsigned i=0; i<L_size; i++) {
			if (a[L[i]] < 0) {
				beta_0_temp = (- arma::dot(B.row(L[i]), arma::trans(beta_w)) - c[L[i]]) / a[L[i]];
				if (beta_0_temp > beta_0_update) {
					beta_0_update = beta_0_temp;
					index_move_to_E = - (int) i - 1;
				}
			}
		}
		for (unsigned i=0; i<R_size; i++) {
			if (a[R[i]] > 0) {
				beta_0_temp = (- arma::dot(B.row(R[i]), arma::trans(beta_w)) - c[R[i]]) / a[R[i]];
				if (beta_0_temp > beta_0_update) {
					beta_0_update = beta_0_temp;
					index_move_to_E = (int) i;
				}
			}
		}
	}
	output[0] = beta_0_update;
	output[1] = index_move_to_E;
	return output;
}


// Update or downdate the inverse matrix by changing only one point
arma::mat update(arma::mat X_Xt_inverse, arma::mat X, arma::uvec E, int E_size) {
       
	arma::uword k = E_size;
	arma::uword index_insert = E[k-1];
	arma::mat X_Xt_inverse_new(k, k);
	arma::uvec E_prev = E.subvec(0, k-2);
	arma::mat X_E_prev = X.rows(E_prev);
	
	arma::vec v = (X.row(index_insert)).t();
	arma::vec u1 = X_E_prev * v;
	arma::vec u2 = X_Xt_inverse * u1;
	double d = 1.0/(arma::dot(v, v) - dot(u1, u2));
	arma::vec u3 = d * u2;
	arma::mat F11_inv = X_Xt_inverse + d * u2 * u2.t();
	X_Xt_inverse_new(k-1, k-1) = d;
	X_Xt_inverse_new(arma::span(0, k-2), k-1) = -u3;
	X_Xt_inverse_new(k-1, arma::span(0, k-2)) = -u3.t();
	X_Xt_inverse_new(arma::span(0, k-2), arma::span(0, k-2)) = F11_inv;

	return X_Xt_inverse_new;
}

arma::mat downdate(arma::mat X_Xt_inverse, arma::mat X, arma::uvec E_prev, int E_prev_size, int index_remove) {

	arma::uword k = E_prev_size;
	arma::mat X_Xt_inverse_new(k-1, k-1);
	
	if (index_remove < k-1) {
		arma::rowvec tmpv1 = X_Xt_inverse.row(index_remove);
		X_Xt_inverse.rows(index_remove, k-2) = X_Xt_inverse.rows(index_remove+1, k-1);
		X_Xt_inverse.row(k-1) = tmpv1;
		arma::vec tmpv2 = X_Xt_inverse.col(index_remove);
		X_Xt_inverse.cols(index_remove, k-2) = X_Xt_inverse.cols(index_remove+1, k-1);
		X_Xt_inverse.col(k-1) = tmpv2;
	}
	arma::mat F11_inv = X_Xt_inverse.submat(0, 0, k-2, k-2);
	double d = X_Xt_inverse(k-1, k-1);
	arma::vec u = - X_Xt_inverse(arma::span(0, k-2), k-1) / d;
	X_Xt_inverse_new = F11_inv - d * u * u.t();
	return X_Xt_inverse_new;
}

//'Case-weight adjusted solution path for L2 regularized nonsmooth problem (quantile regression and svm)
//'
//' @description
//' Path-following algorithm to exactly solve 
//' 	(beta_{0,w}, beta_{w}) = argmin_{beta_0, beta} \sum_{i \neq j} f(g_i(beta_0, beta)) + w*f(g_{j}(beta_0, beta)) + lambda / 2 * \|beta\|_2^2
//' for 0 <= w <= 1, where g_i(beta_0, beta) = a_i beta_0 + b_i^T beta + c_i and f(r) = alpha_0 max(r, 0) + alpha_1 max(-r, 0)
//'
//' @param a A \eqn{n \times 1} vector and a^T = (a_1, ..., a_n)
//' @param B A \eqn{n \times p} matrix and B^T = (b_1, ..., b_n)
//' @param c A \eqn{n \times 1} matrix and c^T = (b_1, ..., b_n) 
//' @param lam Regularization parameter for L2 penalty
//' @param alpha_0 A scalar in the definition of f(r)
//' @param alpha_1 A scalar in the definition of f(r)
//' @param j Index of the observation that is attached a weight
//' @param beta_0_w0 A scalar, which is the true value of beta_{0,w} when w = w_0 = 1
//' @param beta_w0 A \eqn{p \times 1} vector, which is the true value of beta_{w} when w = w_0 = 1
//' @param theta_w0 A \eqn{n \times 1} vector, which is the true value of the dual variable when w = w_0 = 1
//'
//' @details
//' This function will be called by function CaseInfluence_nonsmooth to generate case influence graph for each case.
//'
//' @return W_vec A list of breakout points
//' @return Beta_0 True values of beta_{0,w} at breakout points
//' @return Beta True values of beta_{w} at breakout points
Rcpp::List case_path_nonsmooth_with_init(arma::vec a, arma::mat B, arma::vec c, double lam, double alpha_0, double alpha_1, int j, double beta_0_w0, arma::vec beta_w0, arma::vec theta_w0, 
	arma::uvec E_init, arma::uvec L_init, arma::uvec R_init, arma::vec r_init, arma::mat tilde_B_Bt_inverse_init){
	const int n = B.n_rows;
	const int p = B.n_cols;
	
  // Store breakpoints and solutions
	arma::mat Beta;
	Beta.insert_cols(Beta.n_cols, beta_w0);

	arma::vec Beta_0;
	arma::vec beta_0_insert(1);
	beta_0_insert(0) = beta_0_w0;
	Beta_0.insert_rows(Beta_0.size(), beta_0_insert);

	arma::vec W_vec;
	arma::vec w_insert(1);
	w_insert(0) = 1; 
	W_vec.insert_rows(W_vec.size(), w_insert);

	double beta_0_w = beta_0_w0;
	arma::vec beta_w = beta_w0;
	arma::vec r = r_init;

	// Define \tilde B, a_j, tilde_B_j
	arma::mat tilde_B(n, p+1);
	tilde_B.col(0) = a;
	tilde_B.cols(1, p) = B;

	const double a_j = a(j);
	arma::vec tilde_B_j = tilde_B.row(j).t();
	
	// Declare and initialize three elbow sets, and compute theta_E
	arma::uvec E = E_init;
	arma::uvec L = L_init;
	arma::uvec R = R_init;
	int E_size = E.size();
	int L_size = L.size();
	int R_size = R.size();

	arma::uvec index_insert(1);
	int index_j = -1;
	int j_in_L = -1;
	arma::vec theta_insert(1);
	arma::vec theta_E(E_size);
	theta_E = theta_w0.elem(E);

  // Declare variables 
	int m = 0;
	double w_m = 1.0;
	double w_m_next = 1.0;

	double T = 0.0;
	arma::vec d_0m(1);
	arma::vec d_m;
	arma::vec h_m(n);
	arma::mat tilde_B_Bt_inverse;

	arma::vec w_1_alpha1_temp;
	double w_1_alpha1_max = 0.0;
	int w_1_alpha1_index = 0;
	arma::vec w_1_alpha0_temp;
	double w_1_alpha0_max = 0.0;
	int w_1_alpha0_index = 0;
	double w_1_max = 0.0;
	int w_1_index = 0;

	arma::vec w_2_L_temp;
	double w_2_L_max = 0.0;
	int w_2_L_index = 0;
	arma::vec w_2_R_temp;
	double w_2_R_max = 0.0;
	int w_2_R_index = 0;
	double w_2_max = 0.0;
	int w_2_index = 0;
	int w_2_L_is_max = 1;
	const double INF = 1e8;

	int index_in_elbow = 0;
	arma::vec E_empty_output(2);
	double beta_0_update = 0.0;

	// Initialize the tilde_B_Bt_inverse
	if (E_size) {
		tilde_B_Bt_inverse = tilde_B_Bt_inverse_init;
	}

	// Check where case j is located
	const double epsilon_r = 1e-5;
	const double epsilon_theta = 1e-8;
	if (fabs(theta_w0[j]+alpha_0) < epsilon_theta || r[j] > epsilon_r) {
		j_in_L = 0;		
	} else if (fabs(theta_w0[j]-alpha_1) < epsilon_theta || r[j] < -epsilon_r) {
		j_in_L = 1;
	} else {
		for (arma::uword E_index=0; E_index<E_size; E_index++) {
			if (E[E_index] == j) {
				index_j = E_index;
				break;
			}
		}
	}

	// Manage the case when j is in E
	if (index_j > -1) {
		// Find the next breakpoint
		if (theta_E(index_j) > 0){
			w_m = theta_E(index_j) / alpha_1;
			j_in_L = 1;
		} else {
			w_m = theta_E(index_j) / (-alpha_0);
			j_in_L = 0;
		}

		// Update three sets and theta_E
		if (w_m > 0){
			index_insert(0) = j;
			if (j_in_L){
				L.insert_rows(L_size, index_insert);
				L_size = L_size + 1;
			} else {
				R.insert_rows(R_size, index_insert);
				R_size = R_size + 1;
			}
			if (E_size > 1){
				tilde_B_Bt_inverse = downdate(tilde_B_Bt_inverse, tilde_B, E, E_size, index_j);
      }
			E.shed_row(index_j);
			theta_E.shed_row(index_j);
			E_size = E_size - 1;
			r(j) = 0;
		}
		m = m + 1;
		Beta.insert_cols(m, beta_w0);
		beta_0_insert(0) = beta_0_w0;
		Beta_0.insert_rows(m, beta_0_insert);
		w_insert(0) = w_m; 
		W_vec.insert_rows(m, w_insert);
	}

	// Case when alpha_0 = 0 and j is in R (Algorithm terminates)
	if (alpha_0 == 0 && j_in_L == 0){
		m = m + 1;
		Beta.insert_cols(m, Beta.col(m-1));
		beta_0_insert(0) = Beta_0(m-1);
		Beta_0.insert_rows(m, beta_0_insert);
		w_m = 0;
		w_insert(0) = w_m; 
		W_vec.insert_rows(m, w_insert);
	}

	// Compute T (It will not change)
	if (j_in_L) {
		T = alpha_1;
	} else {
		T = -alpha_0;
	}

	while (w_m > 1e-6) {

		if (E_size > 0) {
     	// Compute three slopes d_0m, d_m, h_m
			d_0m = (a_j - a.elem(E).t()*tilde_B_Bt_inverse*tilde_B.rows(E)*tilde_B_j) / (a.elem(E).t()*tilde_B_Bt_inverse*a.elem(E)) * T;
			d_m = - tilde_B_Bt_inverse * (a.elem(E)*d_0m + tilde_B.rows(E)*tilde_B_j*T);
			h_m = a*d_0m + tilde_B * (arma::trans(tilde_B.rows(E))*d_m + T*tilde_B_j);

			// Compute candidate w_1m
			// w_1_index is the case in E that should be moved to L/R, and w_1_max is the candidate w_1m
			w_1_max = -INF;
			if (E_size > 0) {
				w_1_alpha1_max = -INF;
				w_1_alpha1_temp = (- theta_E + alpha_1) / d_m + w_m;
				for (unsigned int i=0; i<E_size; i++) {
					if ((w_1_alpha1_temp(i) < w_m) && (w_1_alpha1_temp(i) > w_1_alpha1_max)) {
						w_1_alpha1_max = w_1_alpha1_temp(i);
						w_1_alpha1_index = i;
					}
				}

				w_1_alpha0_max = -INF;
				w_1_alpha0_temp = (- theta_E - alpha_0) / d_m + w_m;
				for (unsigned int i=0; i<E_size; i++) {
					if ((w_1_alpha0_temp(i) < w_m) && (w_1_alpha0_temp(i) > w_1_alpha0_max)) {
						w_1_alpha0_max = w_1_alpha0_temp(i);
						w_1_alpha0_index = i;
					}
				}

				if (w_1_alpha0_max > w_1_alpha1_max) {
					w_1_max = w_1_alpha0_max;
					w_1_index = w_1_alpha0_index;
				} else {
					w_1_max = w_1_alpha1_max;
					w_1_index = w_1_alpha1_index;
				}
			}

			// Compute candidate w_2m
			// w_2_index is the case in L/R that should be moved to E, and w_2_L_is_max indicates this case is from L or R
			// w_2_max is the candidate w_2m
			w_2_max = -INF;
			w_2_L_max = -INF;
			w_2_R_max = -INF;
			if (L_size > 0) {
				w_2_L_temp = - lam * r.elem(L) / h_m.elem(L) + w_m;
				for (unsigned int i=0; i<L_size; i++) {
					if ((w_2_L_temp(i) < w_m) && (w_2_L_temp(i) > w_2_L_max)) {
						w_2_L_max = w_2_L_temp(i);
						w_2_L_index = i;
					}
				}
			}

			if (R_size > 0) {
				w_2_R_temp = - lam * r.elem(R) / h_m.elem(R) + w_m;
				for (unsigned int i=0; i<R_size; i++) {
					if ((w_2_R_temp(i) < w_m) && (w_2_R_temp(i) > w_2_R_max)) {
						w_2_R_max = w_2_R_temp(i);
						w_2_R_index = i;
					}
				}
			}

			if (w_2_L_max > w_2_R_max) {
				w_2_max = w_2_L_max;
				w_2_index = w_2_L_index;
				w_2_L_is_max = 1;
			} else {
				w_2_max = w_2_R_max;
				w_2_index = w_2_R_index;
				w_2_L_is_max = 0;
			}

			w_m_next = std::max(w_1_max, w_2_max);
			
			// Compute beta, r at the next breakpoint
			theta_E = theta_E + d_m * (w_m_next - w_m);
			beta_0_w = beta_0_w + d_0m(0) / lam * (w_m_next - w_m);
			r = r + h_m / lam * (w_m_next - w_m);
			beta_w = beta_w + (arma::trans(B.rows(E))*d_m + T*B.row(j).t()) * (w_m_next - w_m) / lam;
			
			// Update three elbow sets and theta_E
			if (w_m_next == w_1_max) {
				// The case when E moves an element to either L or R
				index_insert(0) = E[w_1_index];				
				if (fabs(theta_E[w_1_index] - alpha_1) < epsilon_theta) {
					L.insert_rows(L_size, index_insert);
					L_size = L_size + 1;
				} else {
					R.insert_rows(R_size, index_insert);
					R_size = R_size + 1;
				}
				if (E_size > 1){
					tilde_B_Bt_inverse = downdate(tilde_B_Bt_inverse, tilde_B, E, E_size, w_1_index);
                }
				r(E[w_1_index]) = 0;
				E.shed_row(w_1_index);
				theta_E.shed_row(w_1_index);
				E_size = E_size - 1;
			} else {
				// The case when L/R moves an element to E
				if (w_2_L_is_max) {
					index_insert(0) = L[w_2_index];
					theta_insert(0) = alpha_1;
					E.insert_rows(E_size, index_insert);
					theta_E.insert_rows(E_size, theta_insert);
					E_size = E_size + 1;  
					L.shed_row(w_2_index);
					L_size = L_size - 1;
				} else {
					index_insert(0) = R[w_2_index];
					theta_insert(0) = -alpha_0;
					E.insert_rows(E_size, index_insert);
					theta_E.insert_rows(E_size, theta_insert);
					E_size = E_size + 1;  
					R.shed_row(w_2_index);
					R_size = R_size - 1;					
				}
				tilde_B_Bt_inverse = update(tilde_B_Bt_inverse, tilde_B, E, E_size);
			}
		} else {
			// The case when E is empty
		    E_empty_output = Empty_E(a, B, c, beta_w, L, R, j_in_L, j);
		    beta_0_update = E_empty_output[0];
		    r = r + (beta_0_update - beta_0_w) * a;   // Update the residual
		    beta_0_w = beta_0_update;
		    index_in_elbow = E_empty_output[1];
		    // Move an element from L/R to E
		    if (index_in_elbow < 0) {
					index_insert(0) = L[-index_in_elbow-1];
					theta_insert(0) = alpha_1;
					E.insert_rows(E_size, index_insert);
					theta_E.insert_rows(E_size, theta_insert);
					E_size = E_size + 1;  
					L.shed_row(-index_in_elbow-1);
					L_size = L_size - 1;		
				} else {
					index_insert(0) = R[index_in_elbow];
					theta_insert(0) = -alpha_0;
					E.insert_rows(E_size, index_insert);
					theta_E.insert_rows(E_size, theta_insert);
					E_size = E_size + 1;  
					R.shed_row(index_in_elbow);
					R_size = R_size - 1;
		    }
			Beta_0(m) = beta_0_w;

			tilde_B_Bt_inverse = arma::solve(tilde_B.rows(E)*arma::trans(tilde_B.rows(E)), arma::eye(E_size, E_size));
			continue;
		}	
		m = m + 1;
		Beta.insert_cols(m, beta_w);
		beta_0_insert(0) = beta_0_w;
		Beta_0.insert_rows(m, beta_0_insert);
		w_insert(0) = w_m_next; 
		W_vec.insert_rows(m, w_insert);
		w_m = w_m_next;
	}

  return Rcpp::List::create(Rcpp::Named("W_vec") = W_vec,
                            Rcpp::Named("Beta_0") = Beta_0,
                            Rcpp::Named("Beta") = Beta);  
}

arma::vec Interpolate_solution(double w_grid, int w_grid_position, arma::vec W_vec, arma::vec Beta_0_vec, arma::mat Beta_mat) {
	const int p = Beta_mat.n_rows;
	arma::vec output(p+1, arma::fill::zeros);
	arma::vec beta_w(p, arma::fill::zeros);
	double beta_0_w = 0.0;
	int pos = w_grid_position;

	if (Beta_mat.col(pos)[0] == Beta_mat.col(pos+1)[0]) {
		// Deal with the case when solutions do not change between W_vec[pos+1] and W_vec[pos]. 
		// The reason why we seperate this case is because when |E| = 0 between W_vec[pos+1] and W_vec[pos],
		// Beta_0_vec[pos+1] and Beta_0_vec[pos] are not equal.
		beta_w = Beta_mat.col(pos);
		beta_0_w = Beta_0_vec[pos];
	} else {
		beta_w = (w_grid - W_vec[pos+1]) / (W_vec[pos] - W_vec[pos+1]) * Beta_mat.col(pos) + (W_vec[pos] - w_grid) / (W_vec[pos] - W_vec[pos+1]) * Beta_mat.col(pos+1);
		beta_0_w = (w_grid - W_vec[pos+1]) / (W_vec[pos] - W_vec[pos+1]) * Beta_0_vec[pos] + (W_vec[pos] - w_grid) / (W_vec[pos] - W_vec[pos+1]) * Beta_0_vec[pos+1];
	}
	output(0) = beta_0_w;
	output.tail(p) = beta_w;
	return output;
}

//'Compute global influence for each case under L2 regularized nonsmooth problem 
//'
//' @description
//' Compute the global influence/Cook's distance for each case under L2 regularized nonsmooth problem (quantile regression and svm).
//'
//' @param a A \eqn{n \times 1} vector and a^T = (a_1, ..., a_n)
//' @param B A \eqn{n \times p} matrix and B^T = (b_1, ..., b_n)
//' @param c A \eqn{n \times 1} matrix and c^T = (b_1, ..., b_n) 
//' @param lam Regularization parameter for L2 penalty
//' @param alpha_0 A scalar in the definition of f(r)
//' @param alpha_1 A scalar in the definition of f(r)
//' @param beta_0_w0 A scalar, which is the true value of beta_{0,w} when w = w_0 = 1
//' @param beta_w0 A \eqn{p \times 1} vector, which is the true value of beta_{w} when w = w_0 = 1
//' @param theta_w0 A \eqn{n \times 1} vector, which is the true value of the dual variable when w = w_0 = 1
//' @param model_class The model we consider, two options are "quantile" and "svm"
//' @param influence_measure The influence measure used to compute global influence for each case, two options are "FMD" and "BDLD". It is only used when
//' model_class = "svm"
//'
//' @details
//' This function will be called by the main function Compute_CaseInflu_nonsmooth to generate case influence graph for each case.
//'
//' @return CaseInfluence_vec Global influence for each case
//' @return cook_distance Cook's distance for each case
// [[Rcpp::export(CaseInfluence_nonsmooth)]]
Rcpp::List CaseInfluence_nonsmooth(arma::vec a, arma::mat B, arma::vec c, double lam, double alpha_0, double alpha_1, double beta_0_w0, arma::vec beta_w0, arma::vec theta_w0, std::string model_class, std::string influence_measure) {
  const int n = B.n_rows;
  const int p = B.n_cols;

  // Define \tilde B
	arma::mat tilde_B(n, p+1);
	tilde_B.col(0) = a;
	tilde_B.cols(1, p) = B;

  const int Num_grid = 10;

  arma::vec residual_full = beta_0_w0 * a + B * beta_w0 + c;
  arma::vec residual(n);
  residual.zeros();
  double h = 0.0;
  double w_grid = 0.0;
  arma::vec beta_w(p);
  beta_w.zeros();
  double beta_0_w = 0.0;

  arma::vec CaseInfluence_vec(n);
  CaseInfluence_vec.zeros();
  arma::vec Cook_vec(n);
  Cook_vec.zeros();
  Rcpp::List case_path;

  double beta_0_loo = 0.0;
  arma::vec beta_loo(p);
  beta_loo.zeros();
  arma::vec solution_at_w(p+1, arma::fill::zeros);

  // Initialize E/L/R, residual and matrix inversion
  arma::uvec index_insert(1);
	arma::uvec E_init;
	arma::uvec L_init;
	arma::uvec R_init;
	int E_size = 0;
	int L_size = 0;
	int R_size = 0;
	arma::mat tilde_B_Bt_inverse_init;
	arma::vec r_init = residual_full;

	const double epsilon_r = 1e-5;
	const double epsilon_theta = 1e-8;
	for (arma::uword i=0; i<n; i++){
		index_insert(0) = i;
		if (fabs(theta_w0[i]+alpha_0) < epsilon_theta || r_init[i] > epsilon_r){
			R_init.insert_rows(R_size, index_insert);
			R_size = R_size + 1;
		} else if (fabs(theta_w0[i]-alpha_1) < epsilon_theta || r_init[i] < -epsilon_r){
			L_init.insert_rows(L_size, index_insert);
			L_size = L_size + 1;
		} else {
			E_init.insert_rows(E_size, index_insert);
			E_size = E_size + 1;
		}
	}

	if (E_size) {
		tilde_B_Bt_inverse_init = arma::solve(tilde_B.rows(E_init)*arma::trans(tilde_B.rows(E_init)), arma::eye(E_size, E_size));
	} else {
		// Assign tilde_B_Bt_inverse_init with a random matrix (We choose identity matrix)
		tilde_B_Bt_inverse_init = arma::eye(1, 1);
	}
	
  for (int case_index = 0; case_index < n; case_index++) {
  	case_path = case_path_nonsmooth_with_init(a, B, c, lam, alpha_0, alpha_1, case_index, beta_0_w0, beta_w0, theta_w0, E_init, L_init, R_init, r_init, tilde_B_Bt_inverse_init);
    arma::vec W_vec = case_path[0];
    arma::vec Beta_0_vec = case_path[1];
    arma::mat Beta_mat = case_path[2];

		for (unsigned int i = 0; i < W_vec.size()-1; i++) {
			if (i == W_vec.size()-2) {
				h = W_vec[i] / Num_grid;
			}
			else {
				h = (W_vec[i] - W_vec[i+1]) / Num_grid;
			}
			for (int j = 0; j < Num_grid; j++) {
				w_grid = W_vec[i] - j * h;
				solution_at_w = Interpolate_solution(w_grid, i, W_vec, Beta_0_vec, Beta_mat);
				beta_0_w = solution_at_w[0];
				beta_w = solution_at_w.tail(p);
				residual = beta_0_w * a + B * beta_w + c;
				if (model_class == "quantile") {
					CaseInfluence_vec[case_index] += 1.0 / n * arma::sum(pow(residual_full-residual, 2) ) * h;
				}
				else if (model_class == "svm") {
					if (influence_measure == "FMD") {
						CaseInfluence_vec[case_index] += 1.0 / n * arma::sum(pow(residual_full-residual, 2) ) * h;
					}
					else if (influence_measure == "BDLD") {
						CaseInfluence_vec[case_index] += 1.0 / n * arma::sum(pow(log1p(arma::exp(-residual_full-1))-log1p(arma::exp(-residual-1)), 2) ) * h;
					}
				}				
			} // End j loop
		} // End i loop  

	  // Compute Cook's distance
	  arma::uword Num_break = W_vec.size(); 
	  solution_at_w = Interpolate_solution(0.0, Num_break-2, W_vec, Beta_0_vec, Beta_mat);;
		beta_0_loo = solution_at_w[0];
		beta_loo = solution_at_w.tail(p);
		residual = beta_0_loo * a + B * beta_loo + c;
		if (model_class == "quantile") {
			Cook_vec[case_index] = 1.0 / n * arma::sum(pow(residual_full-residual, 2) );
		}
		else if (model_class == "svm") {
			if (influence_measure == "FMD") {
				Cook_vec[case_index] = 1.0 / n * arma::sum(pow(residual_full-residual, 2) );
			}
			else if (influence_measure == "BDLD") {
				Cook_vec[case_index] = 1.0 / n * arma::sum(pow(log1p(arma::exp(-residual_full-1))-log1p(arma::exp(-residual-1)), 2) );
			}
		}
  }
  return Rcpp::List::create(Rcpp::Named("global_influence") = CaseInfluence_vec,
                            Rcpp::Named("cook_distance") = Cook_vec);
}

//'Compute local influence for each case under L2 regularized nonsmooth problem
//'
//' @description
//' Compute the local influence for each case under L2 regularized nonsmooth problem (quantile regression and svm), 
//' and the procedure only depends on the full-data solution.
//'
//' @param a A \eqn{n \times 1} vector and a^T = (a_1, ..., a_n)
//' @param B A \eqn{n \times p} matrix and B^T = (b_1, ..., b_n)
//' @param c A \eqn{n \times 1} matrix and c^T = (b_1, ..., b_n) 
//' @param lam Regularization parameter for L2 penalty
//' @param alpha_0 A scalar in the definition of f(r)
//' @param alpha_1 A scalar in the definition of f(r)
//' @param beta_0_w0 A scalar, which is the true value of beta_{0,w} when w = w_0 = 1
//' @param beta_w0 A \eqn{p \times 1} vector, which is the true value of beta_{w} when w = w_0 = 1
//' @param theta_w0 A \eqn{n \times 1} vector, which is the true value of the dual variable when w = w_0 = 1
//' @param model_class The model we consider, two options are "quantile" and "svm"
//' @param influence_measure The influence measure used to compute global influence for each case, two options are "FMD" and "BDLD". It is only used when
//' model_class = "svm"
//'
//' @details
//' This function will be called by the main function Compute_LocalInflu_nonsmooth to compute local influence for each case.
//'
//' @return LocalInfluence_vec Local influence for each case
// [[Rcpp::export(LocalInfluence_nonsmooth)]]
arma::vec LocalInfluence_nonsmooth(arma::vec a, arma::mat B, arma::vec c, double lam, double alpha_0, double alpha_1, double beta_0_w0, arma::vec beta_w0, arma::vec theta_w0, std::string model_class, std::string influence_measure) {
  const int n = B.n_rows;
	const int p = B.n_cols;

	// Define \tilde B, a_j, tilde_B_j
	arma::mat tilde_B(n, p+1);
	tilde_B.col(0) = a;
	tilde_B.cols(1, p) = B;
	double a_j = 0.0;
	arma::vec tilde_B_j(p+1);
	tilde_B_j.zeros();

	// Declare and initialize three elbow sets, and compute theta_E
	arma::uvec E;
	arma::uvec L;
	arma::uvec R;
	int E_size = 0;
	int L_size = 0;
	int R_size = 0;
	arma::uvec index_insert(1);
	arma::vec r = beta_0_w0 * a + B * beta_w0 + c;

	const double epsilon_r = 1e-5;
	const double epsilon_theta = 1e-8;
	for (unsigned i=0; i<n; i++){
		index_insert(0) = i;
		if (fabs(theta_w0[i]+alpha_0) < epsilon_theta || r[i] > epsilon_r){
			R.insert_rows(R_size, index_insert);
			R_size = R_size + 1;
		} else if (fabs(theta_w0[i]-alpha_1) < epsilon_theta || r[i] < -epsilon_r){
			L.insert_rows(L_size, index_insert);
			L_size = L_size + 1;
		} else {
			E.insert_rows(E_size, index_insert);
			E_size = E_size + 1;
		}
	}

	// Declare variables 
	int j_in_L = 0;
	double T = 0.0;
	arma::vec d_0m(1);
	arma::vec d_m;
	arma::vec h_m(n);
	arma::mat tilde_B_Bt_inverse;
	arma::vec LocalInfluence_vec(n);
	LocalInfluence_vec.zeros();

	for (unsigned case_index=0; case_index<n; case_index++) {
		arma::uvec E_temp = E;
		arma::uvec L_temp = L;
		arma::uvec R_temp = R;
		arma::vec r_temp = r;
		a_j = a(case_index);
		tilde_B_j = tilde_B.row(case_index).t();

		if (E_size == 0) {
			if (fabs(theta_w0[case_index]+alpha_0) < epsilon_theta || r_temp[case_index] > epsilon_r){
				j_in_L = 0;
			} else {
				j_in_L = 1;
			}
			arma::vec E_empty_output = Empty_E(a, B, c, beta_w0, L_temp, R_temp, j_in_L, case_index);
			r_temp = r_temp + (E_empty_output[0] - beta_0_w0) * a;   // Update the residual
		    int index_in_elbow = E_empty_output[1];
		    if (index_in_elbow < 0) {
				index_insert(0) = L[-index_in_elbow-1];
				E_temp.insert_rows(E_size, index_insert);
				L_temp.shed_row(-index_in_elbow-1);
			} else {
				index_insert(0) = R[index_in_elbow];
				E_temp.insert_rows(E_size, index_insert);
				R_temp.shed_row(index_in_elbow);
		    }
		}
		tilde_B_Bt_inverse = arma::solve(tilde_B.rows(E_temp)*arma::trans(tilde_B.rows(E_temp)), arma::eye(E_temp.size(), E_temp.size()));

		// Check if the case in E or not
		if (fabs(theta_w0[case_index]+alpha_0) < epsilon_theta || r_temp[case_index] > epsilon_r){
			T = -alpha_0;
		} else if (fabs(theta_w0[case_index]-alpha_1) < epsilon_theta || r_temp[case_index] < -epsilon_r){
			T = alpha_1;
		} else {
			continue;
		}

		d_0m = (a_j - a.elem(E_temp).t()*tilde_B_Bt_inverse*tilde_B.rows(E_temp)*tilde_B_j) / (a.elem(E_temp).t()*tilde_B_Bt_inverse*a.elem(E_temp)) * T;
		d_m = - tilde_B_Bt_inverse * (a.elem(E_temp)*d_0m + tilde_B.rows(E_temp)*tilde_B_j*T);
		h_m = a*d_0m + tilde_B * (arma::trans(tilde_B.rows(E_temp))*d_m + T*tilde_B_j);

		// Compute local influence
		if ((model_class == "quantile") || (model_class == "svm" && influence_measure == "FMD")) {
			LocalInfluence_vec[case_index] = 2.0 / n * arma::sum(pow(h_m / lam, 2));
		}
		else if (model_class == "svm" && influence_measure == "BDLD") {
			LocalInfluence_vec[case_index] = 2.0 / n * arma::sum(pow( (arma::exp(-(r_temp+1)) / (1 + arma::exp(-(r_temp+1)))) % (h_m / lam), 2));
		}
	}
	return LocalInfluence_vec;
}
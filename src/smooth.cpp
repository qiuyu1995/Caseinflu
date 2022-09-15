// [[Rcpp::depends(RcppArmadillo)]]
#include <RcppArmadillo.h>


//'Case-weight adjusted solution path for logistic regression Newton method (\eqn{p \le n})
//'
//' @description
//' Generate solution path for case-weight adjusted problem by adopting Newton method at each selected grid points,
//' which are adaptively selected to save computation while maintaining the accuracy of the entire path.
//'
//' @param X A \eqn{n \times p} feature matrix
//' @param Y A \eqn{n \times 1} response vector
//' @param theta_0 A \eqn{p \times 1} vector that corresponds to full-data solution
//' @param t_max Range of solution path \eqn{[0, t_{\max}]}
//' @param lam Regularization parameter for L2 penalty
//' @param case_index Index of the observation that is attached a weight
//' @param epsilon Pre-specified error tolerance of the entire path
//' @param L Lipschitz constant for the gradient of the loss function
//'
//' @details
//' This function will be called by the function CaseInfluence_logistic to generate case influence graph for each case, with method = "Newton".
//'
//' @return t_vec Adaptively selected grid points
//' @return theta Generated soultion path at grid points
//' @return alpha_t Step size between grid points
Rcpp::List  logistic_Newton_low_dimension(arma::mat X, arma::vec Y,arma::vec theta_0, double t_max, double lam, int case_index, double epsilon, double L){

  const int n=X.n_rows;
  const int p=X.n_cols;
  arma::mat X_t(p,n);
  X_t=arma::trans(X);

  // Declare variables for step size computation
  const double C1=1.0/3;
  double norm_gradient_g=0;
  double norm_gradient_g0=0;
  arma::vec alpha_compare_vec(4);
  double alpha_temp=0;
  alpha_compare_vec.zeros();
  alpha_compare_vec(0)=t_max;
  alpha_compare_vec(1)=std::log(1+lam*C1/L);
  alpha_compare_vec(2)=t_max;
  alpha_compare_vec(3)=t_max;

  // Declare variables for Newton method
  double t_Newton=0;
  arma::vec temp_Newton_1(n);
  temp_Newton_1.zeros();
  arma::mat temp_Newton_2(p,p);
  temp_Newton_2.zeros();
  arma::vec temp_Newton_3(p);
  temp_Newton_3.zeros();
  arma::vec weight(n);
  weight.ones();
  weight = weight/n;
  arma::vec b_Newton(n);
  b_Newton.zeros();

  double alpha_t=t_max;
  double t_step=0;
  int index_Newton=0;
  t_Newton=t_step;

  // Store values for termination criterion
  double LHS_condition=0;
  double RHS_condition=0;
  RHS_condition=epsilon;
  LHS_condition=RHS_condition;

  //Update alpha_1
  norm_gradient_g=arma::norm(-Y(case_index)/(1+arma::exp(Y(case_index)*(X.row(case_index)*theta_0)))*X.row(case_index),2)/n;
  norm_gradient_g0=norm_gradient_g;
  alpha_temp=2*lam*epsilon/pow(norm_gradient_g,2);
  alpha_compare_vec(3)=std::log(1+pow(alpha_temp,0.5));
  alpha_t=alpha_compare_vec.min();

  //Decide the storage space based on alpha_1
  const int N=3*std::round(t_max/alpha_compare_vec.min());
  arma::mat theta_Newton(p,N+1);
  theta_Newton.zeros();
  theta_Newton.col(0)=theta_0;
  arma::vec alpha_t_vec(N);
  alpha_t_vec.zeros();
  arma::vec t_vec(N);
  t_vec.zeros();

  alpha_t_vec(index_Newton)=alpha_t;
  t_step=t_step+alpha_t;
  t_vec(index_Newton)=t_step;
  t_Newton=t_step;
  weight[case_index] = std::exp(-t_Newton)/n;
  b_Newton=arma::exp(Y%(X*theta_Newton.col(index_Newton) ) ) ;
  temp_Newton_1=(b_Newton/pow(1+b_Newton,2))%weight;
  temp_Newton_2=X_t*diagmat(temp_Newton_1)*X+lam*arma::eye(p,p);
  temp_Newton_3=X_t*(-1/(1+b_Newton)%Y%weight)+lam*theta_Newton.col(index_Newton);
  theta_Newton.col(index_Newton+1)=theta_Newton.col(index_Newton)- arma::solve(temp_Newton_2,temp_Newton_3);

  index_Newton=index_Newton+1;
  norm_gradient_g=arma::norm(-Y(case_index)/(1+arma::exp(Y(case_index)*(X.row(case_index)*theta_Newton.col(index_Newton))))*X.row(case_index),2)/n;
  LHS_condition=std::exp(-3*t_step)*(std::exp(t_step)/lam+L/(2*pow(lam,2)))*pow(norm_gradient_g,2);

  while(t_step<t_max && 10*LHS_condition>RHS_condition){
    // update alpha;
    alpha_compare_vec(1)=std::log(1+std::exp(t_step)*C1*lam/L);
    alpha_compare_vec(2)=2*alpha_t;
    alpha_temp=std::exp(t_step)*(1-std::exp(-alpha_t_vec(0)))*norm_gradient_g0/norm_gradient_g;
    alpha_compare_vec(3)=std::log(1+alpha_temp);
    alpha_t=alpha_compare_vec.min();
    alpha_t_vec(index_Newton)=alpha_t;
    t_step=t_step+alpha_t;
    t_vec(index_Newton)=t_step;

    t_Newton=t_step;
    weight[case_index] = std::exp(-t_Newton)/n;
    b_Newton=arma::exp(Y%(X*theta_Newton.col(index_Newton) ) ) ;
    temp_Newton_1=(b_Newton/pow(1+b_Newton,2))%weight;
    temp_Newton_2=X_t*diagmat(temp_Newton_1)*X+lam*arma::eye(p,p);
    temp_Newton_3=X_t*(-1/(1+b_Newton)%Y%weight)+lam*theta_Newton.col(index_Newton);
    theta_Newton.col(index_Newton+1)=theta_Newton.col(index_Newton)- arma::solve(temp_Newton_2,temp_Newton_3);

    index_Newton=index_Newton+1;
    norm_gradient_g=arma::norm(-Y(case_index)/(1+arma::exp(Y(case_index)*(X.row(case_index)*theta_Newton.col(index_Newton))))*X.row(case_index),2)/n;
    LHS_condition=std::exp(-3*t_step)*(std::exp(t_step)/lam+L/(2*pow(lam,2)))*pow(norm_gradient_g,2);
  }

  arma::mat theta_Newton_output(p,index_Newton+1);
  theta_Newton_output=theta_Newton.cols(0,index_Newton);

  arma::vec alpha_t_vec_output(index_Newton);
  alpha_t_vec_output=alpha_t_vec.subvec(0,index_Newton-1);

  arma::vec t_vec_output(index_Newton+1);
  t_vec_output(0)=0;
  t_vec_output.subvec(1,index_Newton)=t_vec.subvec(0,index_Newton-1);

  return Rcpp::List::create(Rcpp::Named("theta") = theta_Newton_output,
                            Rcpp::Named("t_vec") = t_vec_output,
                            Rcpp::Named("alpha_t_vec") = alpha_t_vec_output);  
}

//'Case-weight adjusted solution path for logistic regression Newton method (\eqn{n \le p})
//'
//' @description
//' Generate solution path for case-weight adjusted problem by adopting Newton method at each selected grid points,
//' which are adaptively selected to save computation while maintaining the accuracy of the entire path.
//'
//' @param X A \eqn{n \times p} feature matrix
//' @param Y A \eqn{n \times 1} response vector
//' @param theta_0 A \eqn{p \times 1} vector that corresponds to full-data solution
//' @param t_max Range of solution path \eqn{[0, t_{\max}]}
//' @param lam Regularization parameter for L2 penalty
//' @param case_index Index of the observation that is attached a weight
//' @param epsilon Pre-specified error tolerance of the entire path
//' @param L Lipschitz constant for the gradient of the loss function
//'
//' @details
//' This function will be called by the function CaseInfluence_logistic to generate case influence graph for each case, with method = "Newton".
//'
//' @return t_vec Adaptively selected grid points
//' @return theta Generated soultion path at grid points
//' @return alpha_t Step size between grid points
Rcpp::List logistic_Newton_high_dimension(arma::mat X, arma::vec Y,arma::vec theta_0, double t_max, double lam, int case_index, double epsilon, double L){
  const int n=X.n_rows;
  const int p=X.n_cols;
  arma::mat X_t(p,n);
  X_t=arma::trans(X);

  // Declare variables for step size computation
  const double C1=1.0/3;
  double norm_gradient_g=0;
  double norm_gradient_g0=0;
  arma::vec alpha_compare_vec(4);
  double alpha_temp=0;
  alpha_compare_vec.zeros();
  alpha_compare_vec(0)=t_max;
  alpha_compare_vec(1)=std::log(1+lam*C1/L);
  alpha_compare_vec(2)=t_max;
  alpha_compare_vec(3)=t_max;

  // Declare variables for Newton method (use woodbury formula for matrix inversion)
  arma::mat U(p,n);
  U.zeros();
  arma::mat V(n,p);
  V.zeros();
  double t_Newton=0;
  arma::mat temp_Newton_1(p,p);
  temp_Newton_1.zeros();
  arma::vec temp_Newton_2(p);
  temp_Newton_2.zeros();
  arma::vec weight(n);
  weight.ones();
  weight = weight/n;
  arma::vec b_Newton(n);
  b_Newton.zeros();

  double alpha_t=t_max;
  double t_step=0;
  int index_Newton=0;
  t_Newton=t_step;

  // Store values for termination criterion
  double LHS_condition=0;
  double RHS_condition=0;
  RHS_condition=epsilon;
  LHS_condition=RHS_condition;

  //Update alpha_1
  norm_gradient_g=arma::norm(-Y(case_index)/(1+arma::exp(Y(case_index)*(X.row(case_index)*theta_0)))*X.row(case_index),2)/n;
  norm_gradient_g0=norm_gradient_g;
  alpha_temp=2*lam*epsilon/pow(norm_gradient_g,2);
  alpha_compare_vec(3)=std::log(1+pow(alpha_temp,0.5));
  alpha_t=alpha_compare_vec.min();

  //Decide the storage space based on alpha_1
  const int N=3*std::round(t_max/alpha_compare_vec.min());
  arma::mat theta_Newton(p,N+1);
  theta_Newton.zeros();
  theta_Newton.col(0)=theta_0;
  arma::vec alpha_t_vec(N);
  alpha_t_vec.zeros();
  arma::vec t_vec(N);
  t_vec.zeros();

  alpha_t_vec(index_Newton)=alpha_t;
  t_step=t_step+alpha_t;
  t_vec(index_Newton)=t_step;
  t_Newton=t_step;
  weight[case_index] = std::exp(-t_Newton)/n;
  b_Newton=arma::exp(Y%(X*theta_Newton.col(index_Newton) ) ) ;
  U=X_t*diagmat(b_Newton/arma::pow(1+b_Newton,2)%weight/lam);
  V=X;
  temp_Newton_1=1/lam*(arma::eye(p,p)-U*arma::solve(arma::eye(n,n)+V*U,V));
  temp_Newton_2=X_t*(-1/(1+b_Newton)%Y%weight)+lam*theta_Newton.col(index_Newton);
  theta_Newton.col(index_Newton+1)=theta_Newton.col(index_Newton)- temp_Newton_1*temp_Newton_2;

  index_Newton=index_Newton+1;
  norm_gradient_g=arma::norm(-Y(case_index)/(1+arma::exp(Y(case_index)*(X.row(case_index)*theta_Newton.col(index_Newton))))*X.row(case_index),2)/n;
  LHS_condition=std::exp(-3*t_step)*(std::exp(t_step)/lam+L/(2*pow(lam,2)))*pow(norm_gradient_g,2);

  while(t_step<t_max && 10*LHS_condition>RHS_condition){
    // update alpha;
    alpha_compare_vec(1)=std::log(1+std::exp(t_step)*C1*lam/L);
    alpha_compare_vec(2)=2*alpha_t;
    alpha_temp=std::exp(t_step)*(1-std::exp(-alpha_t_vec(0)))*norm_gradient_g0/norm_gradient_g;
    alpha_compare_vec(3)=std::log(1+alpha_temp);
    alpha_t=alpha_compare_vec.min();
    alpha_t_vec(index_Newton)=alpha_t;
    t_step=t_step+alpha_t;
    t_vec(index_Newton)=t_step;

    t_Newton=t_step;
    weight[case_index] = std::exp(-t_Newton)/n;
    b_Newton=arma::exp(Y%(X*theta_Newton.col(index_Newton) ) ) ;
    U=X_t*diagmat(b_Newton/arma::pow(1+b_Newton,2)%weight/lam);
    V=X;
    temp_Newton_1=1/lam*(arma::eye(p,p)-U*arma::solve(arma::eye(n,n)+V*U,V));
    temp_Newton_2=X_t*(-1/(1+b_Newton)%Y%weight)+lam*theta_Newton.col(index_Newton);
    theta_Newton.col(index_Newton+1)=theta_Newton.col(index_Newton)- temp_Newton_1*temp_Newton_2;

    index_Newton=index_Newton+1;
    norm_gradient_g=arma::norm(-Y(case_index)/(1+arma::exp(Y(case_index)*(X.row(case_index)*theta_Newton.col(index_Newton))))*X.row(case_index),2)/n;
    LHS_condition=std::exp(-3*t_step)*(std::exp(t_step)/lam+L/(2*pow(lam,2)))*pow(norm_gradient_g,2);
  }

  arma::mat theta_Newton_output(p,index_Newton+1);
  theta_Newton_output=theta_Newton.cols(0,index_Newton);

  arma::vec alpha_t_vec_output(index_Newton);
  alpha_t_vec_output=alpha_t_vec.subvec(0,index_Newton-1);

  arma::vec t_vec_output(index_Newton+1);
  t_vec_output(0)=0;
  t_vec_output.subvec(1,index_Newton)=t_vec.subvec(0,index_Newton-1);

  return Rcpp::List::create(Rcpp::Named("theta") = theta_Newton_output,
                            Rcpp::Named("t_vec") = t_vec_output,
                            Rcpp::Named("alpha_t_vec") = alpha_t_vec_output);

}

//'Case-weight adjusted solution path for logistic regression gradient descent method
//'
//' @description
//' Generate solution path for case-weight adjusted problem by adopting gradient descent method at each selected grid points,
//' which are adaptively selected to save computation while maintaining the accuracy of the entire path.
//'
//' @param X A \eqn{n \times p} feature matrix
//' @param Y A \eqn{n \times 1} response vector
//' @param theta_0 A \eqn{p \times 1} vector that corresponds to full-data solution
//' @param t_max Range of solution path \eqn{[0, t_{\max}]}
//' @param lam Regularization parameter for L2 penalty
//' @param case_index Index of the observation that is attached a weight
//' @param epsilon Pre-specified error tolerance of the entire path
//' @param L Lipschitz constant for the gradient of the loss function
//'
//' @details
//' This function will be called by the function CaseInfluence_logistic to generate case influence graph for each case, with method = "GD".
//'
//' @return t_vec Adaptively selected grid points
//' @return theta Generated soultion path at grid points
//' @return alpha_t Step size between grid points
Rcpp::List logistic_GD(arma::mat X, arma::vec Y,arma::vec theta_0, double t_max, double lam, int case_index, double epsilon, double L) {
  const int n=X.n_rows;
  const int p=X.n_cols;
  arma::mat X_t(p,n);
  X_t=arma::trans(X);

 // Declare variables for step size computation
  const double C=1.0/3;
  const double C1=1.0/3;
  arma::vec alpha_compare_vec(4);
  double alpha_temp=0;
  alpha_compare_vec.zeros();
  alpha_compare_vec(0)=t_max;
  alpha_compare_vec(1)=std::log(1+lam*C1/L);
  alpha_compare_vec(2)=t_max;
  alpha_compare_vec(3)=t_max;

  // Declare variables for gradient descent
  arma::vec theta_GD_temp(p);
  theta_GD_temp.zeros();
  arma::vec theta_GD_temp1(p);
  theta_GD_temp1.zeros();
  arma::mat b_GD(n,p);
  b_GD.zeros();
  arma::vec weight(n);
  weight.ones();
  weight = weight/n;

  double alpha_t=0;
  double t_step=0;

  int index_GD=0;
  double t_GD=0;
  double norm_gradient_g=0;
  double norm_gradient_g0=0;
  arma::vec g_k(p);
  g_k.zeros();
  double norm_gk=0;

  // declaration of variables that will be used in backtracking line search
  double eta0=0.01;
  double eta_temp=0;
  double c=0.5;
  double tau=0.2;
  double m=0;
  double t=0;
  double f_theta_temp=0;
  double f_theta_temp1=0;
  arma::vec search_direction(p);
  search_direction.zeros();

  // Store values for termination criterion
  double LHS_condition=0;
  double RHS_condition=0;
  RHS_condition=epsilon;
  LHS_condition=RHS_condition;

  //Update alpha_1
  norm_gradient_g=arma::norm(-Y(case_index)/(1+arma::exp(Y(case_index)*(X.row(case_index)*theta_0)))*X.row(case_index),2)/n;
  norm_gradient_g0=norm_gradient_g;
  alpha_temp=2*lam*epsilon/pow(norm_gradient_g,2);
  alpha_compare_vec(3)=std::log(1+pow(alpha_temp,0.5));
  alpha_t=alpha_compare_vec.min();

  //Decide the storage space based on alpha_1
  const int N=3*std::round(t_max/alpha_compare_vec.min());
  arma::mat theta_GD(p,N+1);
  theta_GD.zeros();
  theta_GD.col(0) = theta_0;
  arma::vec alpha_t_vec(N);
  alpha_t_vec.zeros();
  arma::vec t_vec(N);
  t_vec.zeros();
  alpha_t_vec(index_GD)=alpha_t;
  t_step=t_step+alpha_t;
  t_vec(index_GD)=t_step;

  //Calculate first step
  t_GD=t_step;
  theta_GD_temp=theta_GD.col(index_GD);
  b_GD=diagmat(-1/(1+arma::exp(Y%(X*theta_GD_temp)) )%Y)*X;
  weight[case_index] = exp(-t_GD)/n;
  g_k=trans(b_GD)*weight+lam*theta_GD_temp;
  norm_gk=arma::norm(g_k,2);
  norm_gradient_g=arma::norm(-Y(case_index)/(1+arma::exp(Y(case_index)*(X.row(case_index)*theta_GD_temp)))*X.row(case_index),2)/n;
  while (norm_gk>C*(std::exp(alpha_t-t_step)-std::exp(-t_step))*norm_gradient_g){
    eta_temp=eta0;
    search_direction=-g_k/norm_gk;
    m=-norm_gk;
    t=-c*m;
    f_theta_temp=sum(arma::log(1+arma::exp(-Y%(X*theta_GD_temp)) )%weight)+lam/2*pow(arma::norm(theta_GD_temp,2),2);
    theta_GD_temp1=theta_GD_temp+eta_temp*search_direction;
    f_theta_temp1=sum(arma::log(1+arma::exp(-Y%(X*theta_GD_temp1)) )%weight)+lam/2*pow(arma::norm(theta_GD_temp1,2),2);
    while (f_theta_temp-f_theta_temp1<eta_temp*t){
      eta_temp=eta_temp*tau;
      theta_GD_temp1=theta_GD_temp+eta_temp*search_direction;
      f_theta_temp1=sum(arma::log(1+arma::exp(-Y%(X*theta_GD_temp1)) )%weight)+lam/2*pow(arma::norm(theta_GD_temp1,2),2);
    }
    theta_GD_temp=theta_GD_temp1;
    b_GD=diagmat(-1/(1+arma::exp(Y%(X*theta_GD_temp)) )%Y)*X;
    g_k=trans(b_GD)*weight+lam*theta_GD_temp;
    norm_gk=arma::norm(g_k,2);
    norm_gradient_g=arma::norm(-Y(case_index)/(1+arma::exp(Y(case_index)*(X.row(case_index)*theta_GD_temp)))*X.row(case_index),2)/n;
  }
  theta_GD.col(index_GD+1)=theta_GD_temp;
  index_GD=index_GD+1;
  LHS_condition=std::exp(-3*t_step)*(std::exp(t_step)/lam+L/(2*pow(lam,2)))*pow(norm_gradient_g,2);

  while (t_step<t_max && 10*LHS_condition>RHS_condition){
    alpha_compare_vec(1)=std::log(1+std::exp(t_step)*C1*lam/L);
    alpha_compare_vec(2)=2*alpha_t;
    alpha_temp=std::exp(t_step)*(1-std::exp(-alpha_t_vec(0)))*norm_gradient_g0/norm_gradient_g;
    alpha_compare_vec(3)=std::log(1+alpha_temp);
    alpha_t=alpha_compare_vec.min();
    alpha_t_vec(index_GD)=alpha_t;
    t_step=t_step+alpha_t;
    t_vec(index_GD)=t_step;

    t_GD=t_step;
    theta_GD_temp=theta_GD.col(index_GD);
    b_GD=diagmat(-1/(1+arma::exp(Y%(X*theta_GD_temp)) )%Y)*X;
    weight[case_index] = exp(-t_GD)/n;
    g_k=trans(b_GD)*weight+lam*theta_GD_temp;
    norm_gk=arma::norm(g_k,2);
    norm_gradient_g=arma::norm(-Y(case_index)/(1+arma::exp(Y(case_index)*(X.row(case_index)*theta_GD_temp)))*X.row(case_index),2)/n;
    while (norm_gk>C*(std::exp(alpha_t-t_step)-std::exp(-t_step))*norm_gradient_g){
      eta_temp=eta0;
      search_direction=-g_k/norm_gk;
      m=-norm_gk;
      t=-c*m;
      f_theta_temp=sum(arma::log(1+arma::exp(-Y%(X*theta_GD_temp)) )%weight)+lam/2*pow(arma::norm(theta_GD_temp,2),2);
      theta_GD_temp1=theta_GD_temp+eta_temp*search_direction;
      f_theta_temp1=sum(arma::log(1+arma::exp(-Y%(X*theta_GD_temp1)) )%weight)+lam/2*pow(arma::norm(theta_GD_temp1,2),2);
      while (f_theta_temp-f_theta_temp1<eta_temp*t){
    eta_temp=eta_temp*tau;
    theta_GD_temp1=theta_GD_temp+eta_temp*search_direction;
    f_theta_temp1=sum(arma::log(1+arma::exp(-Y%(X*theta_GD_temp1)) )%weight)+lam/2*pow(arma::norm(theta_GD_temp1,2),2);
      }
      theta_GD_temp=theta_GD_temp1;
      b_GD=diagmat(-1/(1+arma::exp(Y%(X*theta_GD_temp)) )%Y)*X;
      g_k=trans(b_GD)*weight+lam*theta_GD_temp;
      norm_gk=arma::norm(g_k,2);
      norm_gradient_g=arma::norm(-Y(case_index)/(1+arma::exp(Y(case_index)*(X.row(case_index)*theta_GD_temp)))*X.row(case_index),2)/n;
    }
    theta_GD.col(index_GD+1)=theta_GD_temp;
    index_GD=index_GD+1;
    LHS_condition=std::exp(-3*t_step)*(std::exp(t_step)/lam+L/(2*pow(lam,2)))*pow(norm_gradient_g,2);
  }

  arma::mat theta_GD_output(p,index_GD+1);
  theta_GD_output=theta_GD.cols(0,index_GD);

  arma::vec alpha_t_vec_output(index_GD);
  alpha_t_vec_output=alpha_t_vec.subvec(0,index_GD-1);

  arma::vec t_vec_output(index_GD+1);
  t_vec_output(0)=0;
  t_vec_output.subvec(1,index_GD)=t_vec.subvec(0,index_GD-1);

  return Rcpp::List::create(Rcpp::Named("theta") = theta_GD_output,
                            Rcpp::Named("t_vec") = t_vec_output,
                            Rcpp::Named("alpha_t_vec") = alpha_t_vec_output);
}

//' Compute the LOO solution for L2 regularized logistic regression 
//'
//' @description
//' Given the initial point, compute the LOO solution for L2 regularized logistic regression by one-step Newton method
//'
//' @param X A \eqn{n \times p} feature matrix
//' @param Y A \eqn{n \times 1} response vector
//' @param theta_init A \eqn{p \times 1} vector, which is the initial point
//' @param lam Regularization parameter for L2 penalty
//' @param case_index Index of the observation that is left out
//'
//' @return theta_loo Computed LOO solutions
arma::vec logistic_compute_LOO(arma::mat X, arma::vec Y, arma::vec theta_init, double lam, int case_index) {
  const int n=X.n_rows;
  const int p=X.n_cols;
  arma::mat X_t(p,n);
  X_t=arma::trans(X);
  arma::vec weight(n);
  weight.ones();
  weight[case_index] = 0;
  weight = weight/n;

  if (n < p) {
    // High dimension (use woodbury formula)
    arma::vec b_Newton(n);
    b_Newton=arma::exp(Y%(X*theta_init) ) ;
    arma::mat U(p, n);
    U=X_t*diagmat(b_Newton/arma::pow(1+b_Newton,2)%weight/lam);
    arma::mat V(n, p);
    V=X;
    arma::mat temp_Newton_1(p, p);
    temp_Newton_1=1/lam*(arma::eye(p,p)-U*arma::solve(arma::eye(n,n)+V*U,V));
    arma::vec temp_Newton_2(p);
    temp_Newton_2=X_t*(-1/(1+b_Newton)%Y%weight)+lam*theta_init;
    return (theta_init-temp_Newton_1*temp_Newton_2);
  }
  else {
    // Low dimension
    arma::vec b_Newton(n);
    b_Newton=arma::exp(Y%(X*theta_init ) ) ;
    arma::vec temp_Newton_1(n);
    temp_Newton_1=(b_Newton/pow(1+b_Newton,2))%weight;
    arma::mat temp_Newton_2(p,p);
    temp_Newton_2=X_t*diagmat(temp_Newton_1)*X+lam*arma::eye(p,p);
    arma::vec temp_Newton_3(p);
    temp_Newton_3=X_t*(-1/(1+b_Newton)%Y%weight)+lam*theta_init;
    return (theta_init-arma::solve(temp_Newton_2,temp_Newton_3));
  }
}

//'Compute global influence for each case under logistic regression
//'
//' @description
//' Compute the global influence/Cook's distance for each case under logistic regression.
//'
//' @param X A \eqn{n \times p} feature matrix
//' @param Y A \eqn{n \times 1} response vector
//' @param theta_0 A \eqn{p \times 1} vector that corresponds to full-data solution
//' @param t_max Range of solution path \eqn{[0, t_{\max}]}
//' @param lam Regularization parameter for L2 penalty
//' @param epsilon Pre-specified error tolerance of the entire path
//' @param method The method to use at each grid point, two options are "Newton" and "GD"
//' @param influence_measure The influence measure used to compute global influence for each case, two options are "FMD" and "BDLD"
//'
//' @details
//' This function will be called by the main function Compute_CaseInflu_GLM, with class = "logistc".
//'
//' @return global_influence Global influence for each case
//' @return cook_distance Cook's distance for each case
// [[Rcpp::export(CaseInfluence_logistic)]]
Rcpp::List CaseInfluence_logistic(arma::mat X, arma::vec Y, arma::vec theta_0, double t_max, double lam, double epsilon, std::string method, std::string influence_measure) {
  const int n=X.n_rows;
  const int p=X.n_cols;
  arma::mat X_t(p,n);
  X_t=arma::trans(X);

  // Compute lipschitz constant for each case
  arma::vec L_list(n);
  L_list = arma::sum(X % X, 1) / (4*n);

  arma::vec margin_full = Y % (X * theta_0);
  arma::vec margin(n);
  margin.zeros();
  arma::vec CaseInfluence_vec(n);
  CaseInfluence_vec.zeros();
  arma::vec Cook_vec(n);
  Cook_vec.zeros();
  Rcpp::List case_path;
  arma::vec theta_loo(p);
  theta_loo.zeros();
  for (int case_index = 0; case_index < n; case_index++) {
    if (method == "Newton" && p >= n) {
        case_path = logistic_Newton_high_dimension(X, Y, theta_0, t_max, lam, case_index, epsilon, L_list[case_index]);
    }
    else if (method == "Newton" && n > p) {
        case_path = logistic_Newton_low_dimension(X, Y, theta_0, t_max, lam, case_index, epsilon, L_list[case_index]);
    }
    else if (method == "GD") {
        case_path = logistic_GD(X, Y, theta_0, t_max, lam, case_index, epsilon, L_list[case_index]);
    }
    arma::mat theta_output = case_path[0];
    arma::vec t_vec = case_path[1];
    arma::vec alpha_t_vec = case_path[2];

    // Compute LOO solution
    theta_loo = logistic_compute_LOO(X, Y, theta_output.tail_cols(1), lam, case_index);
    if (influence_measure == "FMD") {
        for (int i = 1; i < t_vec.size(); i++) {
            margin = Y % (X * theta_output.col(i));
            CaseInfluence_vec[case_index] += 1.0 / n * arma::sum(pow(margin_full-margin, 2) ) * (std::exp(-t_vec[i-1]) - std::exp(-t_vec[i]));
        }
        if (t_vec[t_vec.size()-1] < t_max) {
            CaseInfluence_vec[case_index] += 1.0 / n * arma::sum(pow(margin_full-margin, 2) ) * (std::exp(-t_vec[t_vec.size()-1]) - std::exp(-t_max)); 
        }

        // Compute Cook's distance
        margin = Y % (X * theta_loo);
        Cook_vec[case_index] = 1.0 / n * arma::sum(pow(margin_full-margin, 2) );
    }
    else if (influence_measure == "BDLD") {
        for (int i = 1; i < t_vec.size(); i++) {
            margin = Y % (X * theta_output.col(i));
            CaseInfluence_vec[case_index] += 1.0 / n * arma::sum(pow(log1p(arma::exp(-margin_full))-log1p(arma::exp(-margin)), 2) ) * (std::exp(-t_vec[i-1]) - std::exp(-t_vec[i]));
        }
        if (t_vec[t_vec.size()-1] < t_max) {
            CaseInfluence_vec[case_index] += 1.0 / n * arma::sum(pow(log1p(arma::exp(-margin_full))-log1p(arma::exp(-margin)), 2) ) * (std::exp(-t_vec[t_vec.size()-1]) - std::exp(-t_max)); 
        }

        // Compute Cook's distance
        margin = Y % (X * theta_loo);
        Cook_vec[case_index] = 1.0 / n * arma::sum(pow(log1p(arma::exp(-margin_full))-log1p(arma::exp(-margin)), 2) );
    }
  }
  return Rcpp::List::create(Rcpp::Named("global_influence") = CaseInfluence_vec,
                            Rcpp::Named("cook_distance") = Cook_vec);
}

//'Compute local influence for each case under logistic regression
//'
//' @description
//' Compute the local influence for each case under logistic regression.
//'
//' @param X A \eqn{n \times p} feature matrix
//' @param Y A \eqn{n \times 1} response vector
//' @param theta_0 A \eqn{p \times 1} vector that corresponds to full-data solution
//' @param lam Regularization parameter for L2 penalty
//' @param influence_measure The influence measure used to compute global influence for each case, two options are "FMD" and "BDLD"
//'
//' @details
//' This function will be called by the main function Compute_LocalInflu_GLM, with class = "logistic".
//'
//' @return LocalInfluence_vec Local influence for each case
// [[Rcpp::export(LocalInfluence_logistic)]]
arma::vec LocalInfluence_logistic(arma::mat X, arma::vec Y, arma::vec theta_0, double lam, std::string influence_measure) {
  const int n = X.n_rows;
  const int p = X.n_cols;
  arma::mat X_t(p,n);
  X_t=arma::trans(X);

  arma::vec b_Newton=arma::exp(Y%(X*theta_0)) ;
  arma::vec weight(n);
  weight.ones();
  weight = weight/n;
  arma::mat hessian(p, p);
  hessian.zeros();
  arma::vec theta_grad(p);
  theta_grad.zeros();

  arma::vec LocalInfluence_vec(n);
  LocalInfluence_vec.zeros();

  if (n > p) {    
    arma::vec temp_Newton_1=(b_Newton/pow(1+b_Newton,2))%weight;
    arma::mat temp_Newton_2=X_t*diagmat(temp_Newton_1)*X+lam*arma::eye(p,p);
    hessian=arma::solve(temp_Newton_2,arma::eye(p,p));
  }
  else {
    arma::mat U=X_t*diagmat(b_Newton/arma::pow(1+b_Newton,2)%weight/lam);
    arma::mat V=X;
    hessian=1/lam*(arma::eye(p,p)-U*arma::solve(arma::eye(n,n)+V*U,V));
  }

  for (int case_index = 0; case_index < n; case_index++) {
    theta_grad=hessian*(-1/(1+b_Newton[case_index])*Y[case_index]*weight[case_index]*X_t.col(case_index));
    if (influence_measure == "FMD") {
      LocalInfluence_vec[case_index] = 2.0 / n * arma::sum(pow(Y%(X*theta_grad), 2));
    }
    else {
      LocalInfluence_vec[case_index] = 2.0 / n * arma::sum(pow( (1/(1+b_Newton))%Y%(X*theta_grad), 2));
    }   
  }

  return LocalInfluence_vec;
}

//'Case-weight adjusted solution path for poisson regression Newton method (\eqn{n \le p})
//'
//' @description
//' Generate solution path for case-weight adjusted problem by adopting Newton method at each selected grid points,
//' which are adaptively selected to save computation while maintaining the accuracy of the entire path.
//'
//' @param X A \eqn{n \times p} feature matrix
//' @param Y A \eqn{n \times 1} response vector
//' @param theta_0 A \eqn{p \times 1} vector that corresponds to full-data solution
//' @param t_max Range of solution path \eqn{[0, t_{\max}]}
//' @param lam Regularization parameter for L2 penalty
//' @param case_index Index of the observation that is attached a weight
//' @param epsilon Pre-specified error tolerance of the entire path
//' @param L Lipschitz constant for the gradient of the loss function
//'
//' @details
//' This function will be called by the function CaseInfluence_poisson to generate case influence graph for each case, with method = "Newton".
//'
//' @return t_vec Adaptively selected grid points
//' @return theta Generated soultion path at grid points
//' @return alpha_t Step size between grid points
Rcpp::List poisson_Newton_high_dimension(arma::mat X, arma::vec Y,arma::vec theta_0, double t_max, double lam, int case_index, double epsilon, double L){
  const int n=X.n_rows;
  const int p=X.n_cols;
  arma::mat X_t(p,n);
  X_t=arma::trans(X);

  // Declare variables for step size computation
  const double C1=1.0/3;
  double norm_gradient_g=0;
  double norm_gradient_g0=0;
  arma::vec alpha_compare_vec(4);
  double alpha_temp=0;
  alpha_compare_vec.zeros();
  alpha_compare_vec(0)=t_max;
  alpha_compare_vec(1)=std::log(1+lam*C1/L);
  alpha_compare_vec(2)=t_max;
  alpha_compare_vec(3)=t_max;

  // Declare variables for Newton method (use woodbury formula for matrix inversion)
  arma::mat U(p,n);
  U.zeros();
  arma::mat V(n,p);
  V.zeros();
  double t_Newton=0;
  arma::mat temp_Newton_1(p,p);
  temp_Newton_1.zeros();
  arma::vec temp_Newton_2(p);
  temp_Newton_2.zeros();
  arma::vec weight(n);
  weight.ones();
  weight = weight/n;
  arma::vec b_Newton(n);
  b_Newton.zeros();

  double alpha_t=t_max;
  double t_step=0;
  int index_Newton=0;
  t_Newton=t_step;

  // Store values for termination criterion
  double LHS_condition=0;
  double RHS_condition=0;
  RHS_condition=epsilon;
  LHS_condition=RHS_condition;

  //Update alpha_1
  norm_gradient_g=arma::norm((arma::exp(X.row(case_index)*theta_0)-Y(case_index))*X.row(case_index),2)/n;
  norm_gradient_g0=norm_gradient_g;
  alpha_temp=2*lam*epsilon/pow(norm_gradient_g,2);
  alpha_compare_vec(3)=std::log(1+pow(alpha_temp,0.5));
  alpha_t=alpha_compare_vec.min();

  //Decide the storage space based on alpha_1
  const int N=3*std::round(t_max/alpha_compare_vec.min());
  arma::mat theta_Newton(p,N+1);
  theta_Newton.zeros();
  theta_Newton.col(0)=theta_0;
  arma::vec alpha_t_vec(N);
  alpha_t_vec.zeros();
  arma::vec t_vec(N);
  t_vec.zeros();

  alpha_t_vec(index_Newton)=alpha_t;
  t_step=t_step+alpha_t;
  t_vec(index_Newton)=t_step;
  t_Newton=t_step;
  weight[case_index] = std::exp(-t_Newton)/n;
  b_Newton=arma::exp(X*theta_Newton.col(index_Newton));
  U=X_t*diagmat(b_Newton%weight/lam);
  V=X;
  temp_Newton_1=1/lam*(arma::eye(p,p)-U*arma::solve(arma::eye(n,n)+V*U,V));
  temp_Newton_2=X_t*((b_Newton-Y)%weight)+lam*theta_Newton.col(index_Newton);
  theta_Newton.col(index_Newton+1)=theta_Newton.col(index_Newton)- temp_Newton_1*temp_Newton_2;

  index_Newton=index_Newton+1;
  norm_gradient_g=arma::norm((arma::exp(X.row(case_index)*theta_Newton.col(index_Newton))-Y(case_index))*X.row(case_index),2)/n;
  LHS_condition=std::exp(-3*t_step)*(std::exp(t_step)/lam+L/(2*pow(lam,2)))*pow(norm_gradient_g,2);

  while(t_step<t_max && 10*LHS_condition>RHS_condition){
    // update alpha;
    alpha_compare_vec(1)=std::log(1+std::exp(t_step)*C1*lam/L);
    alpha_compare_vec(2)=2*alpha_t;
    alpha_temp=std::exp(t_step)*(1-std::exp(-alpha_t_vec(0)))*norm_gradient_g0/norm_gradient_g;
    alpha_compare_vec(3)=std::log(1+alpha_temp);
    alpha_t=alpha_compare_vec.min();
    alpha_t_vec(index_Newton)=alpha_t;
    t_step=t_step+alpha_t;
    t_vec(index_Newton)=t_step;

    t_Newton=t_step;
    weight[case_index] = std::exp(-t_Newton)/n;
    b_Newton=arma::exp(X*theta_Newton.col(index_Newton));
    U=X_t*diagmat(b_Newton%weight/lam);
    V=X;
    temp_Newton_1=1/lam*(arma::eye(p,p)-U*arma::solve(arma::eye(n,n)+V*U,V));
    temp_Newton_2=X_t*((b_Newton-Y)%weight)+lam*theta_Newton.col(index_Newton);
    theta_Newton.col(index_Newton+1)=theta_Newton.col(index_Newton)- temp_Newton_1*temp_Newton_2;

    index_Newton=index_Newton+1;
    norm_gradient_g=arma::norm((arma::exp(X.row(case_index)*theta_Newton.col(index_Newton))-Y(case_index))*X.row(case_index),2)/n;
    LHS_condition=std::exp(-3*t_step)*(std::exp(t_step)/lam+L/(2*pow(lam,2)))*pow(norm_gradient_g,2);
  }

  arma::mat theta_Newton_output(p,index_Newton+1);
  theta_Newton_output=theta_Newton.cols(0,index_Newton);

  arma::vec alpha_t_vec_output(index_Newton);
  alpha_t_vec_output=alpha_t_vec.subvec(0,index_Newton-1);

  arma::vec t_vec_output(index_Newton+1);
  t_vec_output(0)=0;
  t_vec_output.subvec(1,index_Newton)=t_vec.subvec(0,index_Newton-1);

  return Rcpp::List::create(Rcpp::Named("theta") = theta_Newton_output,
                            Rcpp::Named("t_vec") = t_vec_output,
                            Rcpp::Named("alpha_t_vec") = alpha_t_vec_output);
}

//'Case-weight adjusted solution path for poisson regression Newton method (\eqn{p \le n})
//'
//' @description
//' Generate solution path for case-weight adjusted problem by adopting Newton method at each selected grid points,
//' which are adaptively selected to save computation while maintaining the accuracy of the entire path.
//'
//' @param X A \eqn{n \times p} feature matrix
//' @param Y A \eqn{n \times 1} response vector
//' @param theta_0 A \eqn{p \times 1} vector that corresponds to full-data solution
//' @param t_max Range of solution path \eqn{[0, t_{\max}]}
//' @param lam Regularization parameter for L2 penalty
//' @param case_index Index of the observation that is attached a weight
//' @param epsilon Pre-specified error tolerance of the entire path
//' @param L Lipschitz constant for the gradient of the loss function
//'
//' @details
//' This function will be called by the function CaseInfluence_poisson to generate case influence graph for each case, with method = "Newton".
//'
//' @return t_vec Adaptively selected grid points
//' @return theta Generated soultion path at grid points
//' @return alpha_t Step size between grid points
Rcpp::List poisson_Newton_low_dimension(arma::mat X, arma::vec Y,arma::vec theta_0, double t_max, double lam, int case_index, double epsilon, double L){
  const int n=X.n_rows;
  const int p=X.n_cols;
  arma::mat X_t(p,n);
  X_t=arma::trans(X);

  // Declare variables for step size computation
  const double C1=1.0/3;
  double norm_gradient_g=0;
  double norm_gradient_g0=0;
  arma::vec alpha_compare_vec(4);
  double alpha_temp=0;
  alpha_compare_vec.zeros();
  alpha_compare_vec(0)=t_max;
  alpha_compare_vec(1)=std::log(1+lam*C1/L);
  alpha_compare_vec(2)=t_max;
  alpha_compare_vec(3)=t_max;

  // Declare variables for Newton method
  double t_Newton=0;
  arma::vec temp_Newton_1(n);
  temp_Newton_1.zeros();
  arma::mat temp_Newton_2(p, p);
  temp_Newton_2.zeros();
  arma::vec temp_Newton_3(p);
  temp_Newton_3.zeros();
  arma::vec weight(n);
  weight.ones();
  weight = weight/n;
  arma::vec b_Newton(n);
  b_Newton.zeros();

  double alpha_t=t_max;
  double t_step=0;
  int index_Newton=0;
  t_Newton=t_step;

  // Store values for termination criterion
  double LHS_condition=0;
  double RHS_condition=0;
  RHS_condition=epsilon;
  LHS_condition=RHS_condition;

  //Update alpha_1
  norm_gradient_g=arma::norm((arma::exp(X.row(case_index)*theta_0)-Y(case_index))*X.row(case_index),2)/n;
  norm_gradient_g0=norm_gradient_g;
  alpha_temp=2*lam*epsilon/pow(norm_gradient_g,2);
  alpha_compare_vec(3)=std::log(1+pow(alpha_temp,0.5));
  alpha_t=alpha_compare_vec.min();

  //Decide the storage space based on alpha_1
  const int N=3*std::round(t_max/alpha_compare_vec.min());
  arma::mat theta_Newton(p,N+1);
  theta_Newton.zeros();
  theta_Newton.col(0)=theta_0;
  arma::vec alpha_t_vec(N);
  alpha_t_vec.zeros();
  arma::vec t_vec(N);
  t_vec.zeros();

  alpha_t_vec(index_Newton)=alpha_t;
  t_step=t_step+alpha_t;
  t_vec(index_Newton)=t_step;
  t_Newton=t_step;
  weight[case_index] = std::exp(-t_Newton)/n;
  b_Newton=arma::exp(X*theta_Newton.col(index_Newton));
  temp_Newton_1=b_Newton%weight;
  temp_Newton_2=X_t*diagmat(temp_Newton_1)*X+lam*arma::eye(p,p);
  temp_Newton_3=X_t*((b_Newton-Y)%weight)+lam*theta_Newton.col(index_Newton);
  theta_Newton.col(index_Newton+1)=theta_Newton.col(index_Newton)- arma::solve(temp_Newton_2,temp_Newton_3);

  index_Newton=index_Newton+1;
  norm_gradient_g=arma::norm((arma::exp(X.row(case_index)*theta_Newton.col(index_Newton))-Y(case_index))*X.row(case_index),2)/n;
  LHS_condition=std::exp(-3*t_step)*(std::exp(t_step)/lam+L/(2*pow(lam,2)))*pow(norm_gradient_g,2);

  while(t_step<t_max && 10*LHS_condition>RHS_condition){
    // update alpha;
    alpha_compare_vec(1)=std::log(1+std::exp(t_step)*C1*lam/L);
    alpha_compare_vec(2)=2*alpha_t;
    alpha_temp=std::exp(t_step)*(1-std::exp(-alpha_t_vec(0)))*norm_gradient_g0/norm_gradient_g;
    alpha_compare_vec(3)=std::log(1+alpha_temp);
    alpha_t=alpha_compare_vec.min();
    alpha_t_vec(index_Newton)=alpha_t;
    t_step=t_step+alpha_t;
    t_vec(index_Newton)=t_step;

    t_Newton=t_step;
    weight[case_index] = std::exp(-t_Newton)/n;
    b_Newton=arma::exp(X*theta_Newton.col(index_Newton));
    temp_Newton_1=b_Newton%weight;
    temp_Newton_2=X_t*diagmat(temp_Newton_1)*X+lam*arma::eye(p,p);
    temp_Newton_3=X_t*((b_Newton-Y)%weight)+lam*theta_Newton.col(index_Newton);
    theta_Newton.col(index_Newton+1)=theta_Newton.col(index_Newton)- arma::solve(temp_Newton_2,temp_Newton_3);

    index_Newton=index_Newton+1;
    norm_gradient_g=arma::norm((arma::exp(X.row(case_index)*theta_Newton.col(index_Newton))-Y(case_index))*X.row(case_index),2)/n;
    LHS_condition=std::exp(-3*t_step)*(std::exp(t_step)/lam+L/(2*pow(lam,2)))*pow(norm_gradient_g,2);
  }

  arma::mat theta_Newton_output(p,index_Newton+1);
  theta_Newton_output=theta_Newton.cols(0,index_Newton);

  arma::vec alpha_t_vec_output(index_Newton);
  alpha_t_vec_output=alpha_t_vec.subvec(0,index_Newton-1);

  arma::vec t_vec_output(index_Newton+1);
  t_vec_output(0)=0;
  t_vec_output.subvec(1,index_Newton)=t_vec.subvec(0,index_Newton-1);

  return Rcpp::List::create(Rcpp::Named("theta") = theta_Newton_output,
                            Rcpp::Named("t_vec") = t_vec_output,
                            Rcpp::Named("alpha_t_vec") = alpha_t_vec_output);
}

//'Case-weight adjusted solution path for poisson regression gradient descent method
//'
//' @description
//' Generate solution path for case-weight adjusted problem by adopting gradient descent method at each selected grid points,
//' which are adaptively selected to save computation while maintaining the accuracy of the entire path.
//'
//' @param X A \eqn{n \times p} feature matrix
//' @param Y A \eqn{n \times 1} response vector
//' @param theta_0 A \eqn{p \times 1} vector that corresponds to full-data solution
//' @param t_max Range of solution path \eqn{[0, t_{\max}]}
//' @param lam Regularization parameter for L2 penalty
//' @param case_index Index of the observation that is attached a weight
//' @param epsilon Pre-specified error tolerance of the entire path
//' @param L Lipschitz constant for the gradient of the loss function
//'
//' @details
//' This function will be called by the function CaseInfluence_poisson to generate case influence graph for each case, with method = "GD".
//'
//' @return t_vec Adaptively selected grid points
//' @return theta Generated soultion path at grid points
//' @return alpha_t Step size between grid points
Rcpp::List poisson_GD(arma::mat X, arma::vec Y,arma::vec theta_0, double t_max, double lam, int case_index, double epsilon, double L) {
  const int n=X.n_rows;
  const int p=X.n_cols;
  arma::mat X_t(p,n);
  X_t=arma::trans(X);

  // Declare variables for step size computation
  const double C=1.0/3;
  const double C1=1.0/3;
  arma::vec alpha_compare_vec(4);
  double alpha_temp=0;
  alpha_compare_vec.zeros();
  alpha_compare_vec(0)=t_max;
  alpha_compare_vec(1)=std::log(1+lam*C1/L);
  alpha_compare_vec(2)=t_max;
  alpha_compare_vec(3)=t_max;

  // Declare variables for gradient descent
  arma::vec theta_GD_temp(p);
  theta_GD_temp.zeros();
  arma::vec theta_GD_temp1(p);
  theta_GD_temp1.zeros();
  arma::mat b_GD(n,p);
  b_GD.zeros();
  arma::vec weight(n);
  weight.ones();
  weight = weight/n;

  double alpha_t=0;
  double t_step=0;

  int index_GD=0;
  double t_GD=0;
  double norm_gradient_g=0;
  double norm_gradient_g0=0;
  arma::vec g_k(p);
  g_k.zeros();
  double norm_gk=0;

  // declaration of variables that will be used in backtracking line search
  double eta0=0.01;
  double eta_temp=0;
  double c=0.5;
  double tau=0.2;
  double m=0;
  double t=0;
  double f_theta_temp=0;
  double f_theta_temp1=0;
  arma::vec search_direction(p);
  search_direction.zeros();

  // Store values for termination criterion
  double LHS_condition=0;
  double RHS_condition=0;
  RHS_condition=epsilon;
  LHS_condition=RHS_condition;

  //Update alpha_1
  norm_gradient_g=arma::norm((arma::exp(X.row(case_index)*theta_0)-Y(case_index))*X.row(case_index),2)/n;
  norm_gradient_g0=norm_gradient_g;
  alpha_temp=2*lam*epsilon/pow(norm_gradient_g,2);
  alpha_compare_vec(3)=std::log(1+pow(alpha_temp,0.5));
  alpha_t=alpha_compare_vec.min();

  //Decide the storage space based on alpha_1
  const int N=3*std::round(t_max/alpha_compare_vec.min());
  arma::mat theta_GD(p,N+1);
  theta_GD.zeros();
  theta_GD.col(0) = theta_0;
  arma::vec alpha_t_vec(N);
  alpha_t_vec.zeros();
  arma::vec t_vec(N);
  t_vec.zeros();

  alpha_t_vec(index_GD)=alpha_t;
  t_step=t_step+alpha_t;
  t_vec(index_GD)=t_step;

  //Calculate first step
  t_GD=t_step;
  theta_GD_temp=theta_GD.col(index_GD);
  b_GD=diagmat(arma::exp(X*theta_GD_temp)-Y)*X;
  weight[case_index] = exp(-t_GD)/n;
  g_k=trans(b_GD)*weight+lam*theta_GD_temp;
  norm_gk=arma::norm(g_k,2);
  norm_gradient_g=arma::norm((arma::exp(X.row(case_index)*theta_GD.col(index_GD))-Y(case_index))*X.row(case_index),2)/n;
  while (norm_gk>C*(std::exp(alpha_t-t_step)-std::exp(-t_step))*norm_gradient_g){
    eta_temp=eta0;
    search_direction=-g_k/norm_gk;
    m=-norm_gk;
    t=-c*m;
    f_theta_temp=sum((arma::exp(X*theta_GD_temp)-Y%(X*theta_GD_temp))%weight)+lam/2*pow(arma::norm(theta_GD_temp,2),2);
    theta_GD_temp1=theta_GD_temp+eta_temp*search_direction;
    f_theta_temp1=sum((arma::exp(X*theta_GD_temp1)-Y%(X*theta_GD_temp1))%weight)+lam/2*pow(arma::norm(theta_GD_temp1,2),2);
    while (f_theta_temp-f_theta_temp1<eta_temp*t){
      eta_temp=eta_temp*tau;
      theta_GD_temp1=theta_GD_temp+eta_temp*search_direction;
      f_theta_temp1=sum((arma::exp(X*theta_GD_temp1)-Y%(X*theta_GD_temp1))%weight)+lam/2*pow(arma::norm(theta_GD_temp1,2),2);
    }
    theta_GD_temp=theta_GD_temp1;
    b_GD=diagmat(arma::exp(X*theta_GD_temp)-Y)*X;
    g_k=trans(b_GD)*weight+lam*theta_GD_temp;
    norm_gk=arma::norm(g_k,2);
    norm_gradient_g=arma::norm((arma::exp(X.row(case_index)*theta_GD.col(index_GD))-Y(case_index))*X.row(case_index),2)/n;
  }
  theta_GD.col(index_GD+1)=theta_GD_temp;
  index_GD=index_GD+1;
  LHS_condition=std::exp(-3*t_step)*(std::exp(t_step)/lam+L/(2*pow(lam,2)))*pow(norm_gradient_g,2);

  while (t_step<t_max && 10*LHS_condition>RHS_condition){
    alpha_compare_vec(1)=std::log(1+std::exp(t_step)*C1*lam/L);
    alpha_compare_vec(2)=2*alpha_t;
    alpha_temp=std::exp(t_step)*(1-std::exp(-alpha_t_vec(0)))*norm_gradient_g0/norm_gradient_g;
    alpha_compare_vec(3)=std::log(1+alpha_temp);
    alpha_t=alpha_compare_vec.min();
    alpha_t_vec(index_GD)=alpha_t;
    t_step=t_step+alpha_t;
    t_vec(index_GD)=t_step;

    t_GD=t_step;
    theta_GD_temp=theta_GD.col(index_GD);
    b_GD=diagmat(arma::exp(X*theta_GD_temp)-Y)*X;
    weight[case_index] = exp(-t_GD)/n;
    g_k=trans(b_GD)*weight+lam*theta_GD_temp;
    norm_gk=arma::norm(g_k,2);
    norm_gradient_g=arma::norm((arma::exp(X.row(case_index)*theta_GD.col(index_GD))-Y(case_index))*X.row(case_index),2)/n;
    while (norm_gk>C*(std::exp(alpha_t-t_step)-std::exp(-t_step))*norm_gradient_g){
      eta_temp=eta0;
      search_direction=-g_k/norm_gk;
      m=-norm_gk;
      t=-c*m;
      f_theta_temp=sum((arma::exp(X*theta_GD_temp)-Y%(X*theta_GD_temp))%weight)+lam/2*pow(arma::norm(theta_GD_temp,2),2);
      theta_GD_temp1=theta_GD_temp+eta_temp*search_direction;
      f_theta_temp1=sum((arma::exp(X*theta_GD_temp1)-Y%(X*theta_GD_temp1))%weight)+lam/2*pow(arma::norm(theta_GD_temp1,2),2);
      while (f_theta_temp-f_theta_temp1<eta_temp*t){
        eta_temp=eta_temp*tau;
        theta_GD_temp1=theta_GD_temp+eta_temp*search_direction;
        f_theta_temp1=sum((arma::exp(X*theta_GD_temp1)-Y%(X*theta_GD_temp1))%weight)+lam/2*pow(arma::norm(theta_GD_temp1,2),2);
      }
      theta_GD_temp=theta_GD_temp1;
      b_GD=diagmat(arma::exp(X*theta_GD_temp)-Y)*X;
      g_k=trans(b_GD)*weight+lam*theta_GD_temp;
      norm_gk=arma::norm(g_k,2);
      norm_gradient_g=arma::norm((arma::exp(X.row(case_index)*theta_GD.col(index_GD))-Y(case_index))*X.row(case_index),2)/n;
    }
    theta_GD.col(index_GD+1)=theta_GD_temp;
    index_GD=index_GD+1;
    LHS_condition=std::exp(-3*t_step)*(std::exp(t_step)/lam+L/(2*pow(lam,2)))*pow(norm_gradient_g,2);
  }

  arma::mat theta_GD_output(p,index_GD+1);
  theta_GD_output=theta_GD.cols(0,index_GD);

  arma::vec alpha_t_vec_output(index_GD);
  alpha_t_vec_output=alpha_t_vec.subvec(0,index_GD-1);

  arma::vec t_vec_output(index_GD+1);
  t_vec_output(0)=0;
  t_vec_output.subvec(1,index_GD)=t_vec.subvec(0,index_GD-1);

  return Rcpp::List::create(Rcpp::Named("theta") = theta_GD_output,
                            Rcpp::Named("t_vec") = t_vec_output,
                            Rcpp::Named("alpha_t_vec") = alpha_t_vec_output);
}

//' Compute the LOO solution for L2 regularized poisson regression 
//'
//' @description
//' Given the initial point, compute the LOO solution for L2 regularized poisson regression by one-step Newton method
//'
//' @param X A \eqn{n \times p} feature matrix
//' @param Y A \eqn{n \times 1} response vector
//' @param theta_init A \eqn{p \times 1} vector, which is the initial point
//' @param lam Regularization parameter for L2 penalty
//' @param case_index Index of the observation that is left out
//'
//' @return theta_loo Computed LOO solutions
arma::vec poisson_compute_LOO(arma::mat X, arma::vec Y, arma::vec theta_init, double lam, int case_index) {
  const int n=X.n_rows;
  const int p=X.n_cols;
  arma::mat X_t(p,n);
  X_t=arma::trans(X);
  arma::vec weight(n);
  weight.ones();
  weight[case_index] = 0;
  weight = weight/n;

  if (n < p) {
    // High dimension (use woodbury formula)
    arma::vec b_Newton(n);
    b_Newton=arma::exp(X*theta_init);
    arma::mat U(p, n);
    U=X_t*diagmat(b_Newton%weight/lam);
    arma::mat V(n, p);
    V=X;
    arma::mat temp_Newton_1(p, p);
    temp_Newton_1=1/lam*(arma::eye(p,p)-U*arma::solve(arma::eye(n,n)+V*U,V));
    arma::vec temp_Newton_2(p);
    temp_Newton_2=X_t*((b_Newton-Y)%weight)+lam*theta_init;
    return (theta_init-temp_Newton_1*temp_Newton_2);
  }
  else {
    // Low dimension
    arma::vec b_Newton(n);
    b_Newton=arma::exp(X*theta_init);
    arma::vec temp_Newton_1(n);
    temp_Newton_1=b_Newton%weight;
    arma::mat temp_Newton_2(p,p);
    temp_Newton_2=X_t*diagmat(temp_Newton_1)*X+lam*arma::eye(p,p);
    arma::vec temp_Newton_3(p);
    temp_Newton_3=X_t*((b_Newton-Y)%weight)+lam*theta_init;
    return (theta_init-arma::solve(temp_Newton_2,temp_Newton_3));
  }
}

//'Compute global influence for each case under Poisson regression
//'
//' @description
//' Compute the global influence/Cook's distance for each case under Poisson regression.
//'
//' @param X A \eqn{n \times p} feature matrix
//' @param Y A \eqn{n \times 1} response vector
//' @param theta_0 A \eqn{p \times 1} vector that corresponds to full-data solution
//' @param t_max Range of solution path \eqn{[0, t_{\max}]}
//' @param lam Regularization parameter for L2 penalty
//' @param epsilon Pre-specified error tolerance of the entire path
//' @param method The method to use at each grid point, two options are "Newton" and "GD"
//'
//' @details
//' This function will be called by the main function Compute_CaseInflu_GLM, with class = "poisson".
//'
//' @return global_influence Global influence for each case
//' @return cook_distance Cook's distance for each case
// [[Rcpp::export(CaseInfluence_poisson)]]
Rcpp::List CaseInfluence_poisson(arma::mat X, arma::vec Y, arma::vec theta_0, double t_max, double lam, double epsilon, std::string method) {
  const int n=X.n_rows;
  const int p=X.n_cols;
  arma::mat X_t(p,n);
  X_t=arma::trans(X);

  // Compute lipschitz constant for each case
  arma::vec L_list(n);
  L_list = 3 * (arma::sum(X % X, 1) % arma::exp(X * theta_0)) / n;

  arma::vec fitted_full = arma::exp(X * theta_0);
  arma::vec fitted(n);
  fitted.zeros();
  arma::vec CaseInfluence_vec(n);
  CaseInfluence_vec.zeros();
  arma::vec Cook_vec(n);
  Cook_vec.zeros();
  Rcpp::List case_path;
  arma::vec theta_loo(p);
  theta_loo.zeros();
  for (int case_index = 0; case_index < n; case_index++) {
    if (method == "Newton" && p >= n) {
        case_path = poisson_Newton_high_dimension(X, Y, theta_0, t_max, lam, case_index, epsilon, L_list[case_index]);
    }
    else if (method == "Newton" && n > p) {
        case_path = poisson_Newton_low_dimension(X, Y, theta_0, t_max, lam, case_index, epsilon, L_list[case_index]);
    }
    else if (method == "GD") {
        case_path = poisson_GD(X, Y, theta_0, t_max, lam, case_index, epsilon, L_list[case_index]);
    }
    arma::mat theta_output = case_path[0];
    arma::vec t_vec = case_path[1];
    arma::vec alpha_t_vec = case_path[2];

    // Compute LOO solution
    theta_loo = poisson_compute_LOO(X, Y, theta_output.tail_cols(1), lam, case_index);
    for (int i = 1; i < t_vec.size(); i++) {
        fitted = arma::exp(X * theta_output.col(i));
        CaseInfluence_vec[case_index] += 1.0 / n * arma::sum(pow(fitted_full-fitted, 2) ) * (std::exp(-t_vec[i-1]) - std::exp(-t_vec[i]));
    }
    if (t_vec[t_vec.size()-1] < t_max) {
        CaseInfluence_vec[case_index] += 1.0 / n * arma::sum(pow(fitted_full-fitted, 2) ) * (std::exp(-t_vec[t_vec.size()-1]) - std::exp(-t_max)); 
    }

    // Compute Cook's distance
    fitted = arma::exp(X * theta_loo);
    Cook_vec[case_index] = 1.0 / n * arma::sum(pow(fitted_full-fitted, 2) );
  }
  return Rcpp::List::create(Rcpp::Named("global_influence") = CaseInfluence_vec,
                            Rcpp::Named("cook_distance") = Cook_vec);
}

//'Compute local influence for each case under Poisson regression
//'
//' @description
//' Compute the local influence for each case under Poisson regression.
//'
//' @param X A \eqn{n \times p} feature matrix
//' @param Y A \eqn{n \times 1} response vector
//' @param theta_0 A \eqn{p \times 1} vector that corresponds to full-data solution
//' @param lam Regularization parameter for L2 penalty
//'
//' @details
//' This function will be called by the main function Compute_LocalInflu_GLM, with class = "poisson".
//'
//' @return LocalInfluence_vec Local influence for each case
// [[Rcpp::export(LocalInfluence_poisson)]]
arma::vec LocalInfluence_poisson(arma::mat X, arma::vec Y, arma::vec theta_0, double lam) {
  const int n = X.n_rows;
  const int p = X.n_cols;
  arma::mat X_t(p,n);
  X_t=arma::trans(X);

  arma::vec b_Newton=arma::exp(X*theta_0);
  arma::vec weight(n);
  weight.ones();
  weight = weight/n;
  arma::mat hessian(p, p);
  hessian.zeros();
  arma::vec theta_grad(p);
  theta_grad.zeros();

  arma::vec LocalInfluence_vec(n);
  LocalInfluence_vec.zeros();

  if (n > p) {    
    arma::vec temp_Newton_1=b_Newton%weight;
    arma::mat temp_Newton_2=X_t*diagmat(temp_Newton_1)*X+lam*arma::eye(p,p);
    hessian=arma::solve(temp_Newton_2,arma::eye(p,p));
  }
  else {
    arma::mat U=X_t*diagmat(b_Newton%weight/lam);
    arma::mat V=X;
    hessian=1/lam*(arma::eye(p,p)-U*arma::solve(arma::eye(n,n)+V*U,V));
  }

  for (int case_index = 0; case_index < n; case_index++) {
    theta_grad=hessian*((b_Newton[case_index]-Y[case_index])*weight[case_index]*X_t.col(case_index));
    LocalInfluence_vec[case_index] = 2.0 / n * arma::sum(pow(b_Newton%(X*theta_grad), 2));
  }

  return LocalInfluence_vec;
}
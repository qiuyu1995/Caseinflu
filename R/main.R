#' Path-following algorithm for L2 penalized nonsmooth problems
#' 
#' @description 
#' Compute exact solution path for case-weight adjusted problem 
#' \loadmathjax
#' \mjseqn{(\beta_{0,w}, \beta_{w}) = argmin_{\beta_0, \beta} \sum_{i \neq j} f(g_i(\beta_0, \beta)) + w*f(g_{j}(\beta_0, \beta)) + \lambda / 2 * \|\beta\|_2^2}
#' for \mjseqn{0 <= w <= 1}, where \mjseqn{g_i(\beta_0, \beta) = a_i \beta_0 + b_i^T \beta + c_i} and \mjseqn{f(r) = \alpha_0 \max(r, 0) + \alpha_1 \max(-r, 0).}
#' The exact solution path is shown to be piece-wise linear 
#' and the function will return all the breakpoints and their solutions.
#' 
#' @param X A \eqn{n \times p} feature matrix
#' @param Y A \eqn{n \times 1} response vector
#' @param lam Regularization parameter for L2 penalty
#' @param obs_index Index of the observation that is attached a weight, ranging from 1 to n
#' @param class Specify the problem class with default value "quantile". Use "quantile" for quantile regression, and "svm" for support vector machine
#' @param tau The parameter for quantile regression, ranging between 0 and 1. No need to specify the value of it if choose class = "svm"
#' 
#' @return W_vec A list of breakout points
#' @return Beta_0 True values of beta_{0,w} at breakout points
#' @return Beta True values of beta_{w} at breakout points
#' @export

nonsmooth_path <- function(X, Y, lam, obs_index, class = "quantile", tau = 0.5){
  n = dim(X)[1]
  p = dim(X)[2]
  if (class == "quantile") {
    a = -rep(1, n)
    B = -X
    c = Y
    alpha_0 = tau
    alpha_1 = 1 - tau
  }
  else if (class == "svm") {
    a = Y
    B = matrix(0, nrow = n, ncol = p)
    for (i in 1:n)
      B[i, ] = X[i, ] * Y[i]
    c = -rep(1, n)
    alpha_0 = 0
    alpha_1 = 1
  }
  else {
    cat("\n The current version only supports 'quantile' and 'svm' class options.\n")
    return(0)
  }
  
  ## Compute the full-data solution
  beta <- CVXR::Variable(p)
  beta_0 = CVXR::Variable()
  xi = CVXR::Variable(n)
  eta = CVXR::Variable(n)
  obj = alpha_0 * CVXR::sum_entries(xi) + alpha_1 * CVXR::sum_entries(eta) + lam / 2 * CVXR::power(CVXR::norm2(beta), 2)
  constraints = list(xi >= 0, eta >= 0, beta_0 * a + B %*% beta + c + eta >= 0, - beta_0 * a - B %*% beta - c + xi >= 0)
  prob <- CVXR::Problem(Minimize(obj), constraints)
  result <- CVXR::solve(prob, abstol=1e-10, reltol=1e-10)
  beta_0_w0 <- result$getValue(beta_0)
  beta_w0 <- result$getValue(beta)
  theta_w0 <- result$getDualValue(constraints[[3]]) - result$getDualValue(constraints[[4]])
  ## Run the path-following algorithm
  .Call(`_Caseinflu_case_path_nonsmooth`, a, B, c, lam, alpha_0, alpha_1, obs_index-1, beta_0_w0, beta_w0, theta_w0)
}

#' Case influence assessment for L2 penalized nonsmooth models
#' 
#' @description 
#' Compute the global influence, which is based on case influence graph, for each case under \eqn{\ell_2} regularized nonsmooth
#' models, including quantile regression and svm. To compute the case influence graph for each case, we need to solve a solution path problem.  
#' In our algorithm, the exact path can be computed and it is piece-wise linear.
#' 
#' @param X A \eqn{n \times p} feature matrix
#' @param Y A \eqn{n \times 1} response vector
#' @param lam Regularization parameter for L2 penalty
#' @param class Specify the problem class with default value "quantile". Use "quantile" for quantile regression, and "svm" for support vector machine
#' @param tau The parameter for quantile regression, ranging between 0 and 1. No need to specify the value of it if choose class = "svm"
#' @param influence_measure Specify the influence measure for computing global influence. For classification problem, "FMD" represents
#' functional margin difference and "BDLD" represents Binomial deviance loss difference. The default is "FMD". For regression problem, no 
#' need to specify the influence_measure
#' 
#' @return global_influence Global influence for each case
#' @return cook_distance Cook's distance for each case
#' @export
#' 
Compute_CaseInflu_nonsmooth <- function(X, Y, lam, class = "svm", tau = 0.5, influence_measure = "FMD"){
  n = dim(X)[1]
  p = dim(X)[2]
  
  if (influence_measure != "FMD" & influence_measure != "BDLD") {
    cat("\n The current version only supports 'BDLD' and 'FMD' influence measure options.\n")
    return(0)
  }
  
  if (class == "quantile") {
    a = -rep(1, n)
    B = -X
    c = Y
    alpha_0 = tau
    alpha_1 = 1 - tau
  }
  else if (class == "svm") {
    a = Y
    B = matrix(0, nrow = n, ncol = p)
    for (i in 1:n)
      B[i, ] = X[i, ] * Y[i]
    c = -rep(1, n)
    alpha_0 = 0
    alpha_1 = 1
  }
  else {
    cat("\n The current version only supports 'quantile' and 'svm' class options.\n")
    return(0)
  }
  
  ## Compute the full-data solution
  beta <- CVXR::Variable(p)
  beta_0 = CVXR::Variable()
  xi = CVXR::Variable(n)
  eta = CVXR::Variable(n)
  obj = alpha_0 * CVXR::sum_entries(xi) + alpha_1 * CVXR::sum_entries(eta) + lam / 2 * CVXR::power(CVXR::norm2(beta), 2)
  constraints = list(xi >= 0, eta >= 0, beta_0 * a + B %*% beta + c + eta >= 0, - beta_0 * a - B %*% beta - c + xi >= 0)
  prob <- CVXR::Problem(Minimize(obj), constraints)
  result <- CVXR::solve(prob, abstol=1e-10, reltol=1e-10)
  beta_0_w0 <- result$getValue(beta_0)
  beta_w0 <- result$getValue(beta)
  theta_w0 <- result$getDualValue(constraints[[3]]) - result$getDualValue(constraints[[4]])
  .Call(`_Caseinflu_CaseInfluence_nonsmooth`, a, B, c, lam, alpha_0, alpha_1, beta_0_w0, beta_w0, theta_w0, class, influence_measure)
}

#' Local influence assessment for L2 penalized nonsmooth models (svm + quantile regression)
#' 
#' @description 
#' Compute the local influence for each case under \eqn{\ell_2} regularized nonsmooth models, including quantile regression and svm.
#' 
#' @param X A \eqn{n \times p} feature matrix
#' @param Y A \eqn{n \times 1} response vector
#' @param lam Regularization parameter for L2 penalty
#' @param class Specify the problem class with default value "quantile". Use "quantile" for quantile regression, and "svm" for support vector machine
#' @param tau The parameter for quantile regression, ranging between 0 and 1. No need to specify the value of it if choose class = "svm"
#' @param influence_measure Specify the influence measure for computing global influence. For classification problem, "FMD" represents
#' functional margin difference and "BDLD" represents Binomial deviance loss diference. The default is "FMD". For regression problem, no 
#' need to specify the influence_measure
#' 
#' @return LocalInfluence_vec Local influence for each case
#' @export
#' 
Compute_LocalInflu_nonsmooth <- function(X, Y, lam, class = "svm", tau = 0.5, influence_measure = "FMD"){
  n = dim(X)[1]
  p = dim(X)[2]
  
  if (influence_measure != "FMD" & influence_measure != "BDLD") {
    cat("\n The current version only supports 'BDLD' and 'FMD' influence measure options.\n")
    return(0)
  }
  
  if (class == "quantile") {
    a = -rep(1, n)
    B = -X
    c = Y
    alpha_0 = tau
    alpha_1 = 1 - tau
  }
  else if (class == "svm") {
    a = Y
    B = matrix(0, nrow = n, ncol = p)
    for (i in 1:n)
      B[i, ] = X[i, ] * Y[i]
    c = -rep(1, n)
    alpha_0 = 0
    alpha_1 = 1
  }
  else {
    cat("\n The current version only supports 'quantile' and 'svm' class options.\n")
    return(0)
  }
  
  ## Compute the full-data solution
  beta <- CVXR::Variable(p)
  beta_0 = CVXR::Variable()
  xi = CVXR::Variable(n)
  eta = CVXR::Variable(n)
  obj = alpha_0 * CVXR::sum_entries(xi) + alpha_1 * CVXR::sum_entries(eta) + lam / 2 * CVXR::power(CVXR::norm2(beta), 2)
  constraints = list(xi >= 0, eta >= 0, beta_0 * a + B %*% beta + c + eta >= 0, - beta_0 * a - B %*% beta - c + xi >= 0)
  prob <- CVXR::Problem(Minimize(obj), constraints)
  result <- CVXR::solve(prob, abstol=1e-10, reltol=1e-10)
  beta_0_w0 <- result$getValue(beta_0)
  beta_w0 <- result$getValue(beta)
  theta_w0 <- result$getDualValue(constraints[[3]]) - result$getDualValue(constraints[[4]])
  .Call(`_Caseinflu_LocalInfluence_nonsmooth`, a, B, c, lam, alpha_0, alpha_1, beta_0_w0, beta_w0, theta_w0, class, influence_measure)
}

#' Path-following algorithm for L2 penalized Generalized linear models
#' 
#' @description 
#' Compute the global influence, which is based on case influence graph, for each case under \eqn{\ell_2} regularized Generalized linear
#' Models. To compute the case influence graph for each case, we need to solve the following solution path problem: 
#' \loadmathjax
#' \mjdeqn{\theta(t) = argmin_{\theta}  e^{-t} L_{j}(\theta) + \sum_{i \neq j} L_{i}(\theta) + \lambda / 2 * \|\theta\|_2^2 \text{ for } t \in [0, t_{max}]}{ASCII representation}
#' In our algorithm, the grid points are adaptively selected to save computation while maintaining the accuracy of the entire path.
#' At each grid point, the optimization problem can be solved by either Newton method or gradient descent method.
#' 
#' @param X A \eqn{n \times p} feature matrix
#' @param Y A \eqn{n \times 1} response vector
#' @param lam Regularization parameter for L2 penalty
#' @param obs_index Index of the observation that is attached a weight, ranging from 1 to n
#' @param epsilon Pre-specified error tolerance of the entire path
#' @param method Specify the method for computing the solution path. Default is "Newton". Use "GD" for gradient descent
#' @param t_max Range of solution path \eqn{[0, t_{\max}]}
#' @param class Specify the problem class with default value "logistic". Use "logistic" for logistic regression, and "poisson" for poisson regression
#' 
#' @return t_vec Adaptively selected grid points
#' @return theta Generated solution path at grid points
#' @return alpha_t Step size between grid points
#' @export
#' 
smooth_path <- function(X, Y, lam, obs_index, epsilon, method = "Newton", t_max = 10, class = "logistic"){
  n = dim(X)[1]
  p = dim(X)[2]
  L_vec = rep(0, n)
  
  if (class == "logistic") {
    for (i in 1:n)
      L_vec[i] = norm(X[i,], '2') ** 2 / (4*n)
    
    ###CVX calculates the full data solution
    beta <- CVXR::Variable(p)
    obj = 1/n*(CVXR::sum_entries(CVXR::logistic(X[Y == -1, ] %*% beta))+CVXR::sum_entries(CVXR::logistic(-X[Y == 1, ] %*% beta)))+lam/2*CVXR::power(CVXR::norm2(beta), 2)
    prob <- CVXR::Problem(Minimize(obj))
    result <- CVXR::solve(prob)
    Beta_0 <- result$getValue(beta)
    if (method == "Newton") {
      if (n < p) {
        .Call(`_Caseinflu_logistic_Newton_high_dimension`, X, Y, Beta_0, t_max, lam, obs_index-1, epsilon, L_vec[obs_index])
      }
      else {
        .Call(`_Caseinflu_logistic_Newton_low_dimension`, X, Y, Beta_0, t_max, lam, obs_index-1, epsilon, L_vec[obs_index])
      }
    }
    else if (method == "GD") {
      .Call(`_Caseinflu_logistic_GD`, X, Y, Beta_0, t_max, lam, obs_index-1, epsilon, L_vec[obs_index])
    }
    else {
      cat("\n The current version only supports 'GD' and 'Newton' method options.\n")
    }
  }
  else if (class == "poisson") {
    for (i in 1:n)
      L_vec[i] = 3 * norm(X[i,], '2') ** 2 / n
    ###CVX calculates the full data solution
    beta <- CVXR::Variable(p)
    obj = 1/n*CVXR::sum_entries(exp(X %*% beta) - CVXR::multiply(Y, X %*% beta))+lam/2*CVXR::power(CVXR::norm2(beta), 2)
    prob <- CVXR::Problem(Minimize(obj))
    result <- CVXR::solve(prob)
    Beta_0 <- result$getValue(beta)
    L_vec = L_vec * exp(X %*% Beta_0)
    if (method == "Newton") {
      if (n < p) {
        .Call(`_Caseinflu_poisson_Newton_high_dimension`, X, Y, Beta_0, t_max, lam, obs_index-1, epsilon, L_vec[obs_index])
      }
      else {
        .Call(`_Caseinflu_poisson_Newton_low_dimension`, X, Y, Beta_0, t_max, lam, obs_index-1, epsilon, L_vec[obs_index])
      }
    }
    else if (method == "GD") {
      .Call(`_Caseinflu_poisson_GD`, X, Y, Beta_0, t_max, lam, obs_index-1, epsilon, L_vec[obs_index])
    }
    else {
      cat("\n The current version only supports 'GD' and 'Newton' method options.\n")
    }
  }
  else {
    cat("\n The current version only supports 'logistic' and 'poisson' class options.\n")
  }
  
}

#' Case influence assessment for L2 penalized GLM
#' 
#' @description 
#' Compute the global influence, which is based on case influence graph, for each case under \eqn{\ell_2} regularized logistic
#' regression. To compute the case influence graph for each case, we need to solve a solution path problem. In our algorithm, 
#' the grid points are adaptively selected to save computation while maintaining the accuracy of the entire path. At each grid point,
#' the optimization problem can be solved by either Newton method or gradient descent method. 
#' 
#' @param X A \eqn{n \times p} feature matrix
#' @param Y A \eqn{n \times 1} response vector
#' @param lam Regularization parameter for L2 penalty
#' @param epsilon Pre-specified error tolerance of the entire path
#' @param t_max Range of solution path \eqn{[0, t_{\max}]}
#' @param class Specify the problem class with default value "logistic". Use "logistic" for logistic regression, and "poisson" for poisson regression
#' @param method Specify the method for computing the solution path. Default is "Newton". Use "GD" for gradient descent
#' @param influence_measure Specify the influence measure for computing global influence. For classification problem, "FMD" represents
#' functional margin difference and "BDLD" represents Binomial deviance loss diference. The default is "FMD". For regression problem, no 
#' need to specify the influence_measure
#' 
#' @return global_influence Global influence for each case
#' @return cook_distance Cook's distance for each case
#' @export
#' 
Compute_CaseInflu_GLM <- function(X, Y, lam, epsilon, t_max = 10, class = "logistic", method = "Newton", influence_measure = "FMD"){
  n = dim(X)[1]
  p = dim(X)[2]
  
  ###Input check
  if (method != "Newton" & method != "GD") {
    cat("\n The current version only supports 'GD' and 'Newton' method options.\n")
    return(0)
  }
  
  if (influence_measure != "FMD" & influence_measure != "BDLD") {
    cat("\n The current version only supports 'BDLD' and 'FMD' influence measure options.\n")
    return(0)
  }
  
  if (class == "logistic") {
    ###CVX calculates the full data solution
    beta <- CVXR::Variable(p)
    obj = 1/n*(CVXR::sum_entries(CVXR::logistic(X[Y == -1, ] %*% beta))+CVXR::sum_entries(CVXR::logistic(-X[Y == 1, ] %*% beta)))+lam/2*CVXR::power(CVXR::norm2(beta), 2)
    prob <- CVXR::Problem(Minimize(obj))
    result <- CVXR::solve(prob)
    Beta_0 <- result$getValue(beta)
    .Call(`_Caseinflu_CaseInfluence_logistic`, X, Y, Beta_0, t_max, lam, epsilon, method, influence_measure)
  }
  else if (class == "poisson") {
    ###CVX calculates the full data solution
    beta <- CVXR::Variable(p)
    obj = 1/n*CVXR::sum_entries(exp(X %*% beta) - CVXR::multiply(Y, X %*% beta))+lam/2*CVXR::power(CVXR::norm2(beta), 2)
    prob <- CVXR::Problem(Minimize(obj))
    result <- CVXR::solve(prob)
    Beta_0 <- result$getValue(beta)
    .Call(`_Caseinflu_CaseInfluence_poisson`, X, Y, Beta_0, t_max, lam, epsilon, method)
  }
  else {
    cat("\n The current version only supports 'logistic' and 'poisson' class options.\n")
  }
}

#' LOO solution computation for L2 penalized GLM
#' 
#' @description 
#' Compute the LOO solution based on the one-step Newton method, using the solution at the last grid point as the initialization
#' 
#' @param X A \eqn{n \times p} feature matrix
#' @param Y A \eqn{n \times 1} response vector
#' @param lam Regularization parameter for L2 penalty
#' @param obs_index Index of the observation that is left out 
#' @param epsilon Pre-specified error tolerance of the entire path
#' @param t_max Range of solution path \eqn{[0, t_{\max}]}
#' @param class Specify the problem class with default value "logistic". Use "logistic" for logistic regression, and "poisson" for poisson regression
#' @param method Specify the method for computing the solution path. Default is "Newton". Use "GD" for gradient descent
#' 
#' @return The computed LOO solution
#' @export
#' 
LOO_GLM <- function(X, Y, lam, obs_index, epsilon, t_max = 10, class = "logistic", method = "Newton"){
  n = dim(X)[1]
  p = dim(X)[2]
  L_vec = rep(0, n)
  
  if (class == "logistic") {
    for (i in 1:n)
      L_vec[i] = norm(X[i,], '2') ** 2 / (4*n)
    
    ###CVX calculates the full data solution
    beta <- CVXR::Variable(p)
    obj = 1/n*(CVXR::sum_entries(CVXR::logistic(X[Y == -1, ] %*% beta))+CVXR::sum_entries(CVXR::logistic(-X[Y == 1, ] %*% beta)))+lam/2*CVXR::power(CVXR::norm2(beta), 2)
    prob <- CVXR::Problem(Minimize(obj))
    result <- CVXR::solve(prob)
    Beta_0 <- result$getValue(beta)
    if (method == "Newton") {
      if (n < p) {
        path = .Call(`_Caseinflu_logistic_Newton_high_dimension`, X, Y, Beta_0, t_max, lam, obs_index-1, epsilon, L_vec[obs_index])
      }
      else {
        path = .Call(`_Caseinflu_logistic_Newton_low_dimension`, X, Y, Beta_0, t_max, lam, obs_index-1, epsilon, L_vec[obs_index])
      }
    }
    else if (method == "GD") {
      path = .Call(`_Caseinflu_logistic_GD`, X, Y, Beta_0, t_max, lam, obs_index-1, epsilon, L_vec[obs_index])
    }
    else {
      cat("\n The current version only supports 'GD' and 'Newton' method options.\n")
      return(0)
    }
    ## Run LOO
    .Call(`_Caseinflu_logistic_compute_LOO`, X, Y, path[[1]][, ncol(path[[1]])], lam, obs_index-1)
  }
  else if (class == "poisson") {
    for (i in 1:n)
      L_vec[i] = 3 * norm(X[i,], '2') ** 2 / n
    ###CVX calculates the full data solution
    beta <- CVXR::Variable(p)
    obj = 1/n*CVXR::sum_entries(exp(X %*% beta) - CVXR::multiply(Y, X %*% beta))+lam/2*CVXR::power(CVXR::norm2(beta), 2)
    prob <- CVXR::Problem(Minimize(obj))
    result <- CVXR::solve(prob)
    Beta_0 <- result$getValue(beta)
    L_vec = L_vec * exp(X %*% Beta_0)
    if (method == "Newton") {
      if (n < p) {
        path = .Call(`_Caseinflu_poisson_Newton_high_dimension`, X, Y, Beta_0, t_max, lam, obs_index-1, epsilon, L_vec[obs_index])
      }
      else {
        path = .Call(`_Caseinflu_poisson_Newton_low_dimension`, X, Y, Beta_0, t_max, lam, obs_index-1, epsilon, L_vec[obs_index])
      }
    }
    else if (method == "GD") {
      path = .Call(`_Caseinflu_poisson_GD`, X, Y, Beta_0, t_max, lam, obs_index-1, epsilon, L_vec[obs_index])
    }
    else {
      cat("\n The current version only supports 'GD' and 'Newton' method options.\n")
      return(0)
    }
    ## Run LOO
    .Call(`_Caseinflu_poisson_compute_LOO`, X, Y, path[[1]][, ncol(path[[1]])], lam, obs_index-1)
  }
  else {
    cat("\n The current version only supports 'logistic' and 'poisson' class options.\n")
  }
}

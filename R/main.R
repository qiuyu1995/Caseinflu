#' Case influence assessment for L2 penalized nonsmooth models
#'
#' @description
#' Compute the global influence, which is based on case influence graph, for each case under L2 regularized nonsmooth
#' models, including quantile regression and svm. To compute the case influence graph for each case, we need to solve a solution path problem.
#' In our algorithm, the exact path can be computed and it is piece-wise linear.
#'
#' @param X A n by p feature matrix
#' @param Y A n by 1 response vector
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
  ## For nonsmooth model, the loss function is not normalized by n in the implementation.
  lam = n * lam
  
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
  result <- CVXR::solve(prob, reltol=1e-14)
  beta_0_w0 <- result$getValue(beta_0)
  beta_w0 <- result$getValue(beta)
  theta_w0 <- result$getDualValue(constraints[[3]]) - result$getDualValue(constraints[[4]])
  .Call(`_Caseinflu_CaseInfluence_nonsmooth`, a, B, c, lam, alpha_0, alpha_1, beta_0_w0, beta_w0, theta_w0, class, influence_measure)
}

#' Local influence assessment for L2 penalized nonsmooth models
#'
#' @description
#' Compute the local influence for each case under L2 regularized nonsmooth models, including quantile regression and svm.
#'
#' @param X A n by p feature matrix
#' @param Y A n by 1 response vector
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
  ## For nonsmooth model, the loss function is not normalized by n in the implementation.
  lam = n * lam

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
  result <- CVXR::solve(prob, reltol=1e-14)
  beta_0_w0 <- result$getValue(beta_0)
  beta_w0 <- result$getValue(beta)
  theta_w0 <- result$getDualValue(constraints[[3]]) - result$getDualValue(constraints[[4]])
  .Call(`_Caseinflu_LocalInfluence_nonsmooth`, a, B, c, lam, alpha_0, alpha_1, beta_0_w0, beta_w0, theta_w0, class, influence_measure)
}

#' Case influence assessment for L2 penalized GLM
#'
#' @description
#' Compute the global influence, which is based on case influence graph, for each case under L2 regularized logistic
#' regression. To compute the case influence graph for each case, we need to solve a solution path problem. In our algorithm,
#' the grid points are adaptively selected to save computation while maintaining the accuracy of the entire path. At each grid point,
#' the optimization problem can be solved by either Newton method or gradient descent method.
#'
#' @param X A n by p feature matrix
#' @param Y A n by 1 response vector
#' @param lam Regularization parameter for L2 penalty
#' @param epsilon Pre-specified error tolerance of the entire path
#' @param t_max Range of solution path [0, t_max]
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

#' Local influence assessment for L2 penalized GLM
#'
#' @description
#' Compute the local influence for each case under L2 regularized GLM.
#'
#' @param X A n by p feature matrix
#' @param Y A n by 1 response vector
#' @param lam Regularization parameter for L2 penalty
#' @param class Specify the problem class with default value "quantile". Use "quantile" for quantile regression, and "svm" for support vector machine
#' @param influence_measure Specify the influence measure for computing local influence. For classification problem, "FMD" represents
#' functional margin difference and "BDLD" represents Binomial deviance loss difference. The default is "FMD". For regression problem, no
#' need to specify the influence_measure
#'
#' @return LocalInfluence_vec Local influence for each case
#' @export
#'
Compute_LocalInflu_GLM <- function(X, Y, lam, class = "logistic", influence_measure = "FMD"){
  n = dim(X)[1]
  p = dim(X)[2]

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
    .Call(`_Caseinflu_LocalInfluence_logistic`, X, Y, Beta_0, lam, influence_measure)
  }
  else if (class == "poisson") {
    ###CVX calculates the full data solution
    beta <- CVXR::Variable(p)
    obj = 1/n*CVXR::sum_entries(exp(X %*% beta) - CVXR::multiply(Y, X %*% beta))+lam/2*CVXR::power(CVXR::norm2(beta), 2)
    prob <- CVXR::Problem(Minimize(obj))
    result <- CVXR::solve(prob)
    Beta_0 <- result$getValue(beta)
    .Call(`_Caseinflu_LocalInfluence_poisson`, X, Y, Beta_0, lam)
  }
  else {
    cat("\n The current version only supports 'logistic' and 'poisson' class options.\n")
  }
}

#' @title Get an xgboost Search Space and Learner
#'
#' @description
#' An xgboost Learner and ParamSet is created that emulates autoxgboost.
#'
#' @param task\cr
#'   The task.
#' @param par.set\cr
#'   Which paramset to use. either \dQuote{unmixed} (default) or \dQuote{mixed} (larger parameter set) but with param dependencies.
#' @param predict.type\cr
#'   Predict type. One of \dQuote{prob}, \dQuote{response} (default). Only \dQuote{response} supported for regression tasks.
#' @param max.nrounds\cr
#'   Maximum number of allowed boosting iterations. Default is \code{10^6}.
#' @param early.stopping.rounds\cr
#'   After how many iterations without an improvement in the boosting OOB error should be stopped?
#'   Default is \code{10}.
#' @param early.stopping.fraction\cr
#'   What fraction of the data should be used for early stopping (i.e. as a validation set).
#'   Default is \code{4/5}.
#' @param impact.encoding.boundary\cr
#'   Defines the threshold on how factor variables are handled. Factors with more levels than the \code{"impact.encoding.boundary"} get impact encoded while factor variables with less or equal levels than the \code{"impact.encoding.boundary"} get dummy encoded.
#'   For \code{impact.encoding.boundary = 0L}, all factor variables get impact encoded while for \code{impact.encoding.boundary = .Machine$integer.max}, all of them get dummy encoded.
#'   Default is \code{10}.
#' @param nthread\cr
#'   Number of cores to use.
#'   If \code{NULL} (default), xgboost will determine internally how many cores to use.
#' @param tune.threshold\cr
#'   Should thresholds be tuned? This has only an effect for classification, see \code{\link[mlr]{tuneThreshold}}.
#'   Default is \code{TRUE}. Only \code{FALSE} is supported currently; if this is \code{TRUE} an error is thrown.
#' @param emulate_exactly\cr
#'   Whether to emulate autoxgboost behaviour for impact.encoding.boundary. Autoxgboost applies the boundary to the *whole* task (behaviour if emulate_exactly==TRUE,
#'   while a more exact approach would be to apply it only to the training set (emulate_exactly==FALSE).
#' @return \code{something}.
#' @export
autoxgboost_space = function(task, par.set = "unmixed", predict.type = "response",
  max.nrounds = 1e6, early.stopping.rounds = 10, early.stopping.fraction = 4/5,
  impact.encoding.boundary = 10, nthread = NULL, tune.threshold = TRUE, emulate_exactly = TRUE) {

  # check inputs
  assertClass(task, "TaskSupervised")
  assertChoice(par.set, c("unmixed", "mixed"))
  assertChoice(predict.type, c("response", "prob"))
  assertIntegerish(max.nrounds, lower = 1L, len = 1L)
  assertIntegerish(early.stopping.rounds, lower = 1L, len = 1L)
  assertNumeric(early.stopping.fraction, lower = 0, upper = 1, len = 1L)
  assertIntegerish(impact.encoding.boundary, lower = 0, len = 1L)
  assertIntegerish(nthread, lower = 1, len = 1L, null.ok = TRUE)
  assertFlag(tune.threshold)
  assertFlag(emulate_exactly)
  if (tune.threshold) {
    stop("threshold tuning not yet supported")
  }

  autoxgbparset = ParamSet$new(list(
    ParamDbl$new("xgboost.eta", 0.01, 0.2),
    ParamDbl$new("xgboost.gamma", -7, 6),
    ParamInt$new("xgboost.max_depth", 3, 20),
    ParamDbl$new("xgboost.colsample_bytree", 0.5, 1),
    ParamDbl$new("xgboost.colsample_bylevel", 0.5, 1),
    ParamDbl$new("xgboost.lambda", -10, 10),
    ParamDbl$new("xgboost.alpha", -10, 10),
    ParamDbl$new("xgboost.subsample", 0.5, 1)
  ))

  autoxgbparset.mixed = autoxgbparset$clone(deep = TRUE)

  lapply(list(
    ParamFct$new("xgboost.booster", c("gbtree", "gblinear", "dart")),
    ParamFct$new("xgboost.sampler_type", c("uniform", "weighted")),
    ParamFct$new("xgboost.normalize_type", c("tree", "forest")),
    ParamDbl$new("xgboost.rate_drop", 0, 1),
    ParamDbl$new("xgboost.skip_drop", 0, 1),
    ParamLgl$new("xgboost.one_drop"),
    ParamFct$new("xgboost.grow_policy", c("depthwise", "lossguide")),
    ParamInt$new("xgboost.max_leaves", 0, 8),
    ParamInt$new("xgboost.max_bin", 2, 9)
  ), autoxgbparset.mixed$add)

  autoxgbparset.mixed$
    add_dep("xgboost.sampler_type", "xgboost.booster", CondEqual$new("dart"))$
    add_dep("xgboost.normalize_type", "xgboost.booster", CondEqual$new("dart"))$
    add_dep("xgboost.rate_drop", "xgboost.booster", CondEqual$new("dart"))$
    add_dep("xgboost.skip_drop", "xgboost.booster", CondEqual$new("dart"))$
    add_dep("xgboost.one_drop", "xgboost.booster", CondEqual$new("dart"))$
    add_dep("xgboost.max_leaves", "xgboost.grow_policy", CondEqual$new("lossguide"))

  parset = switch(par.set,
      mixed = autoxgbparset.mixed,
      unmixed = autoxgbparset)

  parset$trafo = function(x, param_set) {
    if (!is.null(x$xgboost.max_leaves)) {
      x$xgboost.max_leaves = 2^x$xgboost.max_leaves
    }
    if (!is.null(x$xgboost.max_bin)) {
      x$xgboost.max_bin = 2^x$xgboost.max_bin
    }
    if (!is.null(x$xgboost.scale_pos_weight)) {
      x$xgboost.scale_pos_weight = 2^x$xgboost.scale_pos_weight
    }
    x$xgboost.gamma = 2^x$xgboost.gamma
    x$xgboost.lambda = 2^x$xgboost.lambda
    x$xgboost.alpha = 2^x$xgboost.alpha
    x
  }

  if (task$task_type == "classif") {
    if ("twoclass" %in% task$properties) {
      parset$add_param(ParamDbl$new("xgboost.scale_pos_weight", -10, 10))
      objective = "binary:logistic"
      eval_metric = "error"
    } else {
      objective = "multi:softprob"
      eval_metric = "merror"
    }
    xgblearner = lrn("classif.xgboost", eval_metric = eval_metric, objective = objective,
      early_stopping_rounds = early.stopping.rounds, nrounds = max.nrounds, predict_type = predict.type)
  } else if (task$task_type == "regr") {
    if (predict.type != "response") {
      stop("Only response prediction supported for regression tasks")
    }
    objective = "reg:linear"
    eval_metric = "rmse"
    xgblearner = lrn("regr.xgboost", eval_metric = eval_metric, objective = objective,
      early_stopping_rounds = early.stopping.rounds, nrounds = max.nrounds, predict_type = predict.type)
  } else {
    stopf("Unsupported task type %s", task$task_type)
  }
  xgblearner$id = "xgboost"
  xgblearner$param_set$values$nthread = nthread

  # autoxgboost is a bit weird:
  #  - optimization happens with early stopping
  #  - the final model uses the exact same stopping from the best tuning eval
  #  I'm not sure we can emulate that.
  # Also autoxgboost does threshold tuning on the test set FFS


  # two ways to do this: impact.encoding.boundary *inside* the graph -> correct methods
  #                      impact.encoding.boundary *outside* the graph -> cheating but exact emulation

# all cols with levels <= 'impact.encoding.boundary', on training or whole task depending on emulate_exactly
  if (emulate_exactly) {
    encodetarget = selector_name(selector_less_or_equal_levels(impact.encoding.boundary)(task))
  } else {
    encodetarget = selector_less_or_equal_levels(impact.encoding.boundary)
  }

  graph = po("fixfactors") %>>%
    po("encode", affect_columns = encodetarget) %>>%
    po("encodeimpact") %>>%
    po("removeconstants") %>>%
    xgblearner
  list(learner = GraphLearner$new(graph, predict_type = predict.type), searchspace = parset)
}

selector_less_or_equal_levels = function(levels) {
  assert_int(levels)
  mlr3pipelines:::make_selector(function(task) {
    levlens = sapply(task$clone(deep = TRUE)$droplevels()$levels(), length)
    names(levlens[levlens <= levels])
  }, "selector_less_or_equal_levels(%s)", levels)
}





devtools::load_all("mlr3pipelines")

data.table::setDTthreads(1)
devtools::document("autoxgboost3")

library("data.table")
devtools::load_all("autoxgboost3")


# ------------------
# seb's error

library("mlr3tuning")
task <- TaskClassif$new("abalone", readRDS("seb_error_task.rds")$data(), "Class")

axgb_settings = autoxgboost_space(task, tune.threshold = FALSE)
rsmp_inner = axgb_settings$resampling
learner = axgb_settings$learner
ps = axgb_settings$searchspace
ti = TuningInstance$new(task = task, learner = learner, resampling = rsmp_inner, param_set = ps, measures = msr("classif.ce"), terminator = term("evals", n_evals = 100))
tuner = tnr("random_search")
tuner$tune(ti)



# ------------------

library("mlr3")

task = tsk("iris")

space <- autoxgboost_space(task, tune.threshold = FALSE)

space$learner$param_set
names(space$searchspace)

generate_design_random(space$searchspace, 10)$transpose()
suggestion <- generate_design_random(space$searchspace.mixed, 1)$transpose()[[1]]


space$learner$param_set$values <- mlr3misc::insert_named(space$learner$param_set$values, suggestion)

resample(task, space$learner, space$resampling)

for (setup in mlr3misc::transpose_list(CJ(
  task = list("iris", "boston_housing", "pima"),
  predict.type = c("response", "prob"),
  impact.encoding.boundary = c(10, 1000000),
  tune.threshold = FALSE,
  emulate_exactly = c(FALSE, TRUE)))) {
  setup$task = tsk(setup$task)
  if (setup$predict.type == "prob" && setup$task$task_type != "classif") {
    next
  }
  space <- do.call(autoxgboost_space, setup)
  designs <- c(generate_design_random(space$searchspace, 10)$transpose(),
    generate_design_random(space$searchspace.mixed, 10)$transpose())
  for (sug in designs) {
    lrn = space$learner$clone(deep = TRUE)
    lrn$param_set$values <- mlr3misc::insert_named(lrn$param_set$values, sug)
    resample(setup$task, space$learner, space$resampling)
  }
}

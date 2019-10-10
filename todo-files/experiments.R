

devtools::load_all("mlr3pipelines")

data.table::setDTthreads(1)
devtools::document("autoxgboost3")

library("data.table")
devtools::load_all("autoxgboost3")


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
  space <- do.call(autoxgboost_space, setup)
  designs <- c(generate_design_random(space$searchspace, 10)$transpose(),
    generate_design_random(space$searchspace.mixed, 10)$transpose())
  for (sug in designs) {
    lrn = space$learner$clone(deep = TRUE)
    lrn$param_set$values <- mlr3misc::insert_named(lrn$param_set$values, sug)
    resample(setup$task, space$learner, space$resampling)
  }
}
